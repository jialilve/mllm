// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/passes/Qwen3IRGraphFusionPass.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/graph/SubGraphCloneUtils.hpp"
#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

using mllm::DeviceTypes;

namespace {

using ir::val_ptr_t;

std::vector<val_ptr_t> collectOpOutputVals(const ir::op_ptr_t& op) {
  std::vector<val_ptr_t> results;
  if (!op) { return results; }
  for (auto& weak_node : op->outputs()) {
    auto* node = weak_node.get_weak();
    if (!node || !node->isa_<ir::Val>()) { continue; }
    auto val = std::static_pointer_cast<ir::Val>(node->shared_from_this());
    results.push_back(val);
  }
  return results;
}

std::vector<val_ptr_t> collectOpInputVals(const ir::op_ptr_t& op) {
  std::vector<val_ptr_t> results;
  if (!op) { return results; }
  for (auto& weak_node : op->inputs()) {
    auto* node = weak_node.get_weak();
    if (!node || !node->isa_<ir::Val>()) { continue; }
    auto val = std::static_pointer_cast<ir::Val>(node->shared_from_this());
    results.push_back(val);
  }
  return results;
}

ir::cf::ReturnOp::ptr_t getReturnOp(const ir::graph::SubGraphOp::ptr_t& subgraph) {
  if (!subgraph) { return nullptr; }
  auto region = subgraph->getTopRegion();
  if (!region) { return nullptr; }
  for (auto& op : region->ops()) {
    if (op->isa_<ir::cf::ReturnOp>()) {
      return op->cast_<ir::cf::ReturnOp>();
    }
  }
  return nullptr;
}

val_ptr_t cloneTensorValueLike(const ir::IRContext::ptr_t& ctx, const val_ptr_t& src_val) {
  if (!ctx || !src_val) { return nullptr; }
  auto tensor_val = std::dynamic_pointer_cast<ir::tensor::TensorValue>(src_val);
  if (!tensor_val) { return nullptr; }
  auto cloned = ctx->createTemporaryValue<ir::tensor::TensorValue>(tensor_val->tensor());
  cloned->setDevice(tensor_val->getDevice());
  for (const auto& [attr_name, attr_value] : tensor_val->getAttrs()) {
    cloned->setAttr(attr_name, attr_value);
  }
  return cloned;
}

void replaceValueConsumers(const val_ptr_t& old_val, const val_ptr_t& new_val) {
  if (!old_val || !new_val || old_val == new_val) { return; }
  auto consumers = old_val->outputs();
  for (auto& weak_consumer : consumers) {
    auto* consumer_node = weak_consumer.get_weak();
    if (!consumer_node) { continue; }
    auto consumer_ptr = consumer_node->shared_from_this();
    auto& inputs = consumer_ptr->inputs();
    for (auto& weak_input : inputs) {
      if (weak_input.get_weak() == old_val.get()) {
        weak_input = new_val;
        new_val->outputs().emplace_back(consumer_ptr);
      }
    }
  }
  old_val->outputs().clear();
}

ir::graph::SubGraphOp::ptr_t findSubGraph(const ir::Region::ptr_t& region, const std::string& symbol) {
  if (!region) { return nullptr; }
  for (auto& op : region->ops()) {
    auto sub_graph = std::dynamic_pointer_cast<ir::graph::SubGraphOp>(op);
    if (!sub_graph) { continue; }
    auto symbol_attr = sub_graph->getSymbolAttr();
    if (symbol_attr && symbol_attr->str() == symbol) {
      return sub_graph;
    }
  }
  return nullptr;
}

std::vector<ir::graph::CallGraphOp::ptr_t> findCallGraphOps(const ir::Region::ptr_t& region,
                                                            const std::string& symbol) {
  std::vector<ir::graph::CallGraphOp::ptr_t> result;
  if (!region) { return result; }
  for (auto& op : region->ops()) {
    if (auto call_op = std::dynamic_pointer_cast<ir::graph::CallGraphOp>(op)) {
      auto sym = call_op->getSymbolAttr();
      if (sym && sym->str() == symbol) {
        result.push_back(call_op);
      }
    }
  }
  return result;
}

}  // namespace

bool Qwen3IRGraphFusionPass::parseDecoderLayerInfo(const std::string& symbol,
                                                   int& layer_index,
                                                   int& segment_index) {
  // Qwen3 NPU 的 QNN SubGraph 命名规则：
  // - "model.layers.X_1" → segment 1 (proj 段：q/k/v_proj + dequantize + RMSNorm)
  // - "model.layers.X_2" → segment 2 (out+MLP 段：o_proj + MLP + LayerNorm)
  //
  // 解析规则：
  // - 从 "model.layers." 之后读出数字 X 作为 layer_index
  // - 检查后缀是否为 "_1" 或 "_2" 来判定 segment_index
  const std::string prefix = "model.layers.";
  auto pos = symbol.find(prefix);
  if (pos == std::string::npos) { return false; }

  pos += prefix.size();
  // 读取层号
  size_t end_pos = pos;
  while (end_pos < symbol.size() && std::isdigit(static_cast<unsigned char>(symbol[end_pos]))) { end_pos++; }
  if (end_pos == pos) { return false; }

  try {
    layer_index = std::stoi(symbol.substr(pos, end_pos - pos));
  } catch (...) {
    return false;
  }

  // 检查后缀是否为 "_1" 或 "_2"
  const std::string rest = symbol.substr(end_pos);
  if (rest == "_1") {
    segment_index = 1;  // proj 段
    return true;
  }
  if (rest == "_2") {
    segment_index = 2;  // out+MLP 段
    return true;
  }

  // 其它子图暂时不参与 Qwen3 decoder 融合
  return false;
}

uint8_t Qwen3IRGraphFusionPass::run(const ir::node_ptr_t& op) {
  // 只在 ModuleOp 上工作
  if (!op->isa_<ir::ModuleOp>()) {
    MLLM_ERROR("Qwen3IRGraphFusionPass expects ModuleOp as top level op");
    return ir::PASS_RET_FAILURE;
  }

  auto module_op = op->cast_<ir::ModuleOp>();
  auto top_region = module_op->getTopRegion();
  if (!top_region) {
    MLLM_ERROR("ModuleOp has no top region");
    return ir::PASS_RET_FAILURE;
  }

  auto ir_ctx = getCtx();
  if (!ir_ctx) {
    MLLM_ERROR("Qwen3IRGraphFusionPass: IRContext is null");
    return ir::PASS_RET_FAILURE;
  }

  struct Qwen3GraphInfo {
    ir::graph::SubGraphOp::ptr_t sub_graph;
    int layer_index = -1;
    int segment_index = -1;  // 1=proj, 2=out_mlp
  };

  std::vector<Qwen3GraphInfo> candidates;

  // 遍历顶层所有 SubGraphOp，只挑 QNN 设备且符号名符合 Qwen3 decoder pattern 的
  for (auto& region_op : top_region->ops()) {
    auto sub_graph_op = std::dynamic_pointer_cast<ir::graph::SubGraphOp>(region_op);
    if (!sub_graph_op) { continue; }

    // 只处理 device == kQNN 的子图
    if (sub_graph_op->getDevice() != DeviceTypes::kQNN) { continue; }

    auto symbol_attr = sub_graph_op->getSymbolAttr();
    if (!symbol_attr) { continue; }

    const std::string symbol_name = symbol_attr->str();
    int layer_idx = -1;
    int seg_idx = -1;
    if (!parseDecoderLayerInfo(symbol_name, layer_idx, seg_idx)) { continue; }

    Qwen3GraphInfo info;
    info.sub_graph = sub_graph_op;
    info.layer_index = layer_idx;
    info.segment_index = seg_idx;
    candidates.emplace_back(std::move(info));
  }

  if (candidates.empty()) {
    // 对非 Qwen3 NPU 模型，这个 Pass 是 no-op
    return ir::PASS_RET_SUCCESS;
  }

  // 标记所有识别的 SubGraph
  for (auto& g : candidates) {
    auto symbol_attr = g.sub_graph->getSymbolAttr();
    const std::string symbol_name = symbol_attr ? symbol_attr->str() : "<unnamed>";

    // 标记属性：方便在 .mir / debug 中确认哪些 graph 被识别为 Qwen3 decoder 段
    auto layer_attr = ir_ctx->create<ir::IntAttr>(g.layer_index);
    auto seg_attr = ir_ctx->create<ir::IntAttr>(g.segment_index);
    g.sub_graph->setAttr("qwen3_decoder_layer", layer_attr);
    g.sub_graph->setAttr("qwen3_decoder_segment", seg_attr);

    // MLLM_INFO("Qwen3IRGraphFusionPass: detect QNN subgraph '{}' (layer={}, segment={})", symbol_name, g.layer_index,
    //           g.segment_index);
  }

  // 按 layer_index 和 segment_index 排序，方便后续处理
  std::sort(candidates.begin(), candidates.end(),
            [](const Qwen3GraphInfo& a, const Qwen3GraphInfo& b) {
              if (a.layer_index != b.layer_index) return a.layer_index < b.layer_index;
              return a.segment_index < b.segment_index;
            });

  // 识别需要融合的 Graph 对：layer X 的 seg2 和 layer X+1 的 seg1
  // 注意：保存 sub_graph 的 shared_ptr 而不是指针，避免 candidates 移动导致指针失效
  struct FusionPair {
    ir::graph::SubGraphOp::ptr_t graph1;  // layer X, seg2
    ir::graph::SubGraphOp::ptr_t graph2;  // layer X+1, seg1
    int layer_x;
    std::string graph1_name;
    std::string graph2_name;
  };
  std::vector<FusionPair> fusion_pairs;

  struct FusionProduct {
    FusionPair pair;
    ir::graph::SubGraphOp::ptr_t fused_graph;
    std::vector<val_ptr_t> fused_outputs;
  };
  std::vector<FusionProduct> fusion_products;
  // 需要保留符号的集合：当我们让 fused graph 复用 graph1 的名字时，不能再把这个符号从符号表移除
  std::unordered_set<std::string> preserved_symbols;

  for (size_t i = 0; i < candidates.size() - 1; ++i) {
    auto& curr = candidates[i];
    auto& next = candidates[i + 1];

    // 检查是否符合融合条件：当前是 layer X 的 seg2，下一个是 layer X+1 的 seg1
    if (curr.segment_index == 2 && next.segment_index == 1 &&
        next.layer_index == curr.layer_index + 1) {
      // 提前获取符号名，避免后续访问时崩溃
      const std::string graph1_name = curr.sub_graph->getSymbolAttr()->str();
      const std::string graph2_name = next.sub_graph->getSymbolAttr()->str();
      
      FusionPair pair;
      pair.graph1 = curr.sub_graph;  // 保存 shared_ptr，而不是指针
      pair.graph2 = next.sub_graph;  // 保存 shared_ptr，而不是指针
      pair.layer_x = curr.layer_index;
      pair.graph1_name = graph1_name;
      pair.graph2_name = graph2_name;
      fusion_pairs.emplace_back(pair);

      MLLM_INFO("Qwen3IRGraphFusionPass: plan to fuse '{}' (layer={}, seg=2) with '{}' (layer={}, seg=1)",
                graph1_name, curr.layer_index, graph2_name, next.layer_index);
    }
  }

  MLLM_INFO("Qwen3IRGraphFusionPass: identified {} fusion pairs, will reduce graph count from {} to approximately {}",
            fusion_pairs.size(), candidates.size(), candidates.size() - fusion_pairs.size());

  // 第 1 步：基于 clone/inine 的方式，为每个融合对创建新的 fused Graph（暂未接入调用处）
  for (auto& pair : fusion_pairs) {
    auto graph1 = pair.graph1;  // model.layers.X_2
    auto graph2 = pair.graph2;  // model.layers.(X+1)_1
    auto graph1_region = graph1->getTopRegion();
    auto graph2_region = graph2->getTopRegion();
    if (!graph1_region || !graph2_region) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: missing region when preparing fused graph for '{}' and '{}'",
                 pair.graph1_name, pair.graph2_name);
      return ir::PASS_RET_FAILURE;
    }

    ir::graph::SubGraphCloneUtils::CloneOptions options;
    // 直接复用原 seg2 符号，避免额外改名
    options.new_symbol_name = pair.graph1_name;

    std::unordered_map<ir::Val*, ir::val_ptr_t> cloned_value_map;
    auto fused_graph = ir::graph::SubGraphCloneUtils::cloneSubGraph(ir_ctx, graph1, options, &cloned_value_map);
    if (!fused_graph) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: failed to clone '{}'", pair.graph1_name);
      return ir::PASS_RET_FAILURE;
    }
    // 继承原 graph1 的设备 / 抽象节点信息，避免变成 <notype> 导致 QNN 编译/运行异常
    fused_graph->setDevice(graph1->getDevice());
    fused_graph->abstract_nn_node_ = graph1->abstract_nn_node_;

    // 用原符号注册 fused graph，保留符号以兼容旧的 CallGraphOp
    ir_ctx->removeFromSymbolTable(pair.graph1_name);
    ir_ctx->addToSymbolTable(fused_graph, options.new_symbol_name);
    preserved_symbols.insert(pair.graph1_name);

    // 将 fused graph 插入到顶层 Region，放在原 graph1 之后，便于后续 Pass 调试
    auto& top_ops = top_region->ops();
    auto graph1_it = std::find(top_ops.begin(), top_ops.end(), graph1);
    fused_graph->setBelongsTo(top_region->belongsTo());
    if (graph1_it != top_ops.end()) {
      top_ops.insert(std::next(graph1_it), fused_graph);
    } else {
      top_ops.push_back(fused_graph);
    }

    auto fused_region = fused_graph->getTopRegion();
    if (!fused_region) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: fused graph '{}' has no region", options.new_symbol_name);
      return ir::PASS_RET_FAILURE;
    }

    // 移除 cloneSubGraph 自动生成的 ReturnOp，后续会重新构建
    ir::cf::ReturnOp::ptr_t cloned_return = nullptr;
    for (auto& op : fused_region->ops()) {
      if (op->isa_<ir::cf::ReturnOp>()) {
        cloned_return = op->cast_<ir::cf::ReturnOp>();
        break;
      }
    }
    if (!cloned_return) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: cloned graph '{}' has no ReturnOp", options.new_symbol_name);
      return ir::PASS_RET_FAILURE;
    }
    {
      ir::IRWriterGuard guard(ir_ctx, fused_region);
      ir::IRWriter writer(ir_ctx, fused_region);
      writer.removeOp(cloned_return);
    }

    // 记录 graph1 的输出在 fused graph 中的对应值
    std::vector<ir::val_ptr_t> graph1_outputs_cloned;
    graph1_outputs_cloned.reserve(graph1_region->outputs().size());
    for (auto& old_output : graph1_region->outputs()) {
      auto it = cloned_value_map.find(old_output.get());
      if (it == cloned_value_map.end()) {
        MLLM_ERROR("Qwen3IRGraphFusionPass: missing clone map entry for '{}' output '{}'",
                   pair.graph1_name, old_output ? old_output->name() : "<null>");
        return ir::PASS_RET_FAILURE;
      }
      graph1_outputs_cloned.push_back(it->second);
    }

    // 计算 graph2 输入在 fused graph 中应该对应到哪些 Value（由 graph1 输出生成）
    if (graph2_region->inputs().size() != graph1_region->outputs().size()) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: '{}' output count ({}) != '{}' input count ({})",
                 pair.graph1_name, graph1_region->outputs().size(),
                 pair.graph2_name, graph2_region->inputs().size());
      return ir::PASS_RET_FAILURE;
    }

    std::unordered_map<ir::Val*, ir::val_ptr_t> graph2_input_bindings;
    auto out_it = graph1_region->outputs().begin();
    auto in_it = graph2_region->inputs().begin();
    for (; out_it != graph1_region->outputs().end() && in_it != graph2_region->inputs().end(); ++out_it, ++in_it) {
      auto mapped = cloned_value_map.find(out_it->get());
      if (mapped == cloned_value_map.end()) {
        MLLM_ERROR("Qwen3IRGraphFusionPass: missing cloned value for '{}'", (*out_it)->name());
        return ir::PASS_RET_FAILURE;
      }
      graph2_input_bindings[in_it->get()] = mapped->second;
    }

    std::vector<ir::val_ptr_t> graph2_outputs_cloned;
    if (!ir::graph::SubGraphCloneUtils::inlineSubGraphInto(ir_ctx, graph2, fused_graph,
                                                           cloned_value_map, graph2_input_bindings,
                                                           &graph2_outputs_cloned)) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: inline '{}' body into '{}' failed",
                 pair.graph2_name, options.new_symbol_name);
      return ir::PASS_RET_FAILURE;
    }

    // 严格校验 fused 子图的输入 / 输出数量是否符合预期
    const auto expected_inputs = graph1_region->inputs().size();
    const auto fused_inputs = fused_region->inputs().size();
    if (expected_inputs != fused_inputs) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: fused graph '{}' input count mismatch (expected {}, got {})",
                 options.new_symbol_name, expected_inputs, fused_inputs);
      return ir::PASS_RET_FAILURE;
    }

    // 新建 ReturnOp，输出：graph1 的 hidden + graph2 的 Q/K/V
    std::vector<ir::val_ptr_t> fused_outputs = graph1_outputs_cloned;
    fused_outputs.insert(fused_outputs.end(), graph2_outputs_cloned.begin(), graph2_outputs_cloned.end());
    const auto expected_outputs = fused_outputs.size();
    if (expected_outputs != graph1_outputs_cloned.size() + graph2_outputs_cloned.size()) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: fused outputs size mismatch for '{}'", options.new_symbol_name);
      return ir::PASS_RET_FAILURE;
    }
    {
      ir::IRWriterGuard guard(ir_ctx, fused_region);
      ir::IRWriter writer(ir_ctx, fused_region);
      writer.create<ir::cf::ReturnOp>(fused_outputs);
    }

    fused_region->outputs().clear();
    for (auto& output_val : fused_outputs) {
      fused_region->outputs().push_back(output_val);
      (*fused_graph)-- > output_val;
    }

    std::vector<val_ptr_t> fused_outputs_vec(fused_outputs.begin(), fused_outputs.end());

    FusionProduct product;
    product.pair = pair;
    product.fused_graph = fused_graph;
    product.fused_outputs = fused_outputs_vec;
    fusion_products.emplace_back(std::move(product));

    MLLM_INFO("Qwen3IRGraphFusionPass: created fused SubGraph '{}' from '{}' + '{}'",
              options.new_symbol_name, pair.graph1_name, pair.graph2_name);
  }

  if (fusion_products.empty()) {
    return ir::PASS_RET_SUCCESS;
  }

  auto model_subgraph = findSubGraph(top_region, "model");
  if (!model_subgraph) {
    MLLM_ERROR("Qwen3IRGraphFusionPass: cannot find @model CPU wrapper for rewiring");
    return ir::PASS_RET_FAILURE;
  }
  auto model_region = model_subgraph->getTopRegion();
  if (!model_region) {
    MLLM_ERROR("Qwen3IRGraphFusionPass: @model subgraph missing region");
    return ir::PASS_RET_FAILURE;
  }

  std::vector<ir::graph::SubGraphOp::ptr_t> graphs_to_remove;
  auto rewire_single_pair = [&](FusionProduct& product) -> bool {
    const auto& pair = product.pair;
    auto fused_graph = product.fused_graph;
    if (!fused_graph) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: fused graph missing for layer {}", pair.layer_x);
      return false;
    }

    const std::string cpu_layer_symbol = "model.layers." + std::to_string(pair.layer_x);
    const std::string cpu_layer_next_symbol = "model.layers." + std::to_string(pair.layer_x + 1);

    auto cpu_layer_x = findSubGraph(top_region, cpu_layer_symbol);
    auto cpu_layer_next = findSubGraph(top_region, cpu_layer_next_symbol);
    if (!cpu_layer_x || !cpu_layer_next) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: missing CPU layer '{}' or '{}'",
                 cpu_layer_symbol, cpu_layer_next_symbol);
      return false;
    }

    auto cpu_region_x = cpu_layer_x->getTopRegion();
    auto cpu_region_next = cpu_layer_next->getTopRegion();
    if (!cpu_region_x || !cpu_region_next) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: CPU layer '{}' or '{}' missing region",
                 cpu_layer_symbol, cpu_layer_next_symbol);
      return false;
    }

    auto callers_graph1 = findCallGraphOps(cpu_region_x, pair.graph1_name);
    if (callers_graph1.size() != 1) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: expect 1 call to '{}' inside '{}', got {}",
                 pair.graph1_name, cpu_layer_symbol, callers_graph1.size());
      return false;
    }
    auto& fused_call = callers_graph1.front();
    {
      ir::IRWriterGuard guard(ir_ctx, cpu_region_x);
      ir::IRWriter writer(ir_ctx, cpu_region_x);
      fused_call->setSymbolAttr(fused_graph->getSymbolAttr());
    }

    auto call_outputs = collectOpOutputVals(fused_call);
    if (call_outputs.empty()) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: call to '{}' in '{}' has no outputs",
                 pair.graph1_name, cpu_layer_symbol);
      return false;
    }

    while (call_outputs.size() < product.fused_outputs.size()) {
      const auto& template_val = product.fused_outputs[call_outputs.size()];
      auto new_val = cloneTensorValueLike(ir_ctx, template_val);
      if (!new_val) {
        MLLM_ERROR("Qwen3IRGraphFusionPass: failed to create call output placeholder");
        return false;
      }
      (*fused_call)-- > new_val;
      call_outputs.push_back(new_val);
    }

    auto cpu_return_x = getReturnOp(cpu_layer_x);
    if (!cpu_return_x) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: CPU layer '{}' has no ReturnOp", cpu_layer_symbol);
      return false;
    }
    auto& cpu_outputs_list = cpu_region_x->outputs();
    auto& cpu_ret_inputs = cpu_return_x->inputs();

    auto ensure_cpu_output = [&](const val_ptr_t& val) {
      auto exists =
          std::find(cpu_outputs_list.begin(), cpu_outputs_list.end(), val) != cpu_outputs_list.end();
      if (!exists) {
        cpu_outputs_list.emplace_back(val);
        (*cpu_layer_x)-- > val;
        cpu_ret_inputs.emplace_back(val);
      }
    };

    for (size_t idx = 1; idx < call_outputs.size(); ++idx) {
      ensure_cpu_output(call_outputs[idx]);
    }

    // Add new inputs to CPU layer X+1
    std::vector<val_ptr_t> added_inputs;
    for (size_t idx = 1; idx < product.fused_outputs.size(); ++idx) {
      const auto& template_val = product.fused_outputs[idx];
      auto new_input = cloneTensorValueLike(ir_ctx, template_val);
      if (!new_input) {
        MLLM_ERROR("Qwen3IRGraphFusionPass: failed to create new input for '{}'",
                   cpu_layer_next_symbol);
        return false;
      }
      cpu_region_next->inputs().push_back(new_input);
      (*new_input)-- > cpu_layer_next;
      added_inputs.push_back(new_input);
    }

    auto callers_graph2 = findCallGraphOps(cpu_region_next, pair.graph2_name);
    if (callers_graph2.size() != 1) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: expect 1 call to '{}' inside '{}', got {}",
                 pair.graph2_name, cpu_layer_next_symbol, callers_graph2.size());
      return false;
    }
    auto& call_graph2 = callers_graph2.front();
    auto graph2_outputs = collectOpOutputVals(call_graph2);
    if (graph2_outputs.size() != added_inputs.size()) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: '{}' output count {} mismatch new input count {}",
                 pair.graph2_name, graph2_outputs.size(), added_inputs.size());
      return false;
    }
    for (size_t idx = 0; idx < graph2_outputs.size(); ++idx) {
      replaceValueConsumers(graph2_outputs[idx], added_inputs[idx]);
    }
    {
      ir::IRWriterGuard guard(ir_ctx, cpu_region_next);
      ir::IRWriter writer(ir_ctx, cpu_region_next);
      writer.removeOp(call_graph2);
    }

    // Rewire top-level model
    auto cpu_outputs_vector = std::vector<val_ptr_t>(cpu_outputs_list.begin(), cpu_outputs_list.end());
    auto model_call_x = findCallGraphOps(model_region, cpu_layer_symbol);
    if (model_call_x.size() != 1) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: expect 1 model call to '{}', got {}",
                 cpu_layer_symbol, model_call_x.size());
      return false;
    }
    auto& call_model_layer_x = model_call_x.front();
    auto model_call_outputs = collectOpOutputVals(call_model_layer_x);
    if (model_call_outputs.empty()) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: model call to '{}' has no outputs", cpu_layer_symbol);
      return false;
    }
    while (model_call_outputs.size() < cpu_outputs_vector.size()) {
      const auto& template_val = cpu_outputs_vector[model_call_outputs.size()];
      auto new_val = cloneTensorValueLike(ir_ctx, template_val);
      if (!new_val) {
        MLLM_ERROR("Qwen3IRGraphFusionPass: failed to extend outputs for model call '{}'",
                   cpu_layer_symbol);
        return false;
      }
      (*call_model_layer_x)-- > new_val;
      model_call_outputs.push_back(new_val);
    }

    std::vector<val_ptr_t> forwarded_qkv;
    for (size_t idx = 1; idx < model_call_outputs.size(); ++idx) {
      forwarded_qkv.push_back(model_call_outputs[idx]);
    }

    auto model_call_next = findCallGraphOps(model_region, cpu_layer_next_symbol);
    if (model_call_next.size() != 1) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: expect 1 model call to '{}', got {}",
                 cpu_layer_next_symbol, model_call_next.size());
      return false;
    }
    auto& call_model_layer_next = model_call_next.front();
    auto& call_inputs = call_model_layer_next->inputs();
    for (auto& val : forwarded_qkv) {
      call_inputs.emplace_back(val);
    }

    auto add_graph_to_remove = [&](const ir::graph::SubGraphOp::ptr_t& graph) {
      if (!graph) { return; }
      if (std::find(graphs_to_remove.begin(), graphs_to_remove.end(), graph) == graphs_to_remove.end()) {
        graphs_to_remove.emplace_back(graph);
      }
    };
    add_graph_to_remove(pair.graph1);
    add_graph_to_remove(pair.graph2);

    // 同步 Module 调度：移除被融合的 seg1 Module（graph2）的 NN 节点，避免运行时继续调度 *_1
    auto remove_nn_node = [](const nn::AbstractNnNode::ptr_t& node) {
      if (!node) return;
      auto parent = node->refParentNode().get_weak();
      if (parent) {
        auto& siblings = parent->refChildNodes();
        siblings.erase(std::remove_if(siblings.begin(), siblings.end(),
                                      [&](const nn::AbstractNnNode::ptr_t& n) { return n.get() == node.get(); }),
                       siblings.end());
      }
      node->refChildNodes().clear();
      node->setCompiledAsObj(true);     // 标记为已编译，防止再次参与编译/调度
      node->__forceSetDevice(DeviceTypes::kCPU);  // 兜底将设备设为 CPU，避免走 QNN dispatcher
    };

    // layer0 的 _1 未被融合，保持不动；其余被融合的 graph2 全部禁用
    if (pair.layer_x + 1 > 0 && pair.graph2 && pair.graph2->abstract_nn_node_) {
      remove_nn_node(pair.graph2->abstract_nn_node_);
      MLLM_INFO("Qwen3IRGraphFusionPass: pruned NN node for fused-away seg1 '{}'", pair.graph2_name);
    }
    return true;
  };

  for (auto& product : fusion_products) {
    if (!rewire_single_pair(product)) {
      MLLM_ERROR("Qwen3IRGraphFusionPass: rewiring failed for layer {}", product.pair.layer_x);
      return ir::PASS_RET_FAILURE;
    }
  }

  // Remove obsolete graphs
  for (auto& graph : graphs_to_remove) {
    if (!graph) { continue; }
    auto symbol_attr = graph->getSymbolAttr();
    const std::string symbol = symbol_attr ? symbol_attr->str() : "<unnamed>";
    // 如果符号被标记为保留（复用到 fused graph），则仅移除旧 graph，不移除符号表
    if (preserved_symbols.count(symbol) > 0) {
      top_region->ops().remove(graph);
      MLLM_INFO("Qwen3IRGraphFusionPass: removed obsolete SubGraph '{}' but kept symbol (reused by fused graph)",
                symbol);
      continue;
    }

    ir_ctx->removeFromSymbolTable(symbol);
    top_region->ops().remove(graph);
    MLLM_INFO("Qwen3IRGraphFusionPass: removed obsolete SubGraph '{}'", symbol);
  }

  MLLM_INFO("Qwen3IRGraphFusionPass: Graph fusion completed. "
            "Now using {} fused graphs.",
            fusion_products.size());

  return ir::PASS_RET_SUCCESS;
}

}  // namespace mllm::qnn


