// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// 通用的 SubGraph 克隆 / inline 工具的基础实现。

#include "mllm/compile/ir/graph/SubGraphCloneUtils.hpp"

#include <unordered_map>
#include <vector>

#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::ir::graph {
namespace {

using ValueMap = std::unordered_map<Val*, val_ptr_t>;

void copyNodeAttributes(const node_ptr_t& src, const node_ptr_t& dst) {
  if (!src || !dst) { return; }
  for (const auto& [attr_name, attr_value] : src->getAttrs()) {
    dst->setAttr(attr_name, attr_value);
  }
}

tensor::TensorValue::ptr_t cloneTensorValueLike(const IRContext::ptr_t& ctx,
                                                const tensor::TensorValue::ptr_t& src) {
  if (!ctx || !src) { return nullptr; }
  auto cloned = ctx->createTemporaryValue<tensor::TensorValue>(src->tensor());
  copyNodeAttributes(src, cloned);
  return cloned;
}

val_ptr_t cloneValueLike(const IRContext::ptr_t& ctx, const val_ptr_t& src) {
  if (!src) { return nullptr; }
  if (auto tensor_val = std::dynamic_pointer_cast<tensor::TensorValue>(src)) {
    return cloneTensorValueLike(ctx, tensor_val);
  }

  MLLM_ERROR("SubGraphCloneUtils: unsupported Val type when cloning (kind={})",
             static_cast<int>(src->getKind()));
  return nullptr;
}

bool mapGraphInputs(const IRContext::ptr_t& ctx,
                    const SubGraphOp::ptr_t& cloned_graph,
                    const std::list<val_ptr_t>& src_inputs,
                    std::list<val_ptr_t>& cloned_inputs,
                    ValueMap& value_map) {
  cloned_inputs.clear();
  for (auto& src_input : src_inputs) {
    auto cloned_input = cloneValueLike(ctx, src_input);
    if (!cloned_input) {
      MLLM_ERROR("SubGraphCloneUtils: failed to clone graph input '{}'",
                 src_input ? src_input->name() : "<null>");
      return false;
    }
    cloned_inputs.push_back(cloned_input);
    value_map[src_input.get()] = cloned_input;
    // 链接 SubGraph 与输入 Value
    (*cloned_input)-- > cloned_graph;
  }
  return true;
}

template<typename ConcreteOpT>
bool createClonedLinalgOp(IRWriter& writer,
                          const std::vector<tensor::TensorValue::ptr_t>& inputs,
                          const std::vector<tensor::TensorValue::ptr_t>& outputs,
                          const BaseOp::ptr_t& base_op,
                          const node_ptr_t& src_node) {
  auto new_op = writer.create<ConcreteOpT>(base_op, inputs, outputs);
  copyNodeAttributes(src_node, new_op);
  return true;
}

bool cloneLinalgOp(const IRContext::ptr_t& ctx,
                   const ir::linalg::LinalgIROp::ptr_t& src_op,
                   IRWriter& writer,
                   ValueMap& value_map) {
  if (!src_op) { return false; }

  std::vector<tensor::TensorValue::ptr_t> cloned_inputs;
  cloned_inputs.reserve(src_op->inputs().size());

  for (auto& input_weak : src_op->inputs()) {
    auto* input_node = input_weak.get_weak();
    if (!input_node || !input_node->isa_<Val>()) {
      MLLM_ERROR("SubGraphCloneUtils: invalid input node while cloning op");
      return false;
    }
    auto old_val = std::static_pointer_cast<Val>(input_node->shared_from_this());
    auto map_it = value_map.find(old_val.get());
    if (map_it == value_map.end()) {
      // 名字兜底：有些场景下，同一个逻辑 Val 可能被不同 Val* 包装，但 name 保持唯一
      for (auto& kv : value_map) {
        auto existing_val_node = kv.first;
        if (!existing_val_node) { continue; }
        auto existing_val = std::dynamic_pointer_cast<Val>(existing_val_node->shared_from_this());
        if (existing_val && existing_val->name() == old_val->name()) {
          map_it = value_map.find(existing_val_node);
          break;
        }
      }
    }
    if (map_it == value_map.end()) {
      MLLM_ERROR("SubGraphCloneUtils: missing value mapping for input '{}'", old_val->name());
      return false;
    }
    auto tensor_input = std::dynamic_pointer_cast<tensor::TensorValue>(map_it->second);
    if (!tensor_input) {
      MLLM_ERROR("SubGraphCloneUtils: expected TensorValue input for op '{}'", old_val->name());
      return false;
    }
    cloned_inputs.push_back(tensor_input);
  }

  std::vector<tensor::TensorValue::ptr_t> cloned_outputs;
  cloned_outputs.reserve(src_op->outputs().size());
  for (auto& output_weak : src_op->outputs()) {
    auto* output_node = output_weak.get_weak();
    if (!output_node || !output_node->isa_<Val>()) {
      MLLM_ERROR("SubGraphCloneUtils: invalid output node while cloning op");
      return false;
    }
    auto old_val = std::static_pointer_cast<Val>(output_node->shared_from_this());
    auto cloned_val = cloneValueLike(ctx, old_val);
    if (!cloned_val) {
      MLLM_ERROR("SubGraphCloneUtils: failed to clone output value '{}'", old_val->name());
      return false;
    }
    auto tensor_output = std::dynamic_pointer_cast<tensor::TensorValue>(cloned_val);
    if (!tensor_output) {
      MLLM_ERROR("SubGraphCloneUtils: expected TensorValue output for op '{}'", old_val->name());
      return false;
    }
    cloned_outputs.push_back(tensor_output);
    value_map[old_val.get()] = cloned_val;
  }

  auto base_op = src_op->getAOp() ? src_op->getAOp()->shared_from_this() : nullptr;
  if (!base_op) {
    MLLM_ERROR("SubGraphCloneUtils: source Linalg op has no attached BaseOp");
    return false;
  }

  if (src_op->template isa_<ir::linalg::ViewOp>()) {
    return createClonedLinalgOp<ir::linalg::ViewOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }
  if (src_op->template isa_<ir::linalg::CastTypeOp>()) {
    return createClonedLinalgOp<ir::linalg::CastTypeOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }
  if (src_op->template isa_<ir::linalg::LinearOp>()) {
    return createClonedLinalgOp<ir::linalg::LinearOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }
  if (src_op->template isa_<ir::linalg::TransposeOp>()) {
    return createClonedLinalgOp<ir::linalg::TransposeOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }
  if (src_op->template isa_<ir::linalg::RMSNormOp>()) {
    return createClonedLinalgOp<ir::linalg::RMSNormOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }
  if (src_op->template isa_<ir::linalg::SiLUOp>()) {
    return createClonedLinalgOp<ir::linalg::SiLUOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }
  if (src_op->template isa_<ir::linalg::AddOp>()) {
    return createClonedLinalgOp<ir::linalg::AddOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }
  if (src_op->template isa_<ir::linalg::MulOp>()) {
    return createClonedLinalgOp<ir::linalg::MulOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }
  if (src_op->template isa_<ir::linalg::X2XOp>()) {
    return createClonedLinalgOp<ir::linalg::X2XOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }
  if (src_op->template isa_<ir::linalg::CustomizedOp>()) {
    return createClonedLinalgOp<ir::linalg::CustomizedOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }

  MLLM_ERROR("SubGraphCloneUtils: unsupported Linalg op '{}'", static_cast<int>(src_op->getKind()));
  return false;
}

bool cloneReturnOp(const ir::cf::ReturnOp::ptr_t& src_return,
                   IRWriter& writer,
                   ValueMap& value_map) {
  if (!src_return) {
    MLLM_ERROR("SubGraphCloneUtils: missing ReturnOp in source SubGraph");
    return false;
  }
  std::vector<val_ptr_t> cloned_returns;
  cloned_returns.reserve(src_return->inputs().size());
  for (auto& ret_input_weak : src_return->inputs()) {
    auto* ret_node = ret_input_weak.get_weak();
    if (!ret_node || !ret_node->isa_<Val>()) {
      MLLM_ERROR("SubGraphCloneUtils: invalid Return input");
      return false;
    }
    auto old_val = std::static_pointer_cast<Val>(ret_node->shared_from_this());
    auto it = value_map.find(old_val.get());
    if (it == value_map.end()) {
      for (auto& kv : value_map) {
        auto existing_val_node = kv.first;
        if (!existing_val_node) { continue; }
        auto existing_val = std::dynamic_pointer_cast<Val>(existing_val_node->shared_from_this());
        if (existing_val && existing_val->name() == old_val->name()) {
          it = value_map.find(existing_val_node);
          break;
        }
      }
    }
    if (it == value_map.end()) {
      MLLM_ERROR("SubGraphCloneUtils: missing mapping for Return input '{}'", old_val->name());
      return false;
    }
    cloned_returns.push_back(it->second);
  }
  writer.create<ir::cf::ReturnOp>(cloned_returns);
  return true;
}

}  // namespace

SubGraphOp::ptr_t SubGraphCloneUtils::cloneSubGraph(
    const IRContext::ptr_t& ctx,
    const SubGraphOp::ptr_t& src,
    const CloneOptions& options,
    std::unordered_map<Val*, val_ptr_t>* value_map) {
  if (!ctx) {
    MLLM_ERROR("SubGraphCloneUtils::cloneSubGraph: IRContext is null");
    return nullptr;
  }
  if (!src) {
    MLLM_ERROR("SubGraphCloneUtils::cloneSubGraph: source SubGraphOp is null");
    return nullptr;
  }

  auto src_symbol = src->getSymbolAttr();
  if (!src_symbol) {
    MLLM_ERROR("SubGraphCloneUtils::cloneSubGraph: source SubGraphOp has no symbol");
    return nullptr;
  }

  std::string new_symbol_name;
  if (!options.new_symbol_name.empty()) {
    new_symbol_name = options.new_symbol_name;
  } else {
    new_symbol_name = ctx->getUniqueModuleName(src_symbol->str());
  }
  auto new_symbol_attr = ctx->create<SymbolAttr>(new_symbol_name);

  auto cloned = SubGraphOp::build(ctx.get(), new_symbol_attr);
  if (src->hasDevice()) { cloned->setDevice(src->getDevice()); }
  copyNodeAttributes(src, cloned);

  auto src_region = src->getTopRegion();
  auto cloned_region = cloned->getTopRegion();
  if (!src_region || !cloned_region) {
    MLLM_ERROR("SubGraphCloneUtils::cloneSubGraph: source or cloned SubGraph has no top region");
    return nullptr;
  }

  ValueMap local_value_map;
  ValueMap* map_ptr = value_map ? value_map : &local_value_map;
  map_ptr->clear();

  if (!mapGraphInputs(ctx, cloned, src_region->inputs(), cloned_region->inputs(), *map_ptr)) {
    return nullptr;
  }

  ir::IRWriterGuard writer_guard(ctx, cloned_region);
  ir::IRWriter writer(ctx, cloned_region);

  ir::cf::ReturnOp::ptr_t return_op = nullptr;
  for (auto& op : src_region->ops()) {
    if (op->isa_<ir::cf::ReturnOp>()) {
      return_op = op->cast_<ir::cf::ReturnOp>();
      continue;
    }
    if (auto linalg_op = op->cast_<ir::linalg::LinalgIROp>()) {
      if (!cloneLinalgOp(ctx, linalg_op, writer, *map_ptr)) {
        MLLM_ERROR("SubGraphCloneUtils: failed to clone Linalg op inside '{}'", src_symbol->str());
        return nullptr;
      }
      continue;
    }
    MLLM_ERROR("SubGraphCloneUtils: unsupported op kind '{}' in SubGraph '{}'",
               static_cast<int>(op->getKind()), src_symbol->str());
    return nullptr;
  }

  if (!cloneReturnOp(return_op, writer, *map_ptr)) { return nullptr; }

  cloned_region->outputs().clear();
  for (auto& src_output : src_region->outputs()) {
    auto it = map_ptr->find(src_output.get());
    if (it == map_ptr->end()) {
      for (auto& kv : *map_ptr) {
        auto existing_val_node = kv.first;
        if (!existing_val_node) { continue; }
        auto existing_val = std::dynamic_pointer_cast<Val>(existing_val_node->shared_from_this());
        if (existing_val && existing_val->name() == src_output->name()) {
          it = map_ptr->find(existing_val_node);
          break;
        }
      }
    }
    if (it == map_ptr->end()) {
      MLLM_ERROR("SubGraphCloneUtils: missing mapping for graph output '{}'", src_output->name());
      return nullptr;
    }
    cloned_region->outputs().push_back(it->second);
    (*cloned)-- > it->second;
  }

  MLLM_INFO("SubGraphCloneUtils::cloneSubGraph: cloned SubGraph '{}' -> '{}'",
            src_symbol->str(), new_symbol_name);

  return cloned;
}

bool SubGraphCloneUtils::inlineSubGraphInto(
    const IRContext::ptr_t& ctx,
    const SubGraphOp::ptr_t& src,
    const SubGraphOp::ptr_t& dst,
    std::unordered_map<Val*, val_ptr_t>& value_map,
    const std::unordered_map<Val*, val_ptr_t>& preset_input_mapping,
    std::vector<val_ptr_t>* cloned_outputs) {
  if (!ctx || !src || !dst) {
    MLLM_ERROR("SubGraphCloneUtils::inlineSubGraphInto: invalid arguments (ctx/src/dst)");
    return false;
  }
  auto src_region = src->getTopRegion();
  auto dst_region = dst->getTopRegion();
  if (!src_region || !dst_region) {
    MLLM_ERROR("SubGraphCloneUtils::inlineSubGraphInto: missing region while cloning");
    return false;
  }

  // 为 src 的 inputs 建立映射：优先使用外部提供的 preset_input_mapping
  for (auto& src_input : src_region->inputs()) {
    auto preset_it = preset_input_mapping.find(src_input.get());
    if (preset_it != preset_input_mapping.end()) {
      value_map[src_input.get()] = preset_it->second;
      continue;
    }
    if (value_map.find(src_input.get()) == value_map.end()) {
      MLLM_ERROR("SubGraphCloneUtils::inlineSubGraphInto: missing mapping for src input '{}'",
                 src_input ? src_input->name() : "<null>");
      return false;
    }
  }

  ir::IRWriterGuard writer_guard(ctx, dst_region);
  ir::IRWriter writer(ctx, dst_region);

  ir::cf::ReturnOp::ptr_t src_return = nullptr;
  for (auto& op : src_region->ops()) {
    if (op->isa_<ir::cf::ReturnOp>()) {
      src_return = op->cast_<ir::cf::ReturnOp>();
      continue;
    }
    if (auto linalg_op = op->cast_<ir::linalg::LinalgIROp>()) {
      if (!cloneLinalgOp(ctx, linalg_op, writer, value_map)) {
        MLLM_ERROR("SubGraphCloneUtils::inlineSubGraphInto: failed to clone op '{}' from src '{}'",
                   static_cast<int>(linalg_op->getKind()),
                   src->getSymbolAttr() ? src->getSymbolAttr()->str() : "<unnamed>");
        return false;
      }
      continue;
    }
    MLLM_ERROR("SubGraphCloneUtils::inlineSubGraphInto: unsupported op kind '{}' in SubGraph '{}'",
               static_cast<int>(op->getKind()),
               src->getSymbolAttr() ? src->getSymbolAttr()->str() : "<unnamed>");
    return false;
  }

  if (cloned_outputs) {
    cloned_outputs->clear();
    for (auto& output : src_region->outputs()) {
      auto it = value_map.find(output.get());
      if (it == value_map.end()) {
        for (auto& kv : value_map) {
          auto existing_val_node = kv.first;
          if (!existing_val_node) { continue; }
          auto existing_val = std::dynamic_pointer_cast<Val>(existing_val_node->shared_from_this());
          if (existing_val && output && existing_val->name() == output->name()) {
            it = value_map.find(existing_val_node);
            break;
          }
        }
      }
      if (it == value_map.end()) {
        MLLM_ERROR("SubGraphCloneUtils::inlineSubGraphInto: missing mapping for output '{}'",
                   output ? output->name() : "<null>");
        return false;
      }
      cloned_outputs->push_back(it->second);
    }
  }

  // ReturnOp 由调用方负责在 dst 中统一构建
  return true;
}

}  // namespace mllm::ir::graph
