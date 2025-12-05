// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// 通用的 SubGraph 克隆 / inline 工具。
// 第一版只提供「克隆一个 SubGraph 成为新的 SubGraphOp」的骨架实现，
// 先正确处理符号、Region / IO，后续再逐步补充 Region 内具体 Op 的复制逻辑。

#pragma once

#include <string>
#include <unordered_map>

#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/graph/Op.hpp"

namespace mllm::ir::graph {

// SubGraph 克隆相关工具。
class SubGraphCloneUtils {
 public:
  struct CloneOptions {
    // 新 SubGraph 的符号名；为空表示沿用原名并由 IRContext 负责去重。
    std::string new_symbol_name;
  };

  // 克隆一个 SubGraph，返回新的 SubGraphOp。
  //
  // 当前版本只克隆：
  // - SubGraphOp 本身（符号、device）
  // - Region 容器
  // - Region 的 inputs / outputs 骨架（还不复制内部具体 Op）
  //
  // value_map（可选）：
  // - 如果非空，会填充「old_val -> new_val」的映射（目前只包含 Region inputs/outputs）。
  static SubGraphOp::ptr_t cloneSubGraph(
      const IRContext::ptr_t& ctx,
      const SubGraphOp::ptr_t& src,
      const CloneOptions& options,
      std::unordered_map<Val*, val_ptr_t>* value_map = nullptr);

  // 将 src SubGraph 的主体（不含 ReturnOp）克隆并追加到 dst SubGraph 的 Region 中。
  // - value_map: 需要传入并维护 old_val -> new_val 的映射，便于跨多个子图持续复用
  // - preset_input_mapping: 当 src 的 inputs 需要映射到 dst 已存在的 Value 时，通过该映射提供
  // - cloned_outputs (可选)：返回按照 src->outputs() 顺序对应的新 Value 列表
  static bool inlineSubGraphInto(
      const IRContext::ptr_t& ctx,
      const SubGraphOp::ptr_t& src,
      const SubGraphOp::ptr_t& dst,
      std::unordered_map<Val*, val_ptr_t>& value_map,
      const std::unordered_map<Val*, val_ptr_t>& preset_input_mapping,
      std::vector<val_ptr_t>* cloned_outputs = nullptr);
};

}  // namespace mllm::ir::graph




