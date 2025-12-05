// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Qwen3/Qwen NPU 专用 IR Graph 融合 Pass（方案一的落地入口）。
// 当前版本只是识别并标记 Qwen3 decoder 相关的 QNN SubGraph，不修改 IR 语义，
// 方便后续按层做更激进的 Graph 合并。

#pragma once

#include "mllm/compile/passes/Pass.hpp"
#include "mllm/compile/ir/Node.hpp"

namespace mllm::qnn {

// Qwen3 专用 IR Graph 融合 Pass。
// 约束：
// - 只在 Qwen/Qwen3 NPU 的编译流程中显式调用；
// - 只处理 device == kQNN 且名称形如 "model.layers.X..." 的 SubGraph；
// - 其它模型/子图一律跳过。
class Qwen3IRGraphFusionPass final : public ir::Pass {
 public:
  Qwen3IRGraphFusionPass() = default;
  ~Qwen3IRGraphFusionPass() override = default;

  // 顶层期望收到 ModuleOp。
  uint8_t run(const ir::node_ptr_t& op) override;

 private:
  // 尝试根据 symbol name 解析出 decoder 层索引和段号（第 1/2 个 QNN 段）。
  // 典型命名类似：
  //   - "model.layers.0.self_attn_proj.*"
  //   - "model.layers.0.self_attn_out_mlp.*"
  // 返回是否解析成功。
  static bool parseDecoderLayerInfo(const std::string& symbol,
                                    int& layer_index,
                                    int& segment_index);
};

// 工厂函数，便于在 PassManager 中注册。
inline ir::Pass::ptr_t createQwen3IRGraphFusionPass() {
  return std::make_shared<Qwen3IRGraphFusionPass>();
}

}  // namespace mllm::qnn



