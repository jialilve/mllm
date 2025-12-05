// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/op/QNNX2XOp.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/utils/Log.hpp"
#include <cstring>

namespace mllm::qnn {

QNNX2XOp::QNNX2XOp(const aops::X2XOpOptions& options) : aops::X2XOp(options) {}

void QNNX2XOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (inputs.empty() || outputs.empty()) {
    MLLM_ERROR("QNNX2XOp::forward: empty inputs/outputs, inputs={}, outputs={}", inputs.size(), outputs.size());
    return;
  }

  const auto& input = inputs[0];
  auto& output = outputs[0];

  // Detailed logging for debugging
  MLLM_INFO("QNNX2XOp::forward: input shape={}, device={}, bytes={}, impl={}, storage={}", 
            input.shape(), static_cast<int>(input.device()), input.bytes(),
            static_cast<void*>(input.impl().get()), static_cast<void*>(input.impl() ? input.impl()->storage().get() : nullptr));
  
  if (!input.impl() || !output.impl() || !input.impl()->storage() || !output.impl()->storage()) {
    MLLM_ERROR("QNNX2XOp::forward: invalid tensor storage (input impl={}, output impl={}, input storage={}, output storage={})",
               static_cast<void*>(input.impl().get()), static_cast<void*>(output.impl().get()),
               static_cast<void*>(input.impl() ? input.impl()->storage().get() : nullptr),
               static_cast<void*>(output.impl() ? output.impl()->storage().get() : nullptr));
    return;
  }

  // Check input storage pointer before accessing
  void* input_storage_ptr = input.impl()->storage()->ptr_;
  MLLM_INFO("QNNX2XOp::forward: input storage ptr={}, input.ptr<void>()={}", 
            static_cast<void*>(input_storage_ptr), static_cast<const void*>(input.ptr<void>()));

  // For now, only do copy between CPU and QNN shared buffer
  // Ensure output tensor has allocated memory
  if (!output.impl()->storage()->ptr_) { output.alloc(); }

  // Get input data pointer
  const void* src_data = input.ptr<void>();
  void* dst_data = output.ptr<void>();

  // Calculate data size in bytes
  size_t data_size = input.bytes();

  MLLM_INFO("QNNX2XOp::forward: src_data={}, dst_data={}, data_size={}", 
            static_cast<const void*>(src_data), static_cast<void*>(dst_data), data_size);

  if (!src_data || !dst_data || data_size == 0) {
    MLLM_ERROR("QNNX2XOp::forward: invalid src/dst ptr or zero bytes (src_data={}, dst_data={}, bytes={})", 
               static_cast<const void*>(src_data), static_cast<void*>(dst_data), data_size);
    return;
  }

  // Perform memory copy from CPU to QNN shared buffer
  std::memcpy(dst_data, src_data, data_size);
}

bool QNNX2XPattern::addNode(const std::string& graphName, const ir::op_ptr_t& op,
                            const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                            const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  MLLM_ERROR_EXIT(1, "Illegal Modeling Arch, the tensor.to(kQNN/kCPU) should occur in QNN sub graph");

  return false;
}

}  // namespace mllm::qnn
