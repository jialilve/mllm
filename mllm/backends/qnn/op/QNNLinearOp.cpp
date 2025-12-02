// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/op/QNNLinearOp.hpp"
#include "QnnTypes.h"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/utils/Log.hpp"
#include <cstring>
#include <cmath>

namespace mllm::qnn {

// Helper function to manually transpose int8/uint8 weight matrices
// Since CPU TransposeOp doesn't support int8, we implement it manually
static Tensor manualTranspose2D(Tensor input, int dim0, int dim1) {
  auto shape = input.shape();
  if (shape.size() != 2) {
    return input.transpose(dim0, dim1);  // Fallback for non-2D or supported types
  }
  
  DataTypes dtype = input.dtype();
  if (dtype != kInt8 && dtype != kUInt8) {
    return input.transpose(dim0, dim1);  // Use standard transpose for supported types
  }
  
  // Manual transpose for int8/uint8
  // Note: We need to create a new tensor with actual memory allocation.
  // If input is kParamsMMAP (memory-mapped), we use kParamsNormal for the output
  // to ensure proper memory management. This is necessary because transpose
  // requires copying data, not just viewing it.
  int32_t h = shape[0];
  int32_t w = shape[1];
  TensorMemTypes output_mem_type = input.memType();
  // If input is memory-mapped (kParamsMMAP), output needs actual memory (kParamsNormal)
  if (output_mem_type == kParamsMMAP) {
    output_mem_type = kParamsNormal;
  }
  Tensor output = Tensor::empty({w, h}, dtype, input.device())
                      .setMemType(output_mem_type)
                      .setName(input.name())  // Copy tensor name
                      .alloc();
  
  // Copy quantization scale if present
  auto input_copy = input;  // Shadow copy for getQuantScale
  if (input_copy.attachedViews().contains("qnn_quant_scale")) {
    float scale = mllm::qnn::getQuantScale(input_copy);
    mllm::qnn::setQuantScale(output, scale);
  }
  
  if (dtype == kInt8) {
    const int8_t* src = input.ptr<int8_t>();
    int8_t* dst = output.ptr<int8_t>();
    for (int32_t i = 0; i < h; ++i) {
      for (int32_t j = 0; j < w; ++j) {
        dst[j * h + i] = src[i * w + j];
      }
    }
  } else if (dtype == kUInt8) {
    const uint8_t* src = input.ptr<uint8_t>();
    uint8_t* dst = output.ptr<uint8_t>();
    for (int32_t i = 0; i < h; ++i) {
      for (int32_t j = 0; j < w; ++j) {
        dst[j * h + i] = src[i * w + j];
      }
    }
  }
  
  return output;
}

/*
QNN Linear in mllm uses W8A8/W8A16 per-tensor quantization scheme.
The weight and the activation is using offline profiled scale to quantize the float32 data to int8/int16.
In QNN, the quantizeParam is set to QNN_QUANTIZATION_ENCODING_SCALE_OFFSET (with offset=0)

FIXME: Due to history reason, the bias is stored in int8 format.
In w8a8 case, the bias is converted to uint8 with zero-point 128.
In w8a16 case, the bias is needed to convert to int32.
Additionally, when using rotation quant and fused dequantize-add, the bias is not needed here and in float32 format.

TODO: per-channel quantization support, use FullyConnected to replace Conv2d
*/

QNNLinearOp::QNNLinearOp(const aops::LinearOpOptions& options) : LinearOp(options) {}

void QNNLinearOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      // For qwen3-1.7b-int8-rotated.mllm, weights are stored as [out_dim, in_dim] without transpose
      // Need to transpose to [in_dim, out_dim] before reshaping to 4D for QNN Conv2d
      // QNN Conv2d expects weight shape [1, 1, in_channels, out_channels]
      if (weight_.shape().size() == 2) {
        // Transpose from [out_dim, in_dim] to [in_dim, out_dim]
        weight_ = manualTranspose2D(weight_, 0, 1);
      }
      // using Conv2d in QNN, need to reshape the weight to 4D
      weight_ = weight_.view({1, 1, options_.in_channels, options_.out_channels});

      if (options_.bias) {
        bias_ = ploader->pull(getName() + ".bias");
        bias_ = bias_.view({options_.out_channels});

        biasScale_ = ploader->pull(getName() + ".bias.scale");
        biasScale_ = biasScale_.view({1});
      }

      weightScale_ = ploader->pull(getName() + ".weight.scale");
      weightScale_ = weightScale_.view({1});

      outputScale_ = ploader->pull(getName() + ".output_scale");
      outputScale_ = outputScale_.view({1});
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(getName() + ".weight");
      // For qwen3-1.7b-int8-rotated.mllm, weights are stored as [out_dim, in_dim] without transpose
      // Need to transpose to [in_dim, out_dim] before reshaping to 4D for QNN Conv2d
      if (weight_.shape().size() == 2) {
        // Transpose from [out_dim, in_dim] to [in_dim, out_dim]
        weight_ = manualTranspose2D(weight_, 0, 1);
      }
      // using Conv2d in QNN, need to reshape the weight to 4D
      weight_ = weight_.view({1, 1, options_.in_channels, options_.out_channels});
      if (options_.bias) {
        bias_ = ploader->pull(getName() + ".bias");
        bias_ = bias_.view({options_.out_channels});

        biasScale_ = ploader->pull(getName() + ".bias.scale");
      }

      // For Qwen3 converted models, scale is stored as {base_name}.scale, not {base_name}.weight.scale
      // Try both formats for compatibility
      std::string weight_scale_name = getName() + ".weight.scale";
      std::string scale_name = getName() + ".scale";
      if (ploader->has(scale_name)) {
        weightScale_ = ploader->pull(scale_name);
      } else if (ploader->has(weight_scale_name)) {
        weightScale_ = ploader->pull(weight_scale_name);
      } else {
        MLLM_ERROR_EXIT(ExitCode::kIOError, "Parameter does not exist: {} or {}", scale_name, weight_scale_name);
      }

      outputScale_ = ploader->pull(getName() + ".output_scale");
      break;
    }
    default: NYI("Unsupported model file version")
  }

  // set quant scale to tensor's attached view
  setQuantScale(weight_, weightScale_.at<float>({0}));
  if (options_.bias) { setQuantScale(biasScale_, biasScale_.at<float>({0})); }

  // handle bias conversion for history reason
  if (options_.bias) {
    switch (options_.qnn_impl_type) {
      case aops::LinearImplTypes::kQNN_tensor_symm_w8a8: {
        // convert int8 bias to uint8 with zero-point 128
        bias_ = bias_.to(kUInt8);
        break;
      }
      case aops::LinearImplTypes::kQNN_tensor_symm_w8a16: {
        // convert int8 bias to int32
        bias_ = bias_.to(kInt32);
        break;
      }
      default: MLLM_ERROR_EXIT(1, "QNN not support other linear impl");
    }
  }
}

void QNNLinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& i = inputs[0];
  auto i_shape = i.shape();

  MLLM_RT_ASSERT_EQ(i_shape[i_shape.size() - 1], options_.in_channels);

  auto o_shape = i_shape;
  o_shape[o_shape.size() - 1] = options_.out_channels;

  DataTypes o_dtype = i.dtype();

  outputs.emplace_back(Tensor::empty(o_shape, o_dtype, kQNN));

  // attach quant scale to output tensor
  switch (options_.qnn_impl_type) {
    case aops::LinearImplTypes::kQNN_tensor_symm_w8a8: {
      setQuantScale(outputs[0], outputScale_.at<float>({0}) / (std::pow(2, 7) - 1));
      break;
    }
    case aops::LinearImplTypes::kQNN_tensor_symm_w8a16: {
      setQuantScale(outputs[0], outputScale_.at<float>({0}) / (std::pow(2, 15) - 1));
      break;
    }
    default: MLLM_ERROR_EXIT(1, "QNN not support other linear impl");
  }
}

bool QNNLinearPattern::addNode(const std::string& graphName, const ir::op_ptr_t& op,
                               const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
                               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {
  // Get the LinearOp from the IR
  auto* linalgOp = dynamic_cast<ir::linalg::LinearOp*>(op.get());
  if (!linalgOp) {
    MLLM_ERROR("Failed to cast to linalg::LinearOp");
    return false;
  }

  auto* qnnLinearOp = dynamic_cast<QNNLinearOp*>(linalgOp->getAOp());
  if (!qnnLinearOp) {
    MLLM_ERROR("Failed to cast to QNNLinearOp");
    return false;
  }

  const auto& options = qnnLinearOp->options();

  auto qnnBackend = std::dynamic_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));
  if (!qnnBackend) {
    MLLM_ERROR("Failed to get QNN backend.");
    return false;
  }

  auto& weight = qnnLinearOp->weight();
  auto& bias = qnnLinearOp->bias();
  auto& weightScale = const_cast<Tensor&>(qnnLinearOp->weightScale());
  auto& biasScale = const_cast<Tensor&>(qnnLinearOp->biasScale());

  // Check if this is W8A16 quantization type
  if (options.qnn_impl_type != aops::LinearImplTypes::kQNN_tensor_symm_w8a16) {
    MLLM_ERROR("addNode currently only supports W8A16 quantization");
    return false;
  }

  // Create Conv2d parameters (stride and padding)
  auto strideName = qnnLinearOp->getName() + ".stride";
  auto padName = qnnLinearOp->getName() + ".pad";

  auto strideParam = QNNParamTensorWrapper::create("stride", strideName, QNN_DATATYPE_UINT_32, std::vector<uint32_t>{2});
  uint32_t* strideData = static_cast<uint32_t*>(strideParam->alloc());
  strideData[0] = 1;
  strideData[1] = 1;

  auto padParam = QNNParamTensorWrapper::create("pad_amount", padName, QNN_DATATYPE_UINT_32, std::vector<uint32_t>{2, 2});
  uint32_t* padData = static_cast<uint32_t*>(padParam->alloc());
  padData[0] = 0;
  padData[1] = 0;
  padData[2] = 0;
  padData[3] = 0;

  // Add weight tensor to QNN
  float weightScaleValue = weightScale.at<float>({0});

  Qnn_QuantizeParams_t weightQuantParams = {QNN_DEFINITION_DEFINED,
                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                            {.scaleOffsetEncoding = {.scale = weightScaleValue, .offset = 0}}};

  if (!qnnBackend->addStaticTensor(graphName, weight.name(), weight, weightQuantParams)) {
    MLLM_ERROR("Failed to add weight tensor to QNN graph");
    return false;
  }

  // Set output tensor quantization parameters
  float outputScaleValue = getQuantScale(outputs[0]->tensor_);
  Qnn_QuantizeParams_t outputQuantParams = {QNN_DEFINITION_DEFINED,
                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                            {.scaleOffsetEncoding = {.scale = outputScaleValue, .offset = 0}}};

  auto qnn_output_tensor_type = mllm::qnn::getQnnOutputTensorType(outputs[0]);

  if (!qnnBackend->addTensor(graphName, outputs[0]->name(), qnn_output_tensor_type, outputs[0]->tensor_, outputQuantParams)) {
    MLLM_ERROR("Failed to add output tensor to QNN graph");
    return false;
  }

  std::vector<std::string> inputTensorNames = {inputs[0]->name(), weight.name()};
  std::vector<std::string> outputTensorNames = {outputs[0]->name()};

  // Handle bias if supported
  if (options.bias) {
    float biasScaleValue = biasScale.at<float>({0});

    Qnn_QuantizeParams_t biasQuantParams = {QNN_DEFINITION_DEFINED,
                                            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                            {.scaleOffsetEncoding = {.scale = biasScaleValue, .offset = 0}}};

    if (!qnnBackend->addStaticTensor(graphName, bias.name(), bias, biasQuantParams)) {
      MLLM_ERROR("Failed to add bias tensor to QNN graph");
      return false;
    }

    inputTensorNames.push_back(bias.name());
  }

  // Add Conv2d node to graph
  std::string nodeName = qnnLinearOp->getName() + ".linear_w8a16";
  qnnBackend->graphAddNode(graphName, nodeName, "Conv2d", inputTensorNames, outputTensorNames, {strideParam, padParam}, {});
  // TODO_qwen3_npu_modeling
  return true;
}

}  // namespace mllm::qnn
