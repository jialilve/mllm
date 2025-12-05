#include <fmt/core.h>
#include <cstdint>
#include <mllm/mllm.hpp>
#include <mllm/utils/AnyValue.hpp>

#include "mllm/backends/qnn/passes/QNNGraphBuildPass.hpp"
#include "mllm/backends/qnn/passes/QNNGraphIOTensorPass.hpp"
#include "mllm/backends/qnn/passes/QNNOpNamingPass.hpp"
#include "mllm/backends/qnn/passes/Qwen3IRGraphFusionPass.hpp"
#include "mllm/backends/qnn/QNNAllocator.hpp"
#include "mllm/compile/PassManager.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/models/qwen3_npu/tokenization_qwen3_npu.hpp"
#include "mllm/models/qwen3_npu/modeling_qwen3_npu.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

using mllm::Argparse;

MLLM_MAIN({
  mllm::initQnnBackend();

  // Model paths - update these to match your actual file locations on Android device
  // Files should be pushed to /data/local/tmp/zl/mllm-v2/bin_test/ via adb
  const std::string config_path = "./config_qwen3_1.7B_qnn.json";
  const std::string model_path = "./qwen3-1.7b-int8-rotated.mllm";
  const std::string tokenizer_path = "./tokenizer_qwen3_1.7B.json";

  auto qwen3_tokenizer = mllm::models::qwen3_npu::Qwen3Tokenizer(tokenizer_path);

  // Try V2 first (newer format), fallback to V1 if needed
  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV2;
  
  // Note: qwen3-1.7b-int8-rotated.mllm is likely V2 format
  // If you get magic number mismatch, try changing to kV1

  auto cfg = mllm::models::qwen3_npu::Qwen3NPUConfig(config_path);
  auto model = mllm::models::qwen3_npu::Qwen3ForCausalLM("", cfg);

  auto param = mllm::load(model_path, file_version);
  model.load(param);

  const int chunk_size = 32;
  mllm::models::ARGenerationOutputPast inputs{
      {"sequence", mllm::Tensor::empty({1, chunk_size}, mllm::kInt64, mllm::kCPU).alloc()}};

  auto irs = model.trace(inputs, {});

  // QNN Graph Rewrite Pass
  mllm::ir::PassManager rewritePM(irs["model"]);

  // have a look at the IR before QNN Graph Rewrite Pass
  mllm::redirect("qwen3_npu_initial.mir", [&]() { mllm::print(irs["model"]); });

  // Qwen3 专用：在构建 QNN Graph 之前，先识别并标记 decoder 相关的 QNN SubGraph
  rewritePM.reg(mllm::qnn::createQwen3IRGraphFusionPass());
  rewritePM.reg(mllm::qnn::createQNNGraphIOTensorPass());
  rewritePM.reg(mllm::qnn::createQNNOpNamingPass());
  rewritePM.run();

  // have a look at the IR after QNN Graph Rewrite Pass
  // Note: This file is written to the current working directory (usually /data/local/tmp/zl/mllm-v2/bin_test on device)
  // Use adb pull to retrieve it: adb pull /data/local/tmp/zl/mllm-v2/bin_test/qwen3_npu.mir ./android_logs/qwen3_npu_after_fused.mir
  mllm::redirect("qwen3_npu.mir", [&]() { mllm::print(irs["model"]); });

  // QNN Graph Build Pass
  mllm::ir::PassManager graphBuildPM(irs["model"]);
  graphBuildPM.reg(mllm::qnn::createQNNGraphBuildPass());
  graphBuildPM.run();

  // cache has been updated due to trace, clear cache
  model.model.clearKVCache();

  auto raw_input_tokens = qwen3_tokenizer.convertMessage({.prompt = "你好，请介绍一下你自己。"})["sequence"];
  print(raw_input_tokens);
  MLLM_INFO("raw_input_tokens shape: {} {}", raw_input_tokens.shape()[0], raw_input_tokens.shape()[1]);

  const int eos_token_id = cfg.eos_token_id;
  int prompt_tokens = static_cast<int>(raw_input_tokens.shape()[1]);
  if (prompt_tokens <= 0) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kShapeError, "Prompt sequence length must be positive");
  }

  // Prepare reusable [1, chunk_size] CPU buffer for chunked prefill/decode
  mllm::models::ARGenerationOutputPast chunk_inputs{
      {"sequence", mllm::Tensor::empty({1, chunk_size}, mllm::kInt64, mllm::kCPU).alloc()}};
  auto sequence_tensor = chunk_inputs["sequence"];
  auto sequence_ptr = sequence_tensor.ptr<int64_t>();
  auto input_data = raw_input_tokens.ptr<int64_t>();

  const int prompt_chunks = (prompt_tokens + chunk_size - 1) / chunk_size;
  bool reached_eos = false;
  int total_decode_steps = 0;

  for (int chunk_index = 0; chunk_index < prompt_chunks && !reached_eos; ++chunk_index) {
    const int chunk_start = chunk_index * chunk_size;
    const int chunk_prompt_len = std::min(chunk_size, prompt_tokens - chunk_start);
    const bool is_last_prompt_chunk = (chunk_index == prompt_chunks - 1);

    // Copy current chunk prompt tokens and pad remaining positions with -1
    for (int i = 0; i < chunk_prompt_len; ++i) { sequence_ptr[i] = input_data[chunk_start + i]; }
    for (int i = chunk_prompt_len; i < chunk_size; ++i) { sequence_ptr[i] = -1; }

    // Calculate absolute sequence length from the start of the entire sequence
    const int absolute_seq_len = chunk_start + chunk_prompt_len;

    // Align KV cache so StaticCache writes start at the chunk's absolute offset
    model.setKVCacheSeqCnt(chunk_start);

    // Generate position_ids starting from chunk_start for multi-chunk scenarios
    auto position_ids_tensor = mllm::Tensor::empty({1, chunk_size}, mllm::kInt64, mllm::kCPU).alloc();
    auto position_ids_ptr = position_ids_tensor.ptr<int64_t>();
    for (int i = 0; i < chunk_size; ++i) {
      position_ids_ptr[i] = chunk_start + i;
    }
    
    // Prepare input with correct position_ids
    mllm::models::ARGenerationOutputPast prefill_inputs{
        {"sequence", sequence_tensor},
        {"position_ids", position_ids_tensor}};

    // real_seq should be the effective length in the current input tensor (relative position)
    // hidden_states shape is [1, chunk_size, hidden_size], we need to index it with chunk_prompt_len - 1
    auto chunk_output =
        model.forward(prefill_inputs, {{"seq_len", mllm::AnyValue(mllm::any_copy_tag, chunk_prompt_len)}});
    auto& chunk_logits = chunk_output["sequence"];

    if (!is_last_prompt_chunk) {
      chunk_logits.delete_();
      chunk_output.clear();
      continue;
    }

    if (chunk_prompt_len >= chunk_size) {
      MLLM_WARN("Last chunk is fully occupied by prompt tokens; no padding for decode");
      chunk_logits.delete_();
      chunk_output.clear();
      break;
    }

    // Use the prefill logits as the first decode step
    auto next_token = model.sampleGreedy(chunk_logits);
    chunk_logits.delete_();
    
    // Keep full-length position_ids tensor aligned with chunk buffer
    auto position_ids = position_ids_tensor;

    chunk_output.clear();

    auto emit_token = [&](int64_t token_id) {
      std::wcout << qwen3_tokenizer.detokenize(token_id) << std::flush;
      if (token_id == eos_token_id) {
        MLLM_INFO("EOS token detected, stopping decode");
        reached_eos = true;
      }
    };

    int current_chunk_len = chunk_prompt_len;
    emit_token(next_token);
    if (reached_eos) { break; }

    sequence_ptr[current_chunk_len] = next_token;
    current_chunk_len++;

    while (!reached_eos && current_chunk_len < chunk_size) {
      total_decode_steps++;
      
      // Calculate absolute sequence length from the start of the entire sequence
      const int absolute_seq_len = chunk_start + current_chunk_len;
      
      // Keep padding clean for the remaining area
      for (int i = current_chunk_len; i < chunk_size; ++i) { sequence_ptr[i] = -1; }

      // Set KV cache to absolute sequence length (where the next token will be written)
      model.setKVCacheSeqCnt(chunk_start);
      
      // Prepare decode input with position_ids from previous step
      mllm::models::ARGenerationOutputPast decode_inputs{
          {"sequence", sequence_tensor},
          {"position_ids", position_ids}};
      
      // real_seq should be the effective length in the current input tensor (relative position)
      // hidden_states shape is [1, chunk_size, hidden_size], we need to index it with current_chunk_len - 1
      auto decode_output = model.forward(
          decode_inputs, {{"seq_len", mllm::AnyValue(mllm::any_copy_tag, current_chunk_len)}});
      
      auto& decode_logits = decode_output["sequence"];
      next_token = model.sampleGreedy(decode_logits);
      decode_logits.delete_();
      decode_output.erase("sequence");
      decode_output.clear();

      emit_token(next_token);
      if (reached_eos) { break; }

      sequence_ptr[current_chunk_len] = next_token;
      current_chunk_len++;
    }
  }

  std::wcout << L"\n";

  return 0;
})

