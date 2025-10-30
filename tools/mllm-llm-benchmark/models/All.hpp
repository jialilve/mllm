// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <algorithm>

#include "Qwen3_W4A32_KAI.hpp"
#include "Qwen3_W4A8_I8MM_KAI.hpp"
#include "BenchmarkTemplate.hpp"

std::shared_ptr<BenchmarkTemplate> createBenchmark(const std::string& model_name) {
  auto tolower = [](const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
  };
  auto normalized_model_name = tolower(model_name);
  
  // Check for W4A8 I8MM KAI first (more specific)
  if (normalized_model_name.find("qwen3") != std::string::npos && 
      normalized_model_name.find("w4a8") != std::string::npos &&
      normalized_model_name.find("i8mm") != std::string::npos &&
      normalized_model_name.find("kai") != std::string::npos) {
    return std::make_shared<Qwen3_W4A8_I8MM_KAI_Benchmark>();
  }
  
  // Check for W4A32 KAI
  if (normalized_model_name.find("qwen3") != std::string::npos && 
      normalized_model_name.find("w4a32") != std::string::npos &&
      normalized_model_name.find("kai") != std::string::npos) {
    return std::make_shared<Qwen3_W4A32_KAI_Benchmark>();
  }
  
  return nullptr;
}
