// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <vector>
#include <unordered_map>
#include <string>

#include "mllm/preprocessor/tokenizers/BPE.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"

namespace mllm::models::gemma3 {

struct Gemma3Message {
  std::string prompt;
};

class Gemma3Tokenizer final : public mllm::preprocessor::AutoTokenizer {
 public:
  explicit Gemma3Tokenizer(const std::string& file_path) {
    preprocessor::initLocal();
    bpe_.initFromSentencePieceJson(file_path);
    bos_token_id_ = 2;
    eos_token_id_ = 1;
    pad_token_id_ = 0;
  }

  std::vector<std::wstring> _tokenize(const std::string& str) override {
    // Replace all spaces with "▁" (Unicode U+2581)
    std::string new_text = str;
    for (size_t i = 0; i < new_text.length(); ++i) {
      if (new_text[i] == ' ') {
        // Replace space with "▁"
        new_text.replace(i, 1, "▁");
      }
    }

    // Convert to wide string and tokenize with BPE
    std::wstring w_text = preprocessor::utf8string2WideString(new_text);
    auto bpe_tokens = bpe_._bpe(w_text);

    return bpe_tokens;
  }

  std::vector<std::wstring> tokenize(const std::string& str) override {
    // Gemma tokenizer doesn't add prefix space, just tokenize directly
    return _tokenize(str);
  }

  std::wstring _detokenize(int64_t pos_idx) override { return bpe_._lookup_inverse_vocab(pos_idx); }

  std::wstring detokenize(int64_t pos_idx) override {
    auto str = _detokenize(pos_idx);
    // Replace "▁" back to space during detokenization
    std::wstring result;
    for (wchar_t c : str) {
      if (c == L'▁') {
        result += L' ';
      } else {
        result += c;
      }
    }
    return result;
  }

  Tensor convert2Ids(const std::vector<std::wstring>& strs) override {
    std::vector<int64_t> ids;
    ids.reserve(strs.size());
    for (const auto& str : strs) { ids.emplace_back(bpe_._lookup_vocab(str)); }
    Tensor ret = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                     .setMemType(kExtraInput)
                     .setName("gemma3-tokenizer-i0")
                     .alloc();

    auto ptr = ret.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return ret;
  }

  ARGenerationOutputPast convertMessage(const Gemma3Message& message) {
    // Tokenize the prompt
    auto sequence_str = tokenize(message.prompt);
    std::vector<int64_t> ids;
    
    // Insert bos_token at the beginning
    ids.push_back(bos_token_id_);
    
    ids.reserve(sequence_str.size() + 1);
    for (const auto& str : sequence_str) { ids.emplace_back(bpe_._lookup_vocab(str)); }

    // Get sequence Tensor
    Tensor sequence = Tensor::empty({/*batch*/ 1, /*seq*/ (int32_t)ids.size()}, kInt64, kCPU)
                          .setMemType(kNormal)
                          .setName("gemma3-tokenizer-i0")
                          .alloc();

    auto ptr = sequence.ptr<int64_t>();
    for (size_t i = 0; i < ids.size(); ++i) { ptr[i] = ids[i]; }

    return {
        {"sequence", sequence},
    };
  }

  int64_t bos_token_id() const { return bos_token_id_; }
  int64_t eos_token_id() const { return eos_token_id_; }
  int64_t pad_token_id() const { return pad_token_id_; }

 private:
  // For text
  preprocessor::BPE bpe_;
  int64_t bos_token_id_;
  int64_t eos_token_id_;
  int64_t pad_token_id_;
};

}  // namespace mllm::models::gemma3

