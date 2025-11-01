#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/gemma3/modeling_gemma3_fa2.hpp>
#include <mllm/models/gemma3/tokenization_gemma3.hpp>
#include <mllm/models/gemma3/configuration_gemma3.hpp>
#include <mllm/utils/AnyValue.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);

  Argparse::parse(argc, argv);

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::start();
#endif

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
  if (model_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (model_version.get() == "v2") {
    file_version = mllm::ModelFileVersion::kV2;
  }

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

  {
    auto gemma3_cfg = mllm::models::gemma3::Gemma3Config(config_path.get());
    auto gemma3_tokenizer = mllm::models::gemma3::Gemma3Tokenizer(tokenizer_path.get());
    auto gemma3 = mllm::models::gemma3::Gemma3ForCausalLM(gemma3_cfg);

    auto param = mllm::load(model_path.get(), file_version);
    gemma3.load(param);

    fmt::print("\n{:*^60}\n", " Gemma3 Interactive CLI ");
    fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

    std::string prompt_text;

    fmt::print("💬 Prompt text (or 'exit/quit'): ");
    std::getline(std::cin, prompt_text);

    try {
      fmt::print("🔄 Processing...\n");
      auto inputs = gemma3_tokenizer.convertMessage({.prompt = prompt_text});

      fmt::print("\n🤖 Response: ");

      // Use for loop
      for (auto& step : gemma3.chat(inputs)) { std::wcout << gemma3_tokenizer.detokenize(step.cur_token_id) << std::flush; }

      fmt::print("\n{}\n", std::string(60, '-'));
    } catch (const std::exception& e) { fmt::print("\n❌ Error: {}\n{}\n", e.what(), std::string(60, '-')); }
    
    gemma3.perfSummary();
  }

#ifdef MLLM_PERFETTO_ENABLE
  mllm::perf::stop();
  mllm::perf::saveReport("gemma3.perf");
#endif

  mllm::print("\n");
  mllm::memoryReport();
})

