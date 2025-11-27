add_executable(mllm-qwen3-npu main.cpp)
target_link_libraries(mllm-qwen3-npu PRIVATE MllmRT MllmCPUBackend MllmQNNBackend)
target_include_directories(mllm-qwen3-npu PRIVATE ${MLLM_INCLUDE_DIR})

