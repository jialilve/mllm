add_executable(mllm-gemma3-runner main.cpp)
target_link_libraries(mllm-gemma3-runner PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-gemma3-runner PRIVATE ${MLLM_INCLUDE_DIR})

