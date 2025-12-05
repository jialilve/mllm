# CausalMaskOp 测试指南

本文档仅供本地调试使用，并不会被纳入版本库。内容涵盖以下部分：

- CPU Kernel 测试框架的基础知识  
- `CausalMaskOp` 三个核心场景（Prefill / Decode / Append）的验证方式  
- 如何在本地复现实验并排查失败

## 1. CPU Kernel 测试基础

1. **测试可执行文件**  
   - 构建目录：`/root/mllm_v2/build-tests`  
   - 目标：`bin/Mllm-Test-CPUKernel`（链接到 GoogleTest & 内部测试基类）

2. **KernelTest 基类**（`tests/cpu/KernelTest.cpp`）  
   - 统一封装 CPU 设备初始化、Tensor allocator、常用断言工具。  
   - 具体算子测试通过 `TEST_F(<YourTestFixture>, <CaseName>)` 注册。  
   - 常用助手：`mllm::test::allClose`、`Tensor::arange` 等。

3. **运行方式**  
   ```bash
   cd /root/mllm_v2
   build-tests/bin/Mllm-Test-CPUKernel --gtest_filter=<Filter>
   ```  
   - 支持 gtest filter、`--gtest_repeat`、`--gtest_break_on_failure` 等通用参数。  
   - 推荐通过 `GLOG_v=2`/`MLLM_DEBUG=1` 打开详细日志。

## 2. CausalMaskOp 测试结构

文件 `tests/cpu/CausalMaskOpTest.hpp` 定义了 `CausalMaskOpTest` 夹具：

| 关键步骤 | 说明 |
| --- | --- |
| `SetUp()` | `mask_.to(mllm::kCPU)`：显式构建 CPU 后端算子，避免空后端导致的 SIGSEGV。 |
| `runScenario(B, H, S, D)` | 生成 `[B, H, S, D]` 的线性递增输入，调用 `mask_`，并与 `buildExpectedTensor` 的手工实现逐元素比较。 |
| `buildExpectedTensor` | 对每个 batch/head/step，允许的列数为 `min(D, context_offset + s + 1)`，超出位置写入 `-1e10f`。其中 `context_offset = max(0, D - S)` 用于模拟 decode 时更长的 KV cache。 |

因此测试验证了 **算子前向逻辑** 与 **掩码形状推导** 是否正确。

## 3. Prefill / Decode / Append 三个场景

| 场景 | gtest 用例 | 参数 (B,H,S,D) | 覆盖点 |
| --- | --- | --- | --- |
| Prefill | `CausalMaskOpTest.PrefillScenario` | (1, 1, 4, 4) | 典型 prompt + self attention，`D == S`，确保对角掩码正确。 |
| Decode | `CausalMaskOpTest.DecodeScenario` | (1, 1, 1, 6) | 序列长度 1、上下文维更大，验证 `context_offset` 的行为（只允许历史 token 的一小段）。 |
| Append | `CausalMaskOpTest.AppendScenario` | (2, 3, 3, 7) | 多 batch、多 head、`D > S`，检验广播与批量索引逻辑。 |

所有用例都通过 `runScenario()` -> `test::allClose()` 确保输出与期望完全一致；失败时会打印期望值、实际输出以及对比结果，便于定位。

## 4. 复现与调试

1. **单独运行 CausalMask 测试**  
   ```bash
   cd /root/mllm_v2
   build-tests/bin/Mllm-Test-CPUKernel --gtest_filter=CausalMaskOpTest.*
   ```

2. **常见问题**  
   - **忘记切换设备**：`mask_.to(mllm::kCPU)` 缺失会触发空指针或 fallback。  
   - **浮点误差**：若自定义掩码值，请保持 `-1e10f` 级别，防止 softmax 后被误计算。

3. **快速修改后验证**  
   - 重新构建：`cd /root/mllm_v2 && cmake --build build-tests --target Mllm-Test-CPUKernel -j$(nproc)`  
   - 结合 `gdb --args ...` 捕获异常栈；或 `MLLM_DEBUG_MASK=1`（若实现支持）输出详细掩码。
   - cd /root/mllm_v2 && build-tests/bin/Mllm-Test-CPUKernel --gtest_filter=CausalMaskOpTest.*

如需扩展更多场景，可以在同一 Test Fixture 中添加新的 `TEST_F`，沿用 `runScenario()` 的断言逻辑即可。

