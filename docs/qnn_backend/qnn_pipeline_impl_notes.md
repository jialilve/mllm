# QNN Qwen Pipeline 实现说明（v1 经验 + v2 设计）

> 目的：结合 `mllm_v1/examples/demo_qwen_npu_pipeline.cpp`、`mllm_v1/mllm/Parallel.hpp`、`mllm_v2/mllm/models/qwen_npu/modeling_pipeline_trace_simplified.hpp` 以及 `mllm::async` 接口，解释现有/计划中的 pipeline 怎么落地，便于后续开发。

## 1. v1 版本 ChunkPipeline 回顾

### 1.1 运行流程（来自 `demo_qwen_npu_pipeline.cpp`）
1. **初始化**：加载 tokenizer、QNN 模型、decoding 模型，设置 chunk_size=128。
2. **Trace & Warmup**：`Tracer::trace(&model, {input_tensor});` 会将模型切成多个 graph（`Tracer::model_`）。
3. **ChunkPipeline::run**：
   - 把输入 tensor 深拷贝后按 chunk_size 切片，存入 `chunked_tensors`。
   - 遍历每个 chunk，按图编号触发 `Tracer::model_[i]->Forward({}, {chunk_id});`。
   - OMP 并行：`#pragma omp parallel for num_threads(2)`，一次处理两个 chunk，形成“交错执行”。
   - 最后从 CPU graph 结果里取 logits，并做一次 argmax。
4. **Decode 阶段**：prefill 结束后切回 CPU `decoding_model` 做自回归。

### 1.2 关键点
| 组件 | 作用 |
| --- | --- |
| `Tracer::model_` | 保存多个子图（包含 CPU/QNN graph），是 pipeline 的执行单位。 |
| `executeFunc(chunk_id, graphIdx)` | 计算 `i = graphIdx - chunk_id`，控制“斜线型”执行顺序，确保 chunk_i 只在合法 graph 上运行。 |
| OMP 循环 | 使用双线程，每轮推进两个 chunk，实现最小粒度的 pipeline。 |
| `Context::Instance().inference_state()` | 维护 chunk_size、模式（Prompt/Decode）等全局状态，供算子读取。 |

### 1.3 局限
- 并行粒度受限：固定为两个 chunk；CPU/NPU graph 没显式区分。
- 对 v2 Trace 不兼容：`Tracer::model_` 接口在 v2 中已变动，需要新的抽象。
- 难以扩展到 Module Async：OMP 循环写死在单函数里。

## 2. v2 方向：PipelineExecutor + Module Async

### 2.1 `modeling_pipeline_trace_simplified.hpp` 提供的思路
- **ContextManager**：维护 `prefill_128`, `decode_1` 等上下文，便于运行时选择合适的 QNN 图。
- **PipelineExecutor**：内部有两个队列（CPU/QNN），分别由线程消费；任务以 `PipelineTask` 表示（包含 inputs、stage、chunk_id）。
- **HybridLlamaForCausalLM**：示例性地展示如何把 input 分 chunk，交替提交到不同设备，并在 `trace` 中做 chunk 级别记录。
- 目前还是“骨架”实现：`executeCPUTask/executeQNNTask` 里没有真实算子，但结构已经对齐论文提出的 CPU/NPU stage。

#### v1 vs v2 Trace 行为补充
- **v1**：`Tracer::trace` 会直接向 `Tracer::model_` 这个 `std::vector<std::shared_ptr<Module>>` 里 push 子图模块。Trace 期间实际调用的是 `Module::forward`，forward 内部的 `addModule/addOp`（见 `mllm_v1/mllm/Trace.cpp`）会把每个子模块编译成独立 graph，按执行顺序落入 `Tracer::model_`。因此得到的序列里天然混合了 QNN graph 与 CPU graph。
- **v2**：默认 trace 走新的 IR（`ir::lowlevel`），输出的是 `IRContext`，不再自动生成 `Tracer::model_`。如果需要 v1 式的“图列表”，要么复用 v1 Trace 模块，要么扩展 v2 trace pass 生成类似的数据结构。因此在开始实现 pipeline 前，**必须先和师兄确认**：是沿用 v1 Trace 机制，还是在 v2 IR 基础上新增“graph 列表”导出能力，再做 pipeline。

### 2.2 Module Async 接口（`mllm::async`）
- `mllm::async::fork(module, tensors...)`：包装成 Task，提交到 `dispatcherManager()`；不会立即运行，需要调用 `wait`。
- `mllm::async::wait(future_0, future_1, ...)`：阻塞直到所有任务完成，然后返回每个任务的输出向量。
- 适用场景：
  1. 同一个 module（例如单层 decoder）多次执行，可以异步 fork 多个任务，然后统一 wait。
  2. 结合 pipeline：CPU 子图、NPU 子图各自 fork，主线程在合适位置 wait，达到 Stage overlap。
- 注意：async 目前尚未在生产路径验证，需要在示例/测试中逐步引入。

### 2.3 建议的实现策略
1. **拆出 ChunkBuilder**：负责按照 chunk_size 填充 `sequence`、`position_ids`、`real_len`，供 pipeline 阶段使用。
2. **Task Dispatch**：  
   - Prefill：将“QNN heavy graph” 通过 `mllm::async::fork(qnn_submodule, chunk_inputs...)` 投递；  
   - CPU 辅助图（RoPE、KV 合并、采样）则投递到 CPU worker。  
   - `PipelineExecutor` 可以封装这两类 fork/wait，统一返回 future。
3. **调度策略**：  
   - 初版可照 v1：双 chunk，交错执行。  
   - 进一步可改成：`std::future` 或 `mllm::async` 组合，让“准备 chunk_{i+1}”和“执行 chunk_i”重叠。
4. **日志/可视化**：在每次任务提交/完成时打印 `chunk_id`, `stage`, `device`，帮助验证流水线是否按预期推进。

### 2.4 `Tracer::model_` 结构速记
- 所在文件：`mllm_v1/mllm/Trace.hpp|cpp`。`Tracer::model_` 是一个 `std::vector<std::shared_ptr<Module>>`。
- 工作流：当执行 `Tracer::trace(&model, tensors)` 时，会：
  1. 进入 `Tracer::trace` → 调用 `model->forward`，但上下文设置成 “记录模式”；
  2. 每当 `forward` 内部遇到子模块 / 算子，会触发 `Tracer::addModule` 或 `Tracer::addOp`；
  3. 这些 add 函数负责把相应的 module 封装成 graph node 并 push 到 `Tracer::model_`；
  4. 最终 `Tracer::model_` 的顺序即 forward 时的执行顺序，天然保持 CPU/NPU 子图的混合编排。
- 因此：虽然在 `trace()` 函数里看不到 `model_` 的直接操作，但在调用栈更深处的 `addModule/addOp` 已经完成 push，这也是为什么 v1 pipeline 能遍历 `Tracer::model_` 直接驱动 graph。

## 3. Pipeline 伪代码（结合 v1/v2）
```
chunks = chunk_builder.split(prompt_tokens)

// Prefill 阶段
future_cpu = None
future_qnn = mllm::async::fork(qnn_prefill_module, chunks[0].asInputs())

for i in range(len(chunks)):
    if i + 1 < len(chunks):
        future_cpu = std::async(build_next_chunk, chunks[i+1])
    logits_i = mllm::async::wait(future_qnn)[0]
    if i + 1 < len(chunks):
        chunk_next = future_cpu.get()
        future_qnn = mllm::async::fork(qnn_prefill_module, chunk_next.asInputs())

// Decode 阶段
while not finished:
    cpu_future = std::async(prepare_decode_inputs, latest_token)
    future_qnn = mllm::async::fork(qnn_decode_module, cpu_future.get())
    logits = mllm::async::wait(future_qnn)[0]
    token = sampleGreedy(logits)
```

## 4. 文档用途
- 拿去和师兄对齐：“我们计划沿用 v1 ChunkPipeline 的 chunk 管理 + 新增 Module Async 来实现 Stage overlap”。
- 作为 AI prompt：让 AI 根据本说明生成 `PipelineExecutor`、`ChunkBuilder`、`AsyncRunner` 等代码。
- 作为测试清单：日志中应看到 `chunk_id` 随时间交错、NPU/CPU 线程利用率提升。

> 后续如果在 v2 中真正实现 pipeline，请在 PR 中引用本文件和《qnn_pipeline_paper_reference.md》，说明“当前实现处于哪个阶段/与论文的差距”，方便所有人快速了解背景。


