## v2 `modeling_pipeline_trace_simplified.hpp` 流水线思路说明

> 目标：用**纯中文 + 伪代码**，从零解释 v2 中这份示例文件是怎么设计“多 chunk + CPU/QNN 流水线”的，重点是**思路**而不是 C++ 语法，方便你后面按这个思路移植到真正的 Qwen NPU 模型里。

---

## 1. 先回答两个关键对比问题

- **v1：`demo_qwen_npu_pipeline.cpp` + `Parallel.hpp`**
  - Trace 时使用 `Tracer::trace(&model, {input_tensor})`，生成的是一个 **`Tracer::model_`（vector<Module>）**：
    - 里面按执行顺序排好了多个子图（graph）；
    - 这些子图既有 **QNN 子图**（通过 `SubgraphStart/SubgraphFinalize` 切出来），也有 **CPU 子图**（`CPUModuleWrapper`）；
  - `ChunkPipeline::run()` 做的事情：对同一组 graph（`Tracer::model_`）在不同 chunk 上用 OMP 双线程“交错调度”，形成 v1 版的 pipeline。

- **v2：`examples/qwen_npu/main.cpp` + `modeling_qwen_npu.hpp`**
  - v2 的 `trace` 走的是 **新的 IR Trace 路径**：
    - `auto irs = model.trace(inputs, {});`
    - 返回的是一个 `std::map<std::string, ir::IRContext::ptr_t>`，例如 `irs["model"]`；
    - 随后用 `PassManager` 做 `QNNGraphIOTensorPass`、`QNNOpNamingPass`、`QNNGraphBuildPass` 等图级优化。
  - **没有直接暴露出类似 v1 `Tracer::model_` 的“CPU/QNN 子图列表”**。
  - 目前 `examples/qwen_npu/main.cpp` 里，对 multi-chunk 的 prefill+decode 是**串行逻辑**：一块一块地调用 `model.forward()`，没有拆分成“多个子图 + 调度”。

因此：

- **v1 的 pipeline = “Trace 出一堆 graph（CPU/QNN 混合）+ OMP 调度 graph×chunk”**。
- **v2 当前的 Qwen NPU demo = “Trace 出一个统一的 IRContext + 串行按 chunk 调 `forward`”，还没有 graph 级别的 pipeline 调度**。

`modeling_pipeline_trace_simplified.hpp` 就是为了 **在 v2 的 IR / Backend 架构下，设计一个更"正宗"的 CPU/NPU 流水线框架**，它本身是一个**教学/骨架示例**，还没有真正接上 Qwen 模型。

**⚠️ 重要**：
- 这个文件**不是**重新写的 `modeling_qwen_npu.hpp`
- 这个文件**不是**可以直接替换 `modeling_qwen_npu.hpp` 在生产环境使用
- 这个文件**是**一个设计思路的演示，展示如何在 v2 架构下实现 pipeline
- 真正实现时，需要基于 `modeling_qwen_npu.hpp` 改造，集成 pipeline 逻辑
- 详见 `qnn_pipeline_v2_detailed_explanation.md` 第 1 节

---

## 2. 文件中出现的几个核心概念（用中文解释）

### 2.1 ExecutionStage（执行阶段）

```c++
enum class ExecutionStage {
  PREFILL,
  DECODE
};
```

- **含义**：把整个推理流程粗分成两个阶段：
  - **PREFILL**：长输入提示词的“预填充”阶段，序列长度大，一般是 `seq_len > chunk_size`。
  - **DECODE**：自回归生成阶段，一次只处理少量 token，比如 1 或很小的窗口。
- **用途**：后面所有“上下文”、“任务”、“trace”都要根据阶段来区分配置（不同阶段需要的 QNN 图不一样）。

### 2.2 QNNContextInfo / ContextManager

```c++
struct QNNContextInfo {
  std::string context_file_path;
  std::string graph_name;
  ExecutionStage stage;
  int sequence_length;
  void* qnn_model_ptr = nullptr;
};
```

- 可以把它想成一条“**QNN 子图配置记录**”：
  - `context_file_path`：这个 QNN 图存在哪个 `.bin` 文件里；
  - `graph_name`：在 QNN Backend 里识别这张图的名字；
  - `stage`：它是 prefill 用，还是 decode 用；
  - `sequence_length`：适配的序列长度是多少（例如 prefill_128、prefill_256、decode_1、decode_8）。

`ContextManager` 负责：

- **注册上下文**：

```c++
bool registerContext(const std::string& context_key,
                     const std::string& context_file, 
                     ExecutionStage stage, int seq_length);
```

  - 相当于“在字典里登记：某个 stage + 某个长度，对应哪个 QNN 上下文文件、graph 名字”。

- **按需求查找最合适的上下文**：

```c++
QNNContextInfo* getContext(ExecutionStage stage, int seq_length);
```

  - 先尝试精确匹配；
  - 如果没有，就找“同一阶段、长度 ≥ 需求长度中最接近的一条”。
  - 对应论文里的思想：**相同结构但不同序列长度的图，预编译好几份，然后运行时择优选用**。

### 2.3 PipelineTask / PipelineExecutor

```c++
struct PipelineTask {
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  ExecutionStage stage;
  int chunk_id;
  std::promise<std::vector<Tensor>> promise;
};
```

- **PipelineTask**：一条流水线任务：
  - 输入 tensor 列表；
  - 是 prefill 还是 decode；
  - 属于第几个 chunk；
  - 内部带一个 `promise`，方便异步返回结果。

`PipelineExecutor` 是流水线的“调度中心”：

- 内部有两条队列 + 两个线程：

```c++
std::queue<PipelineTask> cpu_task_queue_;
std::queue<PipelineTask> qnn_task_queue_;
std::thread cpu_worker_;
std::thread qnn_worker_;
```

- 对外的主要接口：

```c++
std::future<std::vector<Tensor>> submitTask(
    const std::vector<Tensor>& inputs, 
    ExecutionStage stage, 
    DeviceTypes target_device, 
    int chunk_id = 0);
```

**用中文伪代码描述 `submitTask` 思路：**

1. 来了一条任务：包括输入张量、阶段（prefill/decode）、目标设备（CPU/QNN）、chunk_id。
2. 创建一个 `PipelineTask`，把这些信息塞进去，同时拿出其中的 `future`。
3. 根据 `target_device` 决定丢进哪个队列：
   - 如果是 QNN，就放到 `qnn_task_queue_`，并且通知 `qnn_worker_`；
   - 否则放到 `cpu_task_queue_`，通知 `cpu_worker_`。
4. 把这个 `future` 返回给调用者——调用方以后用 `future.get()` 等待结果即可。

内部两个 worker 线程做的事情（同样用伪代码描述）：

- **CPU worker：**

```text
循环直到 shutdown:
  - 如果队列空，就等待条件变量
  - 取出一条 PipelineTask
  - 调用 executeCPUTask(task) 做实际 CPU 计算
  - 把结果通过 task.promise.set_value(...) 返回
```

- **QNN worker：**

```text
循环直到 shutdown:
  - 如果队列空，就等待条件变量
  - 取出一条 PipelineTask
  - 先找一个合适的 QNNContextInfo（根据 stage + seq_length）
  - 调用 executeQNNTask(task) 做 QNN 推理
  - 把结果通过 task.promise.set_value(...) 返回
```

> 注意：当前文件里 `executeCPUTask/executeQNNTask` 只是打印日志的“空壳”，真正的算子调用还没接上，重点是**结构**。

### 2.4 HybridLlamaForCausalLM —— 一个“示范用虚拟模型”

```c++
class HybridLlamaForCausalLM : public nn::Module, public ARGeneration {
  std::unique_ptr<PipelineExecutor> pipeline_executor_;
  Backend* cpu_backend_;
  Backend* qnn_backend_;
  nn::Linear lm_head_;
  int chunk_size_ = 128;
  bool enable_pipeline_ = true;
};
```

- 这不是你真正要用的 Qwen 模型，而是一个“**教学用的混合 Llama**”：
  - 只挂了一个 `lm_head_`；
  - 主要目的是演示：**如何把序列按 chunk 切开，交给 `PipelineExecutor` 分发到 CPU/QNN 上，并在 trace 中记录下来**。

它的行为可以分三块理解：

1. **构造函数里：**
   - 创建 `PipelineExecutor`；
   - 调用 `registerQNNContexts()` 注册一系列 prefill / decode 的 QNN 上下文占位：

```c++
prefill_lengths = {128, 256, 512, 1024, 2048};
decode_lengths = {1, 8, 16, 32};
// 对每种长度都注册一个 context_file，例如 "qnn_prefill_128.bin"
```

2. **`forward` 里：根据序列长度选择策略**

```c++
int seq_length = sequence.sequence();
ExecutionStage stage = (seq_length > chunk_size_) ? PREFILL : DECODE;

if (enable_pipeline_ && stage == PREFILL) {
  return forwardWithPipeline(...);
} else {
  return forwardDirect(...);
}
```

用中文理解：

- 如果输入序列“长得像 prompt”，就走 **pipeline 版本**；
- 否则走 **直接 QNN 执行的简单版本**。

3. **`forwardWithPipeline`：示范"按 chunk 切分 + 提交异步任务"的流程**

伪代码描述：

```text
seq_length = 输入序列长度
chunk_num = ceil(seq_length / chunk_size_)
futures = []

for chunk_id in 0..chunk_num-1:
  chunk_input = 从大序列中切出 [chunk_id] 这一块
  target_device = (chunk_id 为偶数 -> QNN, 为奇数 -> CPU)   // ⚠️ 这只是演示，实际是错误的！
  future = pipeline_executor.submitTask(
              {chunk_input},
              PREFILL,
              target_device,
              chunk_id)
  futures.push(future)

// 等所有 chunk 跑完
chunk_outputs = []
for f in futures:
  outputs = f.get()
  把 outputs 里的 tensor 全部 append 到 chunk_outputs

merged_output = mergeChunkOutputs(chunk_outputs)
return {"logits": merged_output}
```

**⚠️ 重要说明**：
- `chunk_id % 2` 这个设计**是错误的**！它假设"整个 chunk 在 QNN 或 CPU 上执行"，但实际上：
  - 一个 chunk 内部既有 QNN 部分（AttentionProj、OutProj+MLP），也有 CPU 部分（RoPE、KVCache、Softmax）
  - 不能简单按 chunk_id 分设备
- 这里只是演示"如何提交任务到不同队列"的**框架**，不是真正的 pipeline 逻辑
- 真正接 Qwen NPU 的时候，应该：
  - 把**一个 chunk 的 forward 拆成多个 task**：
    - Task 1：QNN 部分（AttentionProj）→ `submitTask(..., MLLM_QNN, ...)`
    - Task 2：CPU 部分（RoPE、KVCache）→ `submitTask(..., MLLM_CPU, ...)`（等待 Task 1）
    - Task 3：QNN 部分（OutProj+MLP）→ `submitTask(..., MLLM_QNN, ...)`（等待 Task 2）
  - 不同 chunk 的 task 可以交错执行，形成 pipeline
  - 详见 `qnn_pipeline_v2_detailed_explanation.md` 第 3 节

4. **`trace`：如何在 IR 里把“分 chunk + 不同设备执行”的意图记录下来**

```c++
IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) {
  ir::lowlevel::traceStart();

  if (seq_length > chunk_size_) {
    traceWithChunks(sequence);
  } else {
    traceDirect(sequence);
  }

  ir::lowlevel::traceComment("Hybrid CPU-QNN Pipeline Execution");
  llm_ir = ir::lowlevel::traceStop();
  return {{"model", llm_ir}};
}
```

- `traceWithChunks` 里用伪代码：

```text
for chunk_id in 0..chunk_num-1:
  chunk_input = createChunkTensor(sequence, chunk_id)
  if chunk_id 偶数:
    traceComment("QNN Chunk i execution")
    lm_head_(chunk_input)  // 在 IR 中标记：这个 chunk 的这部分在 QNN 上
  else:
    traceComment("CPU Chunk i execution")
    lm_head_(chunk_input)  // 在 IR 中标记：这个 chunk 的这部分在 CPU 上
```

**关键理解**：
- Trace 的作用是**在 IR 中记录模型的计算图结构**，包括：
  - 哪些部分在 QNN 上（例如 AttentionProj、OutProj+MLP）
  - 哪些部分在 CPU 上（例如 RoPE、KVCache、Softmax）
  - 每个 chunk 如何切分，每个部分在哪里执行
- **不是**"整个 chunk 在 QNN 或 CPU 上"，而是"chunk 的不同阶段在不同设备上"
- 后续的 Pass（例如 `QNNGraphBuildPass`）可以根据这些标记，把 QNN 部分编译成独立的图
- 详见 `qnn_pipeline_v2_detailed_explanation.md` 第 6 节

---

## 3. 总结：v2 这份文件的“pipeline 思路”到底是什么？

用一句话概括：  
**“把 v1 那种硬编码 OMP 循环，升级成：任务队列 + 两条 worker（CPU/QNN）+ 可根据阶段/长度选择 QNN 图 + IR 里可见的 chunk 级 trace。”**

拆开来讲：

- **不再直接操作 `Tracer::model_`**：
  - v2 默认走 IR → PassManager → QNNGraphBuild 的链路；
  - 这份文件假设“我们已经有若干个预编译好的 QNN 上下文（prefill_128 / decode_1 等）”，运行时只需挑选并执行。

- **用 `ContextManager` 管理多个"QNN 子图版本"**：
  - 解决"不同 seq_len 需要不同图"的问题；
  - 对应论文中"多图 + runtime 选择"的思想。
  - **关键**：这些图是在**模型加载/初始化时预编译的**，不是 trace 时生成的
  - 例如：`prefill_128.bin`、`prefill_256.bin`、`decode_1.bin` 等
  - 运行时根据实际输入长度，选择最合适的图
  - 详见 `qnn_pipeline_v2_detailed_explanation.md` 第 2 节

- **用 `PipelineExecutor` + 两个 worker 把 CPU/NPU 当成两个 stage**：
  - CPU 阶段（比如 RoPE / KV / 采样）放进 CPU 队列；
  - NPU 阶段（大矩阵乘 / MLP）放进 QNN 队列；
  - 利用 `std::future + promise` 实现真正的异步流水线。
  - **Overlap 的形成**：
    - QNN worker 和 CPU worker 是**两个独立线程**，并行运行
    - 当 QNN worker 执行 chunk0 的 QNN 部分时，CPU worker 可以执行 chunk1 的 CPU 部分（如果数据准备好了）
    - 这就是 pipeline 的 overlap
  - 详见 `qnn_pipeline_v2_detailed_explanation.md` 第 4 节

- **用 `HybridLlamaForCausalLM` 演示 chunk 级切分 + 提交任务 + trace**：
  - 这只是一个"教学模型"，后面真正做 Qwen NPU pipeline 时，你会：
    - 用真正的 `QwenForCausalLM`（基于 `modeling_qwen_npu.hpp` 改造）；
    - 在 prefill 阶段把大 prompt 分 chunk；
    - 对每个 chunk，把 forward 拆成多个 task：
      - Task 1：QNN 部分（AttentionProj）→ 提交到 QNN worker
      - Task 2：CPU 部分（RoPE、KVCache）→ 提交到 CPU worker（等待 Task 1）
      - Task 3：QNN 部分（OutProj+MLP）→ 提交到 QNN worker（等待 Task 2）
    - 不同 chunk 的 task 可以交错执行，形成 pipeline
    - 通过 future 按顺序收集结果并合并
    - 详见 `qnn_pipeline_v2_detailed_explanation.md` 第 5、7 节

---

## 4. 你可以怎么利用这份设计？

从“学习/实现”角度推荐的顺序：

1. **先把这份文件当成“只看中文逻辑，不看 C++ 语法”的教材**：
   - 你可以把上面的伪代码 + 解释抄到自己的笔记里；
   - 自己试着画一个简单的时序图：CPU 阶段 / QNN 阶段各自做什么、怎样错开。

2. **和师兄对齐时，可以这样说**：
   - v1 的 pipeline 是：`Tracer::model_` + OMP 硬编码；
   - v2 想做的是：`IR + ContextManager + PipelineExecutor`，在结构上更接近论文（多 QNN context + CPU/NPU stage overlap）。

3. **等你完全看懂这份思路之后，再讨论“如何把它移植到真正的 Qwen NPU 模型上”**：
   - 哪一段逻辑应该放 QNN worker；
   - 哪些 CPU 步骤可以独立成 task；
   - multi-chunk 的 prefill / decode 分别如何映射到 ExecutionStage。

如果你愿意，下一步我们可以一起写一份“**Qwen NPU 专用的 Pipeline 伪代码文档**”，只用中文和流程图，不写任何 C++，专门描述：

- 多 chunk prompt 的 prefill 如何在 CPU/QNN 两条线上错峰执行；
- decode 阶段如何在一个 chunk 里循环，同时维持 KVCache 和 position_ids 对齐。 


