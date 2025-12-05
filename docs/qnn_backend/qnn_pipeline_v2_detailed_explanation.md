# `modeling_pipeline_trace_simplified.hpp` 详细解答

> 本文档专门解答关于 `modeling_pipeline_trace_simplified.hpp` 的所有疑问，澄清设计思路和实现细节。

---

## 1. 核心问题：这个文件是什么？和 `modeling_qwen_npu.hpp` 的关系？

### 1.1 文件定位

**`modeling_pipeline_trace_simplified.hpp` 是一个教学示例/骨架代码，不是真正的 Qwen 实现。**

- **不是**：重新写的 `modeling_qwen_npu.hpp`
- **不是**：可以直接替换 `modeling_qwen_npu.hpp` 在生产环境使用
- **是**：一个**设计思路的演示**，展示如何在 v2 架构下实现 pipeline

### 1.2 与 `modeling_qwen_npu.hpp` 的关系

```
modeling_qwen_npu.hpp (真正的 Qwen 实现)
    ↓
    ├─ 完整的 Qwen 模型结构（QwenAttentionProjNPU、QwenAttentionMatmul、QwenOutProjAndMLP 等）
    ├─ 真实的 forward 逻辑
    └─ 真实的 trace 逻辑

modeling_pipeline_trace_simplified.hpp (教学示例)
    ↓
    ├─ HybridLlamaForCausalLM（虚拟模型，只有 lm_head_）
    ├─ PipelineExecutor（调度框架）
    └─ ContextManager（上下文管理）
```

**关系**：
- `modeling_pipeline_trace_simplified.hpp` 展示了**如何把 pipeline 框架接入模型**
- 真正实现时，需要把 `HybridLlamaForCausalLM` 替换成 `QwenForCausalLM`，并把 pipeline 逻辑集成进去
- **不是两个文件一起用**，而是**参考这个示例的思路，改造真正的 Qwen 模型**

### 1.3 在 demo 中的使用

**当前 `main.cpp` 使用的是 `modeling_qwen_npu.hpp`，不是 `modeling_pipeline_trace_simplified.hpp`。**

如果要使用 pipeline 版本，需要：
1. 创建一个新的 `QwenForCausalLMWithPipeline`，继承或组合 `PipelineExecutor`
2. 在 `forward` 中集成 pipeline 逻辑
3. 在 `main.cpp` 中使用新类

---

## 2. Trace 的作用：预编译多个 QNN 上下文？

### 2.1 Trace 的两个阶段

**Trace 阶段（编译时）**：
```cpp
// 在 main.cpp 中，trace 时传入的是完整的 prompt（可能很长）
auto irs = model.trace(inputs, {});  // inputs 的 sequence 可能是 [1, 512] 或更长
```

**关键点**：
- Trace 时**不是**用 `chunk_size=128` 的固定输入
- Trace 时传入的是**真实的 prompt 长度**（例如 512 tokens）
- Trace 会记录下"整个模型的计算图"，包括：
  - 哪些部分在 QNN 上（例如 `QwenAttentionProjNPU`、`QwenOutProjAndMLP`）
  - 哪些部分在 CPU 上（例如 `QwenAttentionMatmul` 的 RoPE、KVCache、Softmax）

### 2.2 预编译多个 QNN 上下文

**`ContextManager` 的作用**：

```cpp
// 在构造函数中注册多个预编译的 QNN 图
registerQNNContexts() {
    prefill_lengths = {128, 256, 512, 1024, 2048};  // 不同长度的 prefill 图
    decode_lengths = {1, 8, 16, 32};                 // 不同长度的 decode 图
}
```

**含义**：
- **不是**在 trace 时生成多个图
- **而是**：在**模型加载/初始化时**，已经预编译好了多个不同长度的 QNN 图：
  - `prefill_128.bin`：处理 128 tokens 的 prefill 图
  - `prefill_256.bin`：处理 256 tokens 的 prefill 图
  - `decode_1.bin`：处理 1 token 的 decode 图
  - 等等

**为什么需要多个？**
- QNN 图是**固定形状**的（fixed shape）
- 如果 prompt 是 200 tokens，不能直接用 `prefill_128.bin`（太小），也不能用 `prefill_512.bin`（太大，浪费）
- 所以需要**预编译多个版本**，运行时**选择最合适的**

### 2.3 Trace 和 Context 的关系

```
Trace 阶段：
  - 记录模型结构（哪些在 QNN，哪些在 CPU）
  - 生成 IR（IRContext）
  - PassManager 优化 IR
  - 编译成多个 QNN 图（prefill_128.bin, prefill_256.bin, ...）

运行时：
  - ContextManager 管理这些预编译的图
  - 根据实际输入长度，选择最合适的图
  - PipelineExecutor 使用选定的图执行
```

---

## 3. Pipeline 并行机制：为什么 chunk_id 偶数/奇数分别对应 QNN/CPU？

### 3.1 ⚠️ 重要：这个设计有问题！

**代码中的设计**：
```cpp
DeviceTypes target_device = (chunk_id % 2 == 0) ? MLLM_QNN : MLLM_CPU;
```

**这个设计是错误的！** 原因：

1. **一个 chunk 内部既有 QNN 也有 CPU 部分**：
   - 看 `modeling_qwen_npu.hpp` 的 `QwenDecoder::forward`：
     ```cpp
     x = self_attn_proj_(x);           // QNN
     query_states = states[0].to(kCPU); // 转到 CPU
     x = self_attn_matmul_(...);        // CPU（RoPE、KVCache、Softmax）
     x = x.to(kQNN);                    // 转回 QNN
     x = self_attn_out_mlp_(x, res);    // QNN
     ```
   - 一个 chunk 的执行是：**QNN → CPU → QNN → CPU → ...** 交替的

2. **不能简单按 chunk_id 分设备**：
   - 如果 chunk0 全部在 QNN，chunk1 全部在 CPU，那么：
     - chunk0 的 CPU 部分（RoPE、KVCache）无法执行
     - chunk1 的 QNN 部分（矩阵乘、MLP）无法执行
   - **结果是错误的！**

### 3.2 正确的 Pipeline 思路

**正确的 pipeline 应该是**：

```
时间线：
T0: Chunk0 的 QNN 部分（AttentionProj、OutProj+MLP）在 QNN worker 执行
    Chunk0 的 CPU 部分（RoPE、KVCache）在 CPU worker 执行
    → 这两个是**同一个 chunk 的不同阶段**，需要顺序执行

T1: Chunk0 的 QNN 部分完成 → 结果传给 Chunk0 的 CPU 部分
    Chunk1 的 QNN 部分开始（如果 Chunk0 的 CPU 部分不阻塞）

T2: Chunk0 的 CPU 部分完成 → 结果传给 Chunk0 的下一个 QNN 部分
    Chunk1 的 CPU 部分开始（如果 Chunk1 的 QNN 部分已完成）
```

**关键**：
- **不是**"chunk0 在 QNN，chunk1 在 CPU"并行
- **而是**"chunk0 的 QNN 阶段和 chunk1 的 CPU 阶段"并行（如果数据依赖允许）

### 3.3 为什么示例代码这样写？

**`modeling_pipeline_trace_simplified.hpp` 中的 `chunk_id % 2` 只是演示"如何提交任务到不同队列"，不是真正的 pipeline 逻辑。**

看代码：
```cpp
// 这只是演示：如何把任务提交到不同的 worker
DeviceTypes target_device = (chunk_id % 2 == 0) ? MLLM_QNN : MLLM_CPU;
auto future = pipeline_executor_->submitTask(..., target_device, ...);
```

**真正实现时应该**：
- 一个 chunk 的 forward 会拆成多个 task：
  - Task 1：QNN 部分（AttentionProj）→ 提交到 QNN worker
  - Task 2：CPU 部分（RoPE、KVCache）→ 提交到 CPU worker（等待 Task 1 完成）
  - Task 3：QNN 部分（OutProj+MLP）→ 提交到 QNN worker（等待 Task 2 完成）
- 不同 chunk 的 task 可以交错执行，形成 pipeline

---

## 4. Worker 线程如何形成 Overlap？

### 4.1 PipelineExecutor 的架构

```cpp
PipelineExecutor {
    cpu_task_queue_    // CPU 任务队列
    qnn_task_queue_    // QNN 任务队列
    cpu_worker_        // CPU worker 线程
    qnn_worker_        // QNN worker 线程
}
```

### 4.2 Overlap 的形成

**执行顺序确认**（详细代码分析）：

#### v2 版本的执行顺序

**完整模型流程**（`QwenForCausalLM::forward`，第 475-510 行）：
1. **Embedding**（第 508 行）：`auto input_embeddings = model.embedding_(sequence);` 
   - 在 **QNN 上执行**（第 417 行设置了 `embedding_.to(kQNN)`）
   - **注意**：第 417 行的注释 "execute on CPU" 可能是过时的或不准确的，实际代码 `embedding_.to(kQNN)` 表示 embedding 在 QNN 上执行
2. **QwenText**（第 510 行）：`model(input_embeddings, ...)`
   - 输入 `x` 已经在 QNN 上（来自 embedding）
3. **每个 Decoder 层**（`QwenDecoder::forward`，第 377-399 行）：
   - 第 382 行：`x = x.to(kQNN);` - **显式转换到 QNN**（虽然输入已经在 QNN 上）
   - 第 384 行：`self_attn_proj_(x)` - **QNN 部分**（AttentionProj：QKV 投影）
   - 第 386-388 行：转到 CPU
   - 第 390 行：`self_attn_matmul_(...)` - **CPU 部分**（AttentionMatmul：RoPE + KVCache + Attention）
   - 第 392 行：转回 QNN
   - 第 396 行：`self_attn_out_mlp_(x, res)` - **QNN 部分**（OutProj + MLP）
4. **最终 Norm**（第 434-435 行）：转到 CPU 执行

**结论**：v2 版本每个 decoder 层的执行顺序是 **QNN（AttentionProj）→ CPU（AttentionMatmul）→ QNN（OutProj+MLP）**，从 **QNN 开始**。

#### v1 版本的执行顺序（对比）

**完整模型流程**（`QWenForCausalLM_NPU::Forward`，第 684-695 行）：
1. **Embedding**（第 685 行）：`auto x = embedding(inputs[0]);`
   - 在 **CPU 上执行**（默认）
2. **每个 Decoder 层**（`QwenNPU_CPUDecoder::Forward`，第 480-514 行）：
   - 第 483 行：`x = input_layernorm(inputs[0]);` - **CPU 部分**（LayerNorm）
   - 第 484 行：`x = pre_attn_quantize(x);` - **CPU 部分**（Quantize）
   - 第 488 行：`(*part1)({x})` - **QNN 部分**（Part1：QKV 投影）
   - 第 503 行：`qkv_mm({q, k, v})` - **CPU 部分**（QKVmm：RoPE + KVCache + Attention）
   - 第 507 行：`(*part2)({o_x, res})` - **QNN 部分**（Part2：O_proj + MLP）

**结论**：v1 版本每个 decoder 层的执行顺序是 **CPU（LayerNorm+Quantize）→ QNN（Part1）→ CPU（QKVmm）→ QNN（Part2）**，从 **CPU 开始**。

#### 为什么 v2 和 v1 不一样？

1. **v2 的设计理念**：
   - Embedding 在 QNN 上执行（第 417 行：`embedding_.to(kQNN)`）
   - 因此第一层 decoder 的输入已经在 QNN 上，可以直接开始 QNN 计算
   - 减少了 CPU → QNN 的转换开销

2. **v1 的设计理念**：
   - Embedding 在 CPU 上执行（默认）
   - 第一层 decoder 需要先做 LayerNorm 和 Quantize（CPU），然后才进入 QNN
   - 这种设计可能是为了在 CPU 上做更多的预处理

**代码位置**：
- v2：`/root/mllm_v2/mllm/models/qwen_npu/modeling_qwen_npu.hpp`
  - Embedding：第 416-417 行
  - Decoder forward：第 377-399 行
- v1：`/root/mllm_v1/mllm/models/qwen/modeling_qwen_npu_v2.hpp`
  - Embedding：第 672 行（默认 CPU）
  - Decoder forward：第 480-514 行

**正确的时序图**：

```
主线程：
  T0: submitTask(chunk0_qnn_task) → 返回 future0
  T1: submitTask(chunk0_cpu_task) → 返回 future1（但需要等 future0）
  T2: submitTask(chunk1_qnn_task) → 返回 future2（可以立即开始，因为 QNN worker 可能空闲）
  T3: submitTask(chunk1_cpu_task) → 返回 future3（但需要等 future2）

QNN Worker 线程：
  T0: 从队列取出 chunk0_qnn_task，开始执行
  T1: chunk0_qnn_task 完成，结果返回给 future0
  T2: 从队列取出 chunk1_qnn_task，开始执行（此时 CPU worker 正在执行 chunk0_cpu_task）
  T3: chunk1_qnn_task 完成，结果返回给 future2

CPU Worker 线程：
  T1: 等待 future0.get() → 获取 chunk0_qnn_task 的结果
  T1: 开始执行 chunk0_cpu_task（此时 QNN worker 可以执行 chunk1_qnn_task）
  T2: chunk0_cpu_task 执行中...
  T3: 等待 future2.get() → 获取 chunk1_qnn_task 的结果
  T3: 开始执行 chunk1_cpu_task
```

**关键**：
- QNN worker 和 CPU worker **并行运行**（两个独立线程）
- **正确的 overlap**：当 CPU worker 在执行 chunk0 的 CPU 部分时，QNN worker 可以执行 chunk1 的 QNN 部分
- **为什么这样？**
  - Chunk0 的 CPU 部分（AttentionMatmul）需要等待 Chunk0 的 QNN 部分（AttentionProj）完成
  - Chunk1 的 CPU 部分（AttentionMatmul）需要等待 Chunk1 的 QNN 部分（AttentionProj）完成
  - 但 Chunk1 的 QNN 部分**不依赖于** Chunk0 的 CPU 部分，只依赖于 Chunk0 的 QNN 部分（因为 K/V 已经在 Chunk0 的 QNN 部分生成并写入 KV Cache）
  - 所以当 Chunk0 的 QNN 部分完成后，Chunk1 的 QNN 部分就可以开始执行了，即使 Chunk0 的 CPU 部分还在执行
- **错误的描述**（已修正）：不能说"当 QNN worker 在执行 chunk0 的 QNN 部分时，CPU worker 可以执行 chunk1 的 CPU 部分"，因为 chunk1 的 CPU 部分需要等待 chunk1 的 QNN 部分完成

### 4.3 为什么需要两个队列？

**两个队列的作用**：
- `qnn_task_queue_`：所有需要在 QNN 上执行的任务
- `cpu_task_queue_`：所有需要在 CPU 上执行的任务

**为什么分开？**
- QNN 和 CPU 是**不同的执行资源**（NPU 和 CPU 核心）
- 分开队列可以让两个 worker **独立调度**，不会互相阻塞
- 例如：QNN worker 在执行一个耗时任务时，CPU worker 可以继续处理其他 CPU 任务

---

## 5. Future 如何拼回结果？

### 5.1 Future 的机制

```cpp
// 提交任务，获得 future
auto future = pipeline_executor_->submitTask(...);

// future.get() 会阻塞，直到任务完成，然后返回结果
auto outputs = future.get();
```

### 5.2 拼回结果的代码

```cpp
std::vector<std::future<std::vector<Tensor>>> futures;

// 提交所有 chunk 的任务
for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
    auto future = pipeline_executor_->submitTask(...);
    futures.push_back(std::move(future));
}

// 按顺序收集结果
std::vector<Tensor> chunk_outputs;
for (auto& future : futures) {
    auto outputs = future.get();  // 阻塞等待这个 chunk 完成
    chunk_outputs.insert(chunk_outputs.end(), outputs.begin(), outputs.end());
}

// 合并所有 chunk 的输出
auto merged_output = mergeChunkOutputs(chunk_outputs);
```

**关键点**：
- `future.get()` 会**阻塞**，直到对应的任务完成
- 按 `chunk_id` 顺序收集，保证结果的顺序正确
- 最后用 `mergeChunkOutputs` 把多个 chunk 的输出拼接成一个完整的 tensor

### 5.3 为什么按顺序收集？

**虽然任务是并行提交的，但收集结果必须按顺序**：
- 因为 KV Cache 的写入位置依赖于 chunk 的顺序
- 如果 chunk1 的结果先返回，但 chunk0 还没完成，不能先写入 KV Cache（会破坏顺序）

---

## 6. Trace 时如何记录 Chunk 信息？

### 6.1 Trace 的作用

**Trace 不是"预编译多个 chunk 的 pipeline 逻辑"，而是"记录模型的计算图结构"**。

```cpp
void traceWithChunks(const Tensor& sequence) {
    int seq_length = sequence.sequence();  // 可能是 512
    int chunk_num = (seq_length + chunk_size_ - 1) / chunk_size_;  // 512/128 = 4
    
    for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
        auto chunk_input = createChunkTensor(sequence, chunk_id);
        
        if (chunk_id % 2 == 0) {
            ir::lowlevel::traceComment("QNN Chunk " + std::to_string(chunk_id) + " execution");
            lm_head_(chunk_input);  // 在 IR 中记录：这个 chunk 在 QNN 上执行
        } else {
            ir::lowlevel::traceComment("CPU Chunk " + std::to_string(chunk_id) + " execution");
            lm_head_(chunk_input);  // 在 IR 中记录：这个 chunk 在 CPU 上执行
        }
    }
}
```

**这段代码的作用**：
- 在 IR 中**标记**：哪些 chunk 应该在 QNN 上执行，哪些在 CPU 上执行
- 后续的 Pass（例如 `QNNGraphBuildPass`）可以根据这些标记，把 QNN 部分编译成独立的图

### 6.2 Trace 和 Forward 的区别

**Trace 阶段**：
- 传入完整的 prompt（例如 512 tokens）
- 记录"如果处理 512 tokens，应该怎么切 chunk，每个 chunk 在哪里执行"
- 生成 IR，后续编译成多个 QNN 图

**Forward 阶段**：
- 实际执行时，可能只处理 128 tokens（一个 chunk）
- 使用 trace 时编译好的 QNN 图（例如 `prefill_128.bin`）
- 通过 PipelineExecutor 调度到不同的 worker

---

## 7. 总结：正确的 Pipeline 实现思路

### 7.1 当前示例的问题

1. **`chunk_id % 2` 的设计是错误的**：不能简单按 chunk_id 分设备
2. **`HybridLlamaForCausalLM` 只是骨架**：没有真实的 Qwen 模型逻辑
3. **`executeCPUTask/executeQNNTask` 是空壳**：没有真正的执行逻辑

### 7.2 正确的实现思路

1. **基于 `modeling_qwen_npu.hpp` 改造**：
   - 在 `QwenForCausalLM::forward` 中集成 `PipelineExecutor`
   - 把 `QwenDecoder::forward` 拆成多个 task：
     - Task 1：`self_attn_proj_`（QNN）→ 提交到 QNN worker
     - Task 2：`self_attn_matmul_`（CPU）→ 提交到 CPU worker（等待 Task 1）
     - Task 3：`self_attn_out_mlp_`（QNN）→ 提交到 QNN worker（等待 Task 2）

2. **Pipeline 调度**：
   - 不同 chunk 的 task 可以交错执行
   - 例如：Chunk0 的 Task 1（QNN）和 Chunk1 的 Task 2（CPU）可以并行（如果数据依赖允许）

3. **Trace 阶段**：
   - 传入完整的 prompt，记录 chunk 切分和设备分配
   - 编译成多个 QNN 图（不同长度）

4. **运行时**：
   - 使用 `ContextManager` 选择合适的 QNN 图
   - 通过 `PipelineExecutor` 调度 task 到不同的 worker
   - 用 `future` 收集结果并合并

---

## 8. 与 `qwen_npu_pipeline_design.md` 的关系

**`qwen_npu_pipeline_design.md` 描述的是"chunk 级别的 pipeline"**：
- 在 NPU 处理 chunk N 时，CPU 准备 chunk N+1 的数据
- 这是**数据准备和计算的重叠**，不是"chunk 内部 QNN/CPU 阶段的并行"

**`modeling_pipeline_trace_simplified.hpp` 描述的是"stage 级别的 pipeline"**：
- 把模型拆成多个 stage（QNN stage、CPU stage）
- 不同 chunk 的 stage 可以并行执行
- 这是**更细粒度的并行**

**两者可以结合**：
- Chunk 级别的 pipeline：CPU 准备 chunk N+1 的数据
- Stage 级别的 pipeline：Chunk N 的 QNN stage 和 Chunk N+1 的 CPU stage 并行

---

## 9. 下一步建议

1. **先理解当前示例的框架**：`PipelineExecutor`、`ContextManager` 的作用
2. **明确真正的 pipeline 需求**：
   - 是 chunk 级别的 pipeline（数据准备和计算重叠）？
   - 还是 stage 级别的 pipeline（QNN/CPU stage 并行）？
   - 还是两者结合？
3. **基于 `modeling_qwen_npu.hpp` 改造**：
   - 不要直接用 `HybridLlamaForCausalLM`
   - 而是把 pipeline 逻辑集成到 `QwenForCausalLM` 中
4. **与师兄对齐**：
   - 确认 pipeline 的具体需求
   - 确认 trace 和 context 的管理方式
   - 确认 worker 线程的调度策略


