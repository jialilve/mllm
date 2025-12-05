# Qwen NPU Pipeline（v1 实现详解）

> 目标：面向第一次接触该项目的同学，逐行拆解 v1 `examples/demo_qwen_npu_pipeline.cpp` + `mllm/mllm/Parallel.hpp` 中的 ChunkPipeline，回答“v1 是否把一个 chunk 拆成多个 CPU/NPU 子图并按 DAG 调度”、以及每一步到底在做什么。

---

## 1. 术语预备
| 术语 | 解释 |
| --- | --- |
| **Chunk** | 将长序列按固定长度（v1 中默认为 128）切片，便于 QNN 图复用与 KV Cache 对齐。 |
| **Trace Graph (`Tracer::model_`)** | 调用 `Tracer::trace` 后，模型被切成一系列“子图”。在 v1 中这些子图按照执行顺序排列，既包含 QNN 子图也包含 CPU 子图。 |
| **Prefill / Decode** | Prefill：把 prompt token 一次性送入模型并写入 KV Cache；Decode：循环处理新生成的单 token。 |
| **ChunkPipeline** | v1 自带的“流水线”类，负责把输入拆 chunk，并以“交错/斜线”方式调度多个子图。 |
| **GraphIdx** | `Tracer::model_` 中子图的索引。QNN/CPU graph 已按拓扑排序排好。 |
| **`Context::Instance().inference_state()`** | 全局状态寄存器，供算子读取 chunk_size、当前模式等信息。 |

---

## 2. v1 是否做了"chunk 内多 subgraph 的 DAG 调度"？
- **是的，但粒度有限。** 在 v1 中，一个 chunk 的完整前向会经过若干子图（Graph0, Graph1, …）。这些子图在 trace 阶段就已经固化了拓扑顺序（例如：Embedding/QNN => CPU Softmax => QNN MLP => CPU 输出）。  
- **调度方式**：`ChunkPipeline::run()` 并没有显式构造 DAG，而是利用 `graphIdx - chunk_id` 这一简单算式来决定某个 graph 应该处理哪个 chunk，从而形成类似"斜对角线"的执行顺序（见 §3.3）。  
- 因此，v1 **并没有**像论文那样单独区分 CPU/NPU stage、也没有全局调度器。它只是把 `Tracer::model_` 中的图按照固定规律交错执行，实现"两个 chunk 同时推进不同 graph"的效果。

**⚠️ 重要**：v1 的实现**没有显式检查依赖关系**。理论上，根据 Transformer 的 KV Cache 机制，第 i 个 chunk 的第 j 个子图（Gi,j）只需要依赖于前面所有 chunk 的第 j-1 个子图（G0,j-1, G1,j-1, ..., Gi-1,j-1），而不需要等待前面所有 chunk 的所有子图都执行完。这是因为：
- Attention 计算只依赖于 K/V，而 K/V 在 j-1 子图中就已经生成并写入 KV Cache 了
- MLP、LayerNorm 等子图的输出只影响当前 chunk 的后续计算，不影响其他 chunk 的 Attention
- 详见 `transformer_decoder_dependency_explanation.md` 的详细解释

v1 通过"斜对角线"调度来**隐式地**满足这个依赖关系，但没有显式验证。当 `num_graph <= 5` 时，这个设计基本失效，没有真正的并行。

---

## 3. 代码逐步解析

### 3.1 Demo 主流程（`examples/demo_qwen_npu_pipeline.cpp`）
```c++
auto tokenizer = QWenTokenizer(...);
auto model = v2::QWenForCausalLM_NPU(config, chunk_size);
model.load(model_path);

auto [_, input_tensor] = tokenizer.tokenizePaddingByChunk(trace_string, chunk_size, ...);
Tracer::trace(&model, {input_tensor});    // ⬅️ 生成子图

ChunkPipeline pipeline(real_seq_length, chunk_size);
auto prefill_result = pipeline.run(input_tensor, opt, tokenizer, model, isSwitched);
```
- **关键点**：`Tracer::trace` 会把模型展开成若干 graph 并存入 `Tracer::model_`。之后的 `ChunkPipeline::run()` 就是围绕这些 graph 做调度。

### 3.2 ChunkPipeline 构造（`mllm_v1/mllm/Parallel.hpp`）
```c++
ChunkPipeline::ChunkPipeline(int real_seq_length, int chunk_size) {
    const int seq_length_padding = (chunk_size - real_seq_length % chunk_size) + real_seq_length;
    chunk_num = seq_length_padding / chunk_size;
}
```
- **含义**：即使最后一个 chunk 不满，也会通过 padding 把长度补成 chunk_size 的倍数，方便 QNN graph 复用。

### 3.3 拆 chunk + 调度逻辑

#### 3.3.1 分块
```c++
for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
    chunked_tensors.push_back(...);
    chunked_tensors[chunk_id]->shallowCopyFrom(input_copy_sp, ..., chunk_id * chunk_size, ...);
}
```
- **含义**：把输入 prompt 的 Tensor 按 chunk_size 切片，保存到 `chunked_tensors`。每个 tensor 代表 chunk_i 的输入。

#### 3.3.2 核心调度函数
```c++
std::function<void(int,int)> executeFunc = [&](int chunk_id, int graphIdx) {
    int i = graphIdx - chunk_id;
    if (i < 0 || i >= num_graph) return;
    if (i == num_graph - 1 && chunk_id != chunk_num - 1) return;
    if (i == 0) Tracer::refleshInputTensor({chunked_tensors[chunk_id]});
    auto &graph = Tracer::model_[i];
    graph->Forward({}, {chunk_id});
};
```
- **`i = graphIdx - chunk_id`**：这一行就是斜线调度的核心。随着 `graphIdx` 增长，`i` 会在 `[0, num_graph)` 区间移动，同一时间只会有合法的 `(chunk_id, graph_i)` 被执行。
- **`Tracer::refleshInputTensor`**：当 graph 是第一层时，需要将当前 chunk 的输入 tensor 绑定到 trace 引擎中。
- **`graph->Forward`**：实际执行 QNN/CPU 子图。是否跑在 NPU 取决于 Graph 初始化时绑定的 backend。

> **“交错/斜线”形容的就是：**
> 不同 chunk 的 graph 运行轨迹形成一条条平行斜线。比如同一时刻执行 `(chunk0, graph2)` 和 `(chunk1, graph1)`，这两点在 chunk vs graph 的平面上正好排成对角线，如下图中的黑色斜线。

#### 3.3.3 OMP 并行区
```c++
for (int chunk_id = 0; chunk_id < chunk_num / 2; ++chunk_id) {
    for (int i = chunk_id * 2; i < num_graph + chunk_id * 2 + 5; ++i) {
#pragma omp parallel for num_threads(2)
        for (int pair_idx = 0; pair_idx < 2; ++pair_idx) {
            executeFunc((chunk_id * 2) + pair_idx, i - (pair_idx * 4));
        }
    }
}
```
- 外层 `chunk_id`：每次处理两个 chunk（`pair_idx = 0/1`）。  
- 内层 `graphIdx`：在“图空间”向前推进，并配合 `(chunk_id * 2)` 做偏移。  
- 结果：形成如下并发图示（以 2 chunk + 4 graph 为例）：

```
graphIdx →
            G0   G1   G2   G3
chunk0  ───●────●────●────●──
chunk1      └───●────●────●──

同一列的 ● 可以在 OMP 并行区同时执行（限制为 2 线程）。注意：是否真的“两个 chunk 同时跑不同 graph”，取决于 `graphIdx` 与 `chunk_id` 的映射关系，见下文更精确的推导。
```

##### 循环变量详解（以 `chunk_num=4`, `num_graph=5` 为例，回答“为什么有时 graphIdx 会是负数？”）
| 变量 | 解释 |
| --- | --- |
| `chunk_id` (外层) | 0 → 1 → …；一次迭代覆盖两个 chunk：`chunk = chunk_id*2 + pair_idx`。这里的 `chunk_id` 代表“chunk 对”的编号。 |
| `i` (内层) | 代表“一条对角线”的编号；从 `chunk_id*2` 开始，逐渐增大，确保不会遗漏 graph。 |
| `pair_idx` | 0 或 1，对应本轮要处理的两个 chunk。 |
| `executeFunc((chunk_id * 2) + pair_idx, i - (pair_idx * 4));` | 传入“真实 chunk 索引”和“对应的 graphIdx”。当 `i - (pair_idx * 4)` 较小时，这个 graphIdx 可能是负数，进入 `executeFunc` 后会立即被 `i_local < 0` 的判断过滤掉，相当于“这一次循环该 chunk 不执行任何 graph”。 |
| `num_graph + chunk_id * 2 + 5` | 内层上界；`+5` 给尾部多留几次迭代，用来 flush 最后一个 chunk 的尾部 graph。 |

**执行流程示例（更精确推导，与你的疑问对应）**（假设 `chunk_num = 4`，`num_graph = 5`，即 chunk0~chunk3、每个 chunk 有 5 个 trace graph）

- 对于外层 `chunk_id = 0`：  
  - `pair_idx = 0`（chunk0）传入 `graphIdx = i`：  
    - 在 `executeFunc` 内部变成 `i_local = graphIdx - chunk_id_param = i - 0`。  
    - 要满足 `0 ≤ i_local < num_graph`，即 `i ∈ [0,4]` 时才真正执行 `(chunk0, G0..G4)`；当 `i ≥ 5` 时直接 `return`。  
  - `pair_idx = 1`（chunk1）传入 `graphIdx = i - 4`：  
    - 内部 `i_local = (i - 4) - 1 = i - 5`，所以当 `i < 5` 时 `i_local < 0`，会被 `if (i < 0 || i >= num_graph) return;` 直接跳过，**这正是你提到的“graphIdx 为负数就不会执行”的情况，你理解是对的**。  
    - 只有 `i ∈ [5,9]` 时 `i_local ∈ [0,4]`，才依次执行 `(chunk1, G0..G4)`。  
  - 这意味着在 `chunk_id = 0` 这一轮里：  
    - `i = 0..4` 只有 chunk0 有效；  
    - `i = 5..9` 只有 chunk1 有效；  
    - 每个 `i` 上虽然有两个 `pair_idx` 迭代在线程中运行，但总是有一个立即 return，**不存在 “(chunk0,Gk) 与 (chunk1,Gk-4)” 真正并行计算**。
- 对于外层 `chunk_id = 1`：  
  - 类似地，chunk2 在某一段连续 `i` 上执行 `G0..G3`，chunk3 在稍后的连续 `i` 上执行自己的 `G0..G3`，区间同样不重叠。

结论：对于这一组参数（`chunk_num=4, num_graph=5`），这段 OMP “并行 for” 实际上更多是在**做时间片轮转 + 预留并行框架**：  
- 从语义上，它允许“在同一个 `i` 上两个 chunk 的 `executeFunc` 同时运行”；  
- 但因为其中一个分支的 `graphIdx` 很多时候为负或超界，被 `executeFunc` 的边界检查立刻过滤，所以**实际生效的 graph 调度仍然是每次只跑一个 chunk 的一个 graph**；  
- 真正想要达到论文里那种“(chunkN, CPU subgraph) 与 (chunkN+1, NPU subgraph)” 的重叠，需要在 v2 中设计更明确的 stage 拆分和 `async`/线程池调度，而不能仅靠这段 OMP 代码。

**重要发现**：`num_graph = 5` 恰好是**完全没有并行**的特殊情况。详细分析见 `qwen_pipeline_v1_parallel_analysis.md`：
- 当 `num_graph <= 5` 时，chunk0 和 chunk1 的有效执行区间完全不重叠，**没有真正的并行**
- 当 `num_graph = 6..9` 时，并行度非常有限（只有 1-4 个时间点）
- 只有当 `num_graph >= 10` 时，才有较明显的并行效果
- 这个设计存在明显缺陷，不应该被当作“标准实现”来参考

### 3.4 输出与收尾
```c++
auto cpuModulePtr = std::dynamic_pointer_cast<CPUModuleWrapper>(Tracer::model_.back());
auto result = cpuModulePtr->result();
auto token_idx = postProcessing(result[0], chunked_tensors.back(), real_seq_length);
```
- 最后一个 graph 运行在 CPU 上（通常包含采样逻辑），取出 logits 后做 argmax，得到 prefill 的尾 token，再交给 decode 阶段继续生成。

---

## 4. Markdown 流程图（概览）
```mermaid
flowchart LR
    subgraph Trace 阶段
        A[Tracer::trace] -->|生成| B[Graph 0..N-1]
    end
    subgraph Runtime
        B -->|按照 chunk_id & graphIdx 组合| C{executeFunc}
        C -->|Graph Forward| D[QNN/CPU]
        D --> E[Prefill logits]
        E --> F[Decode (CPU)]
    end
```

---

## 5. 小结：v1 Pipeline 的特点
1. **简单但有效**：通过 `graphIdx - chunk_id` 规则 + OMP 并行，使得两个 chunk 可以交错执行不同的 graph，从而减少 NPU/CPU 间的空闲。
2. **局限**：无法灵活调整 chunk 数量/线程数；CPU/NPU graph 由 trace 固定，无法做更细粒度的 stage scheduling；不支持 module 级别的 async。
3. **对 v2 的启示**：  
   - 保留“按 chunk 切片 + 斜线调度”的思路，但用 `PipelineExecutor`/`mllm::async` 替换 OMP 循环；  
   - 引入 ContextManager 管理不同长度的 QNN graph；  
   - 梳理 CPU/NPU 子图接口，便于未来扩展到论文级流水线。

> 若在阅读过程中遇到不懂的术语，可先回到 §1 术语表再看具体代码；也可与 `qnn_pipeline_impl_notes.md` 对照，理解 v1 → v2 的演进路线。

