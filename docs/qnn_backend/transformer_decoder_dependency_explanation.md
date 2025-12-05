# Transformer Decoder 的跨 Chunk 依赖关系详解

> 本文档解释为什么在 Transformer decoder 中，第 i 个 chunk 的第 j 个子图（Gi,j）只需要依赖于前面所有 chunk 的第 j-1 个子图，而不需要等待前面所有 chunk 的所有子图都执行完。

---

## 1. 核心问题

**论文中的依赖关系**：
```
Gi,j ← G0,j-1, G1,j-1, ..., Gi-1,j-1  (2)
```

**疑问**：为什么 Gi,j 只需要依赖前面 chunk 的 j-1 子图，而不需要等待前面所有 chunk 的所有子图（G0,0..G0,M, G1,0..G1,M, ..., Gi-1,0..Gi-1,M）都执行完？

---

## 2. Transformer Decoder 的基本结构

### 2.1 单层 Decoder 的执行流程

以 Qwen NPU 的实现为例（`modeling_qwen_npu.hpp`），单层 decoder 的执行顺序是：

```
输入 x (来自 embedding 或上一层)
  ↓
1. RMSNorm (Layer Normalization)
  ↓
2. QKV 投影 (AttentionProj) → 生成 Q, K, V
  ↓
3. RoPE + KVCache 更新 + Attention 计算 (AttentionMatmul)
  ↓
4. O_proj (输出投影)
  ↓
5. 残差连接 (x + attention_output)
  ↓
6. RMSNorm (Post-attention Layer Normalization)
  ↓
7. MLP (gate_proj + up_proj + down_proj)
  ↓
8. 残差连接 (x + mlp_output)
  ↓
输出 (传给下一层或 lm_head)
```

### 2.2 子图划分（以 v1 为例）

在 v1 的 pipeline 实现中，一个 chunk 的 forward 被划分成多个子图：

**v1 版本**（`modeling_qwen_npu_v2.hpp`）：
- 每个 decoder 层的执行顺序：**CPU（LayerNorm + Quantize）→ QNN（Part1: QKV投影）→ CPU（QKVmm: RoPE+KVCache+Attention）→ QNN（Part2: O_proj+MLP）**
- 所以 v1 版本从 **CPU 开始**

**v2 版本**（`modeling_qwen_npu.hpp`）：
- 每个 decoder 层的执行顺序：**QNN（AttentionProj: QKV投影）→ CPU（AttentionMatmul: RoPE+KVCache+Attention）→ QNN（OutProj+MLP）**
- 所以 v2 版本从 **QNN 开始**

**v1 的子图划分示例**：
- **Graph 0 (G0)**：第一层的 LayerNorm + Quantize（CPU）
- **Graph 1 (G1)**：第一层的 QKV 投影（QNN，SubgraphStart_1/SubgraphEnd_1）
- **Graph 2 (G2)**：第一层的 RoPE + KVCache + Attention（CPU）
- **Graph 3 (G3)**：第一层的 O_proj + MLP（QNN，SubgraphStart_2/SubgraphEnd_2）
- **Graph 4 (G4)**：第二层的 LayerNorm + Quantize（CPU）
- **Graph 5 (G5)**：第二层的 QKV 投影（QNN）
- ...

**关键观察**：
- **包含 Attention 计算的 graph（G2, G5, G8, ...）**：需要从 KV Cache 读取之前所有 chunk 的 K/V
- **QNN 子图（G1, G3, G6, ...）**：主要是线性变换（QKV 投影、MLP），只依赖当前 chunk 的数据
- **CPU 预处理子图（G0, G4, ...）**：LayerNorm、Quantize 等，只依赖当前 chunk 的数据

---

## 3. 为什么只需要依赖 j-1 子图？

### 3.1 Attention 的计算过程

Attention 的计算公式：
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
```

**关键点**：
- Attention 计算**只依赖于 Q, K, V 三个矩阵**
- Q 来自当前 chunk 的 j-1 子图（QKV 投影）
- K, V 来自：
  - **当前 chunk 的 j-1 子图**（当前 chunk 的 K/V）
  - **之前所有 chunk 的 j-1 子图**（已缓存的 K/V）

### 3.2 KV Cache 的更新时机

看代码 `QwenAttentionMatmul::forward`：

```cpp
// 1. 对 Q, K 应用 RoPE
query_states = q_rope_(query_states, ...);
key_states = k_rope_(key_states, ...);

// 2. 更新 KV Cache（关键！）
auto kv_outputs = kv_cache_(key_states, value_states);
key_states = kv_outputs[0];  // 包含之前所有 chunk 的 K
value_states = kv_outputs[1]; // 包含之前所有 chunk 的 V

// 3. 计算 Attention
attn = matmul(query_states, key_states, ...);
```

**关键理解**：
- **KV Cache 的更新发生在 j-1 子图**（QKV 投影完成后）
- 一旦某个 chunk 的 j-1 子图完成，它的 K/V 就已经**写入 KV Cache**了
- 后续的 Attention 子图（j 子图）可以直接从 KV Cache 中读取这些 K/V

### 3.3 为什么不需要等待其他子图？

**关键洞察**：Attention 计算**不依赖于其他子图的输出**（如 MLP、LayerNorm 等）

让我用一个具体例子说明：

#### 例子：3 个 chunk，每个 chunk 有 3 个子图

```
Chunk 0:
  G0,0: QKV 投影（生成 K0, V0）→ 写入 KV Cache
  G0,1: Attention 计算（使用 K0, V0）
  G0,2: MLP + 输出

Chunk 1:
  G1,0: QKV 投影（生成 K1, V1）→ 写入 KV Cache
  G1,1: Attention 计算（使用 K0, V0, K1, V1）
  G1,2: MLP + 输出

Chunk 2:
  G2,0: QKV 投影（生成 K2, V2）→ 写入 KV Cache
  G2,1: Attention 计算（使用 K0, V0, K1, V1, K2, V2）
  G2,2: MLP + 输出
```

**对于 Chunk 2 的 Attention 子图（G2,1）**：
- 它需要：
  - Chunk 0 的 K0, V0（在 G0,0 中生成并缓存）
  - Chunk 1 的 K1, V1（在 G1,0 中生成并缓存）
  - Chunk 2 的 K2, V2（在 G2,0 中生成并缓存）
- **它不需要**：
  - G0,1, G0,2 的输出（这些只影响 Chunk 0 的后续计算）
  - G1,1, G1,2 的输出（这些只影响 Chunk 1 的后续计算）

**为什么？**
- Attention 计算是**独立的**：它只关心 K/V，不关心其他层的输出
- MLP、LayerNorm 等子图的输出只影响**当前 chunk 的后续层**，不影响其他 chunk 的 Attention 计算
- 只要 K/V 准备好了（在 j-1 子图中），Attention 就可以计算了

### 3.4 依赖关系的可视化

```
时间线：

T0: G0,0 执行 → K0, V0 写入 KV Cache
T1: G0,1 执行（使用 K0, V0）
T2: G0,2 执行（MLP，不影响其他 chunk）
T3: G1,0 执行 → K1, V1 写入 KV Cache
T4: G1,1 执行（使用 K0, V0, K1, V1）
T5: G1,2 执行（MLP，不影响其他 chunk）
T6: G2,0 执行 → K2, V2 写入 KV Cache
T7: G2,1 执行（使用 K0, V0, K1, V1, K2, V2）← 只需要等待 G0,0, G1,0, G2,0
T8: G2,2 执行（MLP）
```

**关键**：G2,1 在 T7 就可以执行了，不需要等待 G0,1, G0,2, G1,1, G1,2 完成，因为这些子图的输出不影响 Attention 计算。

---

## 4. 论文中的公式解释

### 4.1 跨 Chunk 依赖（Cross-chunk dependency）

```
Gi,j ← G0,j-1, G1,j-1, ..., Gi-1,j-1  (2)
```

**含义**：
- 第 i 个 chunk 的第 j 个子图（通常是 Attention 子图）依赖于：
  - 第 0 到第 i-1 个 chunk 的第 j-1 个子图（这些子图生成了 K/V 并写入 KV Cache）
  - 第 i 个 chunk 的第 j-1 个子图（生成当前 chunk 的 K/V）

**为什么是 j-1？**
- 因为 j-1 子图是生成 K/V 的子图（QKV 投影）
- j 子图是使用 K/V 的子图（Attention 计算）

### 4.2 块内依赖（Intra-chunk dependency）

```
Gi,j ← Gi,j-1  (3)
```

**含义**：
- 第 i 个 chunk 的第 j 个子图依赖于同一个 chunk 的第 j-1 个子图
- 这是**顺序依赖**：必须先执行 j-1，再执行 j

**例子**：
- G2,1（Attention）依赖于 G2,0（QKV 投影），因为需要先有 K/V 才能计算 Attention
- G2,2（MLP）依赖于 G2,1（Attention），因为需要先有 Attention 的输出才能做 MLP

---

## 5. v1 Pipeline 的实现方式

### 5.1 v1 的调度策略

v1 使用 `graphIdx - chunk_id` 这个公式来实现"斜对角线"调度：

```cpp
int i = graphIdx - chunk_id;
if (i < 0 || i >= num_graph) return;
graph->Forward({}, {chunk_id});
```

**这个公式的含义**：
- 当 `graphIdx = 5, chunk_id = 1` 时，`i = 4`，表示 chunk1 执行 graph 4
- 当 `graphIdx = 5, chunk_id = 0` 时，`i = 5`，表示 chunk0 执行 graph 5

**为什么可以这样？**
- 因为 graph 4 和 graph 5 是**不同的子图**（例如一个是 Attention，一个是 MLP）
- 它们之间**没有直接依赖**（除了块内依赖）
- 所以可以并行执行

### 5.2 v1 的问题

**v1 的实现并没有严格遵循论文的依赖关系**：
- v1 使用硬编码的偏移量（`pair_idx * 4`）来实现"流水线延迟"
- 这个设计假设了 chunk0 和 chunk1 的执行区间有重叠，但实际上当 `num_graph <= 5` 时没有重叠
- v1 没有显式检查依赖关系，只是通过"斜对角线"调度来尝试实现并行

**正确的实现应该是**：
- 显式检查依赖关系：Gi,j 是否已经满足所有依赖（G0,j-1, G1,j-1, ..., Gi-1,j-1 都完成了）
- 只有当所有依赖都满足时，才执行 Gi,j
- 这样可以实现真正的"乱序执行"（out-of-order execution）

---

## 6. 总结

### 6.1 为什么只需要依赖 j-1 子图？

1. **KV Cache 的更新时机**：
   - K/V 的生成发生在 j-1 子图（QKV 投影）
   - 一旦 j-1 子图完成，K/V 就写入 KV Cache 了
   - 后续的 Attention 子图（j 子图）可以直接从 KV Cache 读取

2. **Attention 的独立性**：
   - Attention 计算只依赖于 Q, K, V，不依赖于其他子图的输出
   - MLP、LayerNorm 等子图的输出只影响当前 chunk 的后续计算，不影响其他 chunk 的 Attention

3. **依赖关系的传递性**：
   - 如果 Gi,j 只需要 G0,j-1, G1,j-1, ..., Gi-1,j-1，那么这些 j-1 子图又各自依赖于它们的 j-2 子图
   - 但这是**间接依赖**，不需要显式等待

### 6.2 对 v2 实现的启示

1. **显式依赖检查**：
   - 在 `PipelineExecutor` 中，应该显式检查每个 task 的依赖是否满足
   - 只有当所有依赖都满足时，才提交 task 到执行队列

2. **KV Cache 的管理**：
   - 确保 K/V 的写入和读取是线程安全的
   - 使用锁或原子操作来保护 KV Cache 的访问

3. **乱序执行的实现**：
   - 使用优先级队列或依赖图来调度 task
   - 优先执行依赖已满足的 task，实现真正的"乱序执行"

---

## 7. 参考资料

- 论文：Fast On-device LLM Inference with NPUs (Section 3.4)
- v1 实现：`mllm_v1/mllm/Parallel.hpp`
- v2 设计：`mllm_v2/mllm/models/qwen_npu/modeling_pipeline_trace_simplified.hpp`

