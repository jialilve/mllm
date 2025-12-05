## Qwen3 NPU —— IR 层 QNN Graph 融合设计说明（方案一）

> 说明：本设计**只针对 Qwen / Qwen3 NPU 系列模型**，特别是 `Qwen3ForCausalLM` 的 decoder 结构，用来解决“QNN Context 子图数量上限”问题。不是一个对所有模型都通用的 Pass。

---

### 1. 先回答：每个 decoder 里到底有多少个 QNN 子图？

#### 1.1 Qwen / Qwen3 NPU 的 decoder 结构

以 `Qwen3Decoder` 为例（`modeling_qwen3_npu.hpp`）：

- **子模块组成**
  - `Qwen3AttentionProjNPU self_attn_proj_;`      → `.to(kQNN);`
  - `Qwen3AttentionMatmul self_attn_matmul_;`     → 留在 CPU
  - `Qwen3OutProjAndMLP self_attn_out_mlp_;`      → `.to(kQNN);`
- **执行顺序**
  - `x` → QNN 上的 `self_attn_proj_` 计算 Q/K/V  
  - Q/K/V 回到 CPU → `self_attn_matmul_` 做注意力计算 + KV Cache  
  - 注意力输出再回到 QNN → `self_attn_out_mlp_`（o_proj + MLP）

也就是说：

- 每个 decoder layer 内部的执行模式是：**QNN → CPU → QNN**
- 对应 IR / QNN 视角，就是 **两个“QNN 子图” + 中间一段 CPU IR**
  - **Graph A**：`self_attn_proj_` 这段（Q/K/V 投影 + DequantizeAdd + Q/K RMSNorm 等）
  - **CPU IR**：`Qwen3AttentionMatmul`（RoPE、CausalMask、Softmax、KVCache、matmul 等）
  - **Graph B**：`Qwen3OutProjAndMLP`（o_proj + gate/up/down_proj + 残差 + LayerNorm）

这一点在老的 `QwenDecoder`（`modeling_qwen_npu.hpp`）中也是完全一致的：

- 同样是 `self_attn_proj_.to(kQNN);` 和 `self_attn_out_mlp_.to(kQNN);`
- 中间的 `self_attn_matmul_` 在 CPU 上跑。

> 结论：**是的**，你理解的是对的——每个 decoder layer 确实有“第一个 QNN 子图（proj）”和“第二个 QNN 子图（out+MLP）”，中间夹着一段 CPU 注意力计算。这也是我们要做 Graph 融合时的基础假设。

---

### 2. IR / QNN Graph 是怎么生成的？

#### 2.1 trace 阶段

在 `Qwen3ForCausalLM::trace` 中：

- `ir::lowlevel::traceStart()` / `traceContinue()` 包住了 `model`（`Qwen3Text`）的调用：

```cpp
auto hidden_states =
    ir::lowlevel::traceModule(model, input_embeddings, llm_embedding_sin, llm_embedding_cos)[0];
```

- `Qwen3Text` 内部会遍历 `decode_blocks_`，逐层调用 `Qwen3Decoder::forward`。
- 每个 `.to(kQNN)` 的子模块（`self_attn_proj_` / `self_attn_out_mlp_` / `embedding_`）在 IR 里会被标记为 `DeviceTypes::kQNN`，CPU 部分保持在 `kCPU`。

#### 2.2 QNNGraphBuildPass 阶段

在 QNN 编译阶段（见 `docs/qnn_backend/core_design.rst`）：

- 对 IR 中 **连续的 QNN 子图**（标记为 `kQNN` 的 SubGraphOp / Region）：
  - 在 QNNBackend 里调用 `createQnnGraph(graphName)` 创建 QNNModel；
  - 按 Pattern（`QNNLinearPattern` / `QNNRMSNormPattern` / `QNNDequantizeAddPattern` …）把 IR Op 映射为 QNN Op；
  - `graphFinalize(graphName)` 编译为 QNN Graph；
- 对 CPU 子图则保持原样，由 CPU backend 执行。

对单个 decoder 来说，IR 上大致是：

```text
[GraphBegin name="layerX_proj"]   // QNN 子图 1
  (QNN ops for Q/K/V projection + DequantizeAdd + RMSNorm Q/K ...)
[GraphEnd   name="layerX_proj"]

  (CPU ops: RoPE, KVCache, CausalMask, Softmax, matmul ...)

[GraphBegin name="layerX_out_mlp"] // QNN 子图 2
  (QNN ops for o_proj + MLP + LayerNorm ...)
[GraphEnd   name="layerX_out_mlp"]
```

> 所以，在“QNN Graph”的层面，你可以近似认为：**每个 decoder 有 2 个 QNN Graph**，分别对应 `self_attn_proj_` 和 `self_attn_out_mlp_`。

---

### 3. 方案一的核心：只对 Qwen NPU 系列做“有意识的” Graph 融合

#### 3.1 不是“通用模型优化”，而是“Qwen NPU 特化优化”

非常重要的一点：  
**Graph 融合的规则是基于 Qwen/Qwen3 decoder 的结构假设设计的**：

- 每层都是“QNN → CPU → QNN”三段；
- QNN 段的名字/布局遵守特定命名（例如 `model.layers.0.self_attn_proj` / `model.layers.0.self_attn_out_mlp` 这样的 pattern）；
- 中间 CPU 部分必须保持完整，不能被错误地合并/移动。

如果把这样的 Pass **无脑挂在全局 QNN 编译流程里**，对其它模型（结构不同、设备标记不同、Graph 拆分方式不同）：

- 轻则什么都融合不到（graph name 对不上）；
- 重则错误地把本不该连在一起的 Graph 融在一起，导致 shape / 依赖错乱。

> 所以，这个 IR Graph 融合 Pass **应当被视为"Qwen NPU 专用优化"**，而不是"所有模型的通用优化"。

---

### 4. 为什么可以跨层融合？数据流分析

#### 4.1 一个常见的疑问

你可能会问：**为什么 `model.layers.26_2`（Layer 26 的第二个 QNN Graph）可以和 `model.layers.27_1`（Layer 27 的第一个 QNN Graph）融合？不需要先经过 Layer 27 的 CPU wrapper 吗？**

答案是：**可以融合，因为数据流是连续的 QNN tensor**。

#### 4.2 从 MIR 文件看数据流

让我们看实际的 MIR 文件（`qwen3_npu.mir`）：

**Layer 26 的输出（第 2310 行）：**
```mir
graph.CallGraphOp @model.layers.26_2 (%3427:QNN, %3388:QNN) 
  -> (%3463:tensor<[1, 32, 2048], Float32, QNN>)
```

**Layer 27 的输入（第 2383 行）：**
```mir
graph.CallGraphOp @model.layers.27_1 (%3463:tensor<[1, 32, 2048], Float32, QNN>...) 
  -> (%3485:QNN, %3486:QNN, %3487:QNN)
```

**关键发现：**
- `model.layers.26_2` 的输出是 **`%3463`（QNN tensor）**
- `model.layers.27_1` 的输入也是 **`%3463`（同一个 QNN tensor）**
- 两者都是 **QNN SubGraph**，数据已经在 QNN 设备上

#### 4.3 数据流图

```
Layer 26 的执行流程：
%3388 (QNN) 
  → model.layers.26_1 (QNN SubGraph) → Q/K/V (QNN)
  → X2X → Q/K/V (CPU)
  → model.layers.26.self_attn (CPU SubGraph) → attention_out (CPU)
  → X2X → attention_out (QNN)
  → model.layers.26_2 (QNN SubGraph) → %3463 (QNN) ✅ 输出

Layer 27 的执行流程：
%3463 (QNN) ✅ 直接输入（来自 Layer 26 的输出！）
  → model.layers.27_1 (QNN SubGraph) → Q/K/V (QNN)
  → X2X → Q/K/V (CPU)
  → model.layers.27.self_attn (CPU SubGraph) → ...
```

#### 4.4 关于"CPU Graph"的澄清

你可能会注意到 MIR 文件中有：
- `graph.SubGraphOp @model.layers.26 <CPU>` - 外层的 CPU SubGraph
- `graph.SubGraphOp @model.layers.27 <CPU>` - 外层的 CPU SubGraph

这些**外层的 CPU SubGraph**的作用是：
- **协调** QNN 和 CPU 之间的数据转换（X2X 操作）
- **调用**内部的 QNN SubGraph 和 CPU SubGraph
- **管理**数据流和依赖关系

但是，**实际的数据传递**是：
- `model.layers.26_2` (QNN) 输出 → `%3463` (QNN tensor)
- `%3463` (QNN tensor) → `model.layers.27_1` (QNN) 输入

**数据已经在 QNN 上，不需要经过 CPU！**

#### 4.5 为什么融合是安全的？

1. **数据类型一致**：两个 Graph 的输入/输出都是 QNN tensor（Float32, QNN device）
2. **数据流连续**：`model.layers.26_2` 的输出直接作为 `model.layers.27_1` 的输入
3. **设备一致**：两个 Graph 都在 QNN 设备上执行
4. **无 CPU 依赖**：Layer 27 的 CPU wrapper 只是用来协调，但数据流本身是 QNN → QNN

#### 4.6 融合后的效果

融合 `model.layers.26_2` 和 `model.layers.27_1` 后：

```
原来的结构：
- model.layers.26_2 (QNN Graph) → %3463 (QNN tensor)
- model.layers.27_1 (QNN Graph) ← %3463 (QNN tensor)

融合后的结构：
- model.layers.26_2_fused_27_1 (QNN Graph) 
  - 包含 model.layers.26_2 的所有操作
  - 直接连接到 model.layers.27_1 的所有操作
  - 输入：model.layers.26_2 的输入
  - 输出：model.layers.27_1 的输出
```

**好处：**
- 减少 QNN Context 中的 Graph 数量（从 56 个减少到约 29 个）
- 减少 Graph 切换的开销
- 数据流更直接，无需经过 CPU wrapper 的协调

---

#### 3.2 更符合项目规范的落点

结合现有代码组织（Qwen / Qwen3 有自己的一套 docs 和 modeling）：

- **代码组织建议**：
  - Pass 实现文件放在：`mllm/backends/qnn/passes/Qwen3IRGraphFusionPass.cpp`（或更通用的 `QwenNpuIRGraphFusionPass.cpp`）；
  - 但：
    - **只对 Graph name / 模块名匹配 Qwen/Qwen3 模式的 IR 生效**；
    - 或者只在 Qwen/Qwen3 的 tracer / compile 入口里调用。
- **调用位置建议**：
  - 在 Qwen3 NPU 的 build 流程里显式调用，而不是在 QNNBackend 的通用 `compileModel` 里无条件执行。
  - 例如：
    - 在 `Qwen3ForCausalLM::trace` 完成 IR 后、写出 `.mir` 或交给 QNNBackend 前，先跑一次 Qwen3 专用 IR pass；
    - 或在 `task.py` / `examples/qwen3_npu/main.cpp` 驱动的“Qwen3 NPU 编译脚本”里，明确插入 `Qwen3IRGraphFusionPass`。

这样有几个好处：

- 对项目其他模型 **零侵入**，不会引入难以排查的 side-effect；
- 非常清晰地表达：**这是“为了 Qwen3 NPU 解 Graph 数量上限”引入的专用优化**；
- 以后如果要给别的模型做类似优化，可以：
  - 新增其它模型专用的 pass；
  - 或者抽象出一层“可配置的 graph-fusion 规则”，在各模型自己的 pipeline 中配置启用。

---

### 4. 针对 Qwen3 的 IR Graph 融合规则（直观版）

> 这一段是“Qwen3 专用规则”，不对外模型生效。

**目标**：在“不改变数学意义”的前提下，减少 QNN Graph 数量。  
**思路**：把“同一层的两个 QNN 子图 + 中间 CPU 段”打包成**一个更大的 QNN Graph**，或者把“层 X 的后半段 + 层 X+1 的前半段”合并，从而减小“单 Context 的 Graph 个数”。

示例（概念图）：

```text
原来 decoder L 层：
  Graph L_qnn_1 (self_attn_proj_, QNN)
  CPU 部分 (self_attn_matmul_)
  Graph L_qnn_2 (self_attn_out_mlp_, QNN)

融合思路 A（同层内三段打成一个大 Graph）：
  Graph L_qnn_fused (包含原来的 qnn_1 + CPU + qnn_2)

融合思路 B（跨层拼接，师兄提到的变种）：
  把 “L_qnn_2 + L+1_qnn_1” 放到一个 Graph 里，减少总 Graph 数。
```

**关键点**：

- 因为 Qwen3 的 decoder 在 IR 层是一个 `Qwen3Decoder` 模块，因此我们可以：  
  - 在“进入 QNNGraphBuildPass 之前”，对 **每个 Qwen3Decoder 的 IR 子图** 做结构分析；
  - 只在满足 pattern 的情况下做 Graph 合并；
  - 严格通过模块名 / graph_name 前缀（如 `model.layers.N.`）来约束作用范围。

---

### 5. 实现位置建议（总结版）

#### 5.1 Pass 文件放哪里？

- **推荐**：`mllm/backends/qnn/passes/Qwen3IRGraphFusionPass.cpp`  
  - 有利于复用 `QNNGraphBuildPass` 里的 IR / QNN 工具；
  - 放在 qnn_backend 的 `passes` 子目录，和已有的 `QNNGraphIOTensorPass.cpp`、`CustomOpPatterns.cpp` 一起，归类为“backend 级 Pass”；
  - 但在实现内部非常明确地写注释：
    - “只针对 Qwen/Qwen3 NPU decoder 的 Graph 名称 / Pattern 生效”；
    - “其它模型如果不符合 pattern，将被安全跳过”。

#### 5.2 什么时候运行？

- **不要**在所有模型通用的 pipeline 入口无条件调用；
- **要**在 “Qwen/Qwen3 NPU 的编译入口” 显式加一行：

伪代码示例：

```cpp
// somewhere in qwen3_npu compile pipeline
ir::ModuleOp module = ...;  // 由 trace / .mir 读入

// 1. Qwen3 专用 IR 优化（Graph 融合，解决 Context 子图上限）
Qwen3IRGraphFusionPass fusionPass;
fusionPass.run(module);

// 2. 通用 QNN Graph build
QNNGraphBuildPass buildPass;
buildPass.run(module);
```

在实现时可以：

- 在 `task.py` 的 Qwen3 NPU build 任务里加一个“启用 Qwen3IRGraphFusionPass”的开关；
- 或在 `examples/qwen3_npu/main.cpp` 调用 trace/compile 的部分增加一个“如果是 qwen3_npu，就跑 fusionPass”的分支。

---

### 6. 回答你的两个具体问题

#### 6.1 “`self_attn_proj_.to(kQNN)` 和 `self_attn_out_mlp_.to(kQNN)` 是不是每个 decoder 里的第一个和第二个 QNN Graph？”

**回答：是的，逻辑上可以这样理解。**

- 这两个子模块都是 `nn::Module`，并被 `.to(kQNN)` 标记为在 QNN 设备上执行；
- trace 时，这两段会被单独切成 QNN SubGraph，在 QNNGraphBuildPass 里各自变成一个 QNN Graph（或一个 Graph 里的两个区块，具体取决于实现，但“两个 QNN 区域”的事实不变）；
- 中间的 `self_attn_matmul_` 始终在 CPU 上执行。

所以，从 IR / QNN graph 粒度来看：  
**每个 decoder 至少包含两个 QNN 子图，分别对应这两个 `.to(kQNN)` 子模块。**

#### 6.2 “IR Graph 融合是所有模型都能这么做，还是只针对这个模型？”

- 当前我们讨论的融合规则，**严格建立在 Qwen/Qwen3 decoder 的结构假设上**：  
  “QNN → CPU → QNN”三段、特定命名、特定数据流。
- 其它模型：
  - 可能有不同的 stage 划分；
  - 可能没有中间 CPU 段，或者 QNN 段的命名不一样；
  - 甚至可能在一个大的 QNN Graph 里已经做过跨层优化。

> 结论：**这不是一个“所有模型通用”的 Graph 融合 Pass，而是“Qwen / Qwen3 NPU 专用优化”**。  
> 实现时应该放在 qnn_backend 的 `passes` 下，但只在 Qwen / Qwen3 编译流程里显式调用，并在代码和文档中写清“仅对 Qwen NPU 生效”的约束。

---

### 7. 后续可以怎么继续？

1. 在这份文档的基础上，我们可以继续细化：
   - 针对 Qwen3 的具体 Graph 命名（从 `.mir` 或 QNN dump 里 grep 出来）；
   - 写出基于 Graph name / 层索引的匹配规则（例如 `model.layers.\d+.(proj|out_mlp)`）。
2. 然后再落到真正的 C++ Pass 实现（`Qwen3IRGraphFusionPass`），一步一步按照文档里的伪代码写。\n

