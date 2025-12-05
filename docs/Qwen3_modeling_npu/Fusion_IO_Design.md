## Qwen3 QNN 跨层融合的 IO 设计方案评估

> 目标：在 **减少 QNN Context graph 数量** 的前提下，保证 Qwen3 decoder 的数据流（特别是 Q/K/V 和 hidden state）清晰可控，方便在 IR Pass 里实现"创建新 Graph + 复制操作"的融合策略。

---

### 0. QNN 编译流程概览：Fusion Pass 发生在哪个阶段？

#### 0.1 完整的编译流程（从模型到可执行 QNN Graph）

对于新手小白，我们先理解整个流程，再定位 Fusion Pass 的位置：

```text
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 1：模型 Trace（生成 IR）                                    │
├─────────────────────────────────────────────────────────────────┤
│ model.trace(inputs)                                             │
│   ↓                                                             │
│ 生成 IR（Intermediate Representation，中间表示）                 │
│   - ModuleOp（顶层模块）                                         │
│   - SubGraphOp（子图，例如 model.layers.0_1, model.layers.0_2）│
│   - CallGraphOp（调用子图的操作）                                │
│   - LinalgIROp（线性代数操作，如 Linear, RMSNorm）              │
│   - 此时还是"抽象的计算图"，没有真正编译成 QNN Graph              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 2：QNN Graph Rewrite Pass（IR 层面的图重写）                │
├─────────────────────────────────────────────────────────────────┤
│ PassManager rewritePM(irs["model"])                            │
│                                                                 │
│ ① Qwen3IRGraphFusionPass（我们正在实现的 Fusion Pass）          │
│   - 识别 Qwen3 decoder 的 QNN 子图（model.layers.X_1, X_2）     │
│   - 将相邻层的 QNN 子图融合（X_2 + (X+1)_1 → X_fused）          │
│   - 修改 IR 结构：删除旧 graph，创建新 fused graph              │
│   - ⚠️ 注意：此时还在 IR 层面，没有真正编译成 QNN Graph         │
│                                                                 │
│ ② QNNGraphIOTensorPass                                         │
│   - 标记每个 SubGraph 的输入输出 tensor                         │
│   - 给 tensor 添加 is_graph_input / is_graph_output 属性       │
│                                                                 │
│ ③ QNNOpNamingPass                                              │
│   - 给未命名的 IR 操作分配唯一名称                               │
│   - 确保每个 op 都有可追踪的名字（用于 QNN 编译）                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 3：QNN Graph Build Pass（IR → QNN Graph 编译）              │
├─────────────────────────────────────────────────────────────────┤
│ PassManager graphBuildPM(irs["model"])                         │
│                                                                 │
│ QNNGraphBuildPass                                               │
│   - 遍历所有标记为 DeviceTypes::kQNN 的 SubGraphOp              │
│   - 对每个 SubGraphOp：                                         │
│     ├─ 调用 backend->createQnnGraph(graphName)                  │
│     ├─ 通过 Pattern Matching 将 IR Op 映射为 QNN Op：           │
│     │   • QNNLinearPattern → QNN FullyConnected                │
│     │   • QNNRMSNormPattern → QNN LayerNorm                     │
│     │   • QNNX2XPattern → QNN Copy                             │
│     ├─ 添加 QNN tensor（输入、输出、权重）                      │
│     └─ 调用 backend->graphFinalize(graphName)                   │
│         └─ QNN SDK 编译优化，生成可执行的 QNN Graph              │
│                                                                 │
│ ⚠️ 关键：只有在这个阶段，IR 中的 SubGraphOp 才会变成真正的      │
│    QNN Graph，注册到 QNN Context 中。                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 4：运行时执行                                               │
├─────────────────────────────────────────────────────────────────┤
│ model.forward(inputs)                                           │
│   ↓                                                             │
│ 根据 IR 调用对应的 QNN Graph 或 CPU 操作                        │
│   - CallGraphOp @model.layers.X_fused → 执行对应的 QNN Graph     │
│   - CPU 操作（如 self_attn matmul）→ CPU backend 执行           │
└─────────────────────────────────────────────────────────────────┘
```

#### 0.2 为什么 Fusion Pass 要在 "Rewrite Pass" 阶段做，而不是在 "Build Pass" 阶段？

**核心原因：IR 层面更容易做图结构重写**

1. **IR 是抽象的、可修改的**：
   - IR 中的 `SubGraphOp`、`CallGraphOp`、`Value` 都是可以创建、删除、修改的对象
   - 你可以自由地合并两个 `SubGraphOp`，修改 `CallGraphOp` 的引用，调整 `Value` 的连接关系
   - 这些操作在 IR 层面是"安全"的，因为还没有真正编译成 QNN Graph

2. **QNN Graph 是编译后的、难以修改的**：
   - 一旦进入 `QNNGraphBuildPass`，IR 中的 `SubGraphOp` 就会被编译成 QNN SDK 的 `QnnGraph_Handle_t`
   - QNN Graph 是已经优化、编译好的二进制结构，很难再拆分或合并
   - 如果在这个阶段才做融合，需要：
     - 撤销已经编译好的 QNN Graph
     - 重新编译融合后的图
     - 处理 QNN Context 中的 graph 注册/注销
   - 这比在 IR 层面做融合复杂得多

3. **设计上的清晰性**：
   - **Rewrite Pass = 图结构优化**（合并、拆分、重命名）
   - **Build Pass = IR → QNN 编译**（模式匹配、tensor 注册、SDK 编译）
   - 职责分离，代码更清晰

#### 0.3 具体到代码：Fusion Pass 的注册位置

在 `examples/qwen3_npu/main.cpp` 中：

```cpp
// 阶段 2：QNN Graph Rewrite Pass
mllm::ir::PassManager rewritePM(irs["model"]);
// ① 首先运行 Fusion Pass（在 IR 层面融合 graph）
rewritePM.reg(mllm::qnn::createQwen3IRGraphFusionPass());
// ② 然后标记 I/O tensor（需要知道哪些是 graph 的输入输出）
rewritePM.reg(mllm::qnn::createQNNGraphIOTensorPass());
// ③ 最后给 op 命名（确保所有 op 都有唯一名称）
rewritePM.reg(mllm::qnn::createQNNOpNamingPass());
rewritePM.run();  // 按顺序执行所有 Pass

// 阶段 3：QNN Graph Build Pass
mllm::ir::PassManager graphBuildPM(irs["model"]);
graphBuildPM.reg(mllm::qnn::createQNNGraphBuildPass());
graphBuildPM.run();  // 将 IR 编译成 QNN Graph
```

**为什么 Fusion Pass 要放在最前面？**

- 因为 `QNNGraphIOTensorPass` 需要知道"哪些 SubGraph 是最终的 QNN Graph"
- 如果先融合，`QNNGraphIOTensorPass` 就能正确标记融合后的 graph 的 I/O
- 如果后融合，可能会标记到已经被删除的旧 graph，导致错误

#### 0.4 总结：Fusion Pass 的作用时机

| 阶段 | 输入 | 输出 | Fusion Pass 的作用 |
|------|------|------|-------------------|
| **Trace** | Python/C++ 模型 | IR（ModuleOp + SubGraphOp） | ❌ 不参与 |
| **Rewrite Pass** | IR（56 个 QNN SubGraph） | IR（29 个 fused QNN SubGraph） | ✅ **在这里融合** |
| **Build Pass** | IR（29 个 fused SubGraph） | QNN Graph（29 个可执行的 QNN Graph） | ❌ 不参与 |
| **Runtime** | 输入 tensor | 输出 tensor | ❌ 不参与 |

**关键理解：**
- Fusion Pass 是在 **IR 层面**做的图结构重写
- 它把 56 个 QNN SubGraph 融合成 29 个，**减少的是 IR 中 SubGraphOp 的数量**
- 当 `QNNGraphBuildPass` 运行时，它看到的是已经融合后的 29 个 SubGraph
- 因此最终注册到 QNN Context 的也是 29 个 QNN Graph，而不是 56 个
- 这就是为什么融合能减少 QNN Context 的 graph 数量，从而避免 6033 错误

---

### 1. 背景回顾（问题是什么）

- 每层 decoder 内部 QNN 结构：
  - `model.layers.X_1`：QNN，做 Q/K/V 投影 + dequant + RMSNorm（输出 Q/K/V，device=QNN）
  - `model.layers.X_2`：QNN，做 out_proj + MLP + LayerNorm（输出新的 hidden state，device=QNN）
- 每层还有一个 CPU wrapper：
  - `graph.SubGraphOp @model.layers.X <CPU>` 里，典型顺序：
    - `CallGraphOp @model.layers.X_1`（QNN）
    - CPU self-attn matmul / softmax / KV cache
    - `CallGraphOp @model.layers.X_2`（QNN）
- 相邻层之间，一个关键的数据流是：
  - `model.layers.X_2` 的 **输出 hidden_state（QNN tensor）** 直接作为 `model.layers.X+1_1` 的输入
  - 这是我们跨层融合 `model.layers.X_2` 和 `model.layers.X+1_1` 的理论基础

我们现在要决定的是：**新建的 fused QNN SubGraph 放在哪里、叫什么名字、输入输出怎么定义、上层 CPU wrapper 怎么接这个 fused graph**。

---

### 2. 关键设计问题拆解

#### 2.1 融合后的 SubGraph 应该放在哪里 / 叫什么？

候选思路：

- **方案 A：直接“覆盖” `model.layers.X_2`**
  - 用 fused graph 完全替代 `model.layers.X_2`，符号名仍叫 `model.layers.X_2`
  - 内部实际执行的是 `X_2 + (X+1)_1` 的组合
  - Layer X 的 CPU wrapper 仍然 `CallGraphOp @model.layers.X_2`
  - Layer X+1 的 CPU wrapper 仍然 `CallGraphOp @model.layers.X+1_1`（此时要么变成空壳，要么完全删除）

  **问题：**
  - 如果继续保留 `model.layers.X+1_1` 并调用，就会 **重复计算** Q/K/V（等于 fused graph 做了一次，X+1_1 又做一遍）。
  - 如果删除 `model.layers.X+1_1` 的调用，Layer X+1 的 CPU wrapper 内部会 **失去 Q/K/V 的来源**，需要我们额外把 Q/K/V 作为 graph 输入接进来，改动很大。

  **结论：**这个方案“名字好看”（不增加符号数量），但**对 CPU wrapper 的改动复杂且容易出错**，不适合作为第一版实现。

- **方案 B：新建一个 `model.layers.X_fused`，完全替代 `X_2` + `X+1_1`**
  - 新建 QNN SubGraph：`@model.layers.X_fused`（或 `@model.layers.X_2_fused`，命名细节可再定）
  - 顶层 QNN graph 数量：原本有 `X_2` 和 `X+1_1` 两个，现在删掉这俩，只保留一个 fused graph → **graph 数量减少 1**
  - CPU/IR 侧的接线：
    - Layer X 的 CPU wrapper：原本 `CallGraphOp @model.layers.X_2`，现在改成 `CallGraphOp @model.layers.X_fused`，但只使用其中 **“新的 hidden state”** 这一部分输出。
    - Layer X+1 的 CPU wrapper：原本第一行 `CallGraphOp @model.layers.X+1_1`，现在删掉；它需要的 Q/K/V 改为从**外部输入**传进来，这些输入来自 `model.layers.X_fused` 的其他输出。

  **优点：**
  - QNN 侧非常清晰：**每一对 (X_2, X+1_1) 被一个新的 fused QNN graph 取代**，Context graph 数量可靠地减少。
  - 不需要“伪装”旧 graph 名字，便于日志和 debug。
  - 保持“**一个 CallGraph 只对应一次执行**”，不会出现重复算的情况。

  **缺点：**
  - 必须改 `@model.layers.X <CPU>` 和 `@model.layers.X+1 <CPU>` 这两个 CPU SubGraph 的 **IO 签名** 和内部 `CallGraphOp`：
    - X 层：`CallGraphOp` 的符号改名（`X_2` → `X_fused`），但输入/输出基本相同；
    - X+1 层：删除 `CallGraphOp @X+1_1`，并把原来从它拿到的 Q/K/V 改成“来自上游 graph 输入”。
  - 这部分 IR 操作重写需要在 Pass 里额外写一点“签名改造”逻辑。

  **结论：**这是比较“工程化”的方案，**逻辑最干净、行为最易理解**，适合作为推荐实现。

- **方案 C：在更外层再包一层“跨层 SubGraph”，由 driver 直接调度**
  - 想象有个新 SubGraph：`@decoder_block_pair_X`，内部再调用 `@X_2` 和 `@X+1_1`
  - Host/driver 直接调用 `decoder_block_pair_X`，CPU wrapper 内部结构尽量不动

  **问题：**
  - 对 QNN Context 来说，底层仍然有 **两个 QNN graph（X_2, X+1_1）**，只是上面又包了一层 CPU/IR 的壳，**并没有从根本上减少 Context 里的子图数量**。
  - 对 QNN 编译器来说，“包外面多一层”不一定会触发真正的 QNN graph merge。

  **结论：**这个方案**无法稳定保证减少 Context graph 数量**，不符合我们现在要解决的核心问题（6033 / Graph limit），可以直接排除。

---

### 3. 新 Graph 的输入 / 输出应该是什么？

我们以下面的记号来描述：

- 原 `model.layers.X_2`：
  - 输入：`(h_X_in, residual_X)`（都在 QNN device 上）
  - 输出：`(h_X_out)`（新的 hidden state，QNN tensor）
- 原 `model.layers.X+1_1`：
  - 输入：`(h_X_out)`（来自上一层的 hidden state，QNN tensor；还有可能有别的一些常量/参数）
  - 输出：`(Q_{X+1}, K_{X+1}, V_{X+1})`（Q/K/V，QNN tensor）

结合上面的数据流，我们希望 fused graph **一次执行**就完成：

1. 对 Layer X 做 out_proj + MLP + LayerNorm，得到 `h_X_out`
2. 立刻用 `h_X_out` 继续做下一层的 Q/K/V 投影，得到 `(Q_{X+1}, K_{X+1}, V_{X+1})`

因此，比较合理的 IO 设计是：

- **输入：**
  - 仍然使用 **`model.layers.X_2` 的全部输入**，即：
    - `h_X_in`（上层传入）
    - `residual_X`（残差）
  - 如果 `model.layers.X+1_1` 额外依赖某些 **非 `h_X_out` 的输入**（比如一些 QNN 常量、bias、scales），这些在 MIR 中通常是**静态权重/常量**，不会通过 SubGraph IO 暴露出来，所以**不需要加到 fused graph 的输入里**。

- **输出：**
  - 输出集合 = `{原 X_2 的输出 h_X_out}` ∪ `{原 X+1_1 的输出 Q_{X+1}, K_{X+1}, V_{X+1}}`
  - 也就是说，fused graph 一共会有 **1 + 3 = 4 个主要输出**（也可以根据实际 MIR 调整数量，但原则是“至少包含这两类信息”）。

对应到外层 CPU wrapper：

- Layer X CPU wrapper：
  - 继续从 fused graph 的输出里 **拿第一个 output（h_X_out）**，当作自己的 layer 输出；
  - 对 `(Q_{X+1}, K_{X+1}, V_{X+1})` 这些额外输出，**在 X 层内部不使用**，而是通过更外层的连接传给 `@model.layers.X+1 <CPU>` 作为其新的 graph 输入。

- Layer X+1 CPU wrapper：
  - 修改 SubGraph IO 签名，使其 **多三个 QNN 输入：Q_{X+1}, K_{X+1}, V_{X+1}`**；
  - 删除原来的 `CallGraphOp @model.layers.X+1_1`，因为它已经被 fused graph 覆盖；
  - 内部 self-attn matmul 直接使用来自 SubGraph 输入的 Q/K/V。

这样一来：

- 所有 QNN 计算 **只在 fused graph 内执行一次**；
- CPU wrapper 之间的数据流变成：
  - `@model.layers.X <CPU>` 的输出不仅包括 `h_X_out`，在顶层 `model` 调用时还能把 fused graph 的 Q/K/V 输出接给 `@model.layers.X+1 <CPU>` 的输入；
  - Q/K/V 明确地**作为 CPU SubGraph 的输入/输出 Value 在 MIR 里出现**，便于追踪和 debug。

---

### 4. 各方案优缺点对比与最终推荐

#### 4.1 方案 A（覆盖 `model.layers.X_2`）总结

- **优点：**
  - 命名简单，不增加符号数量；
  - Layer X 的 CPU wrapper 逻辑几乎不用改（仍然调用 `X_2`）。
- **缺点：**
  - Layer X+1 的 CPU wrapper 需要大幅改造（失去 Q/K/V 的来源）；
  - 如果处理不好，很容易出现“重复执行”或“Q/K/V 丢失”的情况；
  - 对一个刚入门的同学来说，实现和调试成本都很高。

> **结论：不推荐作为第一版实现。**

#### 4.2 方案 B（新建 `model.layers.X_fused`，替代 X_2 + X+1_1）总结

- **优点：**
  - 真正减少 QNN Context 的 graph 数量：**两个 QNN graph → 一个 fused graph**；
  - 行为清晰：一次执行输出两个层次需要的所有 QNN 结果；
  - 数据流显式：Q/K/V 通过 SubGraph IO 显式传递到下一层 CPU wrapper；
  - 有利于后续扩展（比如继续做 intra-layer 融合，只是 fused graph 里多包含一部分 X_1）。
- **缺点：**
  - 需要在 IR Pass 里同时改：
    - 顶层 QNN SubGraph 列表（删除原 `X_2` 和 `X+1_1`，加入 `X_fused`）；
    - `@model.layers.X <CPU>` 的内部 `CallGraphOp` 名称；
    - `@model.layers.X+1 <CPU>` 的 SubGraph 输入签名和内部 self-attn 的输入来源。
  - 对 IR 操作的熟悉程度有一定要求。

> **结论：这是当前情况下最平衡、最工程化、最值得实现的方案，推荐作为正式实现方向。**

#### 4.3 方案 C（外包一层“跨层 SubGraph”）总结

- **优点：**
  - 看起来对现有结构侵入性小（QNN 子图本身不改名）。
- **缺点：**
  - QNN Context 里底层还是两个 graph，对解决“Context graph 上限 / 6033”几乎没帮助；
  - 过度依赖 QNN 编译器的内部 fuse 能力，不可控。

> **结论：不满足当前“硬性减少 graph 数量”的目标，应直接排除。**

---

### 5. 对后续实现的具体建议（给 Pass 编写阶段用）

基于上面的分析，**后续在 `Qwen3IRGraphFusionPass` 里实现“创建新 Graph + 复制操作”时，可以按如下思路落地（对应方案 B）：**

1. **在顶层创建 fused QNN SubGraph：**
   - 符号名建议：`model.layers.{X}_fused` 或 `model.layers.{X}_2_fused`；
   - 输入：沿用 `model.layers.X_2` 的全部输入；
   - 输出：`[h_X_out, Q_{X+1}, K_{X+1}, V_{X+1}]`。

2. **复制 `X_2` 和 `X+1_1` 的 Region 操作到 fused graph：**
   - 用 `value_map` 维护 old→new 的 `Val` 映射；
   - 对 `X+1_1` 中使用 `h_X_out` 的地方，映射到 fused graph 里由 `X_2` 产生的对应新 `Val`；
   - 新建 fused graph 的 `ReturnOp` 同时返回 `[h_X_out, Q, K, V]`。

3. **更新 CPU wrapper 和调用点：**
   - 找到所有 `CallGraphOp @model.layers.X_2`，将其改为 `@model.layers.X_fused`，并且在其 outputs 中保留对 `h_X_out` 的使用；
   - 找到 `@model.layers.X+1 <CPU>`：
     - 修改其 SubGraph 输入列表，增加 3 个 QNN 输入（Q/K/V）；
     - 删除内部的 `CallGraphOp @model.layers.X+1_1`；
     - 将 self-attn 中原本来自 `X+1_1` 的 Q/K/V 输入，改为来自 SubGraph 的这 3 个新输入；
   - 在最外层 `model` 的 SubGraph 调用中，将 `X_fused` 多出来的 3 个输出，接到 `@model.layers.X+1 <CPU>` 的这 3 个新输入上。

4. **删除旧的 QNN SubGraph：**
   - 从符号表中移除 `model.layers.X_2` 与 `model.layers.X+1_1`；
   - 确保没有残留的 `CallGraphOp` 还在引用它们。

整体上，这个方案虽然实现步骤较多，但**每一步的语义都很明确**，你在调试时可以：

- 对照 MIR 检查每个 SubGraph 的 inputs/outputs 是否符合预期；
- 在日志里打印 fused graph 的符号名和 IO 数量；
- 用 grep 检查是否还有遗留的 `CallGraphOp @model.layers.X_2` / `@model.layers.X+1_1`。

---

### 6. 总结给你的结论（简短版）

- **新 fused SubGraph 建议采用方案 B：新建 `model.layers.X_fused`，替代 `X_2` + `X+1_1`。**
- **输入**：沿用 `model.layers.X_2` 的输入即可；
- **输出**：包含 `X_2` 的 hidden 输出 + `X+1_1` 的 Q/K/V 输出；
- Layer X CPU wrapper 只关心 fused 输出里的第一项（hidden），
  Layer X+1 CPU wrapper 通过新增的 3 个输入拿到 Q/K/V，删除原来的 `CallGraphOp @model.layers.X+1_1`。

你后续在写 Pass 时，可以直接把这里的"第 5 节"当成实现 checklist，一步一步对照做。这样既满足师兄"跨层融合"的要求，又能控制好 QNN graph 数量，减少 6033 的风险。

---

### 7. 为什么 Graph Fusion 能减少内存占用？（深入理解 QNN 内存机制）

#### 7.1 一个常见的疑问

你可能会问：**"两个 QNN graph 融合成一个，算子本身还是要跑一遍，为什么内存就会少了？"**

这个问题本质上是在问：**QNN 在"单个 graph 内"可以做哪些内存优化，而"跨 graph"做不到。**

#### 7.2 QNN 官方文档 vs 实际"graph 数量上限"

- **官方文档确实没有写"一个 context 最多 N 张 graph"**，而是说：
  - context binary 的 **heap 使用要 < 3GB 左右**；
  - 一个 context 里可以有多个 graph，数量理论上不限制。
- 但在实际实现里，QNN runtime 会为：
  - 每个 graph 生成自己的 **静态常量 + persistent buffer + scratch（工作区）规划**；
  - 把所有 graph 的需求叠加到同一个 HTP heap 上；
  - 再加上 Fragmentation（内存碎片）、安全裕量等。

因此，"graph 数量太多" 本质上不是一个"逻辑限制"，而是：

> **当 graph 数量变多、每个 graph 又很大时，整体的 heap 需求 + 碎片 就超出了 3GB 左右的物理 / 驱动安全阈值，于是 `graphExecute` 在某个 graph（例如 `model.layers.26_1`）分配 scratch 时失败，返回 6033。**

你师兄说"context 的子图有上限"，实际上是在用一个 **工程化的说法** 概括这件事：  
**对你当前这颗 SoC + 这个导出方式来说，大约在 50 多张 decoder graph 时会爆；压到 28–30 张以内就能跑。**

#### 7.3 为什么"graph 融合"能减内存：关键在于 **"生存期 + 可见范围"**

先看你现在的结构（简化）：

- 每一层 decoder 有两个 QNN 子图：
  - `layerX_1`：QKV proj / RMSNorm on QNN；
  - `layerX_2`：out proj + MLP on QNN；
- 当前的 context 里，大致有 `28 * 2 = 56` 张 QNN graph。

**QNN 编译 / 执行时，per-graph 做的事情是：**

- 对每个 graph G：
  1. 看所有 op 的输入 / 输出 tensor；
  2. 为 **常驻 tensor**（权重、某些中间状态）规划 buffer；
  3. 为 **scratch / workspace** 在 graph 内部做内存复用优化；
  4. 然后把这套规划作为这个 graph 的"私有布局"。

**重要点：**  
这些"scratch 重用 / buffer aliasing"的优化，**只在"单个 graph"内可见**：

- 对 graph G，它知道"Op1 的输出"可以在后面被 Op3 重用；它可以让这两个 tensor 复用同一个 buffer；
- 但对 graph G 和 graph H 之间：  
  - 编译器不知道 G 结束后 H 开始时，哪些 buffer 可以安全回收 / 复用；
  - 为了安全，很多中间结果会被当成"graph 边界的 IO"处理（persistent/外部 tensor），**不能随便覆盖**。

这就导致：

- **两个分开的 graph：**
  - 各自为战，各自规划一套 scratch / persistent buffer；
  - 很多看上去是"短命的中间量"，一旦出现在 graph 边界，就被迫变成"长命 + 不可重用"。

- **一个合并后的大 graph：**
  - 编译器看到的是"一整条链路"的数据流（`layerX_2` → `layerX+1_1`）；
  - 很多中间量在 graph 内可以安全复用 / 释放；
  - 有些原本放在 HTP 上、在 graph 间传递的 tensor，直接变成 graph 内部的 SSA 中间值，不需要单独分配持久 buffer。

**结果：**  
> **虽然算子总数差不多，但"可重用的内存"变多了，"必须常驻 + 不可重用"的部分变少了，整体峰值 heap 下降。**

#### 7.4 结合 Qwen3 结构，具体会省哪些内存？

以 **跨层融合 `layerX_2` + `layerX+1_1`** 为例：

##### 7.4.1 原始结构（未融合）

- `layerX_2` 输出一个 QNN tensor：`hidden_out_X`；
- 这个 `hidden_out_X` 被当作：
  - `model.layers.X` CPU wrapper 的输出（graph 输出，带 `is_graph_output`）；
  - 下一个 QNN graph `layerX+1_1` 的输入（graph 输入，带 `is_graph_input`）。

对 QNN 来说：

- `hidden_out_X` 在 HTP/DDR 需要一个 **graph 边界的 buffer**：
  - 要在 **G = layerX_2** 执行完后仍然存在；
  - 要被 H = layerX+1_1 作为输入使用；
  - 在没有更激进分析的前提下，这通常被当成 **persistent / 不可覆盖** 的 buffer（至少在这两 graph 之间）。

**此外：**

- Graph `layerX_2` 自己内部有一套 scratch；
- Graph `layerX+1_1` 自己也有一套 scratch；
- 这两套 scratch 的生命周期在 runtime 看起来像是"都要准备好，以便这两个 graph 都能执行"。

当这套 pattern **重复 27~28 次** 时：

- 很多像 `hidden_out_X` 这样的"跨 graph 中间结果"，都变成了 "需要单独 buffer 的 QNN tensor"；
- 每个 graph 自己又有一份 scratch 规划，合起来 heap 占用很高；
- 一旦叠加 + 碎片接近 3GB，就在某个 graph（比如层 26）上爆 6033。

##### 7.4.2 融合后的结构（理想情况）

把 `layerX_2` 和 `layerX+1_1` 合成一个 fused graph F：

- 对这个大 graph F 来说：
  - `hidden_out_X` 可能只是一个**内部中间值**：
    - 它只需要在 `layerX_2` 和 `layerX+1_1` 的少数几个 op 间存活；
    - 编译器可以决定它用多小的 buffer、何时复用 / 何时释放；
  - 最终对外只需要输出：
    - Layer X CPU wrapper 要的 `hidden_out_X`（可以是 F 的一个输出）；
    - 或者直接输出 Layer X+1 需要的 Q/K/V（那 hidden_out_X 甚至可以完全不出现在 graph 边界）。

**内存上发生的变化：**

- 原来每一层之间的 `hidden_out_X` 是"跨两个 graph 的边界 tensor"，
  - 需要一个 **大 buffer + 长生命周期**；
- 现在它可以被降级为：
  - Graph 内部的中间 SSA 值，生命周期只在 F 内的一小段；
  - 对应的 buffer 可以被其他中间 tensor 复用。

再加上：

- 每对 `(layerX_2, layerX+1_1)` 的 scratch 可以统一规划；
- 某些重复的结构（如 QNNX2XOp 之间的 copy）在编译优化中可能被消掉 / 合并（视 QNN 编译器能力而定）；

总体来说：

> **原来 56 张 graph 的内存峰值 ≈ 56 份"保守规划"的 scratch + 成堆 graph 间 IO tensor 的 buffer；  
> 融合到 ~29 张 graph 后，scratch 份数减少，IO tensor 边界减少，整体 heap 峰值下降到 3GB 以下，于是 6033 不再出现。**

#### 7.5 和"拆成两个 context"的对比

你也提到第二种方案：**两段 context**，例如：

- Context A：前 14 层；
- Context B：后 14 层。

这个方案的内存收益来自：

- A、B 各自的 context binary / heap 规划互不干扰；
- 一次只需要为当前在跑的那一半 layers 准备 scratch；
- 每个 context 的总 heap 使用明显小于"28 层一起塞一个 context"。

和"graph 融合"比：

- **多 context：**
  - 优点：思想直观，"少装一些东西到一个 context 里"；
  - 缺点：需要多份 context binary，host 侧要管理两个 QNNBackend / QNNContext，调度更复杂。
- **graph 融合（单 context）：**
  - 优点：保持单 context，运行时逻辑比较简单；
  - 缺点：需要你在 IR 层完成一套较复杂的 graph 重写（你正在做的事情）。

**两者都在做同一件事：降低"单 context 的有效 heap 峰值"。**

#### 7.6 总结：算子没变，为什么内存会变？

可以用一句话总结：

> **在 QNN 里，真正决定 heap 使用的是"graph 的划分方式 + 每个 graph 内的内存复用优化能力"，而不是简单的"算子数量"。  
> 把 graph 划分得更好（如跨层融合），能让更多中间张量的生命周期被看见，从而做更 aggressive 的内存复用，最终把总 heap 降到 3GB 以下。**

所以你师兄给出的"第一种方案（IR 融合 graph）能跑 28 层"并不是玄学，而是基于对 QNN 编译/运行机制的理解：  
**减少 graph 数量、拉长单个 graph 的可见范围，就能减少 graph 间的边界张量和重复 scratch，降低整体内存峰值，避免 6033。**


