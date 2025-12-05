# Qwen3 IR Graph 融合机制深度分析

## 1. 崩溃原因分析

### 1.1 问题现象

从日志看，程序在融合阶段崩溃（SIGABRT/SIGSEGV），甚至还没到真正的融合操作就崩溃了。

### 1.2 根本原因

**关键问题：融合顺序和依赖关系处理不当**

1. **融合顺序问题**：
   - 我们按顺序融合：`layer 0_2 + layer 1_1`, `layer 1_2 + layer 2_1`, ...
   - 当我们融合 `layer 0_2` 和 `layer 1_1` 时：
     - 修改了 `graph1` (layer 0_2) 的 Region
     - 删除了 `graph2` (layer 1_1) 的 SubGraphOp
   - 但 `candidates` 列表中，`layer 1_1` 的 `Qwen3GraphInfo` 仍然存在
   - 当后续融合尝试访问 `layer 1_2` 时，可能间接引用了已经被删除的 graph

2. **直接操作 Region 的问题**：
   - 我们直接从 `graph2_ops` 中移除操作：`graph2_ops.remove(op)`
   - 但这些操作可能还在被其他地方引用（如 CallGraphOp）
   - 当我们删除 graph2 的 SubGraphOp 时，可能破坏了 IR 的完整性

3. **Value 替换的问题**：
   - 当我们替换操作的输入 Value 时，需要同时更新：
     - 操作的 inputs 列表
     - 旧 Value 的 outputs 列表
     - 新 Value 的 outputs 列表
   - 如果处理不当，可能导致循环引用或悬空引用

## 2. 应该采用的融合机制

### 2.1 核心原则

1. **不直接修改现有的 SubGraphOp**：避免破坏 IR 的完整性
2. **使用 IRWriter API**：利用框架提供的安全 API 来修改 IR
3. **延迟删除**：等所有融合都完成后再清理
4. **标记已融合的 graph**：避免重复处理

### 2.2 推荐方案：创建新的融合 Graph

**策略**：
1. 为每个融合对创建新的 `SubGraphOp`
2. 使用 IRWriter 来复制和合并操作
3. 更新所有 CallGraphOp 的引用
4. 最后统一删除旧的 SubGraphOp

**优点**：
- 不破坏现有的 IR 结构
- 可以安全地处理依赖关系
- 如果融合失败，可以回滚

**缺点**：
- 需要复制操作（但这是必要的，因为操作属于特定的 Region）

### 2.3 备选方案：使用 IRWriter 的 replaceOp

**策略**：
1. 使用 IRWriter 的 `replaceOp` API 来替换 graph1
2. 使用 IRWriter 的 `insertOpAtPos` 来插入 graph2 的操作
3. 使用 IRWriter 的 `removeOp` 来删除 graph2

**优点**：
- 利用框架提供的安全 API
- 自动处理 prev/next 关系

**缺点**：
- 仍然需要处理 Value 的输入输出关系

## 3. 具体实现建议

### 3.1 融合流程

```
1. 识别融合对（已完成）
2. 对每个融合对：
   a. 创建新的融合 SubGraphOp
   b. 使用 IRWriter 复制 graph1 的所有操作到融合 Graph
   c. 使用 IRWriter 复制 graph2 的所有操作到融合 Graph
   d. 处理 graph1 输出 -> graph2 输入的连接
   e. 更新融合 Graph 的输入输出
3. 更新所有 CallGraphOp 的引用
4. 删除旧的 SubGraphOp（延迟删除）
```

### 3.2 操作复制策略

**关键点**：
- 不能直接复用操作（操作属于特定的 Region）
- 需要创建新的操作实例
- 需要建立 Value 映射关系

**实现**：
- 遍历 graph1 的所有操作
- 为每个操作创建新实例（使用 IRWriter）
- 映射输入输出 Value（使用 value_map）
- 对 graph2 做同样处理

### 3.3 Value 映射策略

**关键点**：
- graph1 的输出 Value 应该映射到 graph2 的输入 Value
- 在融合 Graph 中，它们应该是同一个中间 Value

**实现**：
- 建立 value_map：`graph1_output -> fused_intermediate_value`
- 建立 value_map：`graph2_input -> fused_intermediate_value`
- 在复制 graph2 的操作时，使用 value_map 来替换输入

## 4. 为什么不能直接复用操作？

### 4.1 操作属于特定的 Region

每个操作（Op）都有一个 `belongsTo` 关系，指向它所属的 Region。如果直接复用操作，会导致：
- 操作同时属于两个 Region（违反 IR 的不变量）
- 后续 Pass 可能无法正确处理

### 4.2 操作有输入输出依赖

每个操作都有输入输出的 Value 依赖关系。如果直接移动操作：
- 需要更新所有相关的 Value 引用
- 容易导致循环引用或悬空引用

### 4.3 操作有 prev/next 关系

操作之间有链表式的 prev/next 关系。如果直接移动：
- 需要更新 prev/next 关系
- 容易破坏操作顺序

## 5. 与算子融合、链表插入的区别

### 5.1 图融合 vs 算子融合

- **算子融合**：将多个算子合并成一个算子（如 Conv+BN+ReLU）
  - 通常在编译时完成
  - 不涉及图的整体结构修改
  - 相对简单

- **图融合**：将两个独立的计算图合并成一个图
  - 需要处理输入输出连接
  - 需要处理 Value 映射
  - 需要处理操作依赖关系
  - 更复杂

### 5.2 图融合 vs 链表插入

- **链表插入**：简单的数据结构操作
  - 只需要更新 prev/next 指针
  - 不涉及复杂的依赖关系

- **图融合**：复杂的图结构操作
  - 需要处理 Value 的输入输出关系
  - 需要处理操作的依赖关系
  - 需要保持 IR 的完整性

## 6. 总结

**核心结论**：
1. **必须复制操作**，不能直接复用
2. **应该创建新的融合 Graph**，而不是修改现有的
3. **使用 IRWriter API**，而不是直接操作 Region
4. **延迟删除**，等所有融合都完成后再清理

**下一步**：
重新实现融合逻辑，采用创建新 Graph + 操作复制的策略。

