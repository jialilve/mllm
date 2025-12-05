# Graph 融合方案决策

## 问题分析

### 直接合并 Region 方案的问题

1. **IRWriter 的 Region 限制**：
   - `IRWriter::removeOp` 要求操作在 `cur_region_` 中
   - 递归查找时，`caller` 可能在子 Region 中
   - 需要为每个 Region 创建对应的 IRWriter，复杂且容易出错

2. **Value 映射的复杂性**：
   - 需要正确处理 graph1 输出 -> graph2 输入的连接
   - 需要更新所有相关的 Value 引用
   - 容易导致循环引用或悬空引用

3. **操作移动的复杂性**：
   - 需要更新操作的 belongsTo 关系
   - 需要更新 prev/next 关系
   - 需要处理操作的输入输出依赖

4. **调用者删除的复杂性**：
   - 需要递归查找所有调用者
   - 需要正确处理不同 Region 中的调用者
   - 容易导致断言失败或崩溃

## 决策：采用创建新 Graph 的方案

### 原因

1. **更安全**：
   - 不破坏现有的 IR 结构
   - 可以安全地处理依赖关系
   - 如果融合失败，可以回滚

2. **更清晰**：
   - 逻辑简单明了
   - 易于理解和维护
   - 符合 IR 设计原则

3. **更可靠**：
   - 利用 IRWriter 的安全 API
   - 避免直接操作 Region
   - 减少出错的可能性

### 实现策略

1. **创建新的融合 SubGraphOp**
2. **使用 IRWriter 复制 graph1 的所有操作**
3. **使用 IRWriter 复制 graph2 的所有操作**
4. **处理 graph1 输出 -> graph2 输入的连接（通过 Value 映射）**
5. **更新所有 CallGraphOp 的引用**
6. **删除旧的 SubGraphOp**

### 关键点

- **必须复制操作**：操作属于特定的 Region，不能直接复用
- **建立 Value 映射**：graph1 的输出 Value 应该映射到 graph2 的输入 Value
- **使用 IRWriter API**：利用框架提供的安全 API

