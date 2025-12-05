# QNN Context 子图上限问题与解决方案

> 本文档面向零基础初学者，详细解释 Qwen3-1.7B（28层）在 QNN 上遇到的 Context 子图上限问题，以及两种解决方案的实现方法。

---

## 目录

1. [问题背景](#1-问题背景)
2. [核心概念解释](#2-核心概念解释)
3. [问题根本原因](#3-问题根本原因)
4. [解决方案一：IR 层 Graph 融合](#4-解决方案一ir-层-graph-融合)
5. [解决方案二：多 Context 加载](#5-解决方案二多-context-加载)
6. [实现步骤与伪代码](#6-实现步骤与伪代码)
7. [方案对比与选择建议](#7-方案对比与选择建议)

---

## 1. 问题背景

### 1.1 现象描述

- **Qwen1.5-1.8B（24层）**：可以正常运行 ✅
- **Qwen3-1.7B（28层）**：运行到第 26 层时出现 `graphExecute` 错误（Error 6033）❌

### 1.2 错误信息

```
QnnDsp <E> Failed to prepare graph
QnnDsp <E> Graph ... failed in execution with err 6033
```

### 1.3 师兄的诊断

> "QNN 执行的时候一个 context 的子图有个上限，可能是因为这个。qwen3 1.7B 有 28 层，qwen1.5 1.8b 只有 24 层，所以会出现这个问题。"

---

## 2. 核心概念解释

### 2.1 什么是 Context（上下文）？

**Context** 是 QNN 的执行环境，可以理解为"一个工作空间"。

**简单类比**：
- Context = 一个"工厂车间"
- 一个车间里可以有多台"机器"（Graph）
- 所有机器共享车间的资源（内存、设备等）

**在代码中的体现**：
```cpp
Qnn_ContextHandle_t context_;  // 一个 Context 句柄
```

### 2.2 什么是 Graph（计算图）？

**Graph** 是 QNN 中的一个计算图，包含多个操作节点（nodes）和张量（tensors）。

**简单类比**：
- Graph = 一台"机器"
- 机器内部有多个"零件"（节点）
- 零件之间通过"管道"（张量）连接

**在代码中的体现**：
```cpp
class QNNModel {
    Qnn_GraphHandle_t graph_;  // 一个 Graph 句柄
    std::string graphName_;    // Graph 的名称，如 "model.layers.0"
};
```

### 2.3 Context 和 Graph 的关系

```
一个 Context
    ├── Graph 1: "model.layers.0"
    ├── Graph 2: "model.layers.1"
    ├── Graph 3: "model.layers.2"
    ...
    └── Graph N: "model.layers.N"
```

**关键限制**：
- QNN 对一个 Context 内的 Graph 数量有**上限**（例如 24-26 个）
- 当 Graph 数量超过上限时，会出现资源分配失败（Error 6033）

### 2.4 在 Transformer 模型中的体现

**Qwen3-1.7B 的结构**：
```
Qwen3ForCausalLM
    ├── Embedding 层（1 个 Graph）
    ├── Decoder 层（28 层，每层可能有多个 Graph）
    │   ├── Layer 0: Graph "model.layers.0_1", "model.layers.0_2"
    │   ├── Layer 1: Graph "model.layers.1_1", "model.layers.1_2"
    │   ...
    │   └── Layer 27: Graph "model.layers.27_1", "model.layers.27_2"
    └── LMHead 层（1 个 Graph）
```

**问题**：
- 如果每层有 2 个 Graph，28 层 = 56 个 Graph
- 加上 Embedding 和 LMHead，总共可能超过 60 个 Graph
- **远超 QNN Context 的 Graph 数量上限！**

---

## 3. 问题根本原因

### 3.1 QNN Context 的 Graph 数量限制

QNN SDK 对单个 Context 内的 Graph 数量有硬件/软件限制：

- **硬件限制**：HTP（Hexagon Tensor Processor）的片上内存有限，每个 Graph 都需要分配资源
- **软件限制**：QNN 运行时对 Graph 管理有内部限制（通常约 24-26 个）

### 3.2 为什么 Qwen1.5-1.8B 能跑？

- **24 层** × 每层 Graph 数 < Context 上限
- 即使每层有 2 个 Graph，24 × 2 = 48 个 Graph，但可能：
  - 某些层共享 Graph
  - Graph 组织方式不同
  - 实际 Graph 数量在限制内

### 3.3 为什么 Qwen3-1.7B 不能跑？

- **28 层** × 每层 Graph 数 > Context 上限
- 当 Graph 数量接近或超过上限时：
  - 前 24-26 层可以正常创建和执行
  - 第 26-28 层创建 Graph 时失败
  - 导致 `graphExecute` 返回 Error 6033

---

## 4. 解决方案一：IR 层 Graph 融合

### 4.1 方案概述

**核心思想**：在 IR（中间表示）层，把相邻的 Graph 合并成一个更大的 Graph，从而减少 Graph 总数。

**简单类比**：
- 原来：每层有 2 台小机器（Graph）
- 现在：把相邻层的小机器合并成 1 台大机器
- 结果：Graph 总数减少，不超过 Context 上限

### 4.2 具体操作

**师兄的原话**：
> "在 IR 里把 decoder 里第二个 QNN graph 和下一个 decoder 的第一个 QNN graph 融合到一张图里，这个分图方案能跑 28 层。"

**理解**：
- 假设每层 Decoder 有 2 个 Graph：
  - `model.layers.0_1`（第一个 Graph）
  - `model.layers.0_2`（第二个 Graph）
- 融合策略：
  - 把 `model.layers.0_2` 和 `model.layers.1_1` 合并成一个 Graph
  - 把 `model.layers.1_2` 和 `model.layers.2_1` 合并成一个 Graph
  - 以此类推...

**融合后的结构**：
```
原来：
    Layer 0: Graph_0_1, Graph_0_2
    Layer 1: Graph_1_1, Graph_1_2
    Layer 2: Graph_2_1, Graph_2_2
    ...

融合后：
    Layer 0: Graph_0_1, Graph_0_2_merged_with_1_1
    Layer 1: Graph_1_2_merged_with_2_1, Graph_1_2
    Layer 2: Graph_2_2_merged_with_3_1, Graph_2_2
    ...
```

### 4.3 IR 层是什么？

**IR（Intermediate Representation，中间表示）**是 MLLM 框架在模型编译时生成的中间数据结构。

**执行流程**：
```
Python/C++ 模型定义
    ↓
Model::trace()  [生成 IR]
    ↓
IR Module（包含多个 SubGraphOp）
    ↓
QNNGraphBuildPass::run()  [IR → QNN Graph]
    ↓
QNN Graph（在 Context 中）
```

**IR 中的 Graph 标记**：
- `GraphBeginOp`：标记一个 Graph 的开始
- `GraphEndOp`：标记一个 Graph 的结束
- 两个标记之间的所有操作都属于这个 Graph

### 4.4 实现思路（伪代码）

#### Layer 1：自然语言描述

1. **识别需要融合的 Graph**：
   - 遍历 IR 中的所有 `GraphBeginOp` 和 `GraphEndOp`
   - 找到每个 Decoder 层的第二个 Graph（例如 `model.layers.X_2`）
   - 找到下一个 Decoder 层的第一个 Graph（例如 `model.layers.X+1_1`）

2. **执行融合**：
   - 删除 `model.layers.X_2` 的 `GraphEndOp`
   - 删除 `model.layers.X+1_1` 的 `GraphBeginOp`
   - 把两个 Graph 之间的所有操作合并到一个 Graph 中

3. **更新 Graph 名称**：
   - 新 Graph 的名称可以是 `model.layers.X_2_merged_with_X+1_1`

#### Layer 2：技术中文描述

```cpp
// 伪代码：IR Graph 融合 Pass

class IRGraphFusionPass {
    void run(IRModule& module) {
        // 1. 收集所有 Graph 的边界信息
        std::vector<GraphBoundary> graphBoundaries;
        for (每个 SubGraphOp) {
            GraphBoundary boundary;
            boundary.begin = 找到 GraphBeginOp;
            boundary.end = 找到 GraphEndOp;
            boundary.layerIndex = 提取层索引;
            boundary.graphIndex = 提取 Graph 索引（1 或 2）;
            graphBoundaries.push_back(boundary);
        }
        
        // 2. 识别需要融合的 Graph 对
        for (int i = 0; i < graphBoundaries.size() - 1; i++) {
            auto& current = graphBoundaries[i];
            auto& next = graphBoundaries[i + 1];
            
            // 如果当前是某个层的第二个 Graph，下一个是下一层的第一个 Graph
            if (current.graphIndex == 2 && 
                next.graphIndex == 1 && 
                next.layerIndex == current.layerIndex + 1) {
                
                // 3. 执行融合
                fuseGraphs(current, next);
            }
        }
    }
    
    void fuseGraphs(GraphBoundary& graph1, GraphBoundary& graph2) {
        // 删除 graph1 的 GraphEndOp
        删除 graph1.end;
        
        // 删除 graph2 的 GraphBeginOp
        删除 graph2.begin;
        
        // 更新 graph2 中所有操作的 Graph 名称
        更新 graph2 中所有操作的 graph_name 为新的合并名称;
    }
};
```

#### Layer 3：带函数名的伪代码

```cpp
// 在 QNNGraphBuildPass 或新建一个 Pass 中实现

class IRGraphFusionPass : public Pass {
public:
    void run(ir::ModuleOp& module) override {
        // 1. 遍历 IR 找到所有 GraphBeginOp 和 GraphEndOp
        std::vector<std::pair<GraphBeginOp*, GraphEndOp*>> graphs;
        
        module.walk([&](ir::op::GraphBeginOp* beginOp) {
            // 找到对应的 GraphEndOp
            GraphEndOp* endOp = findCorrespondingEndOp(beginOp);
            graphs.push_back({beginOp, endOp});
        });
        
        // 2. 识别需要融合的 Graph 对
        for (size_t i = 0; i < graphs.size() - 1; ++i) {
            auto [begin1, end1] = graphs[i];
            auto [begin2, end2] = graphs[i + 1];
            
            std::string name1 = begin1->options().graph_name;
            std::string name2 = begin2->options().graph_name;
            
            // 检查是否符合融合条件：layer_X_2 和 layer_X+1_1
            if (shouldFuse(name1, name2)) {
                // 3. 执行融合
                fuseTwoGraphs(begin1, end1, begin2, end2);
            }
        }
    }
    
private:
    bool shouldFuse(const std::string& name1, const std::string& name2) {
        // 解析 name1: "model.layers.X_2"
        // 解析 name2: "model.layers.Y_1"
        // 检查 Y == X + 1
        // 返回 true 如果符合条件
    }
    
    void fuseTwoGraphs(GraphBeginOp* begin1, GraphEndOp* end1,
                       GraphBeginOp* begin2, GraphEndOp* end2) {
        // 1. 删除 end1（第一个 Graph 的结束标记）
        end1->erase();
        
        // 2. 删除 begin2（第二个 Graph 的开始标记）
        begin2->erase();
        
        // 3. 更新新 Graph 的名称
        std::string newName = generateMergedName(begin1->options().graph_name,
                                                 begin2->options().graph_name);
        // 注意：begin1 仍然存在，所以第一个 Graph 的名称可以保持不变
        // 或者也可以更新为新名称
    }
};
```

### 4.5 实现位置

**建议在以下位置实现**：

1. **新建一个 Pass**：`mllm/backends/qnn/passes/IRGraphFusionPass.cpp`
2. **在 QNNGraphBuildPass 之前运行**：确保融合后的 IR 再转换为 QNN Graph
3. **或者在 QNNGraphBuildPass 内部实现**：在构建 Graph 时直接合并

---

## 5. 解决方案二：多 Context 加载

### 5.1 方案概述

**核心思想**：把 28 层分成两组，分别加载到两个不同的 Context 中，每个 Context 的 Graph 数量都在限制内。

**简单类比**：
- 原来：所有机器（Graph）放在一个车间（Context）里，车间满了
- 现在：分成两个车间，每个车间放一半机器
- 结果：每个车间的机器数量都在限制内

### 5.2 具体操作

**师兄的原话**：
> "把模型分成两个 context 加载。"

**理解**：
- Context 1：加载前 14 层（Layer 0-13）
- Context 2：加载后 14 层（Layer 14-27）
- 执行时：先执行 Context 1，再执行 Context 2

### 5.3 实现思路（伪代码）

#### Layer 1：自然语言描述

1. **模型分割**：
   - 把 28 层分成两组（例如前 14 层和后 14 层）
   - 每组包含 Embedding/LMHead 的引用或副本

2. **创建两个 Context**：
   - Context 1：加载前 14 层的 Graph
   - Context 2：加载后 14 层的 Graph

3. **执行流程**：
   - Prefill/Decode 时，先执行 Context 1 的所有 Graph
   - 把 Context 1 的输出作为 Context 2 的输入
   - 再执行 Context 2 的所有 Graph

4. **数据传递**：
   - Context 1 的输出张量需要从 QNN 设备内存拷贝到 CPU
   - 然后作为 Context 2 的输入，再拷贝回 QNN 设备内存

#### Layer 2：技术中文描述

```cpp
// 伪代码：多 Context 模型加载和执行

class MultiContextQwen3Model {
    QNNBackend* context1Backend_;  // 第一个 Context 的 Backend
    QNNBackend* context2Backend_;  // 第二个 Context 的 Backend
    
    // Context 1 包含的层范围
    int context1StartLayer_ = 0;
    int context1EndLayer_ = 13;
    
    // Context 2 包含的层范围
    int context2StartLayer_ = 14;
    int context2EndLayer_ = 27;
    
    void loadModel() {
        // 1. 创建第一个 Context 和 Backend
        context1Backend_ = new QNNBackend();
        // 加载前 14 层的 Graph 到 Context 1
        for (int i = context1StartLayer_; i <= context1EndLayer_; i++) {
            context1Backend_->createQnnGraph("model.layers." + std::to_string(i));
        }
        
        // 2. 创建第二个 Context 和 Backend
        context2Backend_ = new QNNBackend();
        // 加载后 14 层的 Graph 到 Context 2
        for (int i = context2StartLayer_; i <= context2EndLayer_; i++) {
            context2Backend_->createQnnGraph("model.layers." + std::to_string(i));
        }
    }
    
    Tensor forward(const Tensor& input) {
        Tensor current = input;
        
        // 1. 执行 Context 1 的所有层
        for (int i = context1StartLayer_; i <= context1EndLayer_; i++) {
            std::string graphName = "model.layers." + std::to_string(i);
            current = context1Backend_->graphExecute(graphName, current);
        }
        
        // 2. 把 Context 1 的输出从 QNN 设备内存拷贝到 CPU
        Tensor context1Output = current.to(kCPU);
        
        // 3. 把数据拷贝到 Context 2 的输入张量
        Tensor context2Input = context1Output.to(kQNN);  // 重新注册到 Context 2
        
        // 4. 执行 Context 2 的所有层
        Tensor output = context2Input;
        for (int i = context2StartLayer_; i <= context2EndLayer_; i++) {
            std::string graphName = "model.layers." + std::to_string(i);
            output = context2Backend_->graphExecute(graphName, output);
        }
        
        return output;
    }
};
```

#### Layer 3：带函数名的伪代码

```cpp
// 在 modeling_qwen3_npu.hpp 中实现

class Qwen3ForCausalLMNPU {
private:
    // 两个独立的 QNN Backend（每个有自己的 Context）
    std::shared_ptr<QNNBackend> context1Backend_;
    std::shared_ptr<QNNBackend> context2Backend_;
    
    // 层分割配置
    int context1LayerCount_ = 14;  // 前 14 层
    int context2LayerCount_ = 14;  // 后 14 层
    
    // Embedding 和 LMHead（可以放在 Context 1 或单独处理）
    std::shared_ptr<QNNEmbeddingOp> embedding_;
    std::shared_ptr<QNNLinearOp> lmHead_;
    
public:
    Qwen3ForCausalLMNPU(const std::string& modelPath, const Qwen3Config& config) {
        // 1. 创建第一个 Context
        context1Backend_ = std::make_shared<QNNBackend>();
        // 注意：QNNBackend 构造函数会创建 Context
        // 我们需要修改 QNNBackend 支持多个 Context，或者创建两个独立的 Backend 实例
        
        // 2. 加载前 14 层的模型权重和 Graph
        loadLayersToContext(context1Backend_, 0, context1LayerCount_ - 1);
        
        // 3. 创建第二个 Context
        context2Backend_ = std::make_shared<QNNBackend>();
        
        // 4. 加载后 14 层的模型权重和 Graph
        loadLayersToContext(context2Backend_, context1LayerCount_, 
                           context1LayerCount_ + context2LayerCount_ - 1);
    }
    
    Tensor forward(const Tensor& inputIds) {
        // 1. Embedding（在 Context 1 中执行）
        Tensor hidden = embedding_->forward(inputIds);
        
        // 2. Context 1 的前 14 层
        for (int i = 0; i < context1LayerCount_; i++) {
            hidden = executeLayerInContext(context1Backend_, i, hidden);
        }
        
        // 3. 数据传递：从 Context 1 拷贝到 CPU，再注册到 Context 2
        Tensor hiddenCPU = hidden.to(kCPU);  // 拷贝到 CPU
        Tensor hiddenContext2 = registerToContext2(hiddenCPU);  // 注册到 Context 2
        
        // 4. Context 2 的后 14 层
        Tensor output = hiddenContext2;
        for (int i = 0; i < context2LayerCount_; i++) {
            int layerIndex = context1LayerCount_ + i;
            output = executeLayerInContext(context2Backend_, layerIndex, output);
        }
        
        // 5. LMHead（可以在 Context 2 中执行，或单独处理）
        Tensor logits = lmHead_->forward(output);
        
        return logits;
    }
    
private:
    void loadLayersToContext(std::shared_ptr<QNNBackend> backend, 
                            int startLayer, int endLayer) {
        // 加载指定范围的层到指定的 Context
        for (int i = startLayer; i <= endLayer; i++) {
            // 加载权重
            // 创建 Graph
            // 注册到 backend
        }
    }
    
    Tensor executeLayerInContext(std::shared_ptr<QNNBackend> backend,
                                int layerIndex, const Tensor& input) {
        std::string graphName = "model.layers." + std::to_string(layerIndex);
        std::vector<Tensor> inputs = {input};
        std::vector<Tensor> outputs;
        backend->graphExecute(graphName, inputs, outputs);
        return outputs[0];
    }
    
    Tensor registerToContext2(const Tensor& cpuTensor) {
        // 把 CPU tensor 注册到 Context 2 的 QNN 设备内存
        // 这需要调用 context2Backend_ 的 allocator
        return cpuTensor.to(kQNN);  // 简化版本，实际需要指定 backend
    }
};
```

### 5.4 关键技术难点

#### 5.4.1 如何创建多个 Context？

**问题**：`QNNBackend` 构造函数中会创建一个 Context，如何支持多个 Context？

**解决方案**：
1. **修改 QNNBackend**：支持传入已有的 Context，或支持创建多个 Context
2. **创建多个 QNNBackend 实例**：每个实例有自己的 Context（需要确认 QNN SDK 是否支持）

#### 5.4.2 如何在不同 Context 间传递数据？

**问题**：Context 1 的输出在 QNN 设备内存中，如何传递给 Context 2？

**解决方案**：
1. **拷贝到 CPU**：`tensor.to(kCPU)` 把数据从 QNN 设备内存拷贝到 CPU
2. **注册到 Context 2**：`tensor.to(kQNN)` 把数据从 CPU 拷贝到 Context 2 的设备内存
3. **性能开销**：会有额外的内存拷贝，但可以接受

#### 5.4.3 KV Cache 如何处理？

**问题**：KV Cache 需要跨 Context 传递吗？

**解决方案**：
- **方案 A**：KV Cache 也分成两部分，分别存储在两个 Context 中
- **方案 B**：KV Cache 存储在 CPU，每次执行时分别传递给两个 Context
- **推荐**：方案 A，减少数据传递开销

---

## 6. 实现步骤与伪代码

### 6.1 方案一实现步骤

#### 步骤 1：理解 IR 结构

```cpp
// 查看 IR 中 Graph 的组织方式
// 文件：mllm/ir/ 相关文件

// Graph 在 IR 中的表示：
// GraphBeginOp(graph_name="model.layers.0_1")
//   ... 操作节点 ...
// GraphEndOp(graph_name="model.layers.0_1")
```

#### 步骤 2：创建 IR Graph 融合 Pass

```cpp
// 新建文件：mllm/backends/qnn/passes/IRGraphFusionPass.cpp

#include "mllm/ir/Pass.hpp"
#include "mllm/core/aops/GraphOps.hpp"

namespace mllm::qnn::passes {

class IRGraphFusionPass : public ir::Pass {
public:
    void run(ir::ModuleOp& module) override {
        // 实现融合逻辑（见前面的伪代码）
    }
};

}  // namespace mllm::qnn::passes
```

#### 步骤 3：在编译流程中插入 Pass

```cpp
// 在 QNNGraphBuildPass 之前运行
// 文件：mllm/backends/qnn/QNNBackend.cpp 或相关编译入口

void compileModel(ir::ModuleOp& module) {
    // 1. IR Graph 融合
    IRGraphFusionPass fusionPass;
    fusionPass.run(module);
    
    // 2. 转换为 QNN Graph
    QNNGraphBuildPass buildPass;
    buildPass.run(module);
}
```

### 6.2 方案二实现步骤

#### 步骤 1：修改 QNNBackend 支持多 Context

```cpp
// 文件：mllm/backends/qnn/QNNBackend.hpp

class QNNBackend : public Backend {
private:
    // 原来只有一个 Context
    // Qnn_ContextHandle_t context_;
    
    // 改为支持多个 Context
    std::vector<Qnn_ContextHandle_t> contexts_;
    std::map<std::string, int> graphNameToContextIndex_;  // Graph 名称到 Context 索引的映射
    
public:
    // 新增：创建额外的 Context
    int createAdditionalContext();
    
    // 新增：在指定 Context 中创建 Graph
    bool createQnnGraphInContext(const std::string& graphName, int contextIndex);
    
    // 修改：graphExecute 需要知道在哪个 Context 中执行
    void graphExecute(const std::string& graphName, 
                     std::vector<Tensor>& inputs,
                     std::vector<Tensor>& outputs);
};
```

#### 步骤 2：实现模型分割加载

```cpp
// 文件：mllm/models/qwen3_npu/modeling_qwen3_npu.hpp

class Qwen3ForCausalLMNPU {
private:
    std::shared_ptr<QNNBackend> backend_;
    int context1LayerCount_ = 14;
    int context2LayerCount_ = 14;
    
    void loadModel() {
        // 1. 创建第一个 Context（默认 Context，索引 0）
        // QNNBackend 构造函数已创建
        
        // 2. 创建第二个 Context
        int context2Index = backend_->createAdditionalContext();
        
        // 3. 加载前 14 层到 Context 0
        for (int i = 0; i < context1LayerCount_; i++) {
            std::string graphName = "model.layers." + std::to_string(i);
            backend_->createQnnGraphInContext(graphName, 0);
            backend_->registerGraphToContext(graphName, 0);
        }
        
        // 4. 加载后 14 层到 Context 2
        for (int i = context1LayerCount_; i < context1LayerCount_ + context2LayerCount_; i++) {
            std::string graphName = "model.layers." + std::to_string(i);
            backend_->createQnnGraphInContext(graphName, context2Index);
            backend_->registerGraphToContext(graphName, context2Index);
        }
    }
};
```

#### 步骤 3：实现跨 Context 数据传递

```cpp
// 在 forward 中实现

Tensor Qwen3ForCausalLMNPU::forward(const Tensor& input) {
    Tensor hidden = input;
    
    // Context 1 的前 14 层
    for (int i = 0; i < context1LayerCount_; i++) {
        std::string graphName = "model.layers." + std::to_string(i);
        std::vector<Tensor> inputs = {hidden};
        std::vector<Tensor> outputs;
        backend_->graphExecute(graphName, inputs, outputs);
        hidden = outputs[0];
    }
    
    // 跨 Context 数据传递
    Tensor hiddenCPU = hidden.to(kCPU);  // 拷贝到 CPU
    Tensor hiddenContext2 = backend_->registerTensorToContext(hiddenCPU, context2Index_);
    
    // Context 2 的后 14 层
    Tensor output = hiddenContext2;
    for (int i = context1LayerCount_; i < context1LayerCount_ + context2LayerCount_; i++) {
        std::string graphName = "model.layers." + std::to_string(i);
        std::vector<Tensor> inputs = {output};
        std::vector<Tensor> outputs;
        backend_->graphExecute(graphName, inputs, outputs);
        output = outputs[0];
    }
    
    return output;
}
```

---

## 7. 方案对比与选择建议

### 7.1 方案对比

| 对比项 | 方案一：IR Graph 融合 | 方案二：多 Context 加载 |
|--------|---------------------|----------------------|
| **实现复杂度** | 中等（需要理解 IR 结构） | 较高（需要修改 Backend） |
| **性能影响** | 无额外开销（Graph 合并后执行更快） | 有数据拷贝开销（跨 Context 传递） |
| **内存占用** | 不变 | 可能略增（两个 Context 的开销） |
| **灵活性** | 中等（融合策略固定） | 高（可以灵活分割层） |
| **可维护性** | 较好（在编译时处理） | 中等（运行时逻辑复杂） |

### 7.2 选择建议

**推荐方案一（IR Graph 融合）**，原因：

1. **性能更好**：没有跨 Context 数据拷贝的开销
2. **实现更简单**：只需要在编译时修改 IR，不需要修改运行时逻辑
3. **符合师兄的建议**：师兄明确说"这个分图方案能跑 28 层"

**如果方案一实现困难，再考虑方案二**：

- 方案二作为备选，可以快速验证多 Context 的可行性
- 但需要注意数据传递的性能开销

### 7.3 实施建议

1. **先实现方案一**：
   - 理解 IR 中 Graph 的组织方式
   - 实现 Graph 融合 Pass
   - 测试是否能解决 28 层的问题

2. **如果方案一不行，再实现方案二**：
   - 修改 QNNBackend 支持多 Context
   - 实现模型分割加载
   - 测试跨 Context 数据传递

3. **与师兄确认**：
   - 实现前先和师兄确认方案细节
   - 特别是 Graph 融合的具体策略
   - 多 Context 是否被 QNN SDK 支持

---

## 8. 总结

### 8.1 问题本质

Qwen3-1.7B 有 28 层，导致 QNN Context 内的 Graph 数量超过硬件/软件限制，无法正常执行。

### 8.2 解决思路

- **方案一**：减少 Graph 数量（通过融合）
- **方案二**：分散到多个 Context（通过分割）

### 8.3 下一步行动

1. 深入理解 IR 结构和 Graph 组织方式
2. 实现方案一（IR Graph 融合）
3. 测试验证是否能解决 28 层问题
4. 如果不行，再考虑方案二

---

## 附录：相关文件位置

- **IR 相关**：
  - `mllm/ir/`：IR 定义
  - `mllm/core/aops/GraphOps.hpp`：GraphBeginOp/GraphEndOp 定义

- **QNN Backend 相关**：
  - `mllm/backends/qnn/QNNBackend.cpp`：Backend 实现
  - `mllm/backends/qnn/QNNModel.hpp`：Graph 管理
  - `mllm/backends/qnn/passes/`：编译 Pass

- **模型相关**：
  - `mllm/models/qwen3_npu/modeling_qwen3_npu.hpp`：Qwen3 NPU 模型定义

---

> **提示**：本文档遵循《qwen_npu_dev_workflow.md》的三层伪代码方法，从自然语言到技术实现逐步细化。实现时建议先和师兄确认细节，再动手编码。

