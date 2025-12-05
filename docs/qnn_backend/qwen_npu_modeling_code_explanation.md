# Qwen NPU Modeling 代码详解（面向零基础）

> 本文档面向完全没有 C++ 基础的初学者，详细解释 `modeling_qwen_npu.hpp` 的代码组织结构、类的作用、函数的意义，以及代码的执行流程。

---

## 目录

1. [C++ 类的基础概念](#1-c-类的基础概念)
2. [代码整体结构](#2-代码整体结构)
3. [各个类的详细解释](#3-各个类的详细解释)
4. [代码执行流程](#4-代码执行流程)
5. [关键概念解释](#5-关键概念解释)

---

## 1. C++ 类的基础概念

### 1.1 什么是类（Class）？

**类**就像是一个"模板"或"蓝图"，用来描述一类对象应该有什么**属性**（数据）和**行为**（函数）。

**简单类比**：
- 类 = 汽车的"设计图纸"
- 对象 = 根据图纸制造出来的"具体汽车"
- 属性 = 汽车的颜色、品牌、型号
- 行为 = 汽车能做什么（启动、加速、刹车）

### 1.2 类的继承（Inheritance）

**继承**就是"子类"可以"继承"（获得）"父类"的所有属性和行为，然后可以添加自己的新功能。

**简单类比**：
- 父类 = "交通工具"（有轮子、能移动）
- 子类 = "汽车"（继承交通工具，但还有发动机、方向盘）

**代码示例**：
```cpp
class 交通工具 {
    void 移动() { ... }
};

class 汽车 : public 交通工具 {  // 汽车继承交通工具
    void 启动引擎() { ... }
};
```

### 1.3 类的成员函数（Member Function）

**成员函数**就是类里面定义的函数，用来描述这个类能做什么。

**代码示例**：
```cpp
class 汽车 {
    void 启动() {  // 这是一个成员函数
        // 启动汽车的代码
    }
};
```

### 1.4 构造函数（Constructor）

**构造函数**是一个特殊的函数，用来在创建对象时**初始化**对象的属性。

**特点**：
- 函数名和类名相同
- 没有返回值（连 `void` 都不写）
- 在创建对象时自动调用

**代码示例**：
```cpp
class 汽车 {
    int 颜色;
    
    汽车(int c) {  // 构造函数
        颜色 = c;  // 初始化颜色
    }
};

汽车 我的车(红色);  // 创建汽车对象，自动调用构造函数
```

### 1.5 `public` 和 `private`

- **`public`**：公开的，任何人都可以访问
- **`private`**：私有的，只有类内部可以访问

**代码示例**：
```cpp
class 汽车 {
public:
    void 启动() { ... }  // 公开的，外部可以调用
    
private:
    int 内部状态;  // 私有的，外部不能访问
};
```

---

## 2. 代码整体结构

### 2.1 文件概览

`modeling_qwen_npu.hpp` 定义了 Qwen 模型在 NPU 上的实现，包含以下主要类：

```
QwenForCausalLM（最外层，整个模型）
    └── QwenText（文本处理层）
        ├── embedding_（词嵌入）
        ├── decode_blocks_（多个 QwenDecoder 层）
        └── norm_（最终归一化）
        
        QwenDecoder（单个 Transformer 层）
            ├── self_attn_proj_（注意力投影，QNN）
            ├── self_attn_matmul_（注意力计算，CPU）
            └── self_attn_out_mlp_（输出投影+MLP，QNN）
```

### 2.2 类的层次关系

```
nn::Module（基础模块类，框架提供）
    ├── QwenAttentionProjNPU（注意力投影）
    ├── QwenAttentionMatmul（注意力计算）
    ├── QwenOutProjAndMLP（输出+MLP）
    ├── QwenDecoder（单个 Transformer 层）
    ├── QwenText（多层堆叠）
    └── QwenForCausalLM（完整模型）
```

**说明**：
- 所有类都继承自 `nn::Module`，这是框架提供的基础类
- `nn::Module` 提供了注册子模块、前向传播等基础功能

---

## 3. 各个类的详细解释

### 3.1 `QwenAttentionProjNPU`（注意力投影，QNN 执行）

**作用**：将输入转换为 Query、Key、Value 三个向量，在 QNN（NPU）上执行。

**关键成员变量**：
```cpp
nn::RMSNorm input_layer_norm_;        // 输入层归一化
nn::Linear q_proj_, k_proj_, v_proj_; // Q/K/V 投影层（线性变换）
nn::Param quantize_scale_;            // 量化缩放参数
nn::qnn::DequantizeAdd ...;           // 反量化层
```

**关键函数**：
- **`QwenAttentionProjNPU(...)`**（构造函数）：
  - 注册所有子模块（LayerNorm、Linear、DequantizeAdd 等）
  - 设置参数（hidden_size、head_dim 等）

- **`forward(...)`**（前向传播函数）：
  - 输入：`x`（形状 `[B, S, H]`，B=批次，S=序列长度，H=隐藏层大小）
  - 步骤：
    1. 对 `x` 做 LayerNorm（归一化）
    2. 将 `x` 转换为 `kInt16`（量化，准备给 QNN）
    3. 分别计算 `q_proj_(x)`、`k_proj_(x)`、`v_proj_(x)`（在 QNN 上执行）
    4. 将结果反量化回 `kFloat32`
    5. 调整形状并转置
  - 输出：`{query_states, key_states, value_states}`（三个张量）

**代码位置**：第 100-187 行

---

### 3.2 `QwenAttentionMatmul`（注意力计算，CPU 执行）

**作用**：计算注意力权重，更新 KV Cache，在 CPU 上执行。

**关键成员变量**：
```cpp
nn::RoPE q_rope_, k_rope_;  // 旋转位置编码
nn::CausalMask mask_;        // 因果掩码
nn::Softmax softmax_;        // Softmax 激活
nn::KVCache kv_cache_;       // KV 缓存
```

**关键函数**：
- **`QwenAttentionMatmul(...)`**（构造函数）：
  - 注册 RoPE、Mask、Softmax、KVCache

- **`forward(...)`**（前向传播函数）：
  - 输入：`query_states`、`key_states`、`value_states`、`llm_embedding_sin`、`llm_embedding_cos`
  - 步骤：
    1. 对 Q 和 K 应用 RoPE（旋转位置编码）
    2. 更新 KV Cache（将新的 K/V 存入缓存）
    3. 计算注意力权重：`attn = Q @ K^T / sqrt(head_dim)`
    4. 应用因果掩码（防止看到未来信息）
    5. 应用 Softmax
    6. 计算注意力输出：`output = attn @ V`
  - 输出：`{output}`（注意力输出）

**代码位置**：第 189-272 行

---

### 3.3 `QwenOutProjAndMLP`（输出投影+MLP，QNN 执行）

**作用**：注意力输出投影 + 多层感知机（MLP），在 QNN 上执行。

**关键成员变量**：
```cpp
nn::Linear o_proj_;                    // 输出投影
nn::Linear gate_proj_, up_proj_, down_proj_;  // MLP 的三个线性层
nn::SiLU silu_;                        // SiLU 激活函数
nn::RMSNorm post_attention_layer_norm_; // 后注意力层归一化
nn::Param ..._quantize_scale_;         // 量化缩放参数
```

**关键函数**：
- **`QwenOutProjAndMLP(...)`**（构造函数）：
  - 注册所有线性层、激活函数、归一化层

- **`forward(...)`**（前向传播函数）：
  - 输入：`x`（注意力输出）、`res`（残差连接）
  - 步骤：
    1. 对 `x` 做量化，执行 `o_proj_`（输出投影）
    2. 与 `res` 相加（残差连接）
    3. 做 LayerNorm
    4. 执行 MLP：`gate = SiLU(gate_proj_(x))`，`up = up_proj_(x)`，`x = gate * up`，`x = down_proj_(x)`
    5. 再次与 `tmp` 相加（残差连接）
  - 输出：`{x}`（最终输出）

**代码位置**：第 274-359 行

---

### 3.4 `QwenDecoder`（单个 Transformer 层）

**作用**：组合上述三个模块，形成一个完整的 Transformer Decoder 层。

**关键成员变量**：
```cpp
QwenAttentionProjNPU self_attn_proj_;  // 注意力投影（QNN）
QwenAttentionMatmul self_attn_matmul_;  // 注意力计算（CPU）
QwenOutProjAndMLP self_attn_out_mlp_;  // 输出+MLP（QNN）
```

**关键函数**：
- **`QwenDecoder(...)`**（构造函数）：
  - 注册三个子模块
  - 将 `self_attn_proj_` 和 `self_attn_out_mlp_` 设置为在 QNN 上执行（`.to(kQNN)`）

- **`forward(...)`**（前向传播函数）：
  - 输入：`x`（输入张量）、`llm_embedding_sin`、`llm_embedding_cos`（RoPE 参数）
  - 步骤：
    1. 将 `x` 转换到 QNN 设备
    2. 调用 `self_attn_proj_`，得到 Q/K/V（QNN 执行）
    3. 将 Q/K/V 转换到 CPU
    4. 调用 `self_attn_matmul_`，得到注意力输出（CPU 执行）
    5. 将输出转换回 QNN
    6. 调用 `self_attn_out_mlp_`，得到最终输出（QNN 执行）
  - 输出：`{x}`（该层的输出）

**代码位置**：第 361-403 行

**执行顺序**：**QNN → CPU → QNN**

---

### 3.5 `QwenText`（多层堆叠）

**作用**：将多个 `QwenDecoder` 层堆叠起来，加上 Embedding 和最终归一化。

**关键成员变量**：
```cpp
nn::ModuleList<QwenDecoder> decode_blocks_;  // 多个 Decoder 层的列表
nn::RMSNorm norm_;                            // 最终归一化
nn::Embedding embedding_;                     // 词嵌入层
```

**关键函数**：
- **`QwenText(...)`**（构造函数）：
  - 注册 `decode_blocks_`（创建多个 `QwenDecoder` 层）
  - 注册 `norm_`（最终归一化）
  - 注册 `embedding_`，并设置为在 QNN 上执行（`.to(kQNN)`）
  - 初始化 RoPE 的 `inv_freq` 缓冲区

- **`forward(...)`**（前向传播函数）：
  - 输入：`x`（已经嵌入后的张量）、`llm_embedding_sin`、`llm_embedding_cos`
  - 步骤：
    1. 循环遍历所有 `decode_blocks_`，依次调用每一层的 `forward`
    2. 将最终输出转换到 CPU
    3. 执行最终归一化 `norm_`
  - 输出：`{x}`（所有层处理后的输出）

**代码位置**：第 405-456 行

**注意**：`forward` 函数的注释说 "X is already embedded"，意思是输入 `x` 已经是嵌入后的结果，不需要在这里再做 embedding。

---

### 3.6 `QwenForCausalLM`（完整模型）

**作用**：最外层的模型类，包含 Embedding、多层 Decoder、输出层（lm_head）。

**关键成员变量**：
```cpp
QwenText model;           // 文本处理层（包含多层 Decoder）
nn::Linear lm_head_;      // 输出层（将隐藏状态转换为词汇表概率）
bool tie_word_embeddings_; // 是否共享词嵌入权重
```

**关键函数**：

#### 3.6.1 `QwenForCausalLM(...)`（构造函数）
- 注册 `model`（QwenText）
- 如果不需要共享权重，注册 `lm_head_`

#### 3.6.2 `forward(...)`（前向传播函数，运行时调用）

**输入**：
- `input`：包含 `{"sequence": ...}` 的字典
- `args`：包含 `{"seq_len": ...}` 的参数

**步骤**：
1. 从 `input` 中获取 `sequence`（token IDs）
2. 生成 `position_ids`（位置编码）
3. 生成 RoPE 的 sin/cos 嵌入：`makeRotaryPosEmbedding(...)`
4. 调用 `model.embedding_(sequence)`，得到词嵌入（**在 QNN 上执行**）
5. 调用 `model(input_embeddings, llm_embedding_sin, llm_embedding_cos)`，得到隐藏状态
6. 根据 `real_seq` 裁剪隐藏状态（只取最后一个有效位置）
7. 调用 `lm_head_` 或使用共享权重，得到 logits（词汇表概率）
8. 返回 `{"sequence": logits, "position_ids": position_ids}`

**代码位置**：第 475-531 行

#### 3.6.3 `trace(...)`（追踪函数，用于生成 IR）

**作用**：生成中间表示（IR），用于后续的图优化和 QNN 编译。

**步骤**：
1. 开始追踪：`ir::lowlevel::traceStart()`
2. 执行 embedding（不追踪）
3. 暂停追踪：`ir::lowlevel::traceYield()`
4. 准备 RoPE 参数（不追踪）
5. 继续追踪：`ir::lowlevel::traceContinue()`
6. 追踪 `model` 的前向传播：`ir::lowlevel::traceModule(...)`
7. 停止追踪：`ir::lowlevel::traceStop()`
8. 返回 IR

**代码位置**：第 533-573 行

---

## 4. 代码执行流程

### 4.1 模型初始化流程

```
1. 创建 QwenForCausalLM 对象
   └── 调用 QwenForCausalLM 构造函数
       ├── 创建 QwenText 对象
       │   ├── 创建多个 QwenDecoder 对象（每个对应一个 Transformer 层）
       │   │   ├── 创建 QwenAttentionProjNPU（注册 Linear、DequantizeAdd 等）
       │   │   ├── 创建 QwenAttentionMatmul（注册 RoPE、KVCache 等）
       │   │   └── 创建 QwenOutProjAndMLP（注册 Linear、SiLU 等）
       │   ├── 创建 Embedding 层
       │   └── 创建最终归一化层
       └── 创建 lm_head_（如果需要）
```

### 4.2 前向传播流程（运行时）

```
用户调用 model.forward(input, args)
    │
    ├─ 1. 生成 position_ids（位置编码）
    │
    ├─ 2. 生成 RoPE 的 sin/cos 嵌入
    │
    ├─ 3. 执行 embedding（QNN）
    │   └── model.embedding_(sequence)
    │
    ├─ 4. 调用 model.forward(...)（QwenText::forward）
    │   │
    │   └── 循环遍历所有 decode_blocks_（每一层）
    │       │
    │       └── 调用 block.forward(...)（QwenDecoder::forward）
    │           │
    │           ├─ 4.1 转换到 QNN
    │           │
    │           ├─ 4.2 调用 self_attn_proj_.forward(...)（QNN）
    │           │   ├── LayerNorm
    │           │   ├── 量化
    │           │   ├── q_proj_、k_proj_、v_proj_（QNN）
    │           │   └── 反量化
    │           │
    │           ├─ 4.3 转换到 CPU
    │           │
    │           ├─ 4.4 调用 self_attn_matmul_.forward(...)（CPU）
    │           │   ├── RoPE
    │           │   ├── 更新 KV Cache
    │           │   ├── 计算注意力权重
    │           │   ├── 应用掩码和 Softmax
    │           │   └── 计算注意力输出
    │           │
    │           ├─ 4.5 转换回 QNN
    │           │
    │           └─ 4.6 调用 self_attn_out_mlp_.forward(...)（QNN）
    │               ├── o_proj_（输出投影）
    │               ├── 残差连接
    │               ├── LayerNorm
    │               └── MLP（gate_proj_、up_proj_、down_proj_）
    │
    ├─ 5. 转换到 CPU，执行最终归一化
    │
    ├─ 6. 裁剪隐藏状态（只取最后一个有效位置）
    │
    └─ 7. 调用 lm_head_，得到 logits
```

### 4.3 追踪流程（用于生成 IR）

```
用户调用 model.trace(input, args)
    │
    ├─ 1. traceStart()（开始追踪）
    │
    ├─ 2. 执行 embedding（不追踪）
    │
    ├─ 3. traceYield()（暂停追踪）
    │
    ├─ 4. 准备 RoPE 参数（不追踪）
    │
    ├─ 5. traceContinue()（继续追踪）
    │
    ├─ 6. traceModule(model, ...)（追踪 model 的前向传播）
    │   └── 这会记录所有在 QNN 上执行的算子
    │
    └─ 7. traceStop()（停止追踪，返回 IR）
```

---

## 5. 关键概念解释

### 5.1 关于注释 "execute on CPU" 的说明

**问题**：第 417 行的注释说 "execute on CPU"，但代码是 `embedding_.to(kQNN)`，这是否矛盾？

**解释**：
- **`embedding_.to(kQNN)`** 表示 embedding 模块应该在 **QNN 设备上执行**
- 注释中的 "execute on CPU" 可能是**过时的**或**不准确的**
- 更准确的理解是：QNN 版本的 embedding 会**处理 padding token**（padding token 的处理可能在 CPU 上，但主要的 embedding 查找在 QNN 上）

**结论**：**embedding 主要在 QNN 上执行**，注释可能有误导性。

### 5.2 设备转换（`.to(kQNN)` / `.to(kCPU)`）

**作用**：将张量或模块从一个设备转换到另一个设备。

**代码示例**：
```cpp
x = x.to(kQNN);  // 将 x 转换到 QNN 设备
x = x.to(kCPU);  // 将 x 转换到 CPU 设备
```

**在代码中的使用**：
- 第 382 行：`x = x.to(kQNN);` - 将输入转换到 QNN，准备执行 QNN 算子
- 第 386-388 行：`query_states = states[0].to(kCPU);` - 将 Q/K/V 转换到 CPU，准备执行 CPU 算子
- 第 392 行：`x = x.to(kQNN);` - 将注意力输出转换回 QNN，准备执行 QNN 算子

### 5.3 注册子模块（`reg<...>(...)`）

**作用**：向父模块注册一个子模块，这样父模块可以管理子模块的生命周期和参数。

**代码示例**：
```cpp
self_attn_proj_ = reg<QwenAttentionProjNPU>("", cfg);
```

**说明**：
- `reg<类型>("名字", 参数)` 创建一个子模块并注册到父模块
- 第一个参数是子模块的名字（空字符串表示使用默认名字）
- 第二个参数是配置对象

### 5.4 量化与反量化

**量化**：将 `kFloat32` 转换为 `kInt16`，减少数据大小，提高 QNN 执行效率。

**反量化**：将 `kInt16` 转换回 `kFloat32`，用于后续的 CPU 计算。

**代码示例**：
```cpp
x.attach("qnn_quant_scale", quantize_scale_.weight().impl());  // 附加量化缩放参数
x = x.to(kInt16);  // 量化
// ... 在 QNN 上执行 ...
x = x.to(kFloat32);  // 反量化（或通过 DequantizeAdd）
```

### 5.5 KV Cache

**作用**：缓存之前计算的 Key 和 Value，避免重复计算。

**使用场景**：
- Prefill 阶段：计算所有位置的 K/V 并存入缓存
- Decode 阶段：只计算新 token 的 K/V，与缓存中的 K/V 拼接

**代码位置**：
- 第 223-224 行：注册 `nn::KVCache`
- 第 241 行：`kv_cache_(key_states, value_states)` - 更新缓存

### 5.6 残差连接（Residual Connection）

**作用**：将输入直接加到输出上，帮助梯度传播和训练稳定性。

**代码示例**：
```cpp
auto tmp = x + res;  // 残差连接
// ... 做一些变换 ...
x = x + tmp;  // 再次残差连接
```

**在代码中的使用**：
- 第 330 行：`auto tmp = x + res;` - 第一次残差连接
- 第 356 行：`x = x + tmp;` - 第二次残差连接

---

## 6. 总结

### 6.1 类的组织方式

```
QwenForCausalLM（最外层）
    └── QwenText（多层堆叠）
        ├── embedding_（词嵌入，QNN）
        ├── decode_blocks_（多个 QwenDecoder）
        │   └── QwenDecoder（单个层）
        │       ├── self_attn_proj_（QNN）
        │       ├── self_attn_matmul_（CPU）
        │       └── self_attn_out_mlp_（QNN）
        └── norm_（最终归一化，CPU）
```

### 6.2 执行顺序

**每个 Decoder 层**：
1. **QNN**：AttentionProj（Q/K/V 投影）
2. **CPU**：AttentionMatmul（RoPE + KVCache + Attention）
3. **QNN**：OutProj + MLP

**整个模型**：
1. Embedding（QNN）
2. 多层 Decoder（QNN → CPU → QNN）
3. 最终归一化（CPU）
4. lm_head（输出层）

### 6.3 关键函数

- **构造函数**：初始化模块，注册子模块
- **`forward(...)`**：前向传播，执行计算
- **`trace(...)`**：生成 IR，用于图优化和编译

---

> **提示**：如果还有不理解的地方，建议：
> 1. 先理解类的继承关系（谁继承谁）
> 2. 再理解每个类的作用（它负责做什么）
> 3. 最后理解执行流程（代码是怎么一步步执行的）

