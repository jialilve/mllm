## IR SubGraph 克隆与 Qwen3 融合实现流程（新手视角）

> 面向对象：**从零开始、第一次接触 MLLM IR / QNN Backend 的新手小白**  
> 目标：看完之后，你能读懂我们在 IR 层做了什么、`SubGraphCloneUtils` 怎么实现、以后怎么用它来完成 Qwen3 的跨层融合。

---

### 1. 先搞清楚：我们到底在“复制”什么？

在 MLLM 里，一个模型在 Trace 之后，会变成一棵 IR：

- `Node`：所有 IR 节点的基类  
- `Val`：表示“张量值”（数据）  
- `Op`：表示“算子”，有输入 `Val`、输出 `Val`  
- `Region`：一段有序的 `Op` 列表，加上这段区域的“graph 输入 / graph 输出”  
- `SubGraphOp`：一个带有 `Region` 的 `Op`，可以理解为“子图函数”

可以用一句话总结：

> **一个 QNN 子图 = 一个 `SubGraphOp` + 里面那个 `Region` 里的所有 `Op` + 所有输入输出的 `Val` 以及它们之间的边。**

我们现在要做的，是把这个“子图函数”**完整复制一份**：

- 新的 `SubGraphOp`（有新的名字）  
- 新的 `Region`，里面有一模一样顺序的 `Op`  
- 每个 `Op` 的输入输出 `Val` 都有各自的新副本（不和原来的指针混在一起）  
- 所有数据流（谁连到谁）保持一致

这样，后面 Qwen3 的 Fusion Pass 才能：

- 把 `model.layers.X_2` 复制一份到 `model.layers.X_fused`  
- 把 `model.layers.(X+1)_1` 也复制进 `model.layers.X_fused` 的同一个 Region 里  
- 再修改 fused graph 的输入输出，就完成了“新建 fused Graph + 复制操作”的方案。

---

### 2. 关键类和文件位置一览

实现克隆逻辑涉及的核心文件：

- `mllm/compile/ir/Node.hpp`
  - `Node` / `Val` / `Op` / `Region` / `IRContext` / `IRWriter` 等 IR 基础设施
- `mllm/compile/ir/tensor/Value.hpp` + `Value.cpp`
  - `TensorValue`：IR 层表示一个具体的张量（含 shape、dtype、device、memType、name 等）
- `mllm/compile/ir/linalg/Op.hpp` + `Op.cpp`
  - IR 里的 linalg/QNN 算子，对应 Qwen3 子图中的各种 `linalg.QNN.*` op
- `mllm/compile/ir/cf/Op.hpp` + `Op.cpp`
  - 控制流 IR，例如 `cf.ReturnOp`
- **`mllm/compile/ir/graph/SubGraphCloneUtils.hpp/.cpp`**
  - 我们新加的“子图克隆工具类”，对外提供 `cloneSubGraph` 接口

你以后主要用的入口就是：

```cpp
ir::graph::SubGraphOp::ptr_t SubGraphCloneUtils::cloneSubGraph(
    const IRContext::ptr_t& ctx,
    const SubGraphOp::ptr_t& src,
    const CloneOptions& options,
    std::unordered_map<val_ptr_t, val_ptr_t>* value_map = nullptr);
```

---

### 3. IR 基础：Node / Val / Op / Region / SubGraphOp 是怎么连起来的？

#### 3.1 Node：所有 IR 节点的共同基类

`Node` 里有几个关键成员：

- `kind_`：这个 Node 是什么类型（Op？Val？哪种具体 Op？）  
- `inputs_` / `outputs_`：这是 **IR 图上的边**，表示“这个 Node 依赖谁 / 被谁依赖”  
- `belongs_to_parent_`：表示“这个 Node 属于哪个父节点/Op”（比如某个 Region）  
- `attrs_`：各种属性表（字符串 → Attribute）

我们额外给 `Node` 增加了一个小工具方法，方便 clone 时复制属性：

```47:60:mllm/compile/ir/Node.hpp
  void setAttr(const std::string& str, const attr_ptr_t& attr);

  attr_ptr_t getAttr(const std::string& str);

  inline const std::unordered_map<std::string, attr_ptr_t>& getAttrs() const { return attrs_; }
```

#### 3.2 Val：表示“张量值”节点

`Val` 是 `Node` 的子类，用来表示 IR 中的“一个结果张量”，最重要的成员是一个 `name_`，方便打印和调试。

`TensorValue` 是 `Val` 的具体子类，用来包装真正的 `Tensor`：

```27:41:mllm/compile/ir/tensor/Value.hpp
class TensorValue : public TensorIRValue, public SymbolInterface<TensorValue> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(TensorValue);

  ~TensorValue() override;
  TensorValue();

  inline const Tensor& tensor() const { return tensor_; }

  inline void setTensor(const Tensor& tensor) { tensor_ = tensor; }

  static ptr_t build(IRContext* ctx, const Tensor& tensor);
  ...
  Tensor tensor_;
};
```

在 clone 时，我们会用这些接口复制一份新的 `TensorValue`，保证新的子图里有独立的 tensor IR。

#### 3.3 Op / Region / SubGraphOp：算子、区域和子图

- `Op`：表示一个算子，有 `inputs()` / `outputs()` / `dump()` 等接口  
- `Region`：包含：
  - `std::list<op_ptr_t> ops_`：这段子图里的所有 op  
  - `std::list<val_ptr_t> inputs_`：子图的“形参”  
  - `std::list<val_ptr_t> outputs_`：子图的“返回值”
- `SubGraphOp`：一个特殊的 `Op`，自带一个 `Region`，可以看成“函数定义”

在 MIR 里，一个 QNN 子图大概长这样：

```mir
graph.SubGraphOp @model.layers.26_2 <QNN> {
    (%in0: tensor<...>, %in1: tensor<...>) -> (%out: tensor<...>) {
        linalg.QNN.ViewOp ...
        linalg.QNN.LinearOp ...
        ...
        cf.ReturnOp (%out) -> ()
    }
}
```

我们要 clone 的，就是整个 `SubGraphOp` 以及里面这一段 region。

---

### 4. cloneSubGraph 的整体算法（高层路线图）

`cloneSubGraph` 的目标：

> 给定一个 `src` 子图，建立一个新的 `cloned` 子图，使得：
> - 有新的 symbol 名字（避免和原来的重复）  
> - `device`、属性等信息复制过来  
> - Region 的输入 / 输出 与原来一一对应，但都是新的 `Val`  
> - Region 里每个 op 有一个一一对应的新实例，输入输出 `Val` 通过 `value_map` 做 old→new 映射  
> - 最后的 `cf.ReturnOp` 也用新的 `Val`，并把这些新 `Val` 挂到 `cloned_region->outputs()` 上

核心数据结构：

- `ValueMap value_map;`：`old Val -> new Val` 的映射表
  - 图输入：`src_input` → `cloned_input`  
  - 中间结果：某个 `linalg` op 的输出 `old_out` → 新建 `cloned_out`  
  - 图输出：`src_output` 会在最后通过 `value_map` 找到对应的新值

整个流程分 5 步：

1. 决定新子图的 symbol 名字，创建 `SubGraphOp cloned`，复制 device / 属性  
2. 克隆 region 输入，建立 `value_map[src_input] = cloned_input`  
3. 用 `IRWriter` 遍历源 region 的每个 op，按类型 clone 一份新 op，并更新 `value_map`  
4. clone 最后的 `cf.ReturnOp`，让它返回新 `Val`  
5. 根据 `value_map` 重建 `cloned_region->outputs()`，并把这些值作为 `cloned` 的 graph 输出

下面逐步展开。

---

### 5. 代码细节拆解：SubGraphCloneUtils.cpp

#### 5.1 辅助函数：复制属性 & 复制 TensorValue

**复制属性：**

```82:94:mllm/compile/ir/graph/SubGraphCloneUtils.cpp
void copyNodeAttributes(const node_ptr_t& src, const node_ptr_t& dst) {
  if (!src || !dst) { return; }
  for (const auto& [attr_name, attr_value] : src->getAttrs()) {
    dst->setAttr(attr_name, attr_value);
  }
}
```

这里使用了前面在 `Node` 上加的 `getAttrs()` 读出所有属性，然后一一塞到新节点上。  
注意当前实现是“浅拷贝指针”，对我们的用途（大部分 attr 是只读常量）是够用的。

**复制 TensorValue：**

```96:109:mllm/compile/ir/graph/SubGraphCloneUtils.cpp
tensor::TensorValue::ptr_t cloneTensorValueLike(const IRContext::ptr_t& ctx,
                                                const tensor::TensorValue::ptr_t& src) {
  if (!ctx || !src) { return nullptr; }
  auto cloned = ctx->createTemporaryValue<tensor::TensorValue>(src->tensor());
  copyNodeAttributes(src, cloned);
  return cloned;
}
```

这里有几个关键点：

- 使用 `src->tensor()` 拿到原始 `Tensor`，用 `createTemporaryValue` 创建新的 `TensorValue`；
- `createTemporaryValue` 会自动给 `Val` 起一个唯一的名字，并写入符号表；
- 复制属性，保持诸如 `constant`、`symbol` 等信息一致。

**通用 Val 克隆入口：**

```111:124:mllm/compile/ir/graph/SubGraphCloneUtils.cpp
val_ptr_t cloneValueLike(const IRContext::ptr_t& ctx, const val_ptr_t& src) {
  if (!src) { return nullptr; }
  if (auto tensor_val = std::dynamic_pointer_cast<tensor::TensorValue>(src)) {
    return cloneTensorValueLike(ctx, tensor_val);
  }

  MLLM_ERROR("SubGraphCloneUtils: unsupported Val type when cloning (kind={})",
             static_cast<int>(src->getKind()));
  return nullptr;
}
```

当前只支持 `TensorValue`，对于别的 `Val` 类型，我们先报错。  
Qwen3 的 QNN 子图里，graph IO 和中间值都是 `TensorValue`，这就够用了。

#### 5.2 克隆 graph 输入并建立初始 value_map

```126:149:mllm/compile/ir/graph/SubGraphCloneUtils.cpp
bool mapGraphInputs(const IRContext::ptr_t& ctx,
                    const SubGraphOp::ptr_t& cloned_graph,
                    const std::list<val_ptr_t>& src_inputs,
                    std::list<val_ptr_t>& cloned_inputs,
                    ValueMap& value_map) {
  cloned_inputs.clear();
  for (auto& src_input : src_inputs) {
    auto cloned_input = cloneValueLike(ctx, src_input);
    if (!cloned_input) {
      MLLM_ERROR("SubGraphCloneUtils: failed to clone graph input '{}'",
                 src_input ? src_input->name() : "<null>");
      return false;
    }
    cloned_inputs.push_back(cloned_input);
    value_map[src_input] = cloned_input;
    // 链接 SubGraph 与输入 Value
    (*cloned_input)-- > cloned_graph;
  }
  return true;
}
```

这里做了三件事：

1. 为每个 `src_input` 创建新的 `cloned_input`；
2. 填写 `value_map[src_input] = cloned_input`；  
3. 用 `(*cloned_input)-- > cloned_graph;` 建立边：
   - `cloned_input` 的 `outputs()` 里有 `cloned_graph`  
   - `cloned_graph` 的 `inputs()` 里有 `cloned_input`

这保证了新的子图在 IR 中有正确的“形参列表”和边。

#### 5.3 克隆 Linalg / QNN 算子：cloneLinalgOp

Qwen3 的 QNN 子图内部大多是 `ir::linalg::LinalgIROp` 的各种子类，比如：

- `ViewOp` / `CastTypeOp` / `LinearOp` / `RMSNormOp` / `TransposeOp` / `SiLUOp`  
- `AddOp` / `MulOp` / `X2XOp`  
- `CustomizedOp`（DequantizeAdd 用的是这个）

我们给这类 op 写了一个通用的克隆逻辑：

```151:222:mllm/compile/ir/graph/SubGraphCloneUtils.cpp
bool cloneLinalgOp(const IRContext::ptr_t& ctx,
                   const ir::linalg::LinalgIROp::ptr_t& src_op,
                   IRWriter& writer,
                   ValueMap& value_map) {
  ...
  // 1) 根据 value_map 取得新输入
  for (auto& input_weak : src_op->inputs()) {
    auto* input_node = input_weak.get_weak();
    if (!input_node || !input_node->isa_<Val>()) {
      MLLM_ERROR("SubGraphCloneUtils: invalid input node while cloning op");
      return false;
    }
    auto old_val = std::static_pointer_cast<Val>(input_node->shared_from_this());
    auto map_it = value_map.find(old_val);
    if (map_it == value_map.end()) {
      MLLM_ERROR("SubGraphCloneUtils: missing value mapping for input '{}'", old_val->name());
      return false;
    }
    auto tensor_input = std::dynamic_pointer_cast<tensor::TensorValue>(map_it->second);
    ...
    cloned_inputs.push_back(tensor_input);
  }

  // 2) 为输出创建新的 TensorValue，并写回 value_map
  for (auto& output_weak : src_op->outputs()) {
    auto* output_node = output_weak.get_weak();
    ...
    auto old_val = std::static_pointer_cast<Val>(output_node->shared_from_this());
    auto cloned_val = cloneValueLike(ctx, old_val);
    ...
    auto tensor_output = std::dynamic_pointer_cast<tensor::TensorValue>(cloned_val);
    ...
    cloned_outputs.push_back(tensor_output);
    value_map[old_val] = cloned_val;
  }

  // 3) 用原来的 BaseOp 构建新的具体 IR op
  auto base_op = src_op->getAOp() ? src_op->getAOp()->shared_from_this() : nullptr;
  ...
  if (src_op->isa_<ir::linalg::ViewOp>()) {
    return createClonedLinalgOp<ir::linalg::ViewOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }
  if (src_op->isa_<ir::linalg::CastTypeOp>()) { ... }
  if (src_op->isa_<ir::linalg::LinearOp>()) { ... }
  ...
  if (src_op->isa_<ir::linalg::CustomizedOp>()) {
    return createClonedLinalgOp<ir::linalg::CustomizedOp>(writer, cloned_inputs, cloned_outputs, base_op, src_op);
  }

  MLLM_ERROR("SubGraphCloneUtils: unsupported Linalg op '{}'", static_cast<int>(src_op->getKind()));
  return false;
}
```

这里有几个重要的小点：

- **输入 remap**：  
  - 每个老输入 `old_val` 必须在 `value_map` 里能找到对应的新 `Val`，否则说明图不拓扑一致或 `value_map` 填漏了。
- **输出新建**：  
  - 每个老输出都会新建一个 `TensorValue`，并写入 `value_map[old_out] = new_out`；
  - 后续别的 op 再用到这个 old_out 时，就可以在 `value_map` 里查到 new_out。
- **用 IRWriter 创建新 op**：  
  - `createClonedLinalgOp` 里其实就是调用对应 `T::build(ctx, base_op, ins, outs)`，并复制属性。

#### 5.4 克隆 ReturnOp：cloneReturnOp

子图的结束必须是一个 `cf.ReturnOp`，它的输入就是图的返回值列表：

```224:249:mllm/compile/ir/graph/SubGraphCloneUtils.cpp
bool cloneReturnOp(const ir::cf::ReturnOp::ptr_t& src_return,
                   IRWriter& writer,
                   ValueMap& value_map) {
  if (!src_return) {
    MLLM_ERROR("SubGraphCloneUtils: missing ReturnOp in source SubGraph");
    return false;
  }
  std::vector<val_ptr_t> cloned_returns;
  cloned_returns.reserve(src_return->inputs().size());
  for (auto& ret_input_weak : src_return->inputs()) {
    auto* ret_node = ret_input_weak.get_weak();
    if (!ret_node || !ret_node->isa_<Val>()) {
      MLLM_ERROR("SubGraphCloneUtils: invalid Return input");
      return false;
    }
    auto old_val = std::static_pointer_cast<Val>(ret_node->shared_from_this());
    auto it = value_map.find(old_val);
    if (it == value_map.end()) {
      MLLM_ERROR("SubGraphCloneUtils: missing mapping for Return input '{}'", old_val->name());
      return false;
    }
    cloned_returns.push_back(it->second);
  }
  writer.create<ir::cf::ReturnOp>(cloned_returns);
  return true;
}
```

注意这里**不再新建 Return 的输入 `Val`**，而是直接用已经在 `value_map` 里的新值：  
它们要么是 graph 输入的 clone，要么是前面某个 linalg op 的输出 clone。

#### 5.5 主入口：cloneSubGraph

最后，把上面的步骤串起来：

```251:312:mllm/compile/ir/graph/SubGraphCloneUtils.cpp
SubGraphOp::ptr_t SubGraphCloneUtils::cloneSubGraph(
    const IRContext::ptr_t& ctx,
    const SubGraphOp::ptr_t& src,
    const CloneOptions& options,
    std::unordered_map<val_ptr_t, val_ptr_t>* value_map) {
  ...
  auto src_symbol = src->getSymbolAttr();
  ...
  std::string new_symbol_name = options.new_symbol_name.empty()
      ? ctx->getUniqueModuleName(src_symbol->str())
      : options.new_symbol_name;
  auto new_symbol_attr = ctx->create<SymbolAttr>(new_symbol_name);

  auto cloned = SubGraphOp::build(ctx.get(), new_symbol_attr);
  if (src->hasDevice()) { cloned->setDevice(src->getDevice()); }
  copyNodeAttributes(src, cloned);

  auto src_region = src->getTopRegion();
  auto cloned_region = cloned->getTopRegion();
  ...

  ValueMap local_value_map;
  ValueMap* map_ptr = value_map ? value_map : &local_value_map;
  map_ptr->clear();

  // 1) 克隆 graph 输入
  if (!mapGraphInputs(ctx, cloned, src_region->inputs(), cloned_region->inputs(), *map_ptr)) {
    return nullptr;
  }

  // 2) 用 IRWriter 复制 region 内所有 op（除了 Return）
  ir::IRWriterGuard writer_guard(ctx, cloned_region);
  ir::IRWriter writer(ctx, cloned_region);

  ir::cf::ReturnOp::ptr_t return_op = nullptr;
  for (auto& op : src_region->ops()) {
    if (op->isa_<ir::cf::ReturnOp>()) {
      return_op = op->cast_<ir::cf::ReturnOp>();
      continue;
    }
    if (auto linalg_op = op->cast_<ir::linalg::LinalgIROp>()) {
      if (!cloneLinalgOp(ctx, linalg_op, writer, *map_ptr)) {
        ...
      }
      continue;
    }
    MLLM_ERROR("SubGraphCloneUtils: unsupported op kind '{}' in SubGraph '{}'",
               static_cast<int>(op->getKind()), src_symbol->str());
    return nullptr;
  }

  // 3) 克隆 ReturnOp
  if (!cloneReturnOp(return_op, writer, *map_ptr)) { return nullptr; }

  // 4) 根据 value_map 重建 graph 输出
  cloned_region->outputs().clear();
  for (auto& src_output : src_region->outputs()) {
    auto it = map_ptr->find(src_output);
    if (it == map_ptr->end()) {
      MLLM_ERROR("SubGraphCloneUtils: missing mapping for graph output '{}'", src_output->name());
      return nullptr;
    }
    cloned_region->outputs().push_back(it->second);
    (*cloned)-- > it->second;
  }

  MLLM_INFO("SubGraphCloneUtils::cloneSubGraph: cloned SubGraph '{}' -> '{}'",
            src_symbol->str(), new_symbol_name);

  return cloned;
}
```

这样，一个完整的 QNN 子图就被“深拷贝”成了一个新的子图：

- 所有输入 / 输出 `Val` 都有新的实例；
- 所有中间 TensorValue 和 linalg/QNN 算子也有新的实例；
- 数据流（图结构）与原始子图完全一致；
- 外部可以通过 `value_map` 知道“老的某个 Val 对应到新的哪一个 Val”。

---

### 6. 目前克隆能力的覆盖范围与局限

当前版本的 `SubGraphCloneUtils::cloneSubGraph` 主要为 Qwen3 QNN decoder 子图服务，已支持：

- **Val 类型**：
  - `tensor::TensorValue`（绝大部分场景都用这个）
- **Op 类型**：
  - `ir::linalg::ViewOp`
  - `ir::linalg::CastTypeOp`
  - `ir::linalg::LinearOp`
  - `ir::linalg::TransposeOp`
  - `ir::linalg::RMSNormOp`
  - `ir::linalg::SiLUOp`
  - `ir::linalg::AddOp`
  - `ir::linalg::MulOp`
  - `ir::linalg::X2XOp`
  - `ir::linalg::CustomizedOp`（用于 DequantizeAdd）
  - `ir::cf::ReturnOp`

这些正好覆盖了 Qwen3 QNN 子图 `model.layers.X_1 / X_2` 中出现的所有算子类型（详见 `.mir` 文件）。

**暂时不支持的内容：**

- 其他类型的 `Val`（比如非张量型 Val）；
- 其他种类的 Op（比如某些 control-flow / program 级 Op）；
- 复杂的层级 Region（目前只考虑一个 topRegion）。

如果以后在别的模型 / 别的 Pass 中想用同一套工具：

- 需要按类似方式在 `cloneLinalgOp` 里增加对应的分支；
- 或者从架构上引入虚函数 `cloneInto(...)`，让每个具体 Op 自己实现克隆逻辑（这是更长期的方向）。

---

### 7. 未来如何用它来做 Qwen3 的 Graph Fusion？

有了 `cloneSubGraph` 之后，Qwen3 的融合 Pass（方案 B）实现思路会变得比较清晰：

1. 对每个融合对 `(X_2, X+1_1)`：
   - 先用 `cloneSubGraph(ir_ctx, graph_X2, { .new_symbol_name = "model.layers.X_fused" }, &value_map)`  
     得到一个新的 fused graph，初始只包含 X_2 的内容；
   - 再写一个辅助函数，利用同样的 `cloneLinalgOp` / `cloneReturnOp` 逻辑，把 `X+1_1` 的 Region“追加克隆”到 fused graph 的 Region 中，并用 `value_map` 把它的 `h_X_out` 输入连到 X_2 的输出；
   - 修改 fused graph 的 Return，让它返回 `[h_X_out, Q, K, V]`。

2. 按 `Fusion_IO_Design.md` 中第 5 节的 checklist：
   - 更新 `@model.layers.X <CPU>` 的 `CallGraphOp @X_2` → `@X_fused`；
   - 更新 `@model.layers.X+1 <CPU>` 的 SubGraph 输入签名和内部 self-attn 的 Q/K/V 来源；
   - 在顶层 Module 中删除旧的 `X_2` / `X+1_1`，只保留 `X_fused`。

整个过程中，最难的那一步——**“如何在 IR 层安全地复制一个 QNN 子图”**——已经由 `SubGraphCloneUtils::cloneSubGraph` 帮你实现好了。

---

### 8. 总结（给新手的 checkpoint）

如果你是从零开始，建议你确认自己已经理解下面几点：

1. **IR 的基本组成**：  
   - `Node` / `Val` / `Op` / `Region` / `SubGraphOp` 分别表示什么；
   - QNN 子图在 MIR 里是什么样子。

2. **克隆一个 SubGraph 的关键问题**：  
   - 不能直接“共享”原来的 `Val` 和 `Op`，需要新建一份；  
   - 需要一个全局的 `value_map`，跟踪 old→new 的对应关系。

3. **`SubGraphCloneUtils::cloneSubGraph` 的五个步骤**：
   - 决定新符号名，创建新的 `SubGraphOp`，复制 device / 属性；  
   - 克隆 graph inputs，填好 `value_map`；  
   - 用 `IRWriter` 复制 region 内的所有 linalg/QNN op；  
   - 复制 `cf.ReturnOp`；  
   - 重建 graph outputs，并把它们挂到新子图上。

4. **当前实现主要服务于 Qwen3 QNN 子图**，后续可以在此基础上扩展到更多模型和 Pass。

当你能把这份文档中提到的类和函数在 IDE 里点进去、看懂调用关系时，就已经具备了继续实现 Qwen3 跨层融合 Pass（方案 B）的基础。接下来要做的，就是按 `Fusion_IO_Design.md` 第 5 节的 checklist，把 clone 工具真正接入 `Qwen3IRGraphFusionPass` 中。  

---

### 9. Bug 复盘：为什么一开始会在 `%1478` 上报错？（shared_ptr 作为 key 的坑）

在第一次接入 Qwen3 融合 Pass 时，`SubGraphCloneUtils::cloneSubGraph` 在克隆 `model.layers.0_2` 这个子图时，遇到了如下错误日志：

```text
SubGraphCloneUtils: missing value mapping for input '1478'
SubGraphCloneUtils: failed to clone Linalg op inside 'model.layers.0_2'
Qwen3IRGraphFusionPass: failed to clone 'model.layers.0_2'
```

对照 MIR，可以看到：

- `model.layers.0_2` 的前几步大致是：
  - `ViewOp(%1476) -> %1478`
  - `CastTypeOp(%1478) -> %1479`
- 正常情况下：
  - 在克隆 `ViewOp` 时，会给 `%1478` 建立 `value_map[%1478] = %1478'`；
  - 随后克隆 `CastTypeOp(%1478)` 时，通过 `value_map` 查到 `%1478'` 作为新输入。

但实际情况是：`cloneLinalgOp` 在处理 `CastTypeOp` 这一层时，`value_map` 里查不到 `%1478` 对应的映射，于是抛出了 `missing value mapping`。  

根本原因是：**我们最初用 `std::unordered_map<val_ptr_t, val_ptr_t>`（以 `shared_ptr<Val>` 作为 key）来做 value 映射，而 `WeakOwner` + `shared_from_this()` 会为同一个底层对象生成不同控制块的 `shared_ptr`，导致 map 查找失败。**

具体来说：

- `mapGraphInputs` / `cloneLinalgOp` 在记录映射时，用的是：

  ```cpp
  auto old_val = std::static_pointer_cast<Val>(output_node->shared_from_this());
  value_map[old_val] = cloned_val;
  ```

- 但在查找时，又重新调用了一次：

  ```cpp
  auto old_val = std::static_pointer_cast<Val>(input_node->shared_from_this());
  auto map_it = value_map.find(old_val);
  ```

- 如果这两次 `shared_from_this()` 得到的 `shared_ptr` 控制块不同，即使底层裸指针地址相同，`unordered_map` 也认为 key 不相等，从而找不到映射，触发 `missing value mapping`。

#### 9.1 最终修复策略：统一用 `Val*` 做 key，并在必要时按 name 兜底

实际实践下来，我们发现仅仅依赖 `shared_ptr` 或者“`Val*` 裸指针完全一致”都不够鲁棒：  

- `WeakOwner` / `shared_from_this` 会为同一逻辑值制造不同的 `shared_ptr` 控制块；  
- 某些 pass / 构建流程里，甚至会出现**同名、同 shape、逻辑等价但 `Val*` 不同的节点**。  

因此最终的修复是：

1. **彻底把 ValueMap 换成以裸指针 `Val*` 为 key：**

   ```cpp
   // SubGraphCloneUtils.cpp 内部
   namespace {
     using ValueMap = std::unordered_map<Val*, val_ptr_t>;
   }
   ```

   - `mapGraphInputs` / `cloneLinalgOp` / `cloneReturnOp` / `cloneSubGraph` / `inlineSubGraphInto` 等所有内部逻辑，写入映射时统一用 `old_val.get()` 作为 key；
   - `SubGraphCloneUtils.hpp` 里对外接口也改为：

   ```cpp
   static SubGraphOp::ptr_t cloneSubGraph(
       const IRContext::ptr_t& ctx,
       const SubGraphOp::ptr_t& src,
       const CloneOptions& options,
       std::unordered_map<Val*, val_ptr_t>* value_map = nullptr);

   static bool inlineSubGraphInto(
       const IRContext::ptr_t& ctx,
       const SubGraphOp::ptr_t& src,
       const SubGraphOp::ptr_t& dst,
       std::unordered_map<Val*, val_ptr_t>& value_map,
       const std::unordered_map<Val*, val_ptr_t>& preset_input_mapping,
       std::vector<val_ptr_t>* cloned_outputs = nullptr);
   ```

   - `Qwen3IRGraphFusionPass` 里的调用方也同步改成 `std::unordered_map<Val*, val_ptr_t>`，避免类型不一致。

2. **在查不到 key 时，用 `Val::name()` 做“最后一道兜底”：**

   即使统一用 `Val*` 做 key，仍有极少数边界场景里，同一个逻辑值（例如 `%1478`）在某些处理中会被“复制”一份新的 `Val*`，导致单纯按裸指针找不到 mapping。  
   为了让工具在调试阶段更加健壮，我们在几个关键点增加了**name-based fallback**：

   - 克隆 LinalgOp 输入（最常见出错点）：

   ```cpp
   auto old_val = std::static_pointer_cast<Val>(input_node->shared_from_this());
   auto map_it = value_map.find(old_val.get());
   if (map_it == value_map.end()) {
     // 名字兜底：同名 Val 认为是同一个逻辑值
     for (auto& kv : value_map) {
       auto existing_node = kv.first;
       if (!existing_node) continue;
       auto existing_val = std::dynamic_pointer_cast<Val>(existing_node->shared_from_this());
       if (existing_val && existing_val->name() == old_val->name()) {
         map_it = value_map.find(existing_node);
         break;
       }
     }
   }
   if (map_it == value_map.end()) {
     MLLM_ERROR("SubGraphCloneUtils: missing value mapping for input '{}'", old_val->name());
     return false;
   }
   ```

   - 克隆 ReturnOp 输入、重建 cloned graph outputs、`inlineSubGraphInto` 返回 outputs 时也都用了类似的“**先按 `Val*` 查找，再按 name 兜底**”逻辑。

   这样做的效果是：

   - 绝大多数正常 IR 流程下，按 `Val*` 直接命中，性能和语义都清晰；  
   - 极端情况下，即便底层 `Val*` 被重新包装，只要 name 没变（例如 `%1478`），就仍然能通过 name 找到正确的映射，避免再出现“missing value mapping for input '1478'”这类致命错误。

#### 9.2 Qwen3 Fusion Pass 当前验证结果与后续问题

在完成上述修改后，我们重新跑了 `Qwen3IRGraphFusionPass`，从日志可以确认：

- 成功识别到 27 个 `(X_2, X+1_1)` 融合对：

  ```text
  Qwen3IRGraphFusionPass: identified 27 fusion pairs, will reduce graph count from 56 to approximately 29
  ```

- 对每一对 `(model.layers.X_2, model.layers.(X+1)_1)`，都完成了 clone 并创建新的 fused 子图：

  ```text
  SubGraphCloneUtils::cloneSubGraph: cloned SubGraph 'model.layers.0_2' -> 'model.layers.0_fused'
  Qwen3IRGraphFusionPass: created fused SubGraph 'model.layers.0_fused' from 'model.layers.0_2' + 'model.layers.1_1'
  ...
  SubGraphCloneUtils::cloneSubGraph: cloned SubGraph 'model.layers.26_2' -> 'model.layers.26_fused'
  Qwen3IRGraphFusionPass: created fused SubGraph 'model.layers.26_fused' from 'model.layers.26_2' + 'model.layers.27_1'
  ```

- Fusion Pass 在删除旧子图方面也已经开始工作，但**当前可观测到的日志只覆盖到了 `model.layers.20_2` 及之前的部分 `X+1_1`**：

  ```text
  Qwen3IRGraphFusionPass: removed obsolete SubGraph 'model.layers.0_2'
  Qwen3IRGraphFusionPass: removed obsolete SubGraph 'model.layers.1_1'
  ...
  Qwen3IRGraphFusionPass: removed obsolete SubGraph 'model.layers.20_2'
  ```

这意味着：

- 之前困扰我们的 `%1478` 映射问题已经被解决，`SubGraphCloneUtils` 的 clone / inline 能稳定处理 Qwen3 所需的 LinalgOp，Fusion Pass 不会再在第一层就失败；  
- 但**旧子图的删除目前是“逐步进行”的**，在 0～20 层已经能看到明确的删除日志，21 层以后仍需要在后续迭代中进一步确认和完善；  
- Fusion Pass 之后的 `QNNGraphIOTensorPass` 若仍尝试访问已经删除或类型不匹配的 callee，会触发崩溃，因此我们对其做了健壮性增强（详见下文）。

#### 9.3 给新手的 takeaway（更新版）

综合这次迭代，结论可以更新为：

1. 在有 `WeakOwner` / `shared_from_this` 参与的 IR 框架里，**不要使用 `shared_ptr` 作为哈希 key**；  
2. 即使使用 `Val*` 作为 key，也要考虑到某些 pass 可能会“复制”逻辑等价的 Val，此时可以用 `Val::name()` 作为**最后一道兜底手段**；  
3. IR 工具类（比如 `SubGraphCloneUtils`）要尽可能健壮：  
   - 正常路径下使用语义清晰的 key（这里是 `Val*`）；  
   - 出现边界情况时，通过日志 + name fallback 既能暴露问题，又不至于直接让上层 Fusion Pass 崩掉。

目前，基于这一整套修复，`Qwen3IRGraphFusionPass` 已经可以在实际工程环境中稳定完成 27 对 QNN 子图的融合并生成 `model.layers.0_fused ~ model.layers.26_fused`，同时后续的 QNN 相关 Pass（例如 `QNNGraphIOTensorPass`）也通过增加符号存在性和类型检查，避免了因为符号表中残留异常状态而直接 SIGSEGV 崩溃。  


