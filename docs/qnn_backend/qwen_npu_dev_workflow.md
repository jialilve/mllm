# QNN NPU 开发工作流（自然语言伪代码驱动）

## 1. 文档目的
- 给正在调研/实现 QNN NPU 功能的同学一份统一的工作流，确保**先把逻辑说清楚**再动手写代码。
- 这份文档也可以直接作为与 AI/同事沟通的 prompt，帮助在实现前快速校对思路。

## 2. 工作流总览
| 阶段 | 目标 | 主要产出 | 常用工具 |
| --- | --- | --- | --- |
| A. 需求确认 | 聚焦要解决的问题、约束 | 纯自然语言描述（Layer 1） | README、需求文档、师兄同步 |
| B. 方案抽象 | 用项目术语描述数据流/状态机 | 技术中文伪代码（Layer 2） | docs、代码注释、AI 协助 |
| C. 结构校验 | 对照模块、函数名排查缺漏 | 带函数/模块名的伪代码（Layer 3） | IDE、call graph、Anki 记录 |
| D. 代码生成与调试 | 将伪代码翻译成实现 | 代码补丁、日志、测试记录 | Cursor/AI + 本地编译运行 |
| E. 复盘与对齐 | 与师兄确认、写总结 | 对齐邮件/PR 描述、改进清单 | Markdown、Issue tracker |

## 3. 三层自然语言伪代码方法

### Layer 1：纯自然语言（业务层）
- 面向需求评审或师兄确认，不出现函数名。
- 示例：  
  ```
  1. 维持两个“工位”：NPU 负责当前 chunk 推理，CPU 负责准备下一个 chunk。
  2. NPU 正在算 chunk N 时，CPU 把 chunk N+1 的 embedding、position_ids、KV 对齐信息准备好。
  3. NPU 输出完毕后立刻切换到下一个 chunk，整个流程像流水线一样连续推进。
  ```

### Layer 2：技术中文（含项目黑话）
- 引入关键组件/接口名，用来指导 AI 生成骨架。
- 示例提示词：  
  ```
  - 初始化 tokenizer、Qwen3ForCausalLM、chunk buffer。
  - 每轮循环：主线程调用 model.forward(chunk_i)，同时后台 std::async 准备 chunk_{i+1} 的输入 tensor。
  - forward 返回后，立即拿异步结果替换当前 chunk，更新 KV cache 序列长度并进入下一轮。
  ```

### Layer 3：带函数名的伪代码
- 将 Layer 2 映射到真实模块/函数，方便逐行实现。
- 示例：
  ```
  auto cfg = Qwen3NPUConfig(config_path);
  auto model = Qwen3ForCausalLM("", cfg);
  Tensor current = buildChunk(raw_tokens, chunk_idx);
  for (...) {
      auto future_next = std::async([&]{ return buildChunk(raw_tokens, chunk_idx + 1); });
      auto logits = model.forward(current_inputs, {...})["sequence"];
      current = future_next.get();
  }
  ```

## 4. 借助 AI 的具体动作
1. **先用 Layer 1 描述需求** → 发给 AI/同事确认是否理解正确。
2. **把确认过的 Layer 1 + 约束（chunk_size、KV 规则等）给 AI**，让它输出 Layer 2。
3. **审核 AI 产出的 Layer 2**：看术语、步骤是否吻合；必要时补充日志/状态机。
4. **要求 AI 根据 Layer 2 生成 Layer 3 或直接生成代码**；若不满意就指出问题继续 refine。
5. **调试阶段**：把“当前日志、问题”写成自然语言，再请 AI 给定定位思路或可能的 bug 点。

## 5. 与师兄/代码评审的沟通模板
1. **需求确认**  
   ```
   师兄，我的理解是需要实现“两 chunk pipeline”：NPU 算 chunk n，CPU 并行准备 chunk n+1。
   流程如下（Layer 1 摘要）。请帮忙确认是否缺少关键步骤？
   ```
2. **方案确认**  
   ```
   我已经把流程扩展成 Layer 2（附上文档链接）。如果没问题我就按这个结构写代码。
   ```
3. **PR 描述**  
   - 引用文档中的层级描述，让 reviewer 快速对齐期望。

## 6. 建议的日常 checklist
- [ ] 是否先阅读相关 doc（core_design、requirements 等）？  
- [ ] 是否写出 Layer 1，并获得导师/队友确认？  
- [ ] 是否扩展到 Layer 2，并检查和项目术语一致？  
- [ ] 是否在实现前写好 Layer 3/伪代码？  
- [ ] 是否记录了日志点和验证案例？  
- [ ] 是否在 PR 中附上对应文档链接，方便 reviewer？

> **结论**：把“自然语言伪代码”当作强制步骤，既能避免反复返工，也能让 AI/同事快速理解你在做什么。


