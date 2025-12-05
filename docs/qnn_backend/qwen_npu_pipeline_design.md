# QNN Qwen NPU Pipeline（多 Chunk 并行）调研稿

> 本文遵循《qwen_npu_dev_workflow.md》的三层伪代码方法，先讲清楚 **Pipeline 在做什么**，再逐步下钻到工程细节，方便与师兄对齐或作为后续实现的 prompt。

## 1. 背景与目标（Layer 1：纯自然语言）
1. 目前 `examples/qwen_npu/main.cpp` 虽然支持 chunked prefill + decode，但流程是「算完一个 chunk 再准备下一个」，QNN 周期内 CPU 会闲置。
2. 我们希望把流程改成**流水线**：NPU 正在处理 chunk N 时，CPU 同时预处理 chunk N+1（tokenize、padding、position_ids、KV 对齐等）。
3. 当 NPU 输出完 chunk N 的 logits，下一份 chunk 输入已经准备完成，可以立即发起下一次 `forward`，以此降低整体 latency。
4. 初期目标：实现 **双 chunk pipeline**（Pipeline depth=2），未来可以扩展到更多 chunk 或更复杂的调度。

**⚠️ 注意**：本文档描述的是"chunk 级别的 pipeline"（数据准备和计算的重叠），与 `modeling_pipeline_trace_simplified.hpp` 中的"stage 级别的 pipeline"（QNN/CPU stage 并行）不同。两者可以结合使用。详见 `qnn_pipeline_v2_detailed_explanation.md` 第 8 节。

## 2. 场景约束与术语对齐
- **chunk_size**：固定 128（见 `qwen_npu_decoding_requirements.md`）。
- **Prefill/Decode**：维持「prefill 填满 chunk buffer，decode 在 padding 区写入新 token」的现有策略。
- **KV Cache**：通过 `model.setKVCacheSeqCnt(offset)` 控制写入位置，任何 pipeline 方案都必须保证绝对序号一致。
- **异步执行工具**：可优先考虑 `std::future/std::async`，方便与 C++17 标准兼容；若后续需要更细粒度控制，再引入线程池/自研调度。

## 3. Pipeline 工作流（Layer 2：技术中文）

### 3.1 状态流转
1. **初始化**  
   - 加载 tokenizer、`Qwen3ForCausalLM`、配置 chunk buffer。  
   - 生成首个 chunk（chunk0）并完成 prefill。
2. **进入流水线循环**（从 chunk0 的 decode 阶段开始）。每一轮都包含两类任务：
   - **计算任务**（NPU 主线程）  
     `model.forward(current_chunk_inputs)` → 更新 logits/kv cache → 采样 token。  
   - **准备任务**（CPU 后台线程）  
     读取原始 prompt/已生成 token，构建 chunk_{i+1} 所需的 `sequence tensor`、`position_ids`、`seq_len` 元数据。
3. **阶段同步**  
   - 计算任务启动后立即投递下一个准备任务（如果还有 chunk 待处理）。  
   - 计算完成时，等待/获取准备任务的结果，并把 `current_chunk` 指针切换到最新输入。  
   - 如果准备任务尚未完成，主线程会在 `future.get()` 处短暂阻塞，确保顺序正确但尽量减少空闲。
4. **结束条件**  
   - 所有 prompt chunk 都处理完，并且 decode 结束（达到 chunk_size 或生成 EOS）。  
   - 队列为空时回收后台资源、输出结果。

### 3.2 自然语言伪代码（Layer 2.5）
```
输入：raw_tokens（长度 L），chunk_size=128
输出：生成的文本/日志

准备：
  current_chunk = buildChunk(raw_tokens, chunk_index=0)
  next_chunk_job = schedule(buildChunk(raw_tokens, 1))  // 如果还有

循环（chunk_index 从 0 到 prompt_chunks-1）：
  1. model.setKVCacheSeqCnt(chunk_abs_start)
  2. logits = model.forward(current_chunk.inputs, {"seq_len": current_chunk.real_len})
  3. 如果还有后续 chunk：
       wait(next_chunk_job) → next_chunk
       chunk_index += 1
       current_chunk = next_chunk
       next_chunk_job = schedule(buildChunk(raw_tokens, chunk_index+1))
  4. 如果是最后一个 chunk，则进入 decode 阶段：
       while chunk 未满 且 未出现 EOS:
           异步准备：预取“这次 decode 结束后要用的下一段输入/日志” (可选)
           logits = model.forward(current_chunk.buffer, {"seq_len": current_len})
           token = sampleGreedy(logits)
           写入 buffer、更新 position_ids、setKVCacheSeqCnt(...)
```

## 4. Layer 3：模块映射与伪代码

### 4.1 关键结构
```cpp
struct ChunkInputs {
  Tensor sequence;       // [1, chunk_size], padding=-1
  Tensor position_ids;   // [1, chunk_size], 绝对坐标
  int real_len;          // 本 chunk 中真实 token 数
  int chunk_start;       // 在整段 prompt 中的起始 offset
};

ChunkInputs buildChunk(const Tensor& prompt_tokens, int chunk_idx);
std::future<ChunkInputs> scheduleNextChunk(int next_idx);
```

### 4.2 主流程草图
```cpp
auto prompt_tokens = tokenizer(...);                   // CPU
ChunkInputs current = buildChunk(prompt_tokens, 0);    // Chunk 0
std::future<ChunkInputs> next_job;
if (prompt_chunks > 1) {
    next_job = std::async(std::launch::async, buildChunk, prompt_tokens, 1);
}

for (int chunk_idx = 0; chunk_idx < prompt_chunks; ++chunk_idx) {
    model.setKVCacheSeqCnt(current.chunk_start);
    auto logits = model.forward(
        {{"sequence", current.sequence}, {"position_ids", current.position_ids}},
        {{"seq_len", AnyValue(current.real_len)}});

    if (chunk_idx + 1 < prompt_chunks) {
        ChunkInputs next_chunk = next_job.get();
        current = next_chunk;
        if (chunk_idx + 2 < prompt_chunks) {
            next_job = std::async(std::launch::async, buildChunk, prompt_tokens, chunk_idx + 2);
        }
        continue;
    }

    // 进入 decode
    runDecodePipeline(model, current, tokenizer, eos_id);
}
```

### 4.3 解码阶段的流水线思路
- Prefill 阶段已经把最后一个 chunk 填满实时 prompt；decode 阶段改成“单 token + 异步准备日志”的 pipeline。
- 可选优化：把“日志写入/结果 detokenize”放到后台线程，主线程只关注 forward + 采样。

## 5. 后续待确认的问题
1. **Tokenizer/Chunk builder 是否线程安全？**  
   - 如果不是，需要在 `buildChunk` 内部加锁，或改用线程池中的串行任务。
2. **QNN Context 是否允许多个线程同时调用？**  
   - 当前设计只在主线程调用 `model.forward`，后台线程不触碰 QNN，应当安全；仍需在文档中注明。
3. **是否需要更细粒度 pipeline（prefill 的子阶段也重叠）？**  
   - 目前只做 chunk 级别；如果性能仍不满足，可以拆到“embedding → attention → MLP” 层面，需要更多 backend 支持。
4. **日志与调试 hook**  
   - 对齐 `qwen_npu_decoding_requirements.md` 中的日志要求，在 pipeline 中标注 chunk_id、任务状态、KV seq。

## 6. 下一步计划
1. 拿这份文档与师兄确认：  
   - 是否允许使用 `std::async`；  
   - chunk builder/Tokenizer 的线程安全假设是否成立；  
   - 是否需要把 pipeline 控制器抽象成单独类。
2. 根据反馈更新本文，随后才能进入具体实现。
3. 实现阶段再根据 Layer 3 伪代码拆分任务：新建 `ChunkBuilder`、`PipelineExecutor`、日志模块等。

> **提醒**：任何代码实现都必须附上对应的 Layer 2/Layer 3 文档链接，方便 reviewer 快速对齐思路。


