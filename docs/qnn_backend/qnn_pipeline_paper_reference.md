# QNN Pipeline 最终目标（基于《Fast On-device LLM Inference with NPUs》）

> 论文：https://arxiv.org/abs/2407.05858  
> 本文仅提炼与我们项目相关的 pipeline 思路，作为“远期目标”的对齐参考，方便在后续设计中逐步靠近。

## 1. 论文中的核心概念
1. **Chunked Prefill/Decode**  
   - 长序列被拆成固定长度的 chunk，每个 chunk 仍包含完整的注意力、MLP 等子图。  
   - Prefill chunk 之间通过 KV Cache 拼接；Decode chunk（长度 1）复用相同图。
2. **子图划分（Subgraph Partitioning）**  
   - 每个 chunk 会衍生出多个 subgraph：  
     - **CPU subgraph**：tokenizer、embedding、position_ids、KVCache 更新、softmax 等需要较灵活控制流的部分。  
     - **NPU subgraph**：密集矩阵乘、卷积、量化/反量化、RMSNorm 等。  
   - 子图边界由数据依赖和硬件支持矩阵共同决定，论文示例中一个 chunk 至少包含 2~3 个 NPU subgraph + 1 个 CPU subgraph。
3. **流水线调度（Pipeline Scheduling）**  
   - 针对每个 chunk 的子图构建有向无环图（DAG），然后按照 stage 划分：  
     - Stage A：CPU 预处理  
     - Stage B：NPU 主干图（可包含多个 slice）  
     - Stage C：CPU 后处理（logits、采样）  
   - 多个 chunk 的 stage 交错执行，确保 NPU 忙碌时 CPU 在准备下一阶段。  
   - 论文给出的策略类似“分层次拓扑排序”：先看 stage 依赖，再在同 stage 内做 round-robin。
4. **多 QNN 图上下文**  
   - 不同 chunk size / stage 会对应不同的 QNN context file（如 prefill_128, prefill_256, decode_1）。  
   - Pipeline 调度器需要根据 chunk 的实时长度在多个 context 之间切换。

## 2. 论文式 Pipeline 的自然语言伪代码
```
for chunk in chunks:
    // Stage A (CPU)
    preprocess_future = CPU_pool.submit(prepare_chunk, chunk)

    // Stage B (NPU)
    wait(preprocess_future)
    npu_future = QNN_pool.submit(run_qnn_subgraphs, chunk)

    // Stage C (CPU)
    cpu_tail_future = CPU_pool.submit(postprocess, npu_future.result)

    // Pipeline：把上述 Future 放入时间轴
    scheduler.link(chunk, [preprocess_future, npu_future, cpu_tail_future])

scheduler.run_pipeline(policy="stage-overlap")
```

## 3. 对我们项目的启示
| 论文概念 | 对应落地方向 |
| --- | --- |
| Chunk + Subgraph | `Tracer::model_` 在 v1 中已经按图切分，可继续细化到 CPU/NPU 子图。 |
| Stage Overlap | 结合 `mllm::async::fork` 或线程池，将 Stage A/B/C 分别投递。 |
| 多 Context 管理 | `ContextManager`/`PipelineExecutor` 需支持不同 seq_len 的 QNN 图。 |
| CPU/NPU 工作窄化 | NPU 负责纯算子，CPU 负责控制流与 KV；两端通过 DMA/共享内存传递张量。 |

## 4. 下一步建议
1. **定义子图边界**：参考论文附录的图示，先把现有 `Tracer::model_` 中的 QNN graph 和 CPU graph 分类，记录输入输出。  
2. **设计 Stage Scheduler**：可先实现两级调度（CPU 预处理 + NPU 计算），后续再补 CPU 后处理。  
3. **Context 复用计划**：为常见长度（128、256、512、decode=1）预编译 QNN context，并在 runtime 中按需切换。  
4. **指标**：对齐论文中的三类指标——NPU Utilization、端到端 Latency、能耗（如可测）。

> 这份文档定位为“愿景/目标”说明，短期内不需要一次到位，但可以用来校验我们实现的方向是否与论文一致。


