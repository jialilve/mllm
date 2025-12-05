#pragma once

#include "mllm/compile/ir/Trace.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/llama/configuration_llama.hpp"
#include "mllm/utils/Log.hpp"
#include <memory>
#include <unordered_map>
#include <future>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace mllm::models {

// 执行阶段枚举
enum class ExecutionStage {
  PREFILL,
  DECODE
};

// QNN上下文信息（简化版）
struct QNNContextInfo {
  std::string context_file_path;
  std::string graph_name;
  ExecutionStage stage;
  int sequence_length;
  void* qnn_model_ptr = nullptr;  // 避免直接依赖QNN类型
};

// 流水线任务结构
struct PipelineTask {
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  ExecutionStage stage;
  int chunk_id;
  std::promise<std::vector<Tensor>> promise;
};

class ContextManager {
 private:
  std::unordered_map<std::string, QNNContextInfo> qnn_contexts_;
  Backend* qnn_backend_;
  
 public:
  ContextManager(Backend* backend) : qnn_backend_(backend) {}
  
  // 注册不同长度的QNN上下文
  bool registerContext(const std::string& context_key, const std::string& context_file, 
                      ExecutionStage stage, int seq_length) {
    QNNContextInfo info;
    info.context_file_path = context_file;
    info.stage = stage;
    info.sequence_length = seq_length;
    info.graph_name = generateGraphName(stage, seq_length);
    
    // 这里可以调用QNNBackend的retrieveContext来加载预编译图
    qnn_contexts_[context_key] = info;
    MLLM_INFO("Registered QNN context: {} for stage: {}, seq_length: {}", 
              context_key, static_cast<int>(stage), seq_length);
    return true;
  }
  
  // 根据阶段和序列长度获取最合适的QNN上下文
  QNNContextInfo* getContext(ExecutionStage stage, int seq_length) {
    std::string key = generateContextKey(stage, seq_length);
    auto it = qnn_contexts_.find(key);
    if (it != qnn_contexts_.end()) {
      return &it->second;
    }
    
    // 如果没有精确匹配，寻找最接近的上下文
    QNNContextInfo* best_match = nullptr;
    int min_diff = INT_MAX;
    
    for (auto& [ctx_key, ctx_info] : qnn_contexts_) {
      if (ctx_info.stage == stage && ctx_info.sequence_length >= seq_length) {
        int diff = ctx_info.sequence_length - seq_length;
        if (diff < min_diff) {
          min_diff = diff;
          best_match = &ctx_info;
        }
      }
    }
    
    return best_match;
  }
  
  // 公共接口供外部使用
  std::string generateContextKey(ExecutionStage stage, int seq_length) {
    return std::string(stage == ExecutionStage::PREFILL ? "prefill_" : "decode_") + 
           std::to_string(seq_length);
  }
  
 private:
  std::string generateGraphName(ExecutionStage stage, int seq_length) {
    return std::string(stage == ExecutionStage::PREFILL ? "prefill_graph_" : "decode_graph_") + 
           std::to_string(seq_length);
  }
};

class PipelineExecutor {
 private:
  std::unique_ptr<ContextManager> context_manager_;
  std::queue<PipelineTask> cpu_task_queue_;
  std::queue<PipelineTask> qnn_task_queue_;
  std::thread cpu_worker_;
  std::thread qnn_worker_;
  std::mutex cpu_queue_mutex_;
  std::mutex qnn_queue_mutex_;
  std::condition_variable cpu_cv_;
  std::condition_variable qnn_cv_;
  bool shutdown_ = false;
  
  // CPU后端和QNN后端
  Backend* cpu_backend_;
  Backend* qnn_backend_;
  
 public:
  PipelineExecutor(Backend* cpu_backend, Backend* qnn_backend) 
    : cpu_backend_(cpu_backend), qnn_backend_(qnn_backend) {
    context_manager_ = std::make_unique<ContextManager>(qnn_backend);
    startWorkers();
  }
  
  ~PipelineExecutor() {
    shutdown_ = true;
    cpu_cv_.notify_all();
    qnn_cv_.notify_all();
    if (cpu_worker_.joinable()) cpu_worker_.join();
    if (qnn_worker_.joinable()) qnn_worker_.join();
  }
  
  // 注册预编译的QNN上下文
  bool registerQNNContext(const std::string& context_file, ExecutionStage stage, int seq_length) {
    return context_manager_->registerContext(
      context_manager_->generateContextKey(stage, seq_length), 
      context_file, stage, seq_length);
  }
  
  // 异步执行任务
  std::future<std::vector<Tensor>> submitTask(const std::vector<Tensor>& inputs, 
                                             ExecutionStage stage, 
                                             DeviceTypes target_device, 
                                             int chunk_id = 0) {
    PipelineTask task;
    task.inputs = inputs;
    task.stage = stage;
    task.chunk_id = chunk_id;
    
    auto future = task.promise.get_future();
    
    if (target_device == MLLM_QNN) {
      std::lock_guard<std::mutex> lock(qnn_queue_mutex_);
      qnn_task_queue_.push(std::move(task));
      qnn_cv_.notify_one();
    } else {
      std::lock_guard<std::mutex> lock(cpu_queue_mutex_);
      cpu_task_queue_.push(std::move(task));
      cpu_cv_.notify_one();
    }
    
    return future;
  }
  
 private:
  void startWorkers() {
    cpu_worker_ = std::thread([this]() { cpuWorker(); });
    qnn_worker_ = std::thread([this]() { qnnWorker(); });
  }
  
  void cpuWorker() {
    while (!shutdown_) {
      std::unique_lock<std::mutex> lock(cpu_queue_mutex_);
      cpu_cv_.wait(lock, [this]() { return !cpu_task_queue_.empty() || shutdown_; });
      
      if (shutdown_) break;
      
      auto task = std::move(cpu_task_queue_.front());
      cpu_task_queue_.pop();
      lock.unlock();
      
      // 执行CPU任务
      try {
        auto outputs = executeCPUTask(task);
        task.promise.set_value(outputs);
      } catch (const std::exception& e) {
        task.promise.set_exception(std::current_exception());
      }
    }
  }
  
  void qnnWorker() {
    while (!shutdown_) {
      std::unique_lock<std::mutex> lock(qnn_queue_mutex_);
      qnn_cv_.wait(lock, [this]() { return !qnn_task_queue_.empty() || shutdown_; });
      
      if (shutdown_) break;
      
      auto task = std::move(qnn_task_queue_.front());
      qnn_task_queue_.pop();
      lock.unlock();
      
      // 执行QNN任务
      try {
        auto outputs = executeQNNTask(task);
        task.promise.set_value(outputs);
      } catch (const std::exception& e) {
        task.promise.set_exception(std::current_exception());
      }
    }
  }
  
  std::vector<Tensor> executeCPUTask(const PipelineTask& task) {
    // CPU任务执行逻辑
    std::vector<Tensor> outputs;
    MLLM_INFO("Executing CPU task for chunk {}", task.chunk_id);
    
    // 这里添加实际的CPU执行逻辑
    // 可以调用具体的CPU模块进行推理
    
    return outputs;
  }
  
  std::vector<Tensor> executeQNNTask(const PipelineTask& task) {
    // 获取合适的QNN上下文
    int seq_length = task.inputs.empty() ? 0 : task.inputs[0].sequence();
    auto* context_info = context_manager_->getContext(task.stage, seq_length);
    
    if (!context_info) {
      throw std::runtime_error("No suitable QNN context found for stage: " + 
                              std::to_string(static_cast<int>(task.stage)) + 
                              ", seq_length: " + std::to_string(seq_length));
    }
    
    MLLM_INFO("Executing QNN task using context: {} for chunk {}", 
              context_info->graph_name, task.chunk_id);
    
    // 使用选定的QNN模型执行推理
    std::vector<Tensor> outputs;
    
    // 这里需要调用实际的QNN执行逻辑
    // 例如：qnn_backend_->graphExecute(context_info->graph_name, inputs_copy, outputs);
    
    return outputs;
  }
};

class HybridLlamaForCausalLM : public nn::Module, public ARGeneration {
 private:
  std::unique_ptr<PipelineExecutor> pipeline_executor_;
  Backend* cpu_backend_;
  Backend* qnn_backend_;
  
  // 模块组件
  nn::Linear lm_head_;
  
  // 配置
  int chunk_size_ = 128;
  bool enable_pipeline_ = true;
  
 public:
  HybridLlamaForCausalLM() = default;
  
  explicit HybridLlamaForCausalLM(const std::string& name, 
                                 Backend* cpu_backend,
                                 Backend* qnn_backend,
                                 int chunk_size = 128) 
    : nn::Module(name), cpu_backend_(cpu_backend), qnn_backend_(qnn_backend), chunk_size_(chunk_size) {
    
    // 初始化模块
    lm_head_ = reg<nn::Linear>("lm_head", 1024, 1024, false, aops::LinearImplTypes::kDefault);
    
    // 创建流水线执行器
    pipeline_executor_ = std::make_unique<PipelineExecutor>(cpu_backend, qnn_backend);
    
    // 注册不同长度的QNN上下文
    registerQNNContexts();
  }
  
  void registerQNNContexts() {
    // 注册不同长度的prefill和decode上下文
    std::vector<int> prefill_lengths = {128, 256, 512, 1024, 2048};
    std::vector<int> decode_lengths = {1, 8, 16, 32};
    
    for (int len : prefill_lengths) {
      std::string context_file = "qnn_prefill_" + std::to_string(len) + ".bin";
      pipeline_executor_->registerQNNContext(context_file, ExecutionStage::PREFILL, len);
    }
    
    for (int len : decode_lengths) {
      std::string context_file = "qnn_decode_" + std::to_string(len) + ".bin";
      pipeline_executor_->registerQNNContext(context_file, ExecutionStage::DECODE, len);
    }
  }
  
  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");
    int seq_length = sequence.sequence();
    
    // 根据序列长度决定执行策略
    ExecutionStage stage = (seq_length > chunk_size_) ? ExecutionStage::PREFILL : ExecutionStage::DECODE;
    
    if (enable_pipeline_ && stage == ExecutionStage::PREFILL) {
      return forwardWithPipeline(input, args);
    } else {
      return forwardDirect(input, args);
    }
  }
  
  ARGenerationOutputPast forwardWithPipeline(const ARGenerationOutputPast& input, const ARGenerationArgs& args) {
    auto sequence = input.at("sequence");
    int seq_length = sequence.sequence();
    int chunk_num = (seq_length + chunk_size_ - 1) / chunk_size_;
    
    std::vector<std::future<std::vector<Tensor>>> futures;
    
    // 将输入分块并提交到流水线
    for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
      // 创建分块tensor
      auto chunk_input = createChunkTensor(sequence, chunk_id);
      
      // 确定设备类型（这里可以根据策略决定）
      DeviceTypes target_device = (chunk_id % 2 == 0) ? MLLM_QNN : MLLM_CPU;
      
      // 提交异步任务
      auto future = pipeline_executor_->submitTask(
        {chunk_input}, ExecutionStage::PREFILL, target_device, chunk_id);
      futures.push_back(std::move(future));
    }
    
    // 收集结果
    std::vector<Tensor> chunk_outputs;
    for (auto& future : futures) {
      auto outputs = future.get();
      chunk_outputs.insert(chunk_outputs.end(), outputs.begin(), outputs.end());
    }
    
    // 合并分块结果
    auto merged_output = mergeChunkOutputs(chunk_outputs);
    
    return {{"logits", merged_output}};
  }
  
  ARGenerationOutputPast forwardDirect(const ARGenerationOutputPast& input, const ARGenerationArgs& args) {
    auto sequence = input.at("sequence");
    
    // 直接使用QNN执行
    auto future = pipeline_executor_->submitTask(
      {sequence}, ExecutionStage::DECODE, MLLM_QNN, 0);
    
    auto outputs = future.get();
    
    // 创建默认输出tensor
    Tensor default_output(1, 1, 1, 1024, MLLM_CPU, true);
    return {{"logits", outputs.empty() ? default_output : outputs[0]}};
  }
  
  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    ir::IRContext::ptr_t llm_ir = nullptr;
    auto sequence = input.at("sequence");
    
    ir::lowlevel::traceStart();
    
    // 根据序列长度选择不同的trace策略
    int seq_length = sequence.sequence();
    if (seq_length > chunk_size_) {
      // 分块trace
      traceWithChunks(sequence);
    } else {
      // 直接trace
      traceDirect(sequence);
    }
    
    ir::lowlevel::traceComment("Hybrid CPU-QNN Pipeline Execution");
    llm_ir = ir::lowlevel::traceStop();
    
    return {{"model", llm_ir}};
  }
  
 private:
  Tensor createChunkTensor(const Tensor& input, int chunk_id) {
    int chunk_start = chunk_id * chunk_size_;
    int chunk_end = std::min(chunk_start + chunk_size_, input.sequence());
    int actual_chunk_size = chunk_end - chunk_start;
    
    // 创建新的chunk tensor
    Tensor chunk(input.batch(), input.head(), actual_chunk_size, input.dimension(), MLLM_CPU, true);
    
    // 这里需要实现实际的数据拷贝逻辑
    // chunk.shallowCopyFrom(&input, false, {0, 0, chunk_start, 0});
    
    return chunk;
  }
  
  Tensor mergeChunkOutputs(const std::vector<Tensor>& chunk_outputs) {
    if (chunk_outputs.empty()) {
      return Tensor(1, 1, 1, 1024, MLLM_CPU, true);
    }
    
    // 计算总的输出尺寸
    int total_seq_length = 0;
    for (const auto& chunk : chunk_outputs) {
      total_seq_length += chunk.sequence();
    }
    
    Tensor merged(chunk_outputs[0].batch(), 
                  chunk_outputs[0].head(), 
                  total_seq_length, 
                  chunk_outputs[0].dimension(),
                  MLLM_CPU, true);
    
    // 这里需要实现实际的数据合并逻辑
    int offset = 0;
    for (const auto& chunk : chunk_outputs) {
      // merged.copyFrom(chunk, {0, 0, offset, 0});
      offset += chunk.sequence();
    }
    
    return merged;
  }
  
  void traceWithChunks(const Tensor& sequence) {
    int seq_length = sequence.sequence();
    int chunk_num = (seq_length + chunk_size_ - 1) / chunk_size_;
    
    for (int chunk_id = 0; chunk_id < chunk_num; ++chunk_id) {
      auto chunk_input = createChunkTensor(sequence, chunk_id);
      
      // 模拟不同设备的执行
      if (chunk_id % 2 == 0) {
        // QNN执行
        ir::lowlevel::traceComment("QNN Chunk " + std::to_string(chunk_id) + " execution");
        lm_head_(chunk_input);
      } else {
        // CPU执行
        ir::lowlevel::traceComment("CPU Chunk " + std::to_string(chunk_id) + " execution");
        lm_head_(chunk_input);
      }
    }
  }
  
  void traceDirect(const Tensor& sequence) {
    ir::lowlevel::traceComment("Direct execution");
    lm_head_(sequence);
  }
};

}  // namespace mllm::models
