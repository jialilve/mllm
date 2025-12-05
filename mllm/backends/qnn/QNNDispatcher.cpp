// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/nn/Module.hpp"

#ifdef MLLM_PERFETTO_ENABLE
#include "mllm/engine/Perf.hpp"
#endif

namespace mllm::qnn {

QNNDispatcher::QNNDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id, const QNNDispatcherOptions& options)
    : Dispatcher(thread_pool, id), options_(options) {}

void QNNDispatcher::receive(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteOp: {
      process(task);
      break;
    }
    case TaskTypes::kExecuteModule: {
      process(task);
      break;
    }
    default: NYI("Only execute op task is supported receive");
  }
}

TaskResult::sender_t QNNDispatcher::asyncReceive(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteModule: {
      MLLM_EMPTY_SCOPE;
      break;
    }
    default: NYI("Only execute module task is supported asyncReceive");
  }
  auto scheduler = thread_pool_.get_scheduler();
  return stdexec::schedule(scheduler) | stdexec::then([this, task] { process(task); });
}

void QNNDispatcher::process(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteOp: {
      // the reshape should be called to init op output tensors
      task->op->reshape(task->inputs, task->outputs);
      // only X2X op is executed in QNN dispatcher
      if (task->op->getOpType() == OpTypes::kX2X || task->op->getOpType() == OpTypes::kEmbedding) {
        task->op->setup(task->inputs, task->outputs);
        task->op->forward(task->inputs, task->outputs);
      }
      break;
    }
    case TaskTypes::kExecuteModule: {
      auto moduleName = static_cast<nn::Module*>(task->custom_context_ptr)->getModuleName();
      MLLM_INFO("QNNDispatcher: executing QNN module '{}', inputs={}, initial outputs={}", moduleName,
                task->inputs.size(), task->outputs.size());
#ifdef MLLM_PERFETTO_ENABLE
      MLLM_PERF_TRACE_EVENT("mllm.qnn.execute.", perfetto::DynamicString{moduleName}, [&](perfetto::EventContext ctx) {
        int cnt = 0;
        for (auto& i : task->inputs) {
          ctx.AddDebugAnnotation(perfetto::DynamicString{"inputs-" + std::to_string(cnt++)}, i.shape());
        }
      });
#endif
      // here enters in a QNN module, execute it and not dive into its layers
      auto qnnBackend = std::static_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));

      // Module::forward typically returns empty vector for QNN modules
      // graphExecute will populate outputs based on QNN graph definition
      task->outputs = ((nn::Module*)(task->custom_context_ptr))->forward(task->inputs, task->args);
      
      MLLM_INFO("QNNDispatcher: after forward, outputs={}", task->outputs.size());
      
      // graphExecute will resize and populate outputs based on QNN graph outputs
      qnnBackend->graphExecute(moduleName, task->inputs, task->outputs);
      
      MLLM_INFO("QNNDispatcher: after graphExecute, outputs={}", task->outputs.size());
      if (!task->outputs.empty()) {
        for (size_t i = 0; i < task->outputs.size(); ++i) {
          void* ptr = task->outputs[i].ptr<void>();
          void* storage_ptr = task->outputs[i].impl() ? 
                              (task->outputs[i].impl()->storage() ? task->outputs[i].impl()->storage()->ptr_ : nullptr) : 
                              nullptr;
          MLLM_INFO("QNNDispatcher: output[{}] shape={}, device={}, ptr={}, storage_ptr={}, bytes={}, impl={}, storage={}", i, 
                    task->outputs[i].shape(), static_cast<int>(task->outputs[i].device()),
                    static_cast<void*>(ptr), static_cast<void*>(storage_ptr), task->outputs[i].bytes(),
                    static_cast<void*>(task->outputs[i].impl().get()),
                    static_cast<void*>(task->outputs[i].impl() && task->outputs[i].impl()->storage() ? 
                                       task->outputs[i].impl()->storage().get() : nullptr));
          if (!ptr && task->outputs[i].bytes() > 0) {
            MLLM_ERROR("QNNDispatcher: output[{}] has null pointer but non-zero bytes ({}), this will cause X2XOp errors!", 
                       i, task->outputs[i].bytes());
          }
          if (ptr != storage_ptr) {
            MLLM_WARN("QNNDispatcher: output[{}] ptr ({}) != storage_ptr ({}), this may indicate a view tensor issue", 
                      i, static_cast<void*>(ptr), static_cast<void*>(storage_ptr));
          }
        }
      } else {
        MLLM_ERROR("QNNDispatcher: graphExecute did not populate outputs for module '{}'", moduleName);
      }

      break;
    }
    default: NYI("QNNDispatcher::process not supported task type");
  }
}

void QNNDispatcher::syncWait() {
  // TODO
}

QNNDispatcher::ptr_t createQNNDispatcher(exec::static_thread_pool& thread_pool, const QNNDispatcherOptions& options) {
  return std::make_shared<QNNDispatcher>(thread_pool, Dispatcher::qnn_dispatcher_id, options);
}

}  // namespace mllm::qnn
