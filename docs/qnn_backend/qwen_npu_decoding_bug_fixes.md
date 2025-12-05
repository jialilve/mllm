# Qwen NPU Decoding 阶段内存管理错误修复文档

## 概述

本文档详细描述了在实现 Qwen NPU 单 chunk decoding 功能过程中遇到的内存管理相关错误，以及相应的修复方案。这些错误主要发生在 QNN 后端的共享内存缓冲区（shared buffer）注册和管理过程中。

## 错误分析

### 错误 1: Failed to find memHandle 0x1

#### 错误现象

```
[ERROR] (0.0ms, 0) QnnDsp <E> Failed to find memHandle 0x1
[ERROR] (0.0ms, 0) QnnDsp <E> Failed to get HTP memHandle info for tensor
[ERROR] (0.0ms, 0) QnnDsp <E> graphExecuteRpcMemWrite failed for inputs
[ERROR] (0.0ms, 0) QnnDsp <E> preGraph execute failed 6004
[ERROR] (0.0ms, 0) QnnDsp <E> Graph model.layers.0_1 failed in execution with err 6004
```

#### 错误原因

在 decode 阶段，同一个 tensor（通过 tensor ID 或 tensor name 标识）会被多次使用。在 prefill 阶段，tensor 已经被分配内存并注册到 QNN 的共享缓冲区。但在 decode 阶段，代码尝试为同一个 tensor 重新分配新的内存并注册，导致：

1. **重复注册问题**：同一个 tensor ID 对应的 buffer 被多次注册，但 QNN 无法正确追踪这些注册信息
2. **memHandle 丢失**：之前的 memHandle 信息丢失，导致 QNN 无法找到对应的内存句柄
3. **内存泄漏风险**：旧的 buffer 没有被正确释放，造成内存泄漏

#### 修复方案

**核心思路**：实现 buffer 重用机制，避免为同一个 tensor 重复注册 buffer。

1. **添加 tensor ID/name 到 ptr 的映射**：
   - `tensorIdToPtrMap_`：将 tensor ID 映射到已注册的 buffer 指针
   - `tensorNameToPtrMap_`：将 tensor name 映射到已注册的 buffer 指针

2. **在注册前检查是否已存在**：
   ```cpp
   // 检查是否已有相同 tensor ID 的注册
   if (tensorIdToPtrMap_.count(tensor_id) > 0) {
       void* existing_ptr = tensorIdToPtrMap_[tensor_id];
       // 重用已注册的 buffer
       reuseExistingBuffer(existing_ptr);
   }
   ```

3. **实现 buffer 重用逻辑**：
   - 如果找到已注册的 buffer，直接使用其 memHandle
   - 将新分配的内存数据复制到已注册的 buffer
   - 释放新分配的内存，避免重复注册

### 错误 2: FastRPC 内存映射失败

#### 错误现象

```
[ERROR] (0.0ms, 0) QnnDsp <E> fastrpc memory map for fd: 428 with length: 1048576 failed with error: 0x1
[ERROR] (0.0ms, 0) QnnDsp <E> fastrpc memory map error reporting failed
[ERROR] (0.0ms, 0) QnnDsp <E> Mapping buffer fd 428 to FastRpc failed on domain 3
[ERROR] (0.0ms, 0) QnnDsp <E> SharedMemoryMod failed to Map Buffer to SMMU for domain 0
[ERROR] (0.0ms, 0) QnnDsp <E> Mod 4 failed for fd: 428, size: 1048576, pd: 0, with error 8003
[ERROR] (0.0ms, 0) QnnDsp <E> Failed to register memHandles for conext 1 on pdId 0, coreId 0, device 0. Current PD has ~2569.00 MB in use
[ERROR] (0.0ms, 0) QnnDsp <E> qnnMemCreateHandle failed
[ERROR] (0.0ms, 0) QnnDsp <E> Failed to register memHandles
[ERROR] (0.0ms, 0) QnnDsp <E> Failed to register mem with error 0x1f43
```

#### 错误原因

1. **内存资源耗尽**：QNN HTP 设备的内存资源有限，当已注册的 buffer 过多时（约 2569 MB），无法再注册新的 buffer
2. **文件描述符映射失败**：FastRPC 无法将文件描述符（fd）映射到 DSP 域，可能是由于：
   - 内存资源不足
   - 文件描述符无效或已关闭
   - SMMU（System Memory Management Unit）映射失败

#### 修复方案

**核心思路**：实现多级 fallback 机制，当注册失败时尝试重用已注册的 buffer。

1. **记录最后一次成功注册的信息**：
   ```cpp
   struct LastRegistrationInfo {
       uint32_t tensor_id;
       std::string tensor_name;
       void* ptr;
       Qnn_MemHandle_t mem_handle;
       size_t bytes;
   };
   ```

2. **多级 fallback 策略**：
   - **第一级**：尝试通过 tensor ID 重用 buffer
   - **第二级**：尝试通过 tensor name 重用 buffer
   - **第三级**：尝试重用最后一次成功注册的 buffer（如果 tensor ID/name 匹配）

3. **错误处理和日志**：
   - 记录详细的错误信息，包括 tensor ID、name、buffer 大小等
   - 记录当前已注册 buffer 的统计信息（数量和总大小）
   - 在 fallback 成功时记录警告日志

### 错误 3: memDeRegister 失败

#### 错误现象

```
[ASSERT] /root/mllm_v2/mllm/backends/qnn/QNNAllocator.cpp:96 QNN_SUCCESS != qnnInterface_.memDeRegister(&(ptrToFdAndMemHandleMap_[ptr].second), 1)
```

#### 错误原因

1. **重复注销**：尝试注销一个已经被注销的 memHandle
2. **别名问题**：多个 buffer 指针（ptr）共享同一个 memHandle，当其中一个被释放时，不应该立即注销 memHandle，因为其他指针还在使用
3. **生命周期管理错误**：buffer 的释放时机和 memHandle 的注销时机不匹配

#### 修复方案

**核心思路**：实现智能的 memHandle 生命周期管理，避免过早注销。

1. **检查 memHandle 的引用计数**：
   ```cpp
   // 检查是否有其他 ptr 使用同一个 memHandle
   void* alternative_ptr = nullptr;
   for (const auto& [other_ptr, fd_and_handle] : ptrToFdAndMemHandleMap_) {
       if (other_ptr != ptr && fd_and_handle.second == mem_handle) {
           alternative_ptr = other_ptr;
           break;
       }
   }
   ```

2. **延迟注销策略**：
   - 只有当没有任何其他 ptr 使用该 memHandle 时，才执行注销
   - 如果有别名存在，只从映射表中移除当前 ptr，保留 memHandle

3. **更新映射关系**：
   - 当有别名存在时，将 tensor ID/name 的映射更新为指向别名 ptr
   - 确保后续操作能够找到正确的 buffer

4. **清理映射信息**：
   - 统一使用 `eraseTensorMappingsForPtr()` 清理 tensor ID/name 映射
   - 统一使用 `clearLastRegistrationIfMatches()` 清理最后注册信息

## 修复实现细节

### 1. Buffer 重用机制

#### 1.1 映射表管理

```cpp
// tensor ID 到 buffer 指针的映射（更可靠）
std::map<uint32_t, void*> tensorIdToPtrMap_;

// tensor name 到 buffer 指针的映射（备用）
std::map<std::string, void*> tensorNameToPtrMap_;
```

**设计考虑**：
- Tensor ID 是 QNN 内部唯一标识，比 name 更可靠
- Name 作为 fallback，因为某些情况下 ID 可能为 0

#### 1.2 重用逻辑

```cpp
auto reuseExistingBuffer = [&](void* existing_ptr) -> bool {
    // 检查 existing_ptr 是否已注册
    auto fd_handle_iter = ptrToFdAndMemHandleMap_.find(existing_ptr);
    if (fd_handle_iter == ptrToFdAndMemHandleMap_.end()) {
        return false;  // 未注册，无法重用
    }
    
    // 获取已注册的 memHandle
    Qnn_MemHandle_t existing_mem_handle = fd_handle_iter->second.second;
    
    // 复制新数据到已注册的 buffer
    if (existing_ptr != ptr) {
        std::memcpy(existing_ptr, ptr, bytes_to_copy);
        // 释放新分配的 buffer
        rpcmem_free(ptr);
        storage->ptr_ = existing_ptr;
    }
    
    // 设置 tensor 的 memHandle
    QNN_TENSOR_SET_MEM_HANDLE(qnn_tensor, existing_mem_handle);
    updateMappings(existing_ptr);
    return true;
};
```

### 2. Fallback 机制

#### 2.1 最后注册信息记录

```cpp
void rememberLastRegistration(uint32_t tensor_id, const std::string& tensor_name, 
                              void* ptr, Qnn_MemHandle_t mem_handle, size_t total_bytes) {
    if (ptr == nullptr || mem_handle == nullptr) { return; }
    lastRegistrationInfo_.tensor_id = tensor_id;
    lastRegistrationInfo_.tensor_name = tensor_name;
    lastRegistrationInfo_.ptr = ptr;
    lastRegistrationInfo_.mem_handle = mem_handle;
    lastRegistrationInfo_.bytes = total_bytes;
    hasLastRegistrationInfo_ = true;
}
```

#### 2.2 多级 Fallback

```cpp
// 第一级：通过 tensor ID
if (tensorIdToPtrMap_.count(tensor_id) > 0) {
    void* existing_ptr = tensorIdToPtrMap_[tensor_id];
    if (reuseExistingBuffer(existing_ptr)) {
        return true;
    }
}

// 第二级：通过 tensor name
if (tensor_name != "unknown" && tensorNameToPtrMap_.count(tensor_name) > 0) {
    void* existing_ptr = tensorNameToPtrMap_[tensor_name];
    if (reuseExistingBuffer(existing_ptr)) {
        return true;
    }
}

// 第三级：通过最后注册信息
if (hasLastRegistrationInfo_) {
    bool same_tensor_id = tensor_id != 0 && tensor_id == lastRegistrationInfo_.tensor_id;
    bool same_tensor_name = tensor_name != "unknown" && 
                           tensor_name == lastRegistrationInfo_.tensor_name;
    bool ptr_still_registered = lastRegistrationInfo_.ptr != nullptr &&
                               ptrToFdAndMemHandleMap_.count(lastRegistrationInfo_.ptr) > 0;
    if ((same_tensor_id || same_tensor_name) && ptr_still_registered) {
        if (reuseExistingBuffer(lastRegistrationInfo_.ptr)) {
            return true;
        }
    }
}
```

### 3. 内存释放优化

#### 3.1 别名检测

```cpp
void* alternative_ptr = nullptr;
for (const auto& [other_ptr, fd_and_handle] : ptrToFdAndMemHandleMap_) {
    if (other_ptr != ptr && fd_and_handle.second == mem_handle) {
        alternative_ptr = other_ptr;
        break;
    }
}
```

#### 3.2 条件注销

```cpp
if (alternative_ptr == nullptr) {
    // 没有别名，可以安全注销
    auto status = qnnInterface_.memDeRegister(&mem_handle, 1);
    if (status != QNN_SUCCESS) {
        MLLM_WARN("memDeRegister failed, status=0x{:x}", status);
    }
    ptrToFdAndMemHandleMap_.erase(iter);
    rpcmem_free(ptr);  // 释放 buffer
} else {
    // 有别名，只移除映射，保留 buffer 和 memHandle
    ptrToFdAndMemHandleMap_.erase(iter);
    // 更新 tensor 映射指向别名
    for (auto& entry : tensorIdToPtrMap_) {
        if (entry.second == ptr) {
            entry.second = alternative_ptr;
        }
    }
}
```

### 4. 辅助函数

#### 4.1 清理映射

```cpp
void eraseTensorMappingsForPtr(void* ptr, std::string_view reason) {
    if (ptr == nullptr) { return; }
    
    // 清理 tensor ID 映射
    for (auto it = tensorIdToPtrMap_.begin(); it != tensorIdToPtrMap_.end();) {
        if (it->second == ptr) {
            it = tensorIdToPtrMap_.erase(it);
        } else {
            ++it;
        }
    }
    
    // 清理 tensor name 映射
    for (auto it = tensorNameToPtrMap_.begin(); it != tensorNameToPtrMap_.end();) {
        if (it->second == ptr) {
            it = tensorNameToPtrMap_.erase(it);
        } else {
            ++it;
        }
    }
}
```

#### 4.2 清理最后注册信息

```cpp
void clearLastRegistrationIfMatches(void* ptr, std::string_view reason) {
    if (!hasLastRegistrationInfo_ || ptr == nullptr) { return; }
    if (lastRegistrationInfo_.ptr == ptr) {
        lastRegistrationInfo_ = {};
        hasLastRegistrationInfo_ = false;
    }
}
```

## 相关代码文件

- `mllm/backends/qnn/QNNAllocator.cpp` - 内存分配器实现
- `mllm/backends/qnn/QNNAllocator.hpp` - 内存分配器接口
- `mllm/backends/qnn/QNNBackend.cpp` - QNN 后端实现（输入数据复制优化）
- `mllm/backends/qnn/QNNUtils.cpp` - QNN 工具函数（buffer 大小检查）

## 测试验证

修复后的代码应该能够：

1. **正确处理 decode 阶段的重复 tensor**：
   - 重用 prefill 阶段注册的 buffer
   - 避免重复注册导致的 memHandle 丢失

2. **处理内存资源不足的情况**：
   - 通过 fallback 机制重用已注册的 buffer
   - 避免因内存不足导致的注册失败

3. **正确管理 memHandle 生命周期**：
   - 避免重复注销
   - 正确处理别名情况
   - 确保 buffer 和 memHandle 的同步释放

## 总结

这些修复主要解决了 QNN 后端在 decode 阶段的内存管理问题，通过实现 buffer 重用机制、多级 fallback 策略和智能的 memHandle 生命周期管理，确保了 decode 阶段的稳定性和内存效率。这些改进不仅解决了当前的错误，还为未来可能的多 chunk decoding 功能奠定了基础。

