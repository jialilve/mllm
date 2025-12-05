# feat: Implement Qwen NPU Decoding Support with Memory Management Fixes

## Summary

This PR implements complete decoding support for Qwen NPU models on QNN backend, including both single-chunk and multi-chunk decoding capabilities. It also fixes critical memory management issues encountered during decode phase and improves CausalMaskOp for multi-chunk scenarios.

## Features Implemented

### 1. Single-Chunk Decoding Support

Implemented basic decoding functionality for input sequences shorter than chunk size (128 tokens):

- **KV Cache Sequence Management**: Added `setKVCacheSeqCnt()` and `getKVCacheSeqCnt()` methods across the KV cache hierarchy
  - `aops::KVCacheOp`: Added virtual `setCurrentSeqCnt()` and `getCurrentSeqCnt()` methods
  - `CPUKVCacheOp`: Implemented sequence count management using `StaticCache`
  - `nn::KVCache`: Added layer interface for sequence count control
  - `QwenText` and `QwenForCausalLM`: Added model-level APIs for KV cache management

- **Decode Loop Implementation**: 
  - Implemented iterative token generation loop in `examples/qwen_npu/main.cpp`
  - Handles position_ids correctly for decode phase
  - Supports EOS token (151645) termination check
  - Manages input sequence buffer with padding area for new tokens

- **Forward Method Updates**:
  - Enhanced `QwenForCausalLM::forward()` to support decode phase
  - Proper handling of position_ids increment for decode iterations
  - Support for variable sequence lengths during decode

### 2. Multi-Chunk Decoding Support

Extended decoding to handle long input sequences that exceed chunk size:

- **Chunked Prefill**: Processes long prompts in 128-token chunks
- **KV Cache Alignment**: Correctly aligns KV cache offsets for multi-chunk scenarios
  - Uses absolute sequence length from start of entire sequence
  - Sets KV cache sequence count to chunk start offset before each prefill
- **Decode Continuation**: Continues decoding after processing all prompt chunks
- **Position IDs Generation**: Generates position_ids starting from chunk offset for multi-chunk prefill

### 3. CausalMaskOp Improvement

CausalMaskOp improvement by @oreomaker.
Fixed causal mask calculation for multi-chunk decoding scenarios:

**Problem**: Original mask calculation `copy_count = std::min(r + 1, (size_t)D)` was incorrect for multi-chunk scenarios where sequence length (S) and dimension (D) differ.

**Solution**: Changed to `copy_count = D - S + r + 1` to correctly handle cases where S < D (multi-chunk scenarios with padding).

- Applied to both AVX2 (x86_64) and NEON (ARM64) implementations
- Ensures correct masking behavior across chunk boundaries
- Maintains backward compatibility with single-chunk scenarios

### 4. Memory Management Fixes

Fixed critical memory management issues in QNN backend during decode phase:

#### Problem 1: Failed to find memHandle 0x1
- **Root Cause**: Same tensor (by ID/name) was registered multiple times, causing QNN to lose track of memory handles
- **Solution**: Implemented buffer reuse mechanism with multi-level fallback

#### Problem 2: FastRPC Memory Mapping Failures
- **Root Cause**: QNN HTP device memory exhausted (~2.5GB limit) when registering too many buffers
- **Solution**: Multi-level fallback strategy to reuse existing buffers when registration fails

#### Problem 3: memDeRegister Failures
- **Root Cause**: Attempts to de-register memory handles that were already de-registered or shared by multiple pointers
- **Solution**: Implemented alias detection and reference counting for memory handle lifecycle management

## Key Changes

### Core KV Cache Interface (`mllm/core/aops/KVCacheOp.hpp`, `mllm/backends/cpu/ops/KVCacheOp.{hpp,cpp}`)

- Added `setCurrentSeqCnt(int32_t seq)` virtual method to `aops::KVCacheOp`
- Added `getCurrentSeqCnt()` const method to `aops::KVCacheOp`
- Implemented both methods in `CPUKVCacheOp` using `StaticCache` API

### Layer Interface (`mllm/nn/layers/KVCache.{hpp,cpp}`)

- Added `setCurrentSeqCnt(int32_t seq)` method
- Added `getCurrentSeqCnt(int32_t layer_idx)` const method

### Model Interface (`mllm/models/qwen_npu/modeling_qwen_npu.hpp`)

- Added `setKVCacheSeqCnt(int32_t seq)` to `QwenText` and `QwenForCausalLM`
- Added `getKVCacheSeqCnt(int32_t layer_idx)` const method
- Updated `forward()` method to handle decode phase with position_ids

### QNN Backend Memory Management (`mllm/backends/qnn/QNNAllocator.{hpp,cpp}`)

- Added `tensorIdToPtrMap_` and `tensorNameToPtrMap_` for buffer lookup by tensor identity
- Implemented `reuseExistingBuffer()` lambda with multi-level fallback:
  - Level 1: Check exact buffer pointer
  - Level 2: Lookup by tensor ID
  - Level 3: Lookup by tensor name
  - Level 4: Reuse last successfully registered buffer
- Added `LastRegistrationInfo` structure to track last successful registration
- Implemented helper functions:
  - `eraseTensorMappingsForPtr()`: Clean up tensor ID/name mappings
  - `rememberLastRegistration()`: Track successful registrations
  - `clearLastRegistrationIfMatches()`: Clean up last registration info
- Enhanced `free()` method with alias detection and reference counting
- Added multi-level fallback in `registerQnnTensorToSharedBuffer()` when registration fails

### QNN Backend Execution (`mllm/backends/qnn/QNNBackend.cpp`)

- Improved input tensor data copying in `graphExecute()`
- Added size mismatch detection and zero-padding for decode phase inputs
- Enhanced error messages with detailed tensor information

### QNN Utils (`mllm/backends/qnn/QNNUtils.cpp`)

- Added buffer size validation in `QNNTensorWrapper::alloc()`
- Implemented automatic de-registration when registered buffer is too small
- Added checks for buffer validity before reuse

### CausalMaskOp (`mllm/backends/cpu/ops/CausalMaskOp.cpp`)

- Fixed mask calculation for multi-chunk scenarios:
  - Changed from `copy_count = std::min(r + 1, (size_t)D)` 
  - To `copy_count = D - S + r + 1`
- Applied fix to both AVX2 and NEON implementations

### Example Implementation (`examples/qwen_npu/main.cpp`)

- Implemented single-chunk decoding loop with:
  - KV cache sequence count management
  - Position IDs handling
  - EOS token termination
  - Input sequence buffer management
- Extended to multi-chunk decoding with:
  - Chunked prefill processing
  - KV cache alignment across chunks
  - Decode continuation after all chunks processed
  - Proper position IDs generation for multi-chunk scenarios

## Related Commits

This PR consolidates the following commits:

- `1d5d253`: feat: implement Qwen NPU simple single chunk decoding support and Memory management fixes
- `b438b3d`: implement Qwen NPU simple muti-chunk decoding support  
- `e26b11b`: fix: stabilize QNN multi-chunk decoding (including CausalMaskOp improvement)

## Co-authors

This PR is a collaborative effort:

- @oreomaker - Technical guidance, CausalMaskOp improvement for multi-chunk decoding, and code review
- @jialilve - Main implementation including single-chunk/multi-chunk decoding support and memory management fixes
