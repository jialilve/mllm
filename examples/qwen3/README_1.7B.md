# Qwen3 1.7B Support Guide

## Overview
This guide explains how to add Qwen3-1.7B support to the mllm project. Since Qwen3-1.7B shares the same architecture as Qwen3-0.6B, we only need to adjust configuration files.

## What You Need

### ✅ Already Done (No Changes Required)
- **Model Architecture**: `mllm/models/qwen3/modeling_qwen3.hpp` - Same for both 0.6B and 1.7B
- **Tokenizer**: `mllm/models/qwen3/tokenization_qwen3.hpp` - Shared tokenizer
- **Configuration Parser**: `mllm/models/qwen3/configuration_qwen3.hpp` - Generic config loader

### 📝 New Files Created
1. **Runtime Config**: `config_1.7B_w4a32_kai.json` ✅
2. **Quantization Config**: `quant_cfg_1.7B_w4a32_kai.json` ✅

---

## File 1: config_1.7B_w4a32_kai.json

**Purpose**: Runtime configuration that tells mllm how to construct the model.

**Key Parameters Explained**:
```json
{
    "hidden_size": 2048,           // Embedding dimension (0.6B: 1024 → 1.7B: 2048)
    "intermediate_size": 6144,     // MLP hidden size (0.6B: 3072 → 1.7B: 6144)
    "num_attention_heads": 16,     // Query heads (same as 0.6B)
    "num_key_value_heads": 8,      // Key/Value heads for GQA (same as 0.6B)
    "num_hidden_layers": 28,       // Number of transformer layers (same as 0.6B)
    "max_cache_length": 2048,      // KV cache limit
    "vocab_size": 151936,          // Token vocabulary size (same as 0.6B)
    "rope_theta": 1000000.0,       // RoPE base frequency
    "linear_impl_type": "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32"
}
```

---

## File 2: quant_cfg_1.7B_w4a32_kai.json

**Purpose**: Defines quantization strategy for weight compression (W4A32 = 4-bit weights, 32-bit activations).

### Understanding the Pattern

Each entry uses a **regex pattern** to match layer names:

```json
"^model\\.layers\\.\\d+\\.self_attn\\.q_proj.(bias|weight)": { ... }
```
- `^model\\.layers\\.` - Matches "model.layers."
- `\\d+` - Matches any layer number (0-27)
- `\\.self_attn\\.q_proj` - Matches the Q projection layer
- `.(bias|weight)` - Matches either bias or weight parameter

### Shape Calculations

The `shape` field defines the matrix dimensions **[output_dim, input_dim]**:

#### Attention Projections (Self-Attention)
```python
# Q Projection: hidden_size → hidden_size
q_proj: [2048, 2048]

# K Projection: hidden_size → (num_kv_heads * head_dim)
#               2048 → (8 * 128) = 1024
k_proj: [1024, 2048]

# V Projection: Same as K
v_proj: [1024, 2048]

# O Projection: hidden_size → hidden_size
o_proj: [2048, 2048]
```

#### MLP Projections (Feed-Forward Network)
```python
# Gate & Up Projections: hidden_size → intermediate_size
#                        2048 → 6144
gate_proj: [6144, 2048]
up_proj:   [6144, 2048]

# Down Projection: intermediate_size → hidden_size
#                  6144 → 2048
down_proj: [2048, 6144]
```

#### Output Head
```python
# LM Head: hidden_size → vocab_size
#          2048 → 151936
lm_head: [151936, 2048]
```

### Quantization Hints

```json
"hints": {
    "quant_method": "kai",                    // Use Kleidi AI quantization
    "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",  // FP32 output, INT8 activation, INT4 weight
    "kai_matmul_layout": "mxk_nxk",           // Matrix layout format
    "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",  // Tile configuration
    "shape": [output_dim, input_dim],
    "replace": true                            // Replace original layer with quantized version
}
```

---

## Step-by-Step Usage

### Step 1: Download Original Model
```bash
# From ModelScope
git clone https://www.modelscope.cn/Qwen/Qwen3-1.7B.git

# Or from Hugging Face
git clone https://huggingface.co/Qwen/Qwen3-1.7B
```

### Step 2: Convert and Quantize
```bash
# Install pymllm first (if not already installed)
bash ./scripts/install_pymllm.sh

# Convert with quantization
mllm-convertor \
   --input_path  ./Qwen3-1.7B/model.safetensors \
   --output_path ./Qwen3-1.7B/model_w4a32_kai.mllm \
   --cfg_path    ./examples/qwen3/quant_cfg_1.7B_w4a32_kai.json \
   --pipeline    w4a32_kai_pipeline
```

**What happens**:
1. Reads original PyTorch/Safetensors weights
2. Applies quantization based on `quant_cfg_1.7B_w4a32_kai.json`
3. Saves in mllm's binary format (`.mllm`)

### Step 3: Run Inference
```bash
# Build the project
mkdir -p build && cd build
cmake ..
make qwen3-chat

# Run with 1.7B model
./qwen3-chat \
   -m /path/to/Qwen3-1.7B/model_w4a32_kai.mllm \
   -c ../examples/qwen3/config_1.7B_w4a32_kai.json \
   -t path/to/Qwen3-1.7B/tokenizer.model \
   --thread 4
```

---

## Why This Works

### Architecture Compatibility
Both Qwen3-0.6B and 1.7B use:
- ✅ Same transformer architecture (Attention + MLP)
- ✅ Same tokenizer (vocab_size = 151936)
- ✅ Same number of layers (28)
- ✅ Same head configuration (16 Q-heads, 8 KV-heads)

**Only difference**: Hidden dimension scales (1024 → 2048), which affects weight matrix sizes.

### No Code Changes Needed
The `modeling_qwen3.hpp` implementation is **dimension-agnostic**:
```cpp
// Configuration reads hidden_size from JSON
hidden_size = data()["hidden_size"];  // Automatically uses 2048 for 1.7B

// Model layers scale automatically
Linear q_proj(hidden_size, hidden_size, ...);  // Becomes [2048, 2048] for 1.7B
```

---

## Verification Checklist

After conversion, verify:
- [ ] Model file size is reasonable (~1-2GB for W4A32)
- [ ] Inference runs without errors
- [ ] Generated text is coherent
- [ ] Performance matches expectations (TTFT, decode speed)

---

## Troubleshooting

### Error: "Shape mismatch during quantization"
- **Cause**: Shape in `quant_cfg` doesn't match actual weight dimensions
- **Fix**: Double-check dimensions against `config.json` from original model

### Error: "Quantization method not supported"
- **Cause**: Building without Kleidi AI support
- **Fix**: Ensure CMake option `-DMLLM_BUILD_ARM_BACKEND=ON` or similar

### Poor Generation Quality
- **Cause**: Incorrect quantization or mismatched config
- **Fix**: Re-verify all dimensions and retry conversion

---

## Summary

**What you created**:
1. ✅ `config_1.7B_w4a32_kai.json` - Runtime config with correct dimensions
2. ✅ `quant_cfg_1.7B_w4a32_kai.json` - Quantization recipe with updated shapes

**What you don't need to change**:
- ❌ Model code (`modeling_qwen3.hpp`)
- ❌ Tokenizer code (`tokenization_qwen3.hpp`)
- ❌ Config parser (`configuration_qwen3.hpp`)

**Next step**: Download Qwen3-1.7B and run the converter! 🚀
