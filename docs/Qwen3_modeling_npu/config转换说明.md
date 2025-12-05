# Qwen3 Config.json 转换说明

## 问题

**HuggingFace 的 config.json 不能直接用于 MLLM QNN 后端**，因为缺少 MLLM 特有的字段。

## 差异对比

### 重要发现

根据 [HuggingFace Qwen3-1.7B config.json](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json)：

✅ **Qwen3-1.7B 的 HuggingFace config.json 已经包含 `head_dim: 128`**  
✅ **Qwen3-1.7B 的 HuggingFace config.json 已经包含 `attention_bias: false`**

因此，对于 Qwen3-1.7B，**只需要添加 2 个字段**即可用于 MLLM QNN 后端。

### MLLM QNN 需要的额外字段

1. **`head_dim`** (必需)
   - ✅ **Qwen3-1.7B 已包含**：`head_dim: 128`
   - ⚠️ **Qwen1.5 没有**：需要计算 `head_dim = hidden_size / num_attention_heads`
   - 对于 Qwen3-1.7B：`head_dim = 2048 / 16 = 128`（已包含在 HuggingFace config 中）

2. **`max_cache_length`** (必需)
   - MLLM 特有，用于 KV cache 大小
   - 默认值：`2048`（可根据需求调整）

3. **`linear_impl_type`** (必需)
   - NPU 后端特有，指定线性层实现类型
   - 对于 QNN 后端，通常使用：`"Default"` 或根据实际 QNN 配置设置

## 转换方法

### 方法 1: 手动编辑（推荐）

1. 从 HuggingFace 下载 Qwen3-1.7B 的 `config.json`（[链接](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json)）
2. **只需要添加 2 个字段**（`head_dim` 和 `attention_bias` 已存在）：

```json
{
  ... (保留所有原始字段，包括已有的 head_dim: 128 和 attention_bias: false) ...,
  "max_cache_length": 2048,          // 新增：KV cache 大小
  "linear_impl_type": "Default"       // 新增：NPU 线性层实现类型
}
```

**完整示例**（基于 HuggingFace Qwen3-1.7B）：
```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "attention_bias": false,            // ✅ 已存在
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,                    // ✅ 已存在
  "hidden_act": "silu",
  "hidden_size": 2048,
  "intermediate_size": 6144,
  "max_position_embeddings": 40960,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "num_hidden_layers": 28,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000,
  "tie_word_embeddings": true,
  "vocab_size": 151936,
  "max_cache_length": 2048,          // ⭐ 需要添加
  "linear_impl_type": "Default"       // ⭐ 需要添加
}
```

### 方法 2: 使用 Python 脚本转换

创建一个转换脚本 `convert_config.py`：

```python
import json
import sys

def convert_hf_to_mllm(hf_config_path, output_path):
    """将 HuggingFace config.json 转换为 MLLM QNN 格式"""
    with open(hf_config_path, 'r') as f:
        hf_config = json.load(f)
    
    # 计算 head_dim
    hidden_size = hf_config.get("hidden_size", 2048)
    num_attention_heads = hf_config.get("num_attention_heads", 16)
    head_dim = hidden_size // num_attention_heads
    
    # 创建 MLLM config
    mllm_config = hf_config.copy()
    mllm_config["head_dim"] = head_dim
    mllm_config["max_cache_length"] = 2048  # 默认值，可根据需求调整
    mllm_config["linear_impl_type"] = "Default"  # 默认值，可能需要根据实际 QNN 配置调整
    
    # 保存
    with open(output_path, 'w') as f:
        json.dump(mllm_config, f, indent=2)
    
    print(f"转换完成：{output_path}")
    print(f"head_dim = {head_dim} (计算：{hidden_size} / {num_attention_heads})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python convert_config.py <huggingface_config.json> <output_config.json>")
        sys.exit(1)
    
    convert_hf_to_mllm(sys.argv[1], sys.argv[2])
```

使用方法：
```bash
python convert_config.py /path/to/qwen3-1.7b/config.json config_1.7B_w8a16_qnn.json
```

## Qwen3-1.7B vs Qwen1.5-1.8B 对比

| 字段 | Qwen3-1.7B (HF) | Qwen1.5-1.8B-Chat (HF) | MLLM 需要 |
|------|-----------------|------------------------|----------|
| `head_dim` | ✅ **已包含** (128) | ❌ 缺失 | ✅ 必需 |
| `attention_bias` | ✅ **已包含** (false) | ❌ 缺失 | ✅ 必需 |
| `max_cache_length` | ❌ 缺失 | ❌ 缺失 | ✅ 必需 |
| `linear_impl_type` | ❌ 缺失 | ❌ 缺失 | ✅ 必需 |

**结论**：
- **Qwen3-1.7B**：只需要添加 `max_cache_length` 和 `linear_impl_type`（2 个字段）
- **Qwen1.5-1.8B**：需要添加 `head_dim`、`attention_bias`、`max_cache_length`、`linear_impl_type`（4 个字段）

## Qwen3-1.7B 完整配置示例

基于 [HuggingFace Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json) 的完整配置（添加了 2 个必需字段）：

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 6144,
  "max_position_embeddings": 40960,
  "max_window_layers": 28,
  "model_type": "qwen3",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936,
  "max_cache_length": 2048,
  "linear_impl_type": "Default"
}
```

## 注意事项

1. **`head_dim`**：
   - ✅ Qwen3-1.7B 的 HuggingFace config 已包含，值为 128，无需修改
   - ⚠️ 如果手动添加，必须正确：`head_dim = hidden_size / num_attention_heads`
   - 如果值错误，会导致模型加载失败或推理错误
2. **`linear_impl_type`**：可能需要根据实际的 QNN 后端配置调整，建议询问师兄或查看 QNN 文档
3. **`max_cache_length`**：根据实际需求设置，影响内存占用和最大序列长度
4. **`tokenizer.json`**：可以直接使用 HuggingFace 的，Qwen3 只需要这一个文件（不需要 `qwen_merges.txt`）
5. **其他字段**：保留 HuggingFace config 中的所有其他字段（如 `attention_dropout`、`hidden_act` 等），即使 MLLM 不使用它们也不会影响功能

## 验证

转换后，可以使用以下命令验证 JSON 格式是否正确：

```bash
python -m json.tool config_1.7B_w8a16_qnn.json > /dev/null && echo "JSON 格式正确" || echo "JSON 格式错误"
```

## 已创建的文件

已创建可直接使用的配置文件：
- **`docs/Qwen3_modeling_npu/config_1.7B_w8a16_qnn.json`** - Qwen3-1.7B 的完整配置（基于 HuggingFace，已添加 MLLM 必需字段）

可以直接复制使用：
```bash
# 在 Phoenix 上
cp /data/shrelic/mllm_v2/docs/Qwen3_modeling_npu/config_1.7B_w8a16_qnn.json /data/shrelic/mllm_v2/config.json

# 推送到 Android 设备
adb -s 10.29.208.59:9808 push /data/shrelic/mllm_v2/config.json /data/local/tmp/zl/mllm-v2/bin_test/
```

