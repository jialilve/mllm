#!/usr/bin/env python3
"""
将 HuggingFace config.json 转换为 MLLM QNN 格式

用法:
    python convert_config.py <huggingface_config.json> <output_config.json>

示例:
    python convert_config.py /path/to/qwen3-1.7b/config.json config.json
"""

import json
import sys
import os


def convert_hf_to_mllm(hf_config_path, output_path):
    """将 HuggingFace config.json 转换为 MLLM QNN 格式"""
    if not os.path.exists(hf_config_path):
        print(f"错误：文件不存在: {hf_config_path}")
        sys.exit(1)
    
    with open(hf_config_path, 'r', encoding='utf-8') as f:
        hf_config = json.load(f)
    
    # 计算 head_dim
    hidden_size = hf_config.get("hidden_size")
    num_attention_heads = hf_config.get("num_attention_heads")
    
    if hidden_size is None or num_attention_heads is None:
        print("错误：config.json 中缺少 hidden_size 或 num_attention_heads")
        sys.exit(1)
    
    if num_attention_heads == 0:
        print("错误：num_attention_heads 不能为 0")
        sys.exit(1)
    
    head_dim = hidden_size // num_attention_heads
    
    # 创建 MLLM config（保留所有原始字段，添加新字段）
    mllm_config = hf_config.copy()
    
    # 添加必需字段（如果不存在）
    if "head_dim" not in mllm_config:
        mllm_config["head_dim"] = head_dim
        print(f"添加 head_dim = {head_dim} (计算：{hidden_size} / {num_attention_heads})")
    else:
        print(f"head_dim 已存在: {mllm_config['head_dim']}")
        if mllm_config["head_dim"] != head_dim:
            print(f"警告：head_dim 值 ({mllm_config['head_dim']}) 与计算值 ({head_dim}) 不一致")
    
    if "max_cache_length" not in mllm_config:
        mllm_config["max_cache_length"] = 2048
        print(f"添加 max_cache_length = 2048 (默认值)")
    else:
        print(f"max_cache_length 已存在: {mllm_config['max_cache_length']}")
    
    if "linear_impl_type" not in mllm_config:
        mllm_config["linear_impl_type"] = "Default"
        print(f"添加 linear_impl_type = 'Default' (默认值，可能需要根据实际 QNN 配置调整)")
    else:
        print(f"linear_impl_type 已存在: {mllm_config['linear_impl_type']}")
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mllm_config, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换完成！")
    print(f"输入文件: {hf_config_path}")
    print(f"输出文件: {output_path}")
    
    # 验证 JSON 格式
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            json.load(f)
        print("✓ JSON 格式验证通过")
    except json.JSONDecodeError as e:
        print(f"✗ JSON 格式错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python convert_config.py <huggingface_config.json> <output_config.json>")
        print("\n示例:")
        print("  python convert_config.py /path/to/qwen3-1.7b/config.json config.json")
        sys.exit(1)
    
    hf_config_path = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_hf_to_mllm(hf_config_path, output_path)

