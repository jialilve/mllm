注意事项：！
qwen3-1.7b int8 rotated导出到phoenix（10.109.246.210）的/data/shrelic/mllm_v2/qwen3-1.7b-int8-rotated.mllm，这个模型的linear权重没有进行转置，所以在QNNLinearOp的load中需要在加载后调用一下tensor.transpose()。你用这个模型的权重，试一下qnn qwen3的prefill吧

# Qwen3-1.7B

**Embedding 层：**  

- `model.embed_tokens.weight`: [151936, 2048]  (lm_head用的就是这个权重)

**单个 Transformer 层（共 28 层，结构相同，形如model.layers.X.xxx 的通用结构）下面只写出了xxx的部分，X为0-27：**  

- `self_attn.q_proj.weight`: [2048, 2048]  
- `self_attn.k_proj.weight`: [1024, 2048] 
- `self_attn.v_proj.weight`: [1024, 2048]  
- `self_attn.o_proj.weight`: [2048, 2048]  
- `self_attn.q_norm.weight`: [128]  
- `self_attn.k_norm.weight`: [128]  
- `mlp.gate_proj.weight`: [6144, 2048]  
- `mlp.up_proj.weight`: [6144, 2048]  
- `mlp.down_proj.weight`: [2048, 6144]  
- `input_layernorm.weight`: [2048]  
- `post_attention_layernorm.weight`: [2048]

**最终****归一化****层：**  

- `model.norm.weight`: [2048]

**语言头**

- `lm_head.weight`: [151936, 2048] （与embed_tokens.weight共用，在named_params()内看不到）

# Convert

**非Linear层（norm的weight，emb的weight保持fp32）**

对于除了`lm_head`外的所有linear层（q, k, v, o, gate, up, down），量化进行操作如下

暂时无法在飞书文档外展示此内容

1. state_dict中的layer_name.weight先量化再写回state_dict (fp32->int8)
2. 添加layer_name.input_scale、layer_name.output_scale、layer_name.scale，均为fp32，对应输入、输出、weight的scale。三者均写入state_dict。

量化反量化公式为

$$q = round(q/scale) \\ r = q \times scale$$

convert时只需要对weight进行量化和转置即可，量化可参考代码如下

```Python
def quantize_given_scale(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
     使用给定的 scale 对权重张量进行量化（per-tensor）。
    Args:
        w (torch.Tensor): 待量化的权重张量，dtype 可为 float32。
        scale (torch.Tensor): 已知的缩放因子，shape=[1]，dtype float32。
    
    Returns:
        torch.Tensor: 量化后的权重张量，int8
    """
    if scale.numel() != 1:
        raise ValueError(f"scale should be a tensor of shape [1], got {scale.shape}")
    
    # 可以用cuda加速，不需要的话去掉这行
    w = w.to("cuda")
    
    # 量化
    w = w.div(scale.to(w.device))  # 除以给定 scale
    w = w.round_()  # 四舍五入
    
    # 转换为int8
    w_q = w.to("cpu").type(torch.int8)
    
    return w_q
```

convert脚本如下:

```Python
def is_target_layer(name):
    """
    判断是否是需要量化的 Linear 层。
    排除 lm_head。
    目标：layers.X 中的 q, k, v, o, gate, up, down
    """
    if "lm_head" in name:
        return False
    
    # Qwen 的结构通常是 model.layers.X.self_attn.q_proj 等
    target_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    if any(k in name for k in target_keywords) and "weight" in name:
        return True
    return False

def main():
    model = ... # rotation后的模型
    state_dict = model.state_dict()
    new_state_dict = {}
    
    print("开始转换权重...")
    
    # 遍历所有参数
    for name, param in tqdm(state_dict):
        # 如果是需要量化的层
        if is_target_layer(name):
            # 1. 拿到Weight Scale
            # param shape: [out_dim, in_dim] (Linear层的默认存储)
            scale_val = ... # 获得训练得到的该weight的scale
            
            # 2. 量化权重
            # 此时 w_quant shape: [out_dim, in_dim]
            w_quant = quantize_given_scale(param, scale_val)
            
            
            # 3. 构建保存的 Key 名称
            base_name = name.replace(".weight", "")
            
            # 写入量化后的权重 (Int8)
            new_state_dict[name] = w_quant.cpu()
            
            # 写入 Scale (FP32)
            new_state_dict[f"{base_name}.scale"] = scale_val.cpu()
            
            # 写入 Input Scale (FP32)
            input_scale = ... # input scale
            new_state_dict[f"{base_name}.input_scale"] = input_scale
            
            # 写入 Output Scale (FP32) 
            output_scale = ... # output scale
            new_state_dict[f"{base_name}.output_scale"] = output_scale
            
        else:
            # 不需要量化的层（如 Norm, Embeddings, lm_head），直接复制
            # 保持原样 (FP16/FP32)
            new_state_dict[name] = param.cpu()

    torch.save(new_state_dict, "./Qwen3-rotated.pth")


if __name__ == "__main__":
    main()
```

