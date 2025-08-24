//
// GPT-2 Configuration
//

#ifndef CONFIG_GPT2_HPP
#define CONFIG_GPT2_HPP
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class GPT2NameConfig : public TransformerNameConfig {
public:
    std::string blk_name;
    std::string token_embd_name;
    std::string pos_embd_name;
    std::string post_norm_name;
    std::string lm_head_name;

    void init() {
        blk_name = "h.";
        _attn_base_name = "attn.";
        _ffn_base_name = "mlp.";
        _q_proj_name = "c_attn.q_proj";
        _k_proj_name = "c_attn.k_proj";
        _v_proj_name = "c_attn.v_proj";
        _o_proj_name = "c_proj";
        _up_proj_name = "c_fc";
        _down_proj_name = "c_proj";
        _attn_norm_name = "ln_1";
        _ffn_norm_name = "ln_2";
        token_embd_name = "wte";
        pos_embd_name = "wpe";
        post_norm_name = "ln_f";
        lm_head_name = "lm_head";
    }
};

class GPT2Config {
public:
    int vocab_size = 50257;
    int hidden_size = 768;
    int intermediate_size = 3072;
    int num_attention_heads = 12;
    int num_key_value_heads = 12; // same as num_attention_heads for GPT-2
    int num_hidden_layers = 12;
    int max_position_embeddings = 1024;
    RoPEType RoPE_type = RoPEType::NONE; // GPT-2 doesn't use RoPE
    float rope_theta = 10000.0f;
    int cache_limit = 1000;
    string arch = "GPT2Model";
    GPT2NameConfig names_config;

    GPT2Config(int token_limit, const string &type = "117M") {
        names_config.init();
        cache_limit = token_limit;
        
        if (type == "117M") {
            vocab_size = 50257;
            hidden_size = 768;
            intermediate_size = 3072;
            num_attention_heads = 12;
            num_hidden_layers = 12;
            max_position_embeddings = 1024;
        } else if (type == "345M") {
            vocab_size = 50257;
            hidden_size = 1024;
            intermediate_size = 4096;
            num_attention_heads = 16;
            num_hidden_layers = 24;
            max_position_embeddings = 1024;
        } else if (type == "762M") {
            vocab_size = 50257;
            hidden_size = 1280;
            intermediate_size = 5120;
            num_attention_heads = 20;
            num_hidden_layers = 36;
            max_position_embeddings = 1024;
        } else if (type == "1.5B") {
            vocab_size = 50257;
            hidden_size = 1600;
            intermediate_size = 6400;
            num_attention_heads = 25;
            num_hidden_layers = 48;
            max_position_embeddings = 1024;
        }
        
        num_key_value_heads = num_attention_heads; // GPT-2 doesn't use MQA/GQA
    }
};

#endif // CONFIG_GPT2_HPP