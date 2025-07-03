//
// GPT-2 Model Implementation
//

#ifndef MODELING_GPT2_HPP
#define MODELING_GPT2_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_gpt2.hpp"
#include "models/transformer/modeling_transformer.hpp"

using namespace mllm;

class GPT2MLP final : public Module {
    Layer up_proj;
    Layer act;
    Layer down_proj;

public:
    GPT2MLP() = default;
    GPT2MLP(int hidden_dim, int ffn_hidden, const GPT2NameConfig &names, const string &base_name) {
        up_proj = Linear(hidden_dim, ffn_hidden, true, base_name + names._up_proj_name);
        act = GELU(base_name + "act");
        down_proj = Linear(ffn_hidden, hidden_dim, true, base_name + names._down_proj_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = up_proj(inputs[0]);
        x = act(x);
        x = down_proj(x);
        return {x};
    }
};

class GPT2Block final : public Module {
    MultiHeadAttention attention;
    GPT2MLP mlp;
    Layer norm1;
    Layer norm2;

public:
    GPT2Block() = default;
    GPT2Block(int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit, const GPT2NameConfig &names, const string &base_name) {
        attention = MultiHeadAttention(hidden_dim, head_size, kv_head_size, hidden_dim / head_size, SPLIT_NONE, false, false,
                                       RoPE_type, rope_theta, max_position_embeddings, cache_limit, true, true, names, base_name + names._attn_base_name);
        mlp = GPT2MLP(hidden_dim, ffn_hidden, names, base_name + names._ffn_base_name);
        norm1 = LayerNorm(hidden_dim, true, 1e-5, base_name + names._attn_norm_name);
        norm2 = LayerNorm(hidden_dim, true, 1e-5, base_name + names._ffn_norm_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = norm1(inputs[0]);
        x = attention({x, x, x})[0];
        auto tmp = x + inputs[0];
        x = norm2(tmp);
        x = mlp({x})[0];
        x = x + tmp;
        return {x};
    }

    MultiHeadAttention &get_attention() {
        return attention;
    }
};

class GPT2Model final : public Module {
    Layer token_embedding;
    Parameter pos_embedding;
    vector<GPT2Block> blocks;
    Layer norm;
    Layer lm_head;

public:
    explicit GPT2Model(const GPT2Config &config) :
        GPT2Model(config.vocab_size, config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.intermediate_size, config.num_hidden_layers,
                  config.RoPE_type, config.rope_theta, config.max_position_embeddings, config.cache_limit,
                  config.names_config, config.names_config.blk_name) {
    }
    GPT2Model(int vocab_size, int hidden_dim, int head_size, int kv_head_size, int ffn_hidden, int block_num, RoPEType RoPE_type, float rope_theta, int max_position_embeddings, int cache_limit,
              const GPT2NameConfig &names, const string &base_name) {
        token_embedding = Embedding(vocab_size, hidden_dim, names.token_embd_name);
        pos_embedding = Parameter(1, max_position_embeddings, 1, hidden_dim, names.pos_embd_name + ".weight");
        blocks = List<GPT2Block>(block_num, hidden_dim, head_size, kv_head_size, ffn_hidden, RoPE_type, rope_theta, max_position_embeddings, cache_limit, names, base_name);
        norm = LayerNorm(hidden_dim, true, 1e-5, names.post_norm_name);
        lm_head = Linear(hidden_dim, vocab_size, false, names.lm_head_name);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        auto x = token_embedding(inputs[0]);
        
        // Add positional embeddings for GPT-2
        auto positions = pos_embedding();
        auto seq_len = x.sequence();
        
        // Clip positional embeddings to current sequence length
        auto pos_clip = positions.clip({0, 0, 0, 0}, {1, seq_len, 1, positions.dimension()});
        x = x + pos_clip;
        
        for (auto &block : blocks) {
            x = block({x})[0];
        }
        x = norm(x);
        x = lm_head(x);
        return {x};
    }

    void clear_kvcache() override {
        for (auto &block : blocks) {
            auto kvcache = block.get_attention().get_cache();
            for (auto &cache : kvcache) { cache->clearCache(); }
            auto ropes = block.get_attention().get_rope();
            for (auto &rope : ropes) { rope->clearCache(); }
        }
    }
};

class GPT2ForCausalLM final : public Module {
    GPT2Model model;

public:
    explicit GPT2ForCausalLM(const GPT2Config &config) {
        model = GPT2Model(config);
    }
    vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        return model(inputs, args);
    }
    
    void clear_kvcache() override {
        model.clear_kvcache();
    }
};

#endif // MODELING_GPT2_HPP