// Gemma4 (iSWA + MoE) target structs for dflash daemon.
//
// Architecture summary (from Google Gemma-4 config):
//   - Hybrid iSWA: per-layer sliding window pattern (full vs SWA).
//   - MoE on sparse layers (all but lead dense). Routing: softmax-normalized.
//   - Per-layer embeddings: additional per-layer token embedding + projection.
//   - KV sharing: later layers may reuse KV from earlier layers.
//   - Logit softcapping after final lm_head.
//   - Q/K RMSNorm per head, RoPE (with per-layer-type freq base).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "internal.h"  // CpuEmbedder

namespace dflash::common {

struct Gemma4Layer {
    // Pre-attn norm
    ggml_tensor * attn_norm       = nullptr;  // [n_embd]

    // Attention projections
    ggml_tensor * wq              = nullptr;  // [n_embd, n_head * head_dim]
    ggml_tensor * wk              = nullptr;  // [n_embd, n_head_kv * head_dim] (nullptr on KV-reuse layers)
    ggml_tensor * wv              = nullptr;  // [n_embd, n_head_kv * head_dim] (nullptr on KV-reuse layers)
    ggml_tensor * wo              = nullptr;  // [n_head * head_dim, n_embd]
    ggml_tensor * q_norm          = nullptr;  // [head_dim]
    ggml_tensor * k_norm          = nullptr;  // [head_dim]

    // Post-attention norm
    ggml_tensor * attn_post_norm  = nullptr;  // [n_embd]

    // Dense FFN (lead dense layers only)
    ggml_tensor * ffn_norm        = nullptr;  // [n_embd]
    ggml_tensor * ffn_gate        = nullptr;  // [n_embd, n_ff]
    ggml_tensor * ffn_up          = nullptr;  // [n_embd, n_ff]
    ggml_tensor * ffn_down        = nullptr;  // [n_ff, n_embd]
    ggml_tensor * ffn_post_norm   = nullptr;  // [n_embd]

    // MoE FFN (sparse layers): shared expert
    ggml_tensor * ffn_norm_moe    = nullptr;  // [n_embd] (pre-norm for shared exp)
    ggml_tensor * ffn_gate_shexp  = nullptr;  // [n_embd, n_ff_shexp]
    ggml_tensor * ffn_up_shexp    = nullptr;  // [n_embd, n_ff_shexp]
    ggml_tensor * ffn_down_shexp  = nullptr;  // [n_ff_shexp, n_embd]
    ggml_tensor * ffn_post_norm_1 = nullptr;  // [n_embd]

    // MoE FFN (sparse layers): routed experts
    ggml_tensor * ffn_pre_norm_2  = nullptr;  // [n_embd]
    ggml_tensor * ffn_gate_inp    = nullptr;  // [n_embd, n_expert] router weights
    ggml_tensor * ffn_gate_inp_s  = nullptr;  // [n_embd] router scale
    ggml_tensor * ffn_gate_up_exps = nullptr; // packed gate+up for all experts
    ggml_tensor * ffn_down_exps   = nullptr;  // packed down for all experts
    ggml_tensor * ffn_down_exps_s = nullptr;  // scale for quantized down experts
    ggml_tensor * ffn_post_norm_2 = nullptr;  // [n_embd]

    // Per-layer embedding gate + projection
    ggml_tensor * per_layer_inp_gate   = nullptr;  // [n_embd, n_embd_per_layer]
    ggml_tensor * per_layer_proj       = nullptr;  // [n_embd_per_layer, n_embd]
    ggml_tensor * per_layer_post_norm  = nullptr;  // [n_embd]

    // Layer output scale
    ggml_tensor * out_scale       = nullptr;  // scalar or [1]

    // RoPE freq factors (full-attention layers only)
    ggml_tensor * rope_freqs      = nullptr;
};

struct Gemma4Weights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    // Global tensors
    ggml_tensor * tok_embd               = nullptr;  // [n_embd, n_vocab]
    ggml_tensor * out_norm               = nullptr;  // [n_embd]
    ggml_tensor * output                 = nullptr;  // [n_embd, n_vocab] (lm_head)
    ggml_tensor * per_layer_tok_embd     = nullptr;  // [n_embd_per_layer * n_layer, n_vocab]
    ggml_tensor * per_layer_model_proj   = nullptr;  // [n_embd, n_embd_per_layer * n_layer]
    ggml_tensor * per_layer_proj_norm    = nullptr;  // [n_embd_per_layer * n_layer]

    std::vector<Gemma4Layer> layers;

    CpuEmbedder embedder;

    // Architecture metadata
    int n_layer               = 0;
    int n_head                = 0;
    int n_head_kv             = 0;
    int head_dim              = 128;
    int n_embd                = 0;
    int n_ff                  = 0;       // dense FFN intermediate
    int n_ff_exp              = 0;       // expert FFN intermediate
    int n_ff_shexp            = 0;       // shared expert FFN intermediate
    int n_expert              = 0;
    int n_expert_used         = 0;
    int n_layer_dense_lead    = 1;
    int n_embd_per_layer      = 0;       // per-layer embedding dim
    int n_vocab               = 0;

    // iSWA
    int  sliding_window       = 0;
    std::vector<bool> swa_layers;        // true = SWA, false = full attn
    std::vector<bool> has_kv;            // true = layer has own K/V
    int  kv_sharing_start     = 0;       // first layer that reuses KV

    // RoPE
    float rope_freq_base_full = 1000000.0f;
    float rope_freq_base_swa  = 10000.0f;

    // Logit softcapping
    float final_logit_softcap = 0.0f;

    // Tokenizer
    int32_t bos_id      = 2;
    int32_t eos_id      = 1;
    int32_t eos_chat_id = -1;

    float   norm_eps    = 1e-6f;
};

inline bool gemma4_is_swa_layer(const Gemma4Weights & w, int il) {
    return il < (int)w.swa_layers.size() && w.swa_layers[il];
}

inline bool gemma4_has_kv(const Gemma4Weights & w, int il) {
    return il < (int)w.has_kv.size() && w.has_kv[il];
}

// GGUF loader
bool load_gemma4_gguf(const std::string & path,
                       ggml_backend_t backend,
                       Gemma4Weights & out);

void free_gemma4_weights(Gemma4Weights & w);

// KV cache
struct Gemma4Cache {
    int cur_pos  = 0;
    int max_ctx  = 0;
    int n_layer  = 0;

    // Only layers where has_kv[il] == true have real K/V tensors.
    // KV-reuse layers reference an earlier layer's cache.
    std::vector<ggml_tensor *> k;   // n_layer entries (nullptr for reuse layers)
    std::vector<ggml_tensor *> v;
    std::vector<int>           kv_source;  // for each layer, which layer's KV to use

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

bool  create_gemma4_cache(ggml_backend_t backend, const Gemma4Weights & w,
                           int max_ctx, Gemma4Cache & out);
void  free_gemma4_cache(Gemma4Cache & c);

// Snapshot
struct Gemma4Snapshot {
    int cur_pos = 0;
    std::vector<ggml_tensor *> k_snap;
    std::vector<ggml_tensor *> v_snap;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

void free_gemma4_snapshot(Gemma4Snapshot & s);

// Forward: run a single step (prefill chunk or decode token).
// Returns logits for last token.
bool gemma4_step(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    int                     n_tokens,
    int                     kv_start,
    std::vector<float> &    out_logits);

}  // namespace dflash::common
