"""Tiny random-weight Kolibri 1 checkpoints for the plugin tests."""

import json
import os

import torch
from safetensors.torch import save_file

KOLIBRI1_CONFIG = {
    "architectures": ["Kolibri1ForCausalLM"],
    "model_type": "kolibri1",
    "hidden_size": 256,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "hidden_act": "silu",
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-6,
    "vocab_size": 96000,
    "rope_theta": 10000.0,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "shared_expert_intermediate_size": 256,
    "norm_topk_prob": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "use_sliding_window": True,
    "sliding_window": 65,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "full_attention",
    ],
    "bos_token_id": 95900,
    "eos_token_id": 95901,
    "pad_token_id": 95902,
    "tie_word_embeddings": False,
    "dtype": "bfloat16",
    "use_cache": True,
}


def _attention_and_norms(tensors: dict, prefix: str, cfg: dict) -> None:
    H = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]

    tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, H)
    tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(n_kv * head_dim, H)
    tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(n_kv * head_dim, H)
    tensors[f"{prefix}.self_attn.q_norm.weight"] = torch.ones(head_dim)
    tensors[f"{prefix}.self_attn.k_norm.weight"] = torch.ones(head_dim)
    tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(H, n_heads * head_dim)

    tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(H)
    tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(H)
    tensors[f"{prefix}.post_attn_norm.weight"] = torch.ones(H)
    tensors[f"{prefix}.post_ffn_norm.weight"] = torch.ones(H)


def _moe_block(tensors: dict, prefix: str, cfg: dict) -> None:
    H = cfg["hidden_size"]
    n_experts = cfg["num_experts"]
    moe_inter = cfg["moe_intermediate_size"]
    shared_inter = cfg["shared_expert_intermediate_size"]

    tensors[f"{prefix}.mlp.gate.weight"] = torch.randn(n_experts, H)
    tensors[f"{prefix}.mlp.shared_experts.gate_proj.weight"] = torch.randn(
        shared_inter, H
    )
    tensors[f"{prefix}.mlp.shared_experts.up_proj.weight"] = torch.randn(
        shared_inter, H
    )
    tensors[f"{prefix}.mlp.shared_experts.down_proj.weight"] = torch.randn(
        H, shared_inter
    )
    for e in range(n_experts):
        ep = f"{prefix}.mlp.experts.{e}"
        tensors[f"{ep}.gate_proj.weight"] = torch.randn(moe_inter, H)
        tensors[f"{ep}.up_proj.weight"] = torch.randn(moe_inter, H)
        tensors[f"{ep}.down_proj.weight"] = torch.randn(H, moe_inter)


def _write(tensors: dict, cfg: dict, model_dir: str) -> str:
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    save_file(tensors, os.path.join(model_dir, "model.safetensors"))
    return model_dir


def write_kolibri1_checkpoint(model_dir: str, cfg: dict = KOLIBRI1_CONFIG) -> str:
    H = cfg["hidden_size"]

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(cfg["vocab_size"], H),
        "model.norm.weight": torch.ones(H),
        "lm_head.weight": torch.randn(cfg["vocab_size"], H),
    }
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        _attention_and_norms(tensors, p, cfg)
        _moe_block(tensors, p, cfg)
        # torchtitan checkpoint name for the routing correction bias.
        tensors[f"{p}.moe.router.expert_bias"] = torch.randn(cfg["num_experts"])
    return _write(tensors, cfg, model_dir)
