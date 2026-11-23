# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from aleph_alpha_inference.model import vllm_model

from tensor_asserts import assert_equal


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (9, 0)])
def test_supported_gpus_serve(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    vllm_model._require_supported_gpu()


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (12, 0)])
def test_other_gpus_are_rejected(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    with pytest.raises(ValueError, match="requires an SM100 GPU"):
        vllm_model._require_supported_gpu()


class _AttentionWithoutCache(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.impl = SimpleNamespace()


def _model(monkeypatch):
    context = {"source": object(), "consumer": object()}

    def attention(vllm_config, name, *args, kv_source_name):
        module = _AttentionWithoutCache()
        context[name] = module
        return module

    monkeypatch.setattr(vllm_model, "sparse_attention", attention)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=context),
        cache_config=None,
        quant_config=None,
    )
    layers = []
    for role, name in (("indexer", "source"), ("reindexer", "consumer")):
        original = SimpleNamespace(
            qkv_proj=nn.Linear(4, 12, bias=False),
            o_proj=nn.Linear(4, 4, bias=False),
            q_norm=nn.RMSNorm(4),
            k_norm=nn.RMSNorm(4),
            q_size=4,
            kv_size=4,
            head_dim=4,
            num_heads=1,
            num_kv_heads=1,
            attn=SimpleNamespace(layer_name=name),
        )
        layer = nn.Module()
        layer.self_attn = vllm_model.NativeSparseAttention(
            original,
            vllm_config=config,
            source_name="source",
            role=role,
            share_kv=True,
        )
        context[name] = layer.self_attn.attn
        layers.append(layer)
    model = vllm_model.Kolibri1_1ForCausalLM.__new__(vllm_model.Kolibri1_1ForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList(layers)
    return model


def test_native_loader_accepts_shared_kv_without_unused_norm(monkeypatch):
    model = _model(monkeypatch)
    assert model.model.layers[0].self_attn.k_norm is not None
    assert model.model.layers[1].self_attn.k_norm is None
    weights = {
        name: torch.full_like(value, 0.375)
        for name, value in model.state_dict().items()
    }
    loaded = model.load_weights(weights.items())
    assert loaded == set(weights)
    for name, value in model.state_dict().items():
        assert_equal(value, weights[name])


def test_native_loader_still_rejects_other_unexpected_weights(monkeypatch):
    model = _model(monkeypatch)
    with pytest.raises(ValueError):
        model.load_weights(
            [("model.layers.0.self_attn.unexpected.weight", torch.ones(4))]
        )


def test_native_loader_accepts_dense_checkpoint_with_unused_shared_norm(monkeypatch):
    model = _model(monkeypatch)
    weights = {
        name: torch.full_like(value, 0.625)
        for name, value in model.state_dict().items()
    }
    legacy_norm = "model.layers.1.self_attn.k_norm.weight"
    weights[legacy_norm] = torch.full((4,), 17.0)
    loaded = model.load_weights(iter(weights.items()))
    assert loaded == set(weights) - {legacy_norm}
    for name, value in model.state_dict().items():
        assert_equal(value, weights[name])
