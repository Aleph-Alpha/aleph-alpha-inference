# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Attention layers that read another layer's KV cache.

Like DeepSeek-V4.1's KV-sharing layers, a consumer declares no KV cache spec,
so vLLM plans, allocates and builds metadata only for its source. At run time
it reads the source's cache and attention metadata from the forward context.
"""

import torch
from torch.library import Library
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import (
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    LayerNameType,
)

# Own op namespace, so plugin ops cannot collide with vLLM's `vllm::` ops.
_LIBRARY = Library("aleph_alpha_inference", "FRAGMENT")
OP_NAME = "aleph_alpha_inference::shared_kv_attention"


class SharedKVAttention(Attention):
    """Attention over `kv_source_name`'s KV cache, which it never writes."""

    def __init__(self, *args, kv_source_name, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_source_name = kv_source_name
        compilation = get_current_vllm_config().compilation_config
        # Piecewise graphs must break at this attention like at vLLM's own.
        if (
            compilation.splitting_ops_contain_attention()
            and OP_NAME not in compilation.splitting_ops
        ):
            compilation.splitting_ops.append(OP_NAME)

    def get_kv_cache_spec(self, vllm_config):
        return None

    def forward(self, query, key=None, value=None):
        if key is not None or value is not None:
            raise ValueError("A KV-sharing layer writes no KV")
        output = query.new_empty(query.shape[0], self.num_heads * self.head_size_v)
        torch.ops.aleph_alpha_inference.shared_kv_attention(
            query.view(-1, self.num_heads, self.head_size),
            output.view(-1, self.num_heads, self.head_size_v),
            _encode_layer_name(self.layer_name),
        )
        return output


@eager_break_during_capture
def shared_kv_attention(
    query: torch.Tensor, output: torch.Tensor, layer_name: LayerNameType
) -> None:
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    metadata, _, kv_cache, _ = get_attention_context(layer.kv_source_name)
    layer.impl.forward(layer, query, None, None, kv_cache, metadata, output=output)


direct_register_custom_op(
    op_name="shared_kv_attention",
    op_func=shared_kv_attention,
    mutates_args=["output"],
    target_lib=_LIBRARY,
)
