# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""SWA bounded replay for Kolibri encoder-decoder layouts on vLLM's DeepSeek-V4.1
machinery.

Layers up to the last KV producer (the encoder) own SWA KV and write the shared
global KV; the layers after it (the decoder) own only SWA KV. vLLM's bounded
replay runs the decoder on each prefill's trailing window and rebuilds SWA KV
after a prefix hit by recomputing the hit's tail. DeepSeek clips both to one
sliding window; Kolibri uses the exact dependency spans below, so the final
outputs and the KV later steps read equal those of full computation.
"""

from dataclasses import dataclass

from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from aleph_alpha_inference.kernels.sparse_layout import SparseCheckpointLayout

# The K/V producer layers (0-based) of the encoder-decoder layouts that
# bounded replay supports: Kolibri 1.1's.
_REPLAY_KV_PRODUCER_LAYERS = ((4, 19, 24, 29),)


@dataclass(frozen=True)
class ReplayLayout:
    """Where the decoder starts and how many tokens each replay must cover."""

    # First (0-based) decoder replay layer: the last KV producer.
    decoder_start: int
    # Trailing rows of a prefill the decoder replay layers run on.
    decoder_window: int
    # Hit tokens recomputed after a prefix hit to rebuild every SWA cache.
    prefix_replay_tokens: int


def required_replay_tokens(window, swa_layers):
    """Cover the last row's dependencies and the SWA KV the next step reads.

    Shared-global attention and FFNs are position-local with respect to their
    inputs. Each SWA layer extends the required input suffix by W - 1. Its
    next-token KV cache needs W - 1 input rows from the preceding layers,
    which the same bound covers.
    """
    if window < 1 or swa_layers < 0:
        raise ValueError("Replay window must be positive and SWA count nonnegative")
    return 1 + swa_layers * (window - 1)


def bounded_replay_enabled(vllm_config: VllmConfig) -> bool:
    """Whether the deployment serves with SWA bounded replay.

    vLLM's `swa_bounded_replay` switch turns it on for the layouts that
    support it.
    """
    layout = SparseCheckpointLayout.from_hf_config(
        vllm_config.model_config.hf_text_config
    )
    return (
        layout.share_kv
        and layout.kv_producer_layers in _REPLAY_KV_PRODUCER_LAYERS
        and vllm_config.cache_config.swa_bounded_replay
    )


def replay_layout(hf_config: PretrainedConfig) -> ReplayLayout:
    # The decoder starts at the last K/V producer.
    decoder_start = max(
        SparseCheckpointLayout.from_hf_config(hf_config).kv_producer_layers
    )
    window = hf_config.sliding_window
    is_swa = [kind == "sliding_attention" for kind in hf_config.layer_types]
    return ReplayLayout(
        decoder_start,
        required_replay_tokens(window, sum(is_swa[decoder_start:])),
        # Rebuilding the KV after a hit needs no extra row for its output.
        required_replay_tokens(window, sum(is_swa)) - 1,
    )


@dataclass(frozen=True, kw_only=True)
class BoundedReplaySWASpec(SlidingWindowSpec):
    """SWA KV kept out of prefix caching and rebuilt by replaying the hit's last
    `replay_tokens` tokens (`Request.replay_start`)."""

    replay_tokens: int

    @property
    def prefix_cacheable(self) -> bool:
        return False

    @property
    def prefix_replay_tokens(self) -> int:
        return self.replay_tokens


class BoundedReplaySWAManager(SlidingWindowManager):
    """Keeps the blocks a prefix hit's replay writes and reads.

    The scheduler admits a hit with all its tokens computed; the replayed ones
    are recomputed from `replay_start`, whose window must stay allocated.
    SlidingWindowManager would retire it as below the hit's window. DeepSeek's
    one-window replay stays inside the block holding the window start.
    """

    kv_cache_spec: BoundedReplaySWASpec

    def _replay_start(self, num_computed_tokens):
        return max(num_computed_tokens - self.kv_cache_spec.replay_tokens, 0)

    def get_num_blocks_to_allocate(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        total_computed_tokens,
        *args,
        **kwargs,
    ):
        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            self._replay_start(total_computed_tokens),
            *args,
            **kwargs,
        )

    def add_local_computed_blocks(
        self,
        request_id,
        new_computed_blocks,
        num_local_computed_tokens,
        num_external_computed_tokens,
    ):
        # Not prefix-cacheable: a hit brings no blocks of this group, so the
        # computed count only sizes the retired prefix.
        assert not new_computed_blocks
        super().add_local_computed_blocks(
            request_id,
            new_computed_blocks,
            self._replay_start(num_local_computed_tokens),
            num_external_computed_tokens,
        )


def register_bounded_replay_spec():
    # Registering first would leave vLLM's built-in specs unregistered.
    KVCacheSpecRegistry._ensure_registered()
    KVCacheSpecRegistry.register(
        BoundedReplaySWASpec,
        BoundedReplaySWAManager,
        uniform_type_base_spec=BoundedReplaySWASpec,
    )
