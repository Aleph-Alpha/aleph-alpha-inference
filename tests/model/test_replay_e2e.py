# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""SWA bounded replay serves the outputs of full computation.

A reduced Kolibri 1 (production layout and attention, small MLPs and vocab)
serves long prompts in chunked prefill, once with bounded replay and prefix
caching and once computing every row of every layer without prefix caching.
Both run eager (mode 0) or compiled with piecewise CUDA graphs (mode 3); a
short prompt runs in a piecewise graph. A third compilation asks for full CUDA
graphs, as chat deployments do. The sparse attention captures decodes
only, so vLLM runs every prefill outside the graphs. Two short prompts run
together.

Replay chunks a prompt from its replay start, full computation from 0, so a
row can land in chunks of different sizes. Small chunks make the dense GEMMs
(e.g. the 6144-wide output projection) pick other cuBLAS kernels per row count,
which changes the row's bits. All runs therefore chunk at 12288 tokens.

Bounded replay without prefix caching also runs as chat deployments
serve; it replays every prompt from 0.
"""

import json
import os
import random
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from aleph_alpha_inference.kernels import serving_compressed


pytestmark = pytest.mark.gpu

NUM_LAYERS = 50
# One-based: the layer whose K/V each full-attention layer reads.
KV_SOURCES = {5: 5, 10: 5, 15: 5, 20: 20, 25: 25, 30: 30, 35: 30, 40: 30, 45: 30, 50: 5}
VOCAB = 4096
# Chunks of 12288 trim the decoder to its 8193-row window; the last chunk of a
# 30011-token prompt does not and reads the KV the trimmed chunk left.
PROMPT_LENGTH = 30011
SHARED_PREFIX = 28800
CHUNK = 12288
SHORT_PROMPT = 400
# Two prompts in flight together: their decodes share one CUDA graph.
CONCURRENT_PROMPT = 200
# How each engine run serves: bounded replay and prefix caching.
SETUPS: dict[str, dict[str, bool]] = {
    "full_computation": {"swa_bounded_replay": False, "enable_prefix_caching": False},
    "replay": {"swa_bounded_replay": True, "enable_prefix_caching": True},
    "replay_without_prefix_caching": {
        "swa_bounded_replay": True,
        "enable_prefix_caching": False,
    },
}
COMPILATION = {
    "eager": {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4],
    },
    "compiled": {
        "mode": 3,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": [1, 2, 4, 512],
    },
    "full": {
        "mode": 0,
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": [1, 2, 4, 512],
    },
}


def model_config():
    full = set(range(5, NUM_LAYERS + 1, 5))
    return {
        "architectures": ["Kolibri1_1ForCausalLM"],
        "model_type": "kolibri1",
        "hidden_size": 512,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 48,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_act": "silu",
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-06,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128,
        "norm_topk_prob": False,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "use_sliding_window": True,
        "sliding_window": 513,
        "layer_types": [
            "full_attention" if i in full else "sliding_attention"
            for i in range(1, NUM_LAYERS + 1)
        ],
        "eos_token_id": VOCAB - 1,
        "pad_token_id": VOCAB - 1,
        # Indexers 5/15/20/30 and K/V owners 5/20/25/30 (one-based); layer 50
        # reads the K/V of layer 5. The config keys are zero-based.
        "index_layers": [4, 14, 19, 29],
        "kv_source_layers": [
            KV_SOURCES.get(layer, layer) - 1 for layer in range(1, NUM_LAYERS + 1)
        ],
        "index_topk": 2048,
        "index_pool_size": 16384,
        "index_compression": 16,
        "index_num_local_tokens": 128,
        "index_num_sink_tokens": 8,
        "index_num_heads": 12,
        "kv_cache_qat": True,
        "dtype": "bfloat16",
    }


def init_weights(model):
    """Seeded weights by name, scaled so every layer moves the residual."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            generator = torch.Generator(param.device)
            generator.manual_seed(zlib.crc32(name.encode()))
            if param.ndim == 1:
                param.fill_(1.0 if "norm" in name else 0.0)
            else:
                param.normal_(0, param.shape[-1] ** -0.5, generator=generator)


def prompts() -> list[list[list[int]]]:
    rng = random.Random(7)
    prompt = [rng.randrange(VOCAB - 1) for _ in range(PROMPT_LENGTH)]
    other = prompt[:SHARED_PREFIX] + [rng.randrange(VOCAB - 1) for _ in range(1500)]
    short = [rng.randrange(VOCAB - 1) for _ in range(SHORT_PROMPT)]
    concurrent = [
        [rng.randrange(VOCAB - 1) for _ in range(CONCURRENT_PROMPT)] for _ in range(2)
    ]
    # Run in order: the second and third prompts hit the first one's cache.
    # The last group runs together.
    return [[prompt], [other], [prompt], [short], concurrent]


def serve(model_dir: str, setup: str, compilation: str) -> list[dict[str, Any]]:
    # Union-tile prefill groups each chunk's query rows into tiles counted from
    # the chunk start, so a row's rounding depends on how the prompt is chunked,
    # and replay chunks from its replay start. The per-row kernel depends only
    # on the row and the cache, so replay must then match full computation
    # bit for bit. The engine runs in this process (see run_engine).
    serving_compressed.SERVING_UNION_SPAN = 0
    llm = LLM(
        model=model_dir,
        load_format="dummy",
        skip_tokenizer_init=True,
        max_model_len=32768,
        max_num_batched_tokens=CHUNK,
        max_num_seqs=4,
        gpu_memory_utilization=0.4,
        kv_cache_dtype="bfloat16",
        # Opt in; the setup's engine flag still turns caching off per run.
        additional_config={"prefix_caching": True},
        **SETUPS[setup],
        # FlashInfer's MoE needs a JIT toolchain; any one backend serves both.
        kernel_config={"moe_backend": "triton"},
        compilation_config=COMPILATION[compilation],
    )
    llm.apply_model(init_weights)
    params = SamplingParams(max_tokens=4, temperature=0.0, logprobs=8)
    results = []
    for group in prompts():
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt) for prompt in group], params
        )
        for output in outputs:
            completion = output.outputs[0]
            results.append(
                {
                    "tokens": list(completion.token_ids),
                    "logprobs": [
                        {str(t): lp.logprob for t, lp in sorted(step.items())}
                        for step in completion.logprobs
                    ],
                    "cached": output.num_cached_tokens,
                }
            )
    return results


def run_engine(tmp_path: Path, setup: str, compilation: str) -> list[dict[str, Any]]:
    out = tmp_path / f"{setup}-{compilation}.json"
    if out.exists():  # Tests share the module's engine runs.
        return json.loads(out.read_text())
    env = {
        **os.environ,
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_USE_DEEP_GEMM": "0",
        # In-process engine, so serve() can select the per-row prefill kernel.
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }
    subprocess.run(
        [
            sys.executable,
            __file__,
            str(tmp_path / "model"),
            setup,
            compilation,
            str(out),
        ],
        env=env,
        check=True,
        timeout=3600,
    )
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("sparse-replay")
    (path / "model").mkdir()
    (path / "model" / "config.json").write_text(json.dumps(model_config()))
    return path


@pytest.mark.parametrize("compilation", ["eager", "compiled", "full"])
def test_replay_and_prefix_hits_match_full_computation(
    model_dir: Path, compilation: str
) -> None:
    replayed = run_engine(model_dir, "replay", compilation)
    full = run_engine(model_dir, "full_computation", compilation)
    # Both later prompts hit the first one's blocks and replay 20480 tokens.
    assert [r["cached"] for r in replayed] == [
        0,
        28800 - 28800 % 128,
        29952,
        0,
        0,
        0,
    ]
    assert [r["cached"] for r in full] == [0] * 6
    for actual, expected in zip(replayed, full):
        assert actual["tokens"] == expected["tokens"]
        assert actual["logprobs"] == expected["logprobs"]


@pytest.mark.parametrize("compilation", ["compiled", "full"])
def test_replay_without_prefix_caching_matches_full_computation(
    model_dir: Path, compilation: str
) -> None:
    replayed = run_engine(model_dir, "replay_without_prefix_caching", compilation)
    full = run_engine(model_dir, "full_computation", compilation)
    assert [r["cached"] for r in replayed] == [0] * 6
    for actual, expected in zip(replayed, full):
        assert actual["tokens"] == expected["tokens"]
        assert actual["logprobs"] == expected["logprobs"]


if __name__ == "__main__":
    path, setup, compilation, out = sys.argv[1:]
    Path(out).write_text(json.dumps(serve(path, setup, compilation)))
