# aleph-alpha-inference

A [vLLM](https://github.com/vllm-project/vllm) plugin that serves Aleph Alpha
models. It adds:

| Model | Architecture | Reasoning parser | Tool call parser |
|---|---|---|---|
| [Kolibri 1](https://huggingface.co/Aleph-Alpha/Kolibri-1) | `Kolibri1ForCausalLM` | `kolibri1` | `kolibri1` |
| [Kolibri 1.1](https://huggingface.co/Aleph-Alpha/Kolibri-1.1) | `Kolibri1_1ForCausalLM` | `kolibri1` | `kolibri1` |

Kolibri 1.1 is Kolibri 1 with sparse attention: its full-attention layers
attend over a selection of the context. The plugin brings the kernels for it,
a bounded-replay sliding-window KV cache and the serving defaults it needs.

vLLM finds the plugin through its `vllm.general_plugins` entry point, so
installing the package is all the setup there is.

## Install

Each release supports one vLLM minor version, currently **vLLM 0.33**.

```sh
pip install aleph-alpha-inference
```

This installs the supported vLLM as well. To serve Kolibri 1 from an existing
vLLM 0.33 installation, such as the `vllm/vllm-openai:v0.33.0` image, without
touching its vLLM, torch or transformers:

```sh
pip install --no-deps aleph-alpha-inference
```

Kolibri 1.1 needs the package's other requirements too (`flash-attn-4`,
`nvidia-cutlass-dsl`, `quack-kernels`, `flashinfer-python`), so install it
with its dependencies there.

### Requirements of Kolibri 1.1

Kolibri 1.1 runs on NVIDIA GPUs with compute capability 10.x (e.g. B200) or
9.0 (H100). Kolibri 1 runs wherever vLLM does.

Some of its kernels are CUDA C++ extensions that compile on first use, which
takes a few minutes. They need a C++20 compiler and a CUDA 13 toolkit with
`nvcc`; without one, the `cuda` extra installs it:

```sh
pip install "aleph-alpha-inference[cuda]"
```

The plugin finds the extra's toolkit itself when `CUDA_HOME` is unset and
there is no other toolkit. The extra's libraries have only their versioned
names (`libcudart.so.13`), so the plugin adds the unversioned ones the build
links with (`libcudart.so`, `libnvrtc.so`, ...). The links are kept under
`~/.cache/aleph_alpha_inference` (or `$XDG_CACHE_HOME`) and passed to the
linker through `LIBRARY_PATH`. This also covers the kernels FlashInfer builds
on first use. A toolkit you choose with `CUDA_HOME` takes precedence.

The `vllm/vllm-openai` image has a compiler and `nvcc`, but its CUDA toolkit
lacks headers that torch's CUDA headers include, such as `cusparse.h`.
Install them before serving Kolibri 1.1 from that image:

```sh
apt-get install cuda-cccl-13-0 libcusparse-dev-13-0 libcusolver-dev-13-0
```

The compiled extensions go to `TORCH_EXTENSIONS_DIR` (torch's variable,
default `~/.cache/torch_extensions`). Keep that directory across restarts to
compile only once.

## Serve

```sh
vllm serve Aleph-Alpha/Kolibri-1 \
  --kv-cache-dtype fp8 \
  --reasoning-parser kolibri1 \
  --tool-call-parser kolibri1 \
  --enable-auto-tool-choice
```

For the BF16 weights, serve `Aleph-Alpha/Kolibri-1-BF16` and drop
`--kv-cache-dtype fp8`.

Kolibri 1.1 is served the same way, without `--kv-cache-dtype`: its sparse
layers keep their keys and values in their own 4-bit format.

```sh
vllm serve Aleph-Alpha/Kolibri-1.1 \
  --reasoning-parser kolibri1 \
  --tool-call-parser kolibri1 \
  --enable-auto-tool-choice
```

The plugin sets the vLLM options Kolibri 1.1 depends on: 128-token KV cache
blocks, a BF16 KV cache and FlashAttention 4. It rejects other block sizes
and attention backends. Kolibri 1.1 requires a tensor parallel size of 1; data
and expert parallelism work.

vLLM's prefix caching is off by default for Kolibri 1.1, since it pays off
only for very long shared prefixes. Turn it on with
`--additional-config '{"prefix_caching": true}'`.

Thinking is on by default. Turn it off for a request with
`reasoning_effort: "none"` or `enable_thinking: false`:

```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Aleph-Alpha/Kolibri-1",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

The `kolibri1` reasoning parser reads the same switch as the chat template,
so the response's `reasoning` and `content` are split correctly whichever way
thinking is set.

## Changes to vLLM

The plugin changes vLLM itself in every process it loads into, for every model
served there, not only Kolibri. A variable set to a value it does not accept
is ignored with a warning in the log, and its default applies:

| Variable | Purpose |
|---|---|
| `KOLIBRI_VLLM_DECODE_FAST_PATH` | `1` (default): skip vLLM's KV-cache bookkeeping on decode steps that cross no block boundary. `0`: off. `check`: run the bookkeeping anyway and assert that skipping it would have changed nothing. Installed only on the vLLM minor version it was checked against. |
| `KOLIBRI_VLLM_RELAX_FULL_GC` | `1` (default): run Python's full garbage collection 100x less often. `0`: keep the interpreter's threshold. |
| `KOLIBRI_VLLM_GC_UNTRACK` | `1` (default): keep vLLM's per-request token, block-hash and block lists out of Python's cyclic garbage collection, whose pauses otherwise grow with the context length. `0`: leave them tracked. A backport that vLLM 0.34 ships itself; the switch has no effect there. |
| `KOLIBRI_VLLM_DETERMINISTIC_MOE` | `1` (default): FlashInfer's CUTLASS MoE adds expert outputs in a fixed order, so repeated requests give identical logprobs, at up to 4% of the throughput. With more than one process (TP, PP, EP, multi-node), start with a fresh `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`. `0`: vLLM's fused finalize, which is not deterministic. Installed only on the vLLM minor version it was checked against. |
| `KOLIBRI_VLLM_STRIDE` | vLLM's repetition detection scans the output on every n-th generated token only. Default 32; `1` scans on every token, as vLLM does. |

## Development

```sh
uv run pytest -m "not gpu and not image"   # runs anywhere
uv run pytest -m gpu                       # needs a CUDA GPU supported by vLLM;
                                           # the Kolibri 1.1 kernel tests need
                                           # SM100 or SM90
uv run pytest -m image                     # needs nvcc, as the serving image has
```

## License

[Apache-2.0](LICENSE)
