# aleph-alpha-inference

A [vLLM](https://github.com/vllm-project/vllm) plugin that serves Aleph Alpha
models. It adds:

| Model | Architecture | Reasoning parser | Tool call parser |
|---|---|---|---|
| [Kolibri 1](https://huggingface.co/Aleph-Alpha/Kolibri-1) | `Kolibri1ForCausalLM` | `kolibri1` | `kolibri1` |

vLLM finds the plugin through its `vllm.general_plugins` entry point, so
installing the package is all the setup there is.

## Install

Each release supports one vLLM minor version, currently **vLLM 0.29**.

```sh
pip install aleph-alpha-inference
```

This installs the supported vLLM as well. To add the plugin to an existing
vLLM 0.29 installation, such as the `vllm/vllm-openai:v0.29.0` image, without
touching its vLLM, torch or transformers:

```sh
pip install --no-deps aleph-alpha-inference
```

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

## Development

```sh
uv run pytest -m "not gpu"   # runs anywhere
uv run pytest -m gpu         # needs a CUDA GPU supported by vLLM
```

## License

[Apache-2.0](LICENSE)
