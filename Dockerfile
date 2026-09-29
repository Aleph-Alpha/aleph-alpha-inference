# vLLM's OpenAI-compatible server with the aleph-alpha-inference plugin.
# The release workflow builds the wheel into dist/ and passes the vLLM version
# the package supports; the default here is only for local builds.
ARG VLLM_VERSION=0.29.0
FROM vllm/vllm-openai:v${VLLM_VERSION}

# --no-deps: the base image already has the vllm, torch and transformers this
# release supports, and pip must not replace them.
COPY dist/*.whl /tmp/
RUN pip install --no-deps /tmp/*.whl && rm /tmp/*.whl

# The base image's entrypoint stays `vllm serve`, so the image is used like
# vllm/vllm-openai: docker run ... <image> <model> [vllm serve options]
