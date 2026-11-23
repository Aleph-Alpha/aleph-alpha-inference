# vLLM's OpenAI-compatible server with the aleph-alpha-inference plugin.
# The release workflow builds the wheel into dist/ and passes the vLLM version
# the package supports; the default here is only for local builds.
ARG VLLM_VERSION=0.33.0
FROM vllm/vllm-openai:v${VLLM_VERSION}

# The native Kolibri 1.1 kernels are built on first use. They need a host
# compiler, CUDA's unversioned cudart and CCCL headers, and the cuSPARSE and
# cuSOLVER headers that torch's ATen/cuda headers include. The image's CUDA
# runtime packages lack all of them.
RUN cuda=$(echo "$CUDA_VERSION" | cut -d. -f1,2 | tr . -) \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        "cuda-cudart-dev-$cuda" "cuda-cccl-$cuda" \
        "libcusparse-dev-$cuda" "libcusolver-dev-$cuda" \
    && rm -rf /var/lib/apt/lists/*

# A volume mounted at /tmp/aleph-alpha-inference keeps the compiled native
# extensions across restarts.
ENV TORCH_EXTENSIONS_DIR=/tmp/aleph-alpha-inference/torch_extensions
# Kolibri FP8 weights carry fp32 block scales, which vLLM's DeepGEMM path
# would round to powers of two.
ENV VLLM_USE_DEEP_GEMM=0
ENV VLLM_USE_DEEP_GEMM_E8M0=0

# The base image already has the vllm, torch and transformers this release
# supports. uv keeps the base packages that satisfy the plugin's requirements
# and installs the missing ones (the Kolibri 1.1 kernel libraries) at their
# lower bounds. The build fails if a base package changes anyway.
COPY dist/*.whl /tmp/
RUN uv pip freeze --system | sort > /tmp/base.txt \
    && uv pip install --system --resolution lowest-direct /tmp/*.whl \
    && uv pip freeze --system | sort > /tmp/plugin.txt \
    && echo "Added by the plugin:" && comm -13 /tmp/base.txt /tmp/plugin.txt \
    && comm -23 /tmp/base.txt /tmp/plugin.txt > /tmp/changed.txt \
    && if [ -s /tmp/changed.txt ]; then \
        echo "The plugin's requirements change these base packages:" >&2; \
        cat /tmp/changed.txt >&2; exit 1; \
    fi \
    && rm /tmp/*.whl /tmp/base.txt /tmp/plugin.txt /tmp/changed.txt

# The base image's entrypoint stays `vllm serve`, so the image is used like
# vllm/vllm-openai: docker run ... <image> <model> [vllm serve options]
