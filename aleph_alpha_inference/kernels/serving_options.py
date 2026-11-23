# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Serving options of sparse checkpoints.

The options describe a deployment, not the model, so the checkpoint config
does not state them. Set them in vLLM's `additional_config`, e.g.
`--additional-config '{"prefix_caching": true}'`.
"""

from pydantic import BaseModel, ConfigDict
from vllm.config import VllmConfig


class SparseServingOptions(BaseModel):
    """How a deployment serves a sparse checkpoint.

    Attributes:
        prefix_caching: Whether vLLM's prefix caching stays on. It pays off
            only for very long shared prefixes.
    """

    # `strict` rejects values of another type, e.g. the string "false".
    # `extra` skips the entries that are no serving options.
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")

    prefix_caching: bool = False


def sparse_serving_options(vllm_config: VllmConfig) -> SparseServingOptions:
    """Read the serving options of a deployment.

    Args:
        vllm_config: vLLM's config, with the options in its
            `additional_config`.

    Returns:
        The options, with defaults for those not stated.

    Raises:
        pydantic.ValidationError: An option has a value it does not support.
            This is a `ValueError`.
    """
    # vLLM also allows a platform-specific object here, which states no options.
    if not isinstance(vllm_config.additional_config, dict):
        return SparseServingOptions()
    return SparseServingOptions.model_validate(vllm_config.additional_config)
