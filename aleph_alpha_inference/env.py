# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Read the plugin's environment variables without failing on a bad value.

vLLM loads the plugin in every process it starts, so an exception here would
take the whole server down over a typo. A value that cannot be used is
reported with a warning, and the default applies instead.
"""

import logging
import os
from collections.abc import Sequence

# A child of vLLM's logger, without importing vLLM.
logger = logging.getLogger(f"vllm.{__name__}")


def env_choice(name: str, choices: Sequence[str], default: str) -> str:
    """The value of the environment variable `name` if it is one of `choices`,
    else `default`, with a warning unless the variable is unset."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value not in choices:
        logger.warning(
            "Ignoring %s=%r: must be one of %s. Using the default %r",
            name,
            value,
            ", ".join(choices),
            default,
        )
        return default
    return value


def env_int(name: str, default: int) -> int:
    """The environment variable `name` as an integer, or `default`, with a
    warning, if it is set to something else."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: must be an integer. Using the default %d",
            name,
            value,
            default,
        )
        return default
