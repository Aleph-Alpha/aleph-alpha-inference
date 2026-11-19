# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Run Python's full garbage collection 100x less often in vLLM processes.

A full collection scans every tracked container, and a vLLM process serving
long contexts holds each request's token and block-hash lists: at 64k context
and 512 requests one took 0.5-2 s and ran every few seconds, pausing the
engine core and the front-end for a quarter to a third of their time.
Reference cycles are still collected, just rarely.
"""

import gc

from vllm.logger import init_logger

from aleph_alpha_inference.env import env_choice

logger = init_logger(f"vllm.{__name__}")

# "1" (default) relaxes the threshold, "0" keeps Python's.
MODE_ENV = "KOLIBRI_VLLM_RELAX_FULL_GC"
FULL_GC_THRESHOLD = 1000


def relax_full_gc() -> bool:
    """Make full garbage collections rarer for the rest of this process.

    Reads `KOLIBRI_VLLM_RELAX_FULL_GC` ("1", unset or any other value: relax,
    "0": keep the interpreter's setting) and changes the process-wide `gc`
    thresholds.

    Python 3.14 releases with the incremental collector (seen on 3.14.0 to
    3.14.4) have no full-collection threshold: they ignore the value and
    report 0. There is nothing to change on them.

    Returns:
        Whether the threshold was changed.
    """
    mode = env_choice(MODE_ENV, ("0", "1"), "1")
    if mode == "0":
        logger.info("Kolibri full GC relaxation disabled by %s=0", MODE_ENV)
        return False
    threshold0, threshold1, threshold2 = gc.get_threshold()
    gc.set_threshold(threshold0, threshold1, FULL_GC_THRESHOLD)
    if gc.get_threshold()[2] != FULL_GC_THRESHOLD:
        logger.info(
            "This Python has no full GC threshold; Kolibri leaves its GC as it is"
        )
        return False
    logger.info(
        "Kolibri full GC threshold %d -> %d (%s=0 keeps it)",
        threshold2,
        FULL_GC_THRESHOLD,
        MODE_ENV,
    )
    return True


def configure_gc() -> None:
    """Relax full collections and keep vLLM's per-request lists out of the
    cyclic GC."""
    from aleph_alpha_inference.model.vllm_gc_untrack import untrack_request_lists

    relax_full_gc()
    untrack_request_lists()
