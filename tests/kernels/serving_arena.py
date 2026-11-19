# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Compressed serving layers on routing arenas, as `Block8Impl` builds them."""

import itertools

_owners = itertools.count()


def new_owner():
    """A workspace owner no other test uses, so its arenas start private."""
    return ("test", next(_owners))


def decode_scratch(layer):
    """Decode attention scratch for every row and split of `layer`."""
    import torch
    from aleph_alpha_inference.kernels.sparse_decode import workspace_shapes

    return tuple(
        torch.empty(shape, device=layer.table.device, dtype=torch.float32)
        for shape in workspace_shapes(layer.rows, layer.attention.splits)
    )


def served_layer(
    cache, table, rows, span, topk, decode, *, owner, source, pool_tokens=16384
):
    """A `ServingCompressedLayer` on the arena of `owner`.

    A `source` layer gets the owner's source pool; a reindex layer receives
    its pool at run time.
    """
    from aleph_alpha_inference.kernels.serving_compressed import (
        routing_workspace,
        ServingCompressedLayer,
        source_pool,
    )

    device = table.device
    return ServingCompressedLayer(
        cache,
        table,
        rows,
        span,
        topk,
        decode,
        scratch=routing_workspace(device, owner, rows, span, topk, decode, pool_tokens),
        source_pool=(
            source_pool(device, owner, ("source", decode), rows, pool_tokens)
            if source
            else None
        ),
        pool_tokens=pool_tokens,
    )
