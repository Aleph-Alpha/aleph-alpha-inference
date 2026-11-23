# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
"""Check physical cache sharing once, after vLLM binds its allocations."""


def audit_cache(context):
    aliases = {
        name: layer.kv_source_name
        for name, layer in context.items()
        if getattr(layer, "kv_source_name", None)
    }
    owners = {
        name: layer
        for name, layer in context.items()
        if hasattr(layer, "kv_cache") and layer.kv_cache.numel()
    }
    for name, source in aliases.items():
        if name in owners:
            raise RuntimeError(f"KV-sharing layer {name} has a cache of its own")
        if source not in owners:
            raise RuntimeError(f"KV source {source} of {name} has no cache")
    storages = {
        layer.kv_cache.untyped_storage()
        .data_ptr(): layer.kv_cache.untyped_storage()
        .nbytes()
        for layer in owners.values()
    }
    return {
        "physical_layers": len(owners),
        "aliases": aliases,
        "unique_storage_bytes": sum(storages.values()),
        "owner_view_bytes": sum(layer.kv_cache.nbytes for layer in owners.values()),
        "cache_shapes": {
            name: list(layer.kv_cache.shape) for name, layer in owners.items()
        },
    }
