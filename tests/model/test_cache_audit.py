# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
from types import SimpleNamespace

import pytest
import torch
from aleph_alpha_inference.model.cache_audit import audit_cache


def layer(cache, source=None):
    return SimpleNamespace(kv_cache=cache, kv_source_name=source)


def test_cache_audit_counts_physical_storage_once():
    storage = torch.empty(100, dtype=torch.uint8)
    context = {
        "producer": layer(storage[:60]),
        "consumer": layer(torch.empty(0), "producer"),
        "swa": layer(storage[60:]),
        "unused": layer(torch.empty(0)),
    }
    result = audit_cache(context)
    assert result["physical_layers"] == 2
    assert result["aliases"] == {"consumer": "producer"}
    assert result["unique_storage_bytes"] == 100
    assert result["owner_view_bytes"] == 100


def test_cache_audit_rejects_consumer_allocation():
    with pytest.raises(RuntimeError, match="cache of its own"):
        audit_cache(
            {
                "producer": layer(torch.empty(10)),
                "consumer": layer(torch.empty(10), "producer"),
            }
        )


def test_cache_audit_rejects_source_without_cache():
    with pytest.raises(RuntimeError, match="has no cache"):
        audit_cache(
            {
                "producer": layer(torch.empty(0)),
                "consumer": layer(torch.empty(0), "producer"),
            }
        )
