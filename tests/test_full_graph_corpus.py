"""Every full graph in the shipped corpus builds its inputs.

Inputs are allocated on the meta device so the sweep is cheap; one graph per
alias-group layout is also built for real on CPU.
"""
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import full_graph_harness as F  # noqa: E402

GRAPHS = sorted((ROOT / "repros" / "models").glob("*/*/*/full_graph_*.py"))


@pytest.mark.skipif(not GRAPHS, reason="no corpus full graphs in this checkout")
def test_every_corpus_full_graph_builds_inputs():
    failures = {}
    with mock.patch.object(F, "_resolve_device", lambda requested, default: torch.device("meta")):
        for graph in GRAPHS:
            try:
                F.prepare_full_graph_execution(graph, default_device="cpu", dynamic=False).args()
            except Exception as e:  # noqa: BLE001 - collect every failure
                failures[str(graph.relative_to(ROOT))] = f"{type(e).__name__}: {e}"[:200]
    assert not failures, failures


def test_alias_group_without_recorded_capacity_uses_member_extent():
    # Sidecars written before storage_nbytes have storage groups only.
    graph = ROOT / "repros/models/timm/train/deit_tiny_patch16_224.fb_in1k/full_graph_001.py"
    if not graph.exists():
        pytest.skip("corpus graph not present")
    execution = F.prepare_full_graph_execution(graph, default_device="cpu", dynamic=False)
    specs = [s for s in execution.definition.input_specs if s.get("alias_group") is not None]
    assert specs and all(s.get("storage_nbytes") is None for s in specs)
    args = execution.args()
    storages = {}
    for arg, spec in zip(args, execution.definition.input_specs):
        if spec.get("alias_group") is not None:
            storages.setdefault(spec["alias_group"], set()).add(arg.untyped_storage().data_ptr())
    assert all(len(ptrs) == 1 for ptrs in storages.values()), storages
    floats = [a for a, s in zip(args, execution.definition.input_specs)
              if s.get("alias_group") is not None and a.is_floating_point()]
    assert floats and all(a.float().abs().mean() > 0.1 for a in floats)
