"""Invariants of the committed MM pattern corpus (mm_repro/)."""

from __future__ import annotations

import json
from pathlib import Path

from matmul_partitioner import CATEGORIES, MatmulLimits

ROOT = Path(__file__).resolve().parents[1]


def test_committed_snapshot_and_source_references():
    corpus = ROOT / "mm_repro"
    report = json.loads((corpus / "report.json").read_text())
    files = sorted((corpus / "canonical").glob("*/*/repro.py"))
    assert len(files) == report["unique_repros"] == 585
    limits = MatmulLimits()
    points, families = {}, set()
    for path in files:
        meta = json.loads(path.with_name("meta.json").read_text())
        assert path.parent.parent.name == meta["pattern_category"]
        assert meta["pattern_category"] in CATEGORIES
        assert meta["pattern_hash_mode"] == "scalar_literals_v1"
        assert path.parent.name.endswith("_" + meta["pattern_hash"])
        family = (meta["pattern_category"], meta["pattern_family"])
        assert family not in families  # one repro per family
        families.add(family)
        shape_points = json.loads(path.with_name("shapes.json").read_text())["points"]
        points[meta["pattern_hash"]] = {p["shape_hash"] for p in shape_points}
        for p in shape_points:
            assert p["models"]
            for model in p["models"]:
                source = ROOT / "repros" / "models" / model
                assert next(source.glob("full_graph_*.py"), None) is not None
            metrics = p["partition_metrics"]
            assert metrics["mm"] <= limits.max_mm
            assert metrics["contraction_depth"] <= limits.max_stages
            for stage in metrics.get("vertical", {}).get("stages", []):
                assert stage["R"] <= limits.reduction_limit
                assert stage["F"] <= limits.intermediate_limit
    assert sum(len(p) for p in points.values()) == report["shape_points"] == 1019
    assert len(families) == report["families"]
    assert report["deduplicated_repros"] == report["family_members"] - len(families)
    assert {p.name for p in corpus.iterdir()} == {"canonical", "report.json", "README.md"}
