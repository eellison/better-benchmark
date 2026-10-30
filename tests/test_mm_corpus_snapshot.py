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
            metrics = p["partition_metrics"]
            assert metrics["mm"] <= limits.max_mm
            assert metrics["contraction_depth"] <= limits.max_stages
            for stage in metrics.get("vertical", {}).get("stages", []):
                assert stage["R"] <= limits.reduction_limit
                assert stage["F"] <= limits.intermediate_limit
    assert sum(len(p) for p in points.values()) == report["shape_points"] == 1019
    assert len(families) == report["families"]
    rows = json.loads((corpus / "occurrences.json").read_text())
    for row in rows:
        assert (ROOT / row["source"]).is_file()
        for region in row["regions"]:
            if region.get("family_member"):
                # Folded into its family representative: no exported shape.
                assert region["pattern_hash"] in points and region["shape_hash"] is None
            else:
                assert region["shape_hash"] in points[region["pattern_hash"]]
    assert sum(len(r["regions"]) for r in rows) == report["region_occurrences"]
    for manifest in (corpus / "models").rglob("manifest.json"):
        assert set(json.loads(manifest.read_text())["patterns"]) <= points.keys()
