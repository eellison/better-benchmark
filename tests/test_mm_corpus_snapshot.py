"""Invariants of the source-graph snapshot committed with the core generator."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_committed_core_snapshot_and_source_references():
    corpus = ROOT / "mm_repro"
    report = json.loads((corpus / "report.json").read_text())
    files = sorted((corpus / "canonical").glob("*/repro.py"))
    assert len(files) == report["unique_repros"] == 276
    points = {}
    for path in files:
        meta = json.loads(path.with_name("meta.json").read_text())
        assert meta["category"] == "core"
        assert meta["pattern_hash_mode"] == "scalar_literals_v1"
        assert path.parent.name.endswith("_" + meta["pattern_hash"])
        shape_points = json.loads(path.with_name("shapes.json").read_text())["points"]
        points[meta["pattern_hash"]] = {p["shape_hash"] for p in shape_points}
        assert all(
            p["partition_metrics"]["mm"] <= 3
            and p["partition_metrics"]["contraction_depth"] <= 5
            for p in shape_points
        )
    assert sum(len(p) for p in points.values()) == report["shape_points"] == 501
    rows = json.loads((corpus / "occurrences.json").read_text())
    for row in rows:
        assert (ROOT / row["source"]).is_file()
        for region in row["regions"]:
            assert region["shape_hash"] in points[region["pattern_hash"]]
    assert sum(len(r["regions"]) for r in rows) == report["region_occurrences"]
    for manifest in (corpus / "models").rglob("manifest.json"):
        assert set(json.loads(manifest.read_text())["patterns"]) <= points.keys()
