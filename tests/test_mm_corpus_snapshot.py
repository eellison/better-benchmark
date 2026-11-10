"""Invariants of the committed four-category MM corpus."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from matmul_partitioner import CATEGORIES, MatmulLimits

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "prologue": (16, 48),
    "epilogue": (1102, 1756),
    "horizontal": (56, 96),
    "horizontal_combine": (5, 7),
}


def test_committed_snapshot_and_source_references():
    corpus = ROOT / "mm_repro"
    report = json.loads((corpus / "report.json").read_text())
    files = sorted((corpus / "canonical").glob("*/*/repro.py"))
    assert len(files) == report["unique_repros"] == 1179
    assert tuple(EXPECTED) == CATEGORIES
    assert {p.name for p in (corpus / "canonical").iterdir()} == set(CATEGORIES)
    assert set(report["pattern_categories"]) == set(CATEGORIES)
    limits = MatmulLimits()
    assert report["policy"] == asdict(limits)
    points, families = {}, set()
    observed = {category: [0, 0] for category in CATEGORIES}
    for path in files:
        meta = json.loads(path.with_name("meta.json").read_text())
        category = meta["pattern_category"]
        assert path.parent.parent.name == category and category in CATEGORIES
        normal = category in {"prologue", "epilogue"}
        assert meta["pattern_hash_mode"] == ("ordered_fx_v1" if normal else "scalar_literals_v1")
        assert path.parent.name.endswith("_" + meta["pattern_hash"])
        family = (category, meta["pattern_family"])
        assert family not in families
        families.add(family)
        shape_points = json.loads(path.with_name("shapes.json").read_text())["points"]
        points[(category, meta["pattern_hash"])] = {p["shape_hash"] for p in shape_points}
        assert len(points[(category, meta["pattern_hash"])]) == len(shape_points)
        observed[category][0] += 1
        observed[category][1] += len(shape_points)
        for point in shape_points:
            assert point["models"]
            for model in point["models"]:
                source = ROOT / "repros" / "models" / model
                assert next(source.glob("full_graph_*.py"), None) is not None
            metrics = point["partition_metrics"]
            if normal:
                assert metrics["mm"] == 1
            else:
                assert 2 <= metrics["mm"] <= limits.max_mm
                assert metrics["contraction_depth"] == 1
    assert {category: tuple(counts) for category, counts in observed.items()} == EXPECTED
    assert sum(len(p) for p in points.values()) == report["shape_points"] == 1907
    assert len(families) == report["families"]
    assert report["family_members"] == sum(c["family_members"] for c in report["pattern_categories"].values())
    assert report["deduplicated_repros"] == report["family_members"] - len(families)
    assert {p.name for p in corpus.iterdir()} == {"canonical", "report.json", "README.md"}
