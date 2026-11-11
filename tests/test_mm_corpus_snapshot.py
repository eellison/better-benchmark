"""Invariants of the committed five-category MM corpus."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from matmul_partitioner import CATEGORIES, MatmulLimits

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "prologue": (25, 36),
    "epilogue": (378, 763),
    "horizontal": (91, 156),
    "horizontal_combine": (5, 7),
    "grouped": (3, 3),
}
TOTAL = tuple(map(sum, zip(*EXPECTED.values())))


def test_committed_snapshot_and_source_references():
    corpus = ROOT / "mm_repro"
    report = json.loads((corpus / "report.json").read_text())
    # Flat canonical/<category>_<mm ops>_<hash>/, like repros/canonical/.
    files = sorted((corpus / "canonical").glob("*/repro.py"))
    assert len(files) == report["unique_repros"] == TOTAL[0]
    assert all(p.is_dir() for p in (corpus / "canonical").iterdir())
    assert tuple(EXPECTED) == CATEGORIES
    assert set(report["pattern_categories"]) == set(CATEGORIES)
    limits = MatmulLimits()
    assert report["policy"] == asdict(limits)
    points, hashes = {}, set()
    observed = {category: [0, 0] for category in CATEGORIES}
    for path in files:
        meta = json.loads(path.with_name("meta.json").read_text())
        category = meta["pattern_category"]
        assert category in CATEGORIES and path.parent.name.startswith(category + "_")
        normal = category in {"prologue", "epilogue", "grouped"}
        assert meta["pattern_hash_mode"] == "dag_v1"
        assert path.parent.name.endswith("_" + meta["pattern_hash"])
        # One repro per pattern hash, in every category.
        assert meta["pattern_hash"] not in hashes
        hashes.add(meta["pattern_hash"])
        shape_points = json.loads(path.with_name("shapes.json").read_text())["points"]
        points[meta["pattern_hash"]] = {p["shape_hash"] for p in shape_points}
        assert len(points[meta["pattern_hash"]]) == len(shape_points)
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
    assert sum(len(p) for p in points.values()) == report["shape_points"] == TOTAL[1]
    assert {p.name for p in corpus.iterdir()} == {"canonical", "report.json", "README.md"}
