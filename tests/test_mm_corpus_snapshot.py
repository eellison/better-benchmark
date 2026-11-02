"""Invariants of the committed five-category MM corpus."""
from __future__ import annotations

import json
from pathlib import Path

from matmul_partitioner import CATEGORIES, MatmulLimits

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "prologue": (105, 408),
    "epilogue": (345, 650),
    "horizontal": (74, 112),
    "horizontal_combine": (5, 7),
    "vertical": (93, 102),
}


def test_committed_snapshot_and_source_references():
    corpus = ROOT / "mm_repro"
    report = json.loads((corpus / "report.json").read_text())
    files = sorted((corpus / "canonical").glob("*/*/repro.py"))
    assert len(files) == report["unique_repros"] == 622
    assert tuple(EXPECTED) == CATEGORIES
    limits = MatmulLimits()
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
                assert metrics["mm"] <= limits.max_mm
                assert metrics["contraction_depth"] <= limits.max_stages
            for stage in metrics.get("vertical", {}).get("stages", []):
                assert stage["R"] <= limits.reduction_limit
                assert stage["F"] <= limits.intermediate_limit
    assert {category: tuple(counts) for category, counts in observed.items()} == EXPECTED
    assert sum(len(p) for p in points.values()) == report["shape_points"] == 1279
    assert len(families) == report["families"]
    assert report["deduplicated_repros"] == report["family_members"] - len(families)
    assert {p.name for p in corpus.iterdir()} == {"canonical", "report.json", "README.md"}
    # Actual singleton-axis checks are performed by validate_matmul_repros.py,
    # not inferred from total R (a [1,6] reduction has R=6 but is excluded).
