"""Final size limits (mm_corpus_io) and coverage labels (mm_corpus_features)."""

from __future__ import annotations

import torch
from mm_corpus_features import analyze_graph
from mm_corpus_io import admits, filter_patterns, load_records, write_json
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import get_fusion_partitions
from matmul_partitioner import MatmulLimits

# The limits curate_mm_corpus applies to exported repros.
FINAL_LIMITS = MatmulLimits(
    max_mm=3, max_compute=32, max_nodes=42, max_inputs=8, max_outputs=2
)


def test_filter_keeps_every_record_within_the_limits_without_a_count_cap():
    at_limit = {
        "mm": 3,
        "compute": 32,
        "nodes": 42,
        "inputs": 8,
        "outputs": 2,
        "contraction_depth": 5,
    }
    assert admits(at_limit, FINAL_LIMITS)
    for key, value in at_limit.items():
        assert not admits({**at_limit, key: value + 1}, FINAL_LIMITS)
    records = [
        {"id": i, "metrics": at_limit, "horizontal_shapes_match": True}
        for i in range(350)
    ]
    records.append({"id": "rule3", "metrics": at_limit, "horizontal_shapes_match": False})
    assert [r["id"] for r in filter_patterns(records, FINAL_LIMITS)] == list(range(350))


def test_coverage_labels_follow_value_dependencies():
    gm = make_fx(
        lambda x, y, a, b, c: (((x @ a).relu() + y @ b) @ c).relu(), tracing_mode="fake"
    )(*(torch.randn(8, 64) for _ in range(2)), *(torch.randn(64, 64) for _ in range(3)))
    coverage = set(analyze_graph(gm)["coverage"])
    assert {"horizontal", "horizontal_pw_join", "horizontal_then_mm", "mm_pw_mm"} <= coverage


def test_gather_norm_mm_residual_norm_fits_the_final_limits():
    def norm(x, gamma, beta):
        xf = x.float()
        mean = xf.sum(-1, keepdim=True) / x.size(-1)
        centered = xf - mean
        variance = (centered * centered).sum(-1, keepdim=True) / x.size(-1)
        normalized = centered * torch.rsqrt(variance + 1e-5)
        return (normalized * gamma.float() + beta.float()).to(x.dtype)

    def fn(table, ids, w, residual, pg, pb, qg, qb):
        return norm(norm(table.index_select(0, ids), pg, pb) @ w + residual, qg, qb)

    bf16 = dict(dtype=torch.bfloat16)
    gm = make_fx(fn, tracing_mode="fake")(
        torch.randn(64, 128, **bf16),
        torch.arange(8),
        torch.randn(128, 128, **bf16),
        torch.randn(8, 128, **bf16),
        *(torch.randn(128, **bf16) for _ in range(4)),
    )
    analysis = analyze_graph(gm)
    assert "gather_reduction_mm_reduction" in analysis["coverage"]
    assert admits(analysis["metrics"], FINAL_LIMITS)
    assert any(
        r.metrics["contraction_depth"] == 5 and admits(r.metrics, FINAL_LIMITS)
        for r in get_fusion_partitions(gm, only_include_mm=True)
    )


def test_a_repro_is_dropped_if_any_shape_point_breaks_rule_3(tmp_path):
    # A raw repro whose horizontal MMs match at one shape but not the other.
    source = tmp_path / "raw"
    directory = source / "canonical" / "mm_pair"
    directory.mkdir(parents=True)
    (directory / "repro.py").write_text(
        "import torch\n"
        "class Repro(torch.nn.Module):\n"
        "    def forward(self, x, a, y, b):\n"
        "        return (x @ a).relu(), (y @ b).relu()\n"
    )
    write_json(directory / "meta.json", {"pattern_hash": "pair"})
    metrics = {"mm": 2, "compute": 4, "nodes": 4, "inputs": 4, "outputs": 2, "contraction_depth": 1}
    points = [
        {
            "shape_hash": str(width),
            "inputs": [[[8, 256], "f32"], [[256, 64], "f32"], [[8, 512], "f32"], [[512, width], "f32"]],
            "partition_metrics": metrics,
        }
        for width in [64, 32]
    ]
    write_json(directory / "shapes.json", {"points": points})
    write_json(source / "occurrences.json", [])
    records, _ = load_records(source)
    assert len(records) == 1 and not records[0]["horizontal_shapes_match"]
    assert filter_patterns(records, MatmulLimits()) == []
