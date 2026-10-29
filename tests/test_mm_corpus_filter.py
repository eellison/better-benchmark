from __future__ import annotations

import torch
from mm_corpus_features import analyze_graph
from mm_corpus_io import admits, filter_patterns
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import get_fusion_partitions
from matmul_partitioner import MatmulLimits


def test_filter_keeps_every_passing_pattern_without_a_count_limit():
    limits = MatmulLimits(
        max_mm=3, max_compute=32, max_nodes=42, max_inputs=8, max_outputs=2
    )
    valid = {
        "mm": 1,
        "compute": 2,
        "nodes": 2,
        "inputs": 2,
        "outputs": 1,
        "contraction_depth": 1,
    }
    records = [
        {"id": str(i), "metrics": valid.copy(), "horizontal_shapes_match": True}
        for i in range(350)
    ]
    records.append(
        {
            "id": "too_large",
            "metrics": {**valid, "compute": 33},
            "horizontal_shapes_match": True,
        }
    )
    kept = filter_patterns(records, limits)
    assert len(kept) == 350
    assert all(r["id"] != "too_large" for r in kept)


def test_admission_caps_keep_boundary_values_and_reject_oversized_regions():
    limits = MatmulLimits(
        max_mm=3, max_compute=32, max_nodes=42, max_inputs=8, max_outputs=2
    )
    metrics = {
        "mm": 3,
        "compute": 32,
        "nodes": 42,
        "inputs": 8,
        "outputs": 2,
        "contraction_depth": 5,
    }
    assert admits(metrics, limits)
    for key, value in metrics.items():
        oversized = {**metrics, key: value + 1}
        assert not admits(oversized, limits)


def test_graph_features_recognize_generic_parallel_and_vertical_structure():
    gm = make_fx(
        lambda x, y, a, b, c: (((x @ a).relu() + y @ b) @ c).relu(), tracing_mode="fake"
    )(
        torch.randn(8, 64),
        torch.randn(8, 64),
        torch.randn(64, 64),
        torch.randn(64, 64),
        torch.randn(64, 64),
    )
    analysis = analyze_graph(gm)
    assert {"horizontal", "horizontal_pw_join", "mm_pw_mm"} <= set(analysis["coverage"])


def test_heuristic_caps_allow_full_gather_affine_layernorm_mm_residual_layernorm():
    def norm(x, gamma, beta):
        xf = x.float()
        mean = xf.sum(-1, keepdim=True) / x.size(-1)
        centered = xf - mean
        variance = (centered * centered).sum(-1, keepdim=True) / x.size(-1)
        return (
            centered * torch.rsqrt(variance + 1e-5) * gamma.float() + beta.float()
        ).to(x.dtype)

    def fn(table, ids, w, residual, pg, pb, qg, qb):
        x = norm(table.index_select(0, ids), pg, pb)
        return norm(x @ w + residual, qg, qb)

    args = (
        torch.randn(64, 128, dtype=torch.bfloat16),
        torch.arange(8),
        torch.randn(128, 128, dtype=torch.bfloat16),
        torch.randn(8, 128, dtype=torch.bfloat16),
        *(torch.randn(128, dtype=torch.bfloat16) for _ in range(4)),
    )
    gm = make_fx(fn, tracing_mode="fake")(*args)
    analysis = analyze_graph(gm)
    assert {"reduction_mm_reduction", "gather_reduction_mm_reduction"} <= set(
        analysis["coverage"]
    )
    limits = MatmulLimits(
        max_mm=3, max_compute=32, max_nodes=42, max_inputs=8, max_outputs=2
    )
    assert admits(analysis["metrics"], limits)
    regions = get_fusion_partitions(gm, only_include_mm=True)
    assert any(
        r.metrics["contraction_depth"] == 5 and admits(r.metrics, limits)
        for r in regions
    )


def test_block_reduction_requires_a_view_after_the_relevant_mm():
    before = make_fx(
        lambda x, w: (x.view(2, 4, 8).view(8, 8) @ w).sum(-1), tracing_mode="fake"
    )(torch.randn(8, 8), torch.randn(8, 8))
    after = make_fx(lambda x, w: (x @ w).view(8, 2, 4).sum(-1), tracing_mode="fake")(
        torch.randn(8, 8), torch.randn(8, 8)
    )
    assert "mm_block_amax_sum" not in analyze_graph(before)["coverage"]
    assert "mm_block_amax_sum" in analyze_graph(after)["coverage"]


def test_filter_discards_the_whole_repro_when_any_horizontal_shape_point_differs(
    tmp_path,
):
    from mm_corpus_io import load_records, write_json

    source = tmp_path / "source"
    # Use direct extraction to construct a historical corpus containing a now
    # rejected horizontal pair, without invoking today's partitioner policy.
    directory = source / "canonical" / "mm_pair"
    directory.mkdir(parents=True)
    ir = "import torch\nclass Repro(torch.nn.Module):\n    def forward(self, x, a, y, b):\n        return (x @ a).relu(), (y @ b).relu()\n"
    (directory / "repro.py").write_text(ir)
    write_json(directory / "meta.json", {"pattern_hash": "pair"})
    points = []
    for width in [64, 32]:
        inputs = [
            [[8, 256], "f32"],
            [[256, 64], "f32"],
            [[8, 512], "f32"],
            [[512, width], "f32"],
        ]
        points.append(
            {
                "shape_hash": str(width),
                "inputs": inputs,
                "partition_metrics": {
                    "mm": 2,
                    "compute": 4,
                    "nodes": 4,
                    "inputs": 4,
                    "outputs": 2,
                    "contraction_depth": 1,
                },
            }
        )
    write_json(directory / "shapes.json", {"points": points})
    write_json(source / "occurrences.json", [])
    records, _ = load_records(source, tmp_path / "cache.json")
    assert len(records) == 1 and records[0]["shape_points"] == 2
    assert not records[0]["horizontal_shapes_match"]
    assert filter_patterns(records, MatmulLimits()) == []
