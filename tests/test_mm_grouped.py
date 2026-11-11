"""Grouped MMs absorb their whole producer and consumer partitions."""

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import partition_node_is_supported
from input_codec import compact_from_spec, spec_from_compact
from matmul_partitioner import (
    MatmulLimits,
    classify_pattern_graph,
    get_matmul_regions,
    op_name,
)
from mm_input_constraints import annotate_grouped_mm_offsets
from repro_harness import make_inputs_from_config


def _moe(x, w1, w2, offs):
    # Back-to-back grouped MMs: the h * sigmoid(h) partition is the epilogue
    # of the first and the prologue of the second.
    h = torch._grouped_mm(torch.relu(x), w1, offs)
    y = torch._grouped_mm(h * torch.sigmoid(h), w2, offs)
    return y * 2


def _graph():
    return make_fx(_moe, tracing_mode="fake")(
        torch.randn(64, 16, dtype=torch.bfloat16),
        torch.randn(4, 16, 32, dtype=torch.bfloat16),
        torch.randn(4, 32, 16, dtype=torch.bfloat16),
        torch.tensor([16, 32, 48, 64], dtype=torch.int32),
    )


def test_back_to_back_grouped_mms_share_the_middle_partition():
    gm = _graph()
    regions = [
        r for r in get_matmul_regions(gm, partition_node_is_supported)
        if r.category == "grouped"
    ]
    assert len(regions) == 2
    ops = [{op_name(n) for n in r} for r in regions]
    assert {"aten::relu", "aten::sigmoid", "aten::mul"} <= ops[0]
    assert {"aten::sigmoid", "aten::mul"} <= ops[1]
    assert [r.metrics["non_mm_compute"] for r in regions] == [3, 3]


def _large_epilogue(mm):
    # 5 extra inputs and 5 + 30 = 35 non-MM ops in one consumer partition,
    # over both max_epilogue_extra_inputs (4) and max_normal_compute (31).
    def fn(*args):
        *operands, b1, b2, b3, b4, b5 = args
        y = mm(*operands) + b1 + b2 + b3 + b4 + b5
        for _ in range(15):
            y = torch.sin(torch.cos(y))
        return y
    return fn


def test_grouped_regions_are_exempt_from_the_normal_size_bounds():
    limits = MatmulLimits()
    bias = [torch.randn(64, 32, dtype=torch.bfloat16) for _ in range(5)]
    grouped = make_fx(
        _large_epilogue(lambda x, w, o: torch._grouped_mm(torch.relu(x), w, o)),
        tracing_mode="fake",
    )(
        torch.randn(64, 16, dtype=torch.bfloat16),
        torch.randn(4, 16, 32, dtype=torch.bfloat16),
        torch.tensor([16, 32, 48, 64], dtype=torch.int32),
        *bias,
    )
    (region,) = [
        r for r in get_matmul_regions(grouped, partition_node_is_supported)
        if r.category == "grouped"
    ]
    assert region.metrics["non_mm_compute"] == 36  # relu + 35
    assert region.metrics["non_mm_compute"] > limits.max_normal_compute
    assert region.metrics["extra_inputs"] == 5 > limits.max_epilogue_extra_inputs
    # The same epilogue after an mm is still dropped by the bounds.
    plain = make_fx(_large_epilogue(lambda x, w: x @ w), tracing_mode="fake")(
        torch.randn(64, 16, dtype=torch.bfloat16),
        torch.randn(16, 32, dtype=torch.bfloat16),
        *bias,
    )
    assert get_matmul_regions(plain, partition_node_is_supported) == []


def test_classify_grouped():
    gm = make_fx(lambda x, w, o: torch.relu(torch._grouped_mm(x, w, o)), tracing_mode="fake")(
        torch.randn(64, 16, dtype=torch.bfloat16),
        torch.randn(4, 16, 32, dtype=torch.bfloat16),
        torch.tensor([16, 32, 48, 64], dtype=torch.int32),
    )
    assert classify_pattern_graph(gm, expected="grouped") == "grouped"
    assert classify_pattern_graph(gm, expected="epilogue") is None


def test_offs_inputs_are_sorted_group_ends():
    gm = make_fx(lambda x, w, o: torch._grouped_mm(x, w, o), tracing_mode="fake")(
        torch.randn(64, 16, dtype=torch.bfloat16),
        torch.randn(4, 16, 32, dtype=torch.bfloat16),
        torch.tensor([16, 32, 48, 64], dtype=torch.int32),
    )
    offs = [n for n in gm.graph.nodes if n.op == "placeholder"][2]
    infos = {offs.name: {"kind": "tensor", "shape": [4], "dtype": "torch.int32"}}
    assert annotate_grouped_mm_offsets(gm, infos) == [offs.name]
    assert infos[offs.name]["gen"] == {"kind": "offsets", "high": 65, "end": 64}
    entry = compact_from_spec({**infos[offs.name], "device": "cpu"})
    assert entry[2]["gen"] == ["offsets", 65, 64]
    for _ in range(20):
        (value,) = make_inputs_from_config({"inputs": [spec_from_compact(entry)]})
        assert int(value[-1]) == 64 and bool((value.diff() >= 0).all())
        assert int(value.min()) >= 0
