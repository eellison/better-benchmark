"""Horizontal groups contain independent MMs; source absorption labels stay explicit."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import get_fusion_partitions, partition_node_is_supported
from matmul_partitioner import CATEGORIES, classify_pattern_graph, get_matmul_regions


def test_matmul_categories_are_exactly_the_supported_four():
    assert CATEGORIES == ("prologue", "epilogue", "horizontal", "horizontal_combine")


def test_dependent_mms_are_not_horizontal_regions():
    gm = make_fx(lambda x, a, b: (x @ a) @ b, tracing_mode="fake")(
        torch.randn(4, 8), torch.randn(8, 8), torch.randn(8, 8)
    )
    diagnostics = {}
    assert get_matmul_regions(gm, partition_node_is_supported, diagnostics=diagnostics) == []
    assert get_fusion_partitions(gm, only_include_mm=True) == []
    assert classify_pattern_graph(gm) is None
    assert set(diagnostics["regions"]) == set(CATEGORIES)


def test_mm_reduction_is_only_an_epilogue_with_source_metadata():
    gm = make_fx(lambda x, w: (x @ w).sum(-1), tracing_mode="fake")(
        torch.randn(4, 8), torch.randn(8, 6)
    )
    (region,) = get_matmul_regions(gm, partition_node_is_supported)
    assert region.category == "epilogue"
    assert classify_pattern_graph(gm) is None
    assert classify_pattern_graph(gm, expected="epilogue") == "epilogue"


@pytest.mark.parametrize("combine", [False, True])
def test_singleton_k_is_allowed_in_horizontal_groups(combine):
    def fn(x, a, b):
        first, second = x @ a, x @ b
        return first + second if combine else (first, second)

    gm = make_fx(fn, tracing_mode="fake")(
        torch.randn(4, 1), torch.randn(1, 6), torch.randn(1, 6)
    )
    found = get_matmul_regions(
        gm, partition_node_is_supported, categories=("horizontal", "horizontal_combine")
    )
    expected = "horizontal_combine" if combine else "horizontal"
    assert [r.category for r in found] == [expected]
    assert classify_pattern_graph(gm) == expected
