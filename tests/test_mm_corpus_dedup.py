"""Curation normalization (normalize_graph) and DAG matching (match_graphs)."""

from __future__ import annotations

import torch
from mm_corpus_dedup import match_graphs, normalize_graph, semantic_graph
from torch.fx.experimental.proxy_tensor import make_fx


def trace(fn, *args):
    gm = make_fx(fn, tracing_mode="fake")(*args)
    infos = {
        n.name: {
            "kind": "tensor",
            "shape": list(n.meta["val"].shape),
            "stride": list(n.meta["val"].stride()),
            "dtype": str(n.meta["val"].dtype),
            "device": "cpu",
        }
        for n in gm.graph.nodes
        if n.op == "placeholder"
    }
    return normalize_graph(gm, infos)


def same_dag(a, b):
    return match_graphs(semantic_graph(a), semantic_graph(b)) is not None


def test_input_transpose_becomes_a_strided_input():
    x, w = torch.randn(3, 5), torch.randn(7, 5)
    left = trace(lambda x, w: (x @ w.t()).relu(), x, w)
    right = trace(lambda x, w: (x @ w).relu(), x, w.t())
    assert same_dag(left[0], right[0])
    assert list(left[1].values())[1]["stride"] == [1, 5]
    torch.testing.assert_close(left[0](x, w.t()), (x @ w.t()).relu(), atol=0, rtol=0)


def test_input_transpose_through_a_float_cast_keeps_rounding_and_layout():
    x, w = torch.randn(3, 5, dtype=torch.bfloat16), torch.randn(7, 5)
    a = trace(lambda x, w: (x @ w.to(torch.bfloat16).t()).relu(), x, w)
    b = trace(lambda x, w: (x @ w.to(torch.bfloat16)).relu(), x, w.t())
    assert same_dag(a[0], b[0])
    actual, expected = a[0](x, w.t()), (x @ w.to(torch.bfloat16).t()).relu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.stride() == expected.stride()


def test_only_symmetric_ops_match_with_swapped_operands():
    x, w, s = torch.randn(3, 5), torch.randn(5, 7), torch.randn(3, 7)
    assert same_dag(
        trace(lambda x, w, s: (x @ w) * s, x, w, s)[0],
        trace(lambda s, x, w: s * (x @ w), s, x, w)[0],
    )
    assert not same_dag(
        trace(lambda x, w, s: torch.add(x @ w, s, alpha=2), x, w, s)[0],
        trace(lambda x, w, s: torch.add(s, x @ w, alpha=2), x, w, s)[0],
    )


def test_scalar_literals_keep_graphs_apart():
    x, w = torch.randn(3, 5), torch.randn(5, 7)
    assert not same_dag(
        trace(lambda x, w: (x @ w) / 49, x, w)[0],
        trace(lambda x, w: (x @ w) / 64, x, w)[0],
    )


def test_equivalent_view_chains_normalize_identically():
    x, w = torch.randn(3, 5), torch.randn(5, 7)
    a = trace(lambda x, w: (x @ w).relu().view(3, 7, 1, 1), x, w)[0]
    b = trace(lambda x, w: (x @ w).relu().unsqueeze(2).unsqueeze(3), x, w)[0]
    assert same_dag(a, b)
    av, bv = a(x, w, [3, 7, 1, 1]), b(x, w, [3, 7, 1, 1])
    torch.testing.assert_close(av, bv, atol=0, rtol=0)
    assert av.stride() == bv.stride()


def test_reduction_axes_and_keepdim_are_canonical():
    x = torch.randn(4, 8, 3, 3, dtype=torch.bfloat16)
    w = torch.randn(8, 5, dtype=torch.bfloat16)

    def a(x, w):
        return x.mean((2, 3)) @ w

    def b(x, w):
        return x.mean((-1, -2), keepdim=True).view(4, 8) @ w

    ga, _, pa = trace(a, x, w)
    gb, _, pb = trace(b, x, w)
    assert same_dag(ga, gb)
    torch.testing.assert_close(ga(x, w, *pa.values()), a(x, w), atol=0, rtol=0)
    torch.testing.assert_close(gb(x, w, *pb.values()), b(x, w), atol=0, rtol=0)


def test_identity_view_removal_keeps_output_aliasing():
    x, w = torch.randn(3, 5), torch.randn(5, 7)

    def fn(x, w):
        y = (x @ w).relu()
        return y, y[-4095:]

    gm, _, _ = trace(fn, x, w)
    a, b = gm(x, w)
    assert torch._C._is_alias_of(a, b)
    assert a.stride() == b.stride()


def test_pointwise_cse_merges_internal_values_but_not_returned_buffers():
    x, w, labels = torch.randn(3, 5), torch.randn(5, 7), torch.tensor([0, 1, 2])

    def internal(x, w, labels):
        first, second = labels.ne(-100), labels.ne(-100)
        return (x @ w).relu() + first[:, None].float() + second[:, None].float()

    gm, _, params = trace(internal, x, w, labels)
    assert sum(n.target == torch.ops.aten.ne.Scalar for n in gm.graph.nodes) == 1
    torch.testing.assert_close(
        gm(x, w, labels, *params.values()), internal(x, w, labels), atol=0, rtol=0
    )

    def returned(x, w):
        y = x @ w
        return y.relu(), y.relu()

    gm, _, _ = trace(returned, x, w)
    a, b = gm(x, w)
    assert not torch._C._is_alias_of(a, b)


def test_identity_adaptive_pool_is_removed_only_when_internal():
    x, w = torch.randn(2, 3, 2, 2), torch.randn(12, 4)

    def internal(x, w):
        return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten(1) @ w

    gm, _, params = trace(internal, x, w)
    pool_ops = {torch.ops.aten._adaptive_avg_pool2d.default, torch.ops.aten.adaptive_avg_pool2d.default}
    assert not any(n.target in pool_ops for n in gm.graph.nodes)
    torch.testing.assert_close(gm(x, w, *params.values()), internal(x, w), atol=0, rtol=0)

    def exposed(x, w):
        return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)), x.flatten(1) @ w

    exposed_gm = trace(exposed, x, w)[0]
    assert any(n.target in pool_ops for n in exposed_gm.graph.nodes)


def test_matching_symmetric_graphs_is_deterministic():
    # Two identical branches have two isomorphisms; always pick the identity.
    fn = lambda x, a, y, b: ((x @ a).relu(), (y @ b).relu())
    args = torch.randn(8, 16), torch.randn(16, 32), torch.randn(8, 16), torch.randn(16, 32)
    first = semantic_graph(make_fx(fn, tracing_mode="fake")(*args))
    second = semantic_graph(make_fx(fn, tracing_mode="fake")(*args))
    mapping = match_graphs(first, second)
    inputs = [n for n in first if getattr(n, "op", None) == "placeholder"]
    assert [mapping[n].name for n in inputs] == [n.name for n in inputs]
