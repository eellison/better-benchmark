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


def test_input_transpose_becomes_same_strided_interface_and_values():
    x, w = torch.randn(3, 5), torch.randn(7, 5)
    left = trace(lambda x, w: (x @ w.t()).relu(), x, w)
    right = trace(lambda x, w: (x @ w).relu(), x, w.t())
    assert match_graphs(semantic_graph(left[0]), semantic_graph(right[0])) is not None
    torch.testing.assert_close(left[0](x, w.t()), (x @ w.t()).relu(), atol=0, rtol=0)
    assert list(left[1].values())[1]["stride"] == [1, 5]


def test_commutative_operand_order_and_input_order_are_equivalent():
    x, w, s = torch.randn(3, 5), torch.randn(5, 7), torch.randn(7)
    a = trace(lambda x, w, s: (x @ w) * s, x, w, s)[0]
    b = trace(lambda s, x, w: s * (x @ w), s, x, w)[0]
    assert match_graphs(semantic_graph(a), semantic_graph(b)) is not None


def test_nonunit_add_alpha_is_not_commutative():
    x, w, s = torch.randn(3, 5), torch.randn(5, 7), torch.randn(3, 7)
    a = trace(lambda x, w, s: torch.add(x @ w, s, alpha=2), x, w, s)[0]
    b = trace(lambda x, w, s: torch.add(s, x @ w, alpha=2), x, w, s)[0]
    assert match_graphs(semantic_graph(a), semantic_graph(b)) is None


def test_scalar_literals_are_preserved_in_deduplication():
    x, w = torch.randn(3, 5), torch.randn(5, 7)
    a = trace(lambda x, w: (x @ w) / 49, x, w)[0]
    b = trace(lambda x, w: (x @ w) / 64, x, w)[0]
    assert match_graphs(semantic_graph(a), semantic_graph(b)) is None


def test_contiguous_rank_view_and_unsqueeze_chain_normalize_identically():
    x, w = torch.randn(3, 5), torch.randn(5, 7)
    a = trace(lambda x, w: (x @ w).relu().view(3, 7, 1, 1), x, w)[0]
    b = trace(lambda x, w: (x @ w).relu().unsqueeze(2).unsqueeze(3), x, w)[0]
    assert match_graphs(semantic_graph(a), semantic_graph(b)) is not None
    s = [3, 7, 1, 1]
    av, bv = a(x, w, s), b(x, w, s)
    torch.testing.assert_close(av, bv, atol=0, rtol=0)
    assert av.stride() == bv.stride()


def test_identity_slice_removal_preserves_output_alias_contract():
    x, w = torch.randn(3, 5), torch.randn(5, 7)
    gm, _, _, _, changes = trace(lambda x, w: ((x @ w).relu(),), x, w)

    # Build the aliased output case separately so the source MM is shared.
    def fn(x, w):
        y = (x @ w).relu()
        return y, y[-4095:]

    gm, _, _, _, changes = trace(fn, x, w)
    a, b = gm(x, w)
    assert "identity_view" in changes
    assert torch._C._is_alias_of(a, b)
    assert a.stride() == b.stride()


def test_cse_only_merges_internal_pointwise_results():
    x, w, labels = torch.randn(3, 5), torch.randn(5, 7), torch.tensor([0, 1, 2])

    def fn(x, w, labels):
        first = labels.ne(-100)
        second = labels.ne(-100)
        return (x @ w).relu() + first[:, None].float() + second[:, None].float()

    gm, _, params, _, changes = trace(fn, x, w, labels)
    assert "internal_pointwise_cse" in changes
    torch.testing.assert_close(
        gm(x, w, labels, *params.values()), fn(x, w, labels), atol=0, rtol=0
    )


def test_input_transpose_through_float_cast_preserves_rounding_and_layout():
    x = torch.randn(3, 5, dtype=torch.bfloat16)
    w = torch.randn(7, 5)
    a = trace(lambda x, w: (x @ w.to(torch.bfloat16).t()).relu(), x, w)
    b = trace(lambda x, w: (x @ w.to(torch.bfloat16)).relu(), x, w.t())
    assert match_graphs(semantic_graph(a[0]), semantic_graph(b[0])) is not None
    av = a[0](x, w.t())
    bv = (x @ w.to(torch.bfloat16).t()).relu()
    torch.testing.assert_close(av, bv, atol=0, rtol=0)
    assert av.stride() == bv.stride()


def test_pointwise_cse_does_not_alias_distinct_returned_buffers():
    x, w = torch.randn(3, 5), torch.randn(5, 7)

    def fn(x, w):
        y = x @ w
        return y.relu(), y.relu()

    gm, _, _, _, _changes = trace(fn, x, w)
    a, b = gm(x, w)
    assert not torch._C._is_alias_of(a, b)


def test_reduction_axes_and_keepdim_view_preserve_bfloat16_values():
    x = torch.randn(4, 8, 3, 3, dtype=torch.bfloat16)
    w = torch.randn(8, 5, dtype=torch.bfloat16)

    def a(x, w):
        return x.mean((2, 3)) @ w

    def b(x, w):
        return x.mean((-1, -2), keepdim=True).view(4, 8) @ w

    ga, _, pa, _, _ = trace(a, x, w)
    gb, _, pb, _, _ = trace(b, x, w)
    assert match_graphs(semantic_graph(ga), semantic_graph(gb)) is not None
    torch.testing.assert_close(ga(x, w, *pa.values()), a(x, w), atol=0, rtol=0)
    torch.testing.assert_close(gb(x, w, *pb.values()), b(x, w), atol=0, rtol=0)


def test_internal_identity_adaptive_pool_is_removed_but_exposed_copy_is_kept():
    x, w = torch.randn(2, 3, 2, 2), torch.randn(12, 4)

    def fn(x, w):
        return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten(1) @ w

    gm, _, params, _, changes = trace(fn, x, w)
    assert "internal_identity_adaptive_pool" in changes
    torch.testing.assert_close(gm(x, w, *params.values()), fn(x, w), atol=0, rtol=0)

    def exposed(x, w):
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))
        return pooled, (x.flatten(1) @ w)

    gm, _, params, _, changes = trace(exposed, x, w)
    assert "internal_identity_adaptive_pool" not in changes


def test_mm_hash_ignores_unused_tuple_projection_retrace_noise():
    import copy

    from capture_hook import pattern_hash_for_subgraph

    x, w = torch.randn(2, 3, 8, 8), torch.randn(48, 5)
    gm = make_fx(
        lambda x, w: (torch.nn.functional.max_pool2d(x, 2).flatten(1) @ w).relu(),
        tracing_mode="fake",
    )(x, w)
    trimmed = copy.deepcopy(gm)
    trimmed.graph.eliminate_dead_code()
    trimmed.recompile()
    assert pattern_hash_for_subgraph(
        gm, scalar_sensitive=True
    ) == pattern_hash_for_subgraph(trimmed, scalar_sensitive=True)
