"""Index bounds for index consumers beyond index/gather/scatter/embedding.

An integer repro input with no inferred bound falls back to
randint(0, max(own shape)), which can index out of range; each consumer
below must bound its index operand by the indexed dim instead.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.fx as fx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from full_graph_harness import infer_index_bounds_from_gm
from mask_rules import region_approx_index_rule

INFO = {"self": {"dtype": "torch.float32"}, "idx": {"dtype": "torch.int64"},
        "mask": {"dtype": "torch.bool"}, "src": {"dtype": "torch.float32"}}


def _graph(build, self_shape=(7, 5)):
    g = fx.Graph()
    phs = {}
    for name, val in (
        ("self", torch.empty(self_shape)),
        ("idx", torch.empty(3, dtype=torch.int64)),
        ("mask", torch.empty(3, dtype=torch.bool)),
        ("src", torch.empty(self_shape)),
    ):
        phs[name] = g.placeholder(name)
        phs[name].meta["val"] = val
    g.output(build(g, **phs))
    return fx.GraphModule({}, g)


@pytest.mark.parametrize("build, bound", [
    (lambda g, self, idx, mask, src: g.call_function(
        torch.ops.aten._unsafe_index.Tensor, (self, [None, idx])), 5),
    (lambda g, self, idx, mask, src: g.call_function(
        torch.ops.aten._unsafe_index_put.default, (self, [idx], src)), 7),
    (lambda g, self, idx, mask, src: g.call_function(
        torch.ops.aten._unsafe_masked_index.default, (self, mask, [idx], 0.0)), 7),
    (lambda g, self, idx, mask, src: g.call_function(
        torch.ops.aten._unsafe_masked_index_put_accumulate.default,
        (self, mask, [None, idx], src)), 5),
    (lambda g, self, idx, mask, src: g.call_function(
        torch.ops.aten.scatter_reduce.two, (self, 1, idx, src, "sum")), 5),
    (lambda g, self, idx, mask, src: g.call_function(
        torch.ops.aten.scatter.reduce, (self, 0, idx, src, "add")), 7),
    (lambda g, self, idx, mask, src: g.call_function(
        torch.ops.aten.index_add.default, (self, 0, idx, src)), 7),
    (lambda g, self, idx, mask, src: g.call_function(
        torch.ops.aten.index_copy.default, (self, 1, idx, src)), 5),
    (lambda g, self, idx, mask, src: g.call_function(
        torch.ops.aten.index_fill.int_Scalar, (self, 1, idx, 0.0)), 5),
    (lambda g, self, idx, mask, src: g.call_function(
        torch.ops.aten.embedding_dense_backward.default, (src, idx, 11, -1, False)), 11),
])
def test_consumer_bounds_its_index_operand(build, bound):
    gm = _graph(build)
    assert infer_index_bounds_from_gm(gm, INFO)["idx"] == bound


def test_offset_index_bound_is_inverted():
    def build(g, self, idx, mask, src):
        shifted = g.call_function(torch.ops.aten.add.Tensor, (idx, 2))
        return g.call_function(torch.ops.aten.index_add.default, (self, 0, shifted, src))

    assert infer_index_bounds_from_gm(_graph(build), INFO)["idx"] == 5


def test_unbounded_index_input_is_still_approx_index():
    # take() indexes the flattened tensor: no per-dim bound is inferred, so
    # the input keeps the randint fallback, and the repro must be flagged.
    gm = _graph(lambda g, self, idx, mask, src: g.call_function(
        torch.ops.aten.take.default, (self, idx)))
    assert "idx" not in infer_index_bounds_from_gm(gm, INFO)
    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    assert region_approx_index_rule(nodes) == "approx_index"


def test_unbacked_indexed_dim_leaves_index_unbounded():
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch._subclasses.fake_tensor import FakeTensorMode

    shape_env = ShapeEnv()
    u0 = shape_env.create_unbacked_symint()
    with FakeTensorMode(shape_env=shape_env):
        table = torch.empty(u0, 5)
    gm = _graph(lambda g, self, idx, mask, src: (
        g.call_function(torch.ops.aten.index.Tensor, (self, [idx])),
        g.call_function(torch.ops.aten.embedding.default, (self, idx)),
    ))
    self_ph = next(n for n in gm.graph.nodes if n.name == "self")
    self_ph.meta["val"] = table

    assert "idx" not in infer_index_bounds_from_gm(gm, INFO)
