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

from full_graph_harness import _ANY_INDEX, infer_index_bounds_from_gm
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


@pytest.mark.parametrize("op, extra", [
    (torch.ops.aten._to_copy.default, {"dtype": torch.int32}),
])
def test_bound_passes_through_identity_copies(op, extra):
    # A graph constant used as an index reaches index() through
    # lift_fresh_copy; once a partition lifts it to an input, the bound must
    # still be found, or the input falls back to a range past the dim.
    def build(g, self, idx, mask, src):
        kwargs = extra if isinstance(extra, dict) else {}
        copied = g.call_function(op, (idx,), kwargs)
        return g.call_function(torch.ops.aten.index.Tensor, (self, [None, copied]))

    assert infer_index_bounds_from_gm(_graph(build), INFO)["idx"] == 5


def test_bound_does_not_pass_through_low_precision_float_copy():
    # fp16 rounds 4099 up to 4100, so a value drawn below the bound can come
    # back from an int64 -> fp16 -> int64 round trip equal to it.
    def build(g, self, idx, mask, src):
        half = g.call_function(
            torch.ops.aten._to_copy.default, (idx,), {"dtype": torch.float16})
        back = g.call_function(
            torch.ops.aten._to_copy.default, (half,), {"dtype": torch.int64})
        return g.call_function(torch.ops.aten.index.Tensor, (self, [None, back]))

    assert "idx" not in infer_index_bounds_from_gm(_graph(build, (7, 4100)), INFO)


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


def _bit_field_graph(fields, offset=1):
    """idx -> (idx - offset) split into bit fields, each indexing a table.

    fields: (shift, mask, table_rows) per field. The fields are stacked with
    cat and read back with select, as a hashed id is split across tables.
    """
    aten = torch.ops.aten
    g = fx.Graph()
    idx = g.placeholder("idx")
    idx.meta["val"] = torch.empty(3, dtype=torch.int64)
    value = g.call_function(aten.sub.Tensor, (idx, offset)) if offset else idx
    columns = []
    for shift, mask, _ in fields:
        shifted = g.call_function(aten.__rshift__.Scalar, (value, shift))
        masked = g.call_function(aten.bitwise_and.Scalar, (shifted, mask))
        column = g.call_function(aten.unsqueeze.default, (masked, 1))
        column.meta["val"] = torch.empty(3, 1, dtype=torch.int64)
        columns.append(column)
    stacked = g.call_function(aten.cat.default, (columns, -1))
    stacked.meta["val"] = torch.empty(3, len(fields), dtype=torch.int64)
    outputs = []
    for k, (_, _, rows) in enumerate(fields):
        table = g.placeholder(f"table{k}")
        table.meta["val"] = torch.empty(rows, 2)
        picked = g.call_function(aten.select.int, (stacked, 1, k))
        outputs.append(g.call_function(aten.embedding.default, (table, picked)))
    g.output(tuple(outputs))
    return fx.GraphModule({}, g)


def test_masked_bit_fields_leave_their_input_unconstrained():
    # Each field's mask fits its own table (field 0's would not fit field
    # 1's table, so cat columns must be paired with their selects), so any
    # id is safe, including the ones the offset makes negative.
    gm = _bit_field_graph([(0, 7, 8), (3, 3, 4)])
    assert infer_index_bounds_from_gm(gm, INFO)["idx"] == _ANY_INDEX


def test_comparisons_and_outputs_of_a_masked_input_are_harmless():
    gm = _bit_field_graph([(0, 7, 8), (3, 3, 4)])
    g = gm.graph
    idx = next(n for n in g.nodes if n.name == "idx")
    output = next(n for n in g.nodes if n.op == "output")
    with g.inserting_before(output):
        padding = g.call_function(torch.ops.aten.eq.Scalar, (idx, 0))
    output.args = ((*output.args[0], padding, idx),)
    gm.recompile()
    assert infer_index_bounds_from_gm(gm, INFO)["idx"] == _ANY_INDEX


def test_mask_wider_than_its_table_does_not_unbound_the_input():
    gm = _bit_field_graph([(0, 15, 8), (4, 3, 4)])
    assert infer_index_bounds_from_gm(gm, INFO).get("idx") != _ANY_INDEX


@pytest.mark.parametrize("extra_use", ["abs", "unmasked_index", "narrowing_cast"])
def test_any_use_outside_the_masked_paths_keeps_the_input_bounded(extra_use):
    gm = _bit_field_graph([(0, 7, 8)])
    g = gm.graph
    idx = next(n for n in g.nodes if n.name == "idx")
    table = next(n for n in g.nodes if n.name == "table0")
    output = next(n for n in g.nodes if n.op == "output")
    with g.inserting_before(output):
        if extra_use == "abs":
            index = g.call_function(torch.ops.aten.abs.default, (idx,))
        elif extra_use == "unmasked_index":
            index = idx
        else:
            narrow = g.call_function(
                torch.ops.prims.convert_element_type.default, (idx, torch.int32))
            index = g.call_function(torch.ops.aten.bitwise_and.Scalar, (narrow, 7))
        used = g.call_function(torch.ops.aten.embedding.default, (table, index))
    output.args = ((*output.args[0], used),)
    gm.recompile()
    assert infer_index_bounds_from_gm(gm, INFO).get("idx") != _ANY_INDEX


def test_masked_int32_input_is_not_unbounded():
    gm = _bit_field_graph([(0, 7, 8)])
    info = {**INFO, "idx": {"dtype": "torch.int32"}}
    assert infer_index_bounds_from_gm(gm, info).get("idx") != _ANY_INDEX


def test_masked_index_next_to_a_bool_mask_index_is_not_unbounded():
    # index(t, [bool[100, 4], x & 3]): the bool mask covers dims 0-1, so
    # x & 3 indexes dim 2 (size 2), not dim 1 (size 4).
    aten = torch.ops.aten
    g = fx.Graph()
    idx = g.placeholder("idx")
    idx.meta["val"] = torch.empty(3, dtype=torch.int64)
    t = g.placeholder("t")
    t.meta["val"] = torch.empty(100, 4, 2)
    keep = g.placeholder("keep")
    keep.meta["val"] = torch.empty(100, 4, dtype=torch.bool)
    masked = g.call_function(aten.bitwise_and.Scalar, (idx, 3))
    g.output(g.call_function(aten.index.Tensor, (t, [keep, masked])))
    info = {**INFO, "t": {"dtype": "torch.float32"}, "keep": {"dtype": "torch.bool"}}
    assert infer_index_bounds_from_gm(fx.GraphModule({}, g), info).get("idx") != _ANY_INDEX


def test_unindexed_int64_input_is_not_unbounded():
    g = fx.Graph()
    idx = g.placeholder("idx")
    idx.meta["val"] = torch.empty(3, dtype=torch.int64)
    g.output((g.call_function(torch.ops.aten.eq.Scalar, (idx, 0)), idx))
    assert infer_index_bounds_from_gm(fx.GraphModule({}, g), INFO).get("idx") != _ANY_INDEX
