"""Partitions of symbolically traced graphs that read a size comparison."""
import sys
from pathlib import Path

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import capture_hook


def _symbolic_graph():
    def f(x):
        n = x.shape[0]
        ge = torch.ops.aten.scalar_tensor.default(n >= 4, dtype=torch.float32)
        eq = torch.ops.aten.scalar_tensor.default(n == 5, dtype=torch.float32)
        return (x * 2 + ge + eq).sum(0)

    return make_fx(f, tracing_mode="symbolic")(torch.randn(6, 3))


def _region_with(captured, op):
    (entry,) = [e for e in captured if op in e["origin_ops"]]
    return entry


def test_size_comparison_becomes_a_scalar_input(tmp_path):
    state = capture_hook._CaptureState(str(tmp_path), label="m", validate=False)
    state.process_graph(_symbolic_graph())

    entry = _region_with(state.captured, "aten.scalar_tensor.default")
    assert ["sc", True] in entry["inputs"]
    assert ["sc", False] in entry["inputs"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="validates the repro on CUDA")
def test_size_comparison_region_validates(tmp_path):
    state = capture_hook._CaptureState(str(tmp_path), label="m")
    state.process_graph(_symbolic_graph())

    assert state.dropped == []
    assert _region_with(state.captured, "aten.scalar_tensor.default")


def test_unbacked_symbols_resolve_from_input_specs():
    import torch.fx as fx
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    u0 = shape_env.create_unbacked_symint()
    assert capture_hook._sym_hint(u0) is None

    g = fx.Graph()
    ph = g.placeholder("n")
    ph.meta["val"] = u0
    g.output(ph)
    gm = fx.GraphModule({}, g)
    gm.meta["better_benchmark_input_specs"] = [{"kind": "symint", "value": 7}]

    known = capture_hook._input_symbol_values(gm)
    assert capture_hook._sym_hint(u0, known) == 7
    assert capture_hook._sym_hint(u0 * 2 + 1, known) == 15
    assert capture_hook._sym_hint(u0 == 7, known) is True


def test_partition_input_without_value_is_dropped_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_hook, "_sym_hint", lambda x, known=None: None)
    state = capture_hook._CaptureState(str(tmp_path), label="m", validate=False)
    state.process_graph(_symbolic_graph())

    assert any("no concrete value" in d["reason"] for d in state.dropped)


def test_view_shape_with_symbolic_dim_is_not_lifted(tmp_path):
    def f(x, n):
        return (x.view(n, 2, 3) * 2).sum(1)

    gm = make_fx(f, tracing_mode="symbolic")(torch.randn(4, 6), 4)
    state = capture_hook._CaptureState(str(tmp_path), label="m", validate=False)
    state.process_graph(gm)

    (entry,) = [e for e in state.captured if "aten.view.default" in e["origin_ops"]]
    # The view's shape reads the symbolic input, so it is passed as that
    # input (a "sym" entry), not lifted into a shape literal.
    assert ["sym", 4] in entry["inputs"]
    assert not [item for item in entry["inputs"] if item[0] == "S"]


def test_index_into_unbacked_dim_is_flagged_approx_index(tmp_path):
    import torch.fx as fx
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    u0 = shape_env.create_unbacked_symint()
    with FakeTensorMode(shape_env=shape_env):
        table = torch.empty(u0, 5)
        idx = torch.empty(3, dtype=torch.int64)
        out = torch.empty(3, 5)
    g = fx.Graph()
    table_ph = g.placeholder("table")
    table_ph.meta["val"] = table
    idx_ph = g.placeholder("idx")
    idx_ph.meta["val"] = idx
    gathered = g.call_function(torch.ops.aten.index.Tensor, (table_ph, [idx_ph]))
    gathered.meta["val"] = out
    doubled = g.call_function(torch.ops.aten.mul.Tensor, (gathered, 2))
    doubled.meta["val"] = out
    g.output(doubled)
    gm = fx.GraphModule({}, g)
    # The ingest loader records the concrete input shapes; u0 is 7 here.
    gm.meta["better_benchmark_input_specs"] = [
        {"kind": "tensor", "shape": [7, 5]},
        {"kind": "tensor", "shape": [3]},
    ]

    state = capture_hook._CaptureState(str(tmp_path), label="m", validate=False)
    state.process_graph(gm)

    entry = _region_with(state.captured, "aten.index.Tensor")
    assert entry["masked_reason"] == "approx_index"
    # No bound is inferred for an index into a data-dependent dim.
    assert [[7, 5], "f32"] in entry["inputs"]
    assert [[3], "i64"] in entry["inputs"]
