"""Input synthesis for captured regions: values the region's ops can consume.

  1. fp8 inputs are drawn in fp32 and rounded (randn has no fp8 kernels).
  2. index_put values are cast to the destination dtype, as Inductor's lowering
     does, so a post-grad region with mismatched dtypes runs in eager.
  3. A float label reaching gather through a long where/convert chain is bounded.
  4. A lifted index constant reaching index() through lift_fresh_copy is bounded.
  5. Only a scatter's index argument is bounded, not its float src.
  6. Inductor's split scatter (idx + (iota & (P-1)) * N into P*N rows) bounds idx
     by N.
  7. bucketize/searchsorted boundaries are generated sorted, and the sorted
     generator survives the T()/S() text form written into repro.py.

Run: python tests/test_input_synthesis.py  (CPU only)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import cast_index_put_values
from full_graph_harness import (
    infer_index_bounds_from_gm,
    infer_sorted_inputs_from_gm,
    placeholder_info_from_gm,
)
from repro_harness import make_inputs_from_config


def test_fp8_inputs_are_generated():
    (x, y) = make_inputs_from_config({"inputs": [
        {"kind": "tensor", "shape": [4, 8], "dtype": "torch.float8_e4m3fn",
         "device": "cpu"},
        {"kind": "tensor", "shape": [4, 8], "stride": [16, 1],
         "dtype": "torch.float8_e5m2", "device": "cpu"},
    ]})
    assert x.dtype == torch.float8_e4m3fn and y.dtype == torch.float8_e5m2
    assert x.float().std() > 0.1
    print("PASS fp8 inputs are generated")


def test_index_put_values_cast_to_destination():
    def f(dst, idx, v):
        return torch.ops.aten.index_put.default(dst, [idx], v.to(dst.dtype))

    gm = make_fx(f, tracing_mode="fake")(
        torch.zeros(4, 3, dtype=torch.bfloat16), torch.tensor([1]),
        torch.randn(1, 3))
    (put,) = [n for n in gm.graph.nodes
              if n.target == torch.ops.aten.index_put.default]
    put.update_arg(2, put.args[2].args[0])  # post-grad graphs drop the cast
    cast_index_put_values(gm)
    out = gm(torch.zeros(4, 3, dtype=torch.bfloat16), torch.tensor([1]),
             torch.randn(1, 3))
    assert out.dtype == torch.bfloat16
    print("PASS index_put values are cast to the destination dtype")


def test_float_label_through_long_chain_is_bounded():
    convert = torch.ops.prims.convert_element_type.default

    def f(logits, label, fill):
        lab = convert(label, torch.float32)
        lab = torch.where(torch.isnan(lab), fill, lab)
        lab = torch.where(lab == float("-inf"), fill, lab)
        lab = torch.where(lab == float("inf"), fill, lab)
        idx = convert(lab, torch.int64).squeeze()
        idx = torch.where(idx != -100, idx, torch.zeros_like(idx))
        return logits.log_softmax(1).gather(1, idx.unsqueeze(1))

    gm = make_fx(f, tracing_mode="fake")(
        torch.randn(16, 19), torch.randn(16, dtype=torch.float64),
        torch.tensor(0.0))
    info = placeholder_info_from_gm(gm)
    bounds = infer_index_bounds_from_gm(gm, info)
    label = [n.name for n in gm.graph.nodes if n.op == "placeholder"][1]
    assert bounds.get(label) == 19, bounds
    print("PASS float label through a long chain is bounded")


def test_lifted_index_constant_is_bounded():
    def f(x, rows):
        return torch.ops.aten.index.Tensor(
            x, [None, torch.ops.aten.lift_fresh_copy.default(rows)])

    gm = make_fx(f, tracing_mode="fake")(
        torch.randn(5, 9, 4), torch.zeros(8, dtype=torch.int64))
    info = placeholder_info_from_gm(gm)
    bounds = infer_index_bounds_from_gm(gm, info)
    rows = [n.name for n in gm.graph.nodes if n.op == "placeholder"][1]
    assert bounds.get(rows) == 9, bounds
    print("PASS lifted index constant is bounded")


def test_scatter_src_is_not_treated_as_an_index():
    convert = torch.ops.prims.convert_element_type.default

    def f(idx, src):
        pos = idx.unsqueeze(1).expand(-1, 3)
        return torch.zeros(7, 3).scatter_add(0, pos, convert(src, torch.float32))

    gm = make_fx(f, tracing_mode="fake")(
        torch.zeros(50, dtype=torch.int64), torch.randn(50, 3, dtype=torch.bfloat16))
    info = placeholder_info_from_gm(gm)
    bounds = infer_index_bounds_from_gm(gm, info)
    idx, src = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    assert bounds.get(idx) == 7, bounds
    assert src not in bounds and "gen" not in info[src], (bounds, info[src])
    print("PASS scatter src is not treated as an index")


def test_split_scatter_index_is_bounded_by_the_split():
    convert = torch.ops.prims.convert_element_type.default

    def f(idx, src):
        rows, splits = 7, 4
        part = torch.ops.prims.iota.default(
            idx.shape[0], start=0, step=1, dtype=torch.int64,
            device=idx.device, requires_grad=False)
        pos = idx.unsqueeze(1).expand(-1, 3) + (part & (splits - 1)).view(-1, 1) * rows
        acc = torch.zeros(rows * splits, 3).scatter_add(0, pos, convert(src, torch.float32))
        return acc.view(splits, rows, 3).sum(0)

    gm = make_fx(f, tracing_mode="fake")(
        torch.zeros(50, dtype=torch.int64), torch.randn(50, 3, dtype=torch.bfloat16))
    info = placeholder_info_from_gm(gm)
    bounds = infer_index_bounds_from_gm(gm, info)
    idx, src = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    assert bounds.get(idx) == 7, bounds
    assert src not in bounds and "gen" not in info[src], (bounds, info[src])
    print("PASS split scatter index is bounded by the split")


def test_split_scatter_bound_follows_a_symbolic_split():
    """On a symbolic graph the split is a SymInt (the table's rows), so the
    bound is that symbol, not half of the split buffer (32*s0)."""
    import sympy

    convert = torch.ops.prims.convert_element_type.default

    def f(idx, src, table):
        rows, splits = table.shape[0], 64
        part = torch.ops.prims.iota.default(
            idx.shape[0], start=0, step=1, dtype=torch.int64,
            device=idx.device, requires_grad=False)
        pos = idx.unsqueeze(1).expand(-1, 3) + (part & (splits - 1)).view(-1, 1) * rows
        acc = torch.zeros(rows * splits, 3).scatter_add(0, pos, convert(src, torch.float32))
        return acc.view(splits, rows, 3).sum(0) + table

    gm = make_fx(f, tracing_mode="symbolic")(
        torch.zeros(500, dtype=torch.int64), torch.randn(500, 3, dtype=torch.bfloat16), torch.randn(7, 3))
    idx, src, table = [n for n in gm.graph.nodes if n.op == "placeholder"]
    exprs = {}
    bounds = infer_index_bounds_from_gm(gm, placeholder_info_from_gm(gm), exprs_out=exprs)
    assert bounds == {idx.name: 7}, bounds
    n, rows = idx.meta["val"].shape[0].node.expr, table.meta["val"].shape[0].node.expr
    bound = sympy.sympify(exprs[idx.name], locals={str(n): n, str(rows): rows})
    assert bound.subs({n: 500, rows: 10}) == 10, exprs
    print("PASS split scatter bound follows a symbolic split")


def test_bucketize_boundaries_are_sorted():
    gm = make_fx(torch.bucketize, tracing_mode="fake")(
        torch.randn(32), torch.randn(9))
    info = placeholder_info_from_gm(gm)
    infer_sorted_inputs_from_gm(gm, info)
    names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    assert "gen" not in info[names[0]], info
    assert info[names[1]]["gen"] == {"kind": "sorted"}, info
    (b,) = make_inputs_from_config({"inputs": [
        {"kind": "tensor", "shape": [9], "dtype": "torch.float32",
         "device": "cpu", "gen": {"kind": "sorted"}}]})
    assert torch.equal(b, b.sort().values)
    print("PASS bucketize boundaries are sorted")


def test_sorted_generator_survives_the_text_signature():
    import tempfile

    from input_codec import render_signature
    from repro_harness import load_shape_configs

    line = render_signature([[[9], "f32", {"gen": ["sorted"]}]])
    assert line == "(T([9], f32, gen=Sorted()))", line
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "repro.py").write_text("# repro\n")
        (Path(tmp) / "shapes.txt").write_text(f"case: {line}\n")
        configs = load_shape_configs(str(Path(tmp) / "repro.py"))
    (spec,) = configs["case"]["inputs"]
    assert spec["gen"] == {"kind": "sorted"}, spec
    (b,) = make_inputs_from_config({"inputs": [dict(spec, device="cpu")]})
    assert torch.equal(b, b.sort().values)
    print("PASS sorted generator survives the text signature")


def test_reshaped_sorted_boundaries_stay_sorted():
    # The detector tags a placeholder reached through reshape, so the generated
    # tensor must be sorted in its flat (storage) order, not just along the last dim.
    gm = make_fx(lambda x, b: torch.bucketize(x, b.reshape(-1)), tracing_mode="fake")(
        torch.randn(32), torch.randn(3, 3))
    info = placeholder_info_from_gm(gm)
    infer_sorted_inputs_from_gm(gm, info)
    names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    assert info[names[1]]["gen"] == {"kind": "sorted"}, info
    (b,) = make_inputs_from_config({"inputs": [
        {"kind": "tensor", "shape": [3, 3], "dtype": "torch.float32",
         "device": "cpu", "gen": {"kind": "sorted"}}]})
    flat = b.reshape(-1)
    assert torch.equal(flat, flat.sort().values), b
    print("PASS reshaped sorted boundaries stay sorted")


def test_sorted_generator_handles_overlapping_and_padded_strides():
    expanded, padded = make_inputs_from_config({"inputs": [
        {"kind": "tensor", "shape": [4], "stride": [0], "dtype": "torch.float32",
         "device": "cpu", "gen": {"kind": "sorted"}},
        {"kind": "tensor", "shape": [3, 3], "stride": [4, 1], "dtype": "torch.bfloat16",
         "device": "cpu", "gen": {"kind": "sorted"}}]})
    assert expanded.stride() == (0,) and torch.equal(expanded, expanded.sort().values)
    assert padded.stride() == (4, 1)
    flat = padded.float().reshape(-1)
    assert torch.equal(flat, flat.sort().values), padded
    print("PASS sorted generator handles overlapping and padded strides")


def test_sorted_generator_survives_shapes_txt_writer():
    from canonicalize_repros import _spec_to_T

    line = _spec_to_T({"shape": [9], "dtype": "torch.float32", "gen": {"kind": "sorted"}})
    assert line == "T([9], f32, gen=Sorted())", line
    print("PASS sorted generator survives the shapes.txt writer")



if __name__ == "__main__":
    test_fp8_inputs_are_generated()
    test_index_put_values_cast_to_destination()
    test_float_label_through_long_chain_is_bounded()
    test_lifted_index_constant_is_bounded()
    test_scatter_src_is_not_treated_as_an_index()
    test_split_scatter_index_is_bounded_by_the_split()
    test_split_scatter_bound_follows_a_symbolic_split()
    test_bucketize_boundaries_are_sorted()
    test_sorted_generator_survives_the_text_signature()
    test_reshaped_sorted_boundaries_stay_sorted()
    test_sorted_generator_handles_overlapping_and_padded_strides()
    test_sorted_generator_survives_shapes_txt_writer()
