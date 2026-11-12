"""End-to-end dynamic capture: capture a model compiled with dynamic shapes,
merge its regions into families, and replay them at the recorded and at new
bindings against eager.

Capture and replay run on CPU (Inductor's C++ backend); the benchmark test
at the end needs a GPU.
"""
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture_hook import install_capture_hook, uninstall_capture_hook  # noqa: E402
from dynamic_benchmark import check_family  # noqa: E402
from input_codec import validate_bindings  # noqa: E402
from merge_captures import temporary_capture_for_merge  # noqa: E402


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(1000, 64)
        self.fc1 = torch.nn.Linear(64, 128)
        self.ln = torch.nn.LayerNorm(128)
        self.fc2 = torch.nn.Linear(128, 32)

    def forward(self, idx, x):
        h = self.emb(idx) + x
        h = self.ln(torch.nn.functional.gelu(self.fc1(h)))
        return self.fc2(h).softmax(-1)


def _mlp_args(batch):
    return torch.randint(0, 1000, (batch,)), torch.randn(batch, 64)


def _capture(fn, arg_sets, root, name="m", dynamic=True):
    """Compile ``fn`` once and run it on each argument set, merging the
    regions into ``root``. Returns the raw capture index."""
    with temporary_capture_for_merge(root, name, suite="toy", mode="infer") as cap:
        install_capture_hook(str(cap.capture_dir), label=name, validate=False,
                             graph_dir=str(Path(root) / "models/toy/infer" / name))
        try:
            torch._dynamo.reset()
            compiled = torch.compile(fn, dynamic=dynamic)
            with torch.no_grad():
                for args in arg_sets:
                    compiled(*args)
        finally:
            uninstall_capture_hook()
            torch._dynamo.reset()
        index = json.loads((cap.capture_dir / "index.json").read_text())
        cap.merge()
    return index


def _regions(root):
    return {
        d.name: json.loads((d / "shapes.json").read_text())
        for d in sorted((Path(root) / "canonical").iterdir())
    }


def _repro_module(repro_file):
    spec = importlib.util.spec_from_file_location("repro_under_test", repro_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repro_cls(repro_file):
    return _repro_module(repro_file).Repro


def _check(root, name, bindings):
    repro = str(Path(root) / "canonical" / name / "repro.py")
    return check_family(repro, _repro_cls(repro), bindings, device="cpu")


@pytest.fixture(scope="module")
def mlp_root(tmp_path_factory):
    """The MLP captured at batch 8, then (fresh compile) at batch 16."""
    root = tmp_path_factory.mktemp("mlp")
    torch.manual_seed(0)
    model = MLP().eval()
    _capture(model, [_mlp_args(8)], root, name="mlp")
    _capture(model, [_mlp_args(16)], root, name="mlp")
    return root


def test_regions_carry_symbols_ranges_and_bindings(mlp_root):
    regions = _regions(mlp_root)
    assert {name.split("_")[0] for name in regions} == {"amax", "pointwise", "var"}
    for name, shapes in regions.items():
        assert shapes["symbols"] == {"s0": {"range": [2, None], "hint": 8}}, name
        for point in shapes["points"]:
            assert point["captured_dynamic"] is True
            # The batch dim is the symbol, not the hint, in every slot.
            assert any(e[0][0] == "s0" for e in point["inputs"]
                       if isinstance(e[0], list) and e[0]), point["inputs"]


def test_same_region_at_two_batch_sizes_is_one_family(mlp_root):
    for name, shapes in _regions(mlp_root).items():
        assert [p["bindings"] for p in shapes["points"]] == [{"s0": 8}, {"s0": 16}], name
        assert len({p["shape_hash"] for p in shapes["points"]}) == 2


def test_replay_matches_eager_at_recorded_and_new_bindings(mlp_root):
    bindings = [{"s0": 8}, {"s0": 16}, {"s0": 3}, {"s0": 37}]
    for name in _regions(mlp_root):
        rows = _check(mlp_root, name, bindings)
        assert [r["numerics"] for r in rows] == ["pass"] * 4, (name, rows)
        assert not any(r["recompiled"] for r in rows), (name, rows)


def test_range_rejects_specialized_sizes(mlp_root):
    name = next(iter(_regions(mlp_root)))
    with pytest.raises(ValueError, match="below range min 2"):
        _check(mlp_root, name, [{"s0": 1}])


def test_different_regions_do_not_merge(tmp_path):
    def norm(x):
        return torch.nn.functional.layer_norm(x, (64,))

    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    _capture(norm, [(torch.randn(8, 64),)], tmp_path, name="norm")
    _capture(rms, [(torch.randn(8, 64),)], tmp_path, name="rms")
    # The same layer norm captured statically is a different kernel than the
    # dynamic one and keeps its own directory.
    _capture(norm, [(torch.randn(8, 64),)], tmp_path, name="norm_static", dynamic=False)
    regions = _regions(tmp_path)
    assert len(regions) == 3, sorted(regions)
    assert sum(1 for s in regions.values() if s.get("symbols")) == 2


def test_guards_restrict_valid_bindings(tmp_path):
    def f(x):
        if x.shape[0] > 16:
            return (x * 2).relu().sum(-1)
        return (x + 1).sum(-1)

    _capture(f, [(torch.randn(32, 64),)], tmp_path)
    (name, shapes), = _regions(tmp_path).items()
    assert shapes["guards"] == ["s0 > 16"]
    assert shapes["symbols"]["s0"]["range"] == [17, None]
    with pytest.raises(ValueError, match="below range min 17"):
        validate_bindings(shapes["symbols"], {"s0": 8, "s1": 64}, shapes["guards"])
    rows = _check(tmp_path, name, [{"s0": 17, "s1": 129}, {"s0": 40, "s1": 3}])
    assert [r["numerics"] for r in rows] == ["pass", "pass"]


def test_symint_scalar_and_symbolic_view_inputs(tmp_path):
    def f(x):
        b = x.shape[0]
        positions = torch.arange(b, dtype=x.dtype).unsqueeze(-1) * (b // 2)
        return (x + positions).view(b, 4, 16).softmax(-1).view(b, 64)

    _capture(f, [(torch.randn(12, 64),)], tmp_path)
    regions = _regions(tmp_path)
    entries = [e for s in regions.values() for p in s["points"] for e in p["inputs"]]
    symints = [e for e in entries if e[0] == "I"]
    assert ["I", 12, "s0"] in symints, entries
    assert ["I", 6, "floor(s0/2)"] in symints, entries
    for name in regions:
        rows = _check(tmp_path, name, [{"s0": 12}, {"s0": 5}, {"s0": 40}])
        assert [r["numerics"] for r in rows] == ["pass"] * 3, (name, rows)
        assert not any(r["recompiled"] for r in rows)


def test_view_input_replays_at_its_symbolic_offset(tmp_path):
    def f(x):
        return x.relu().sum(-1)

    _capture(f, [(torch.randn(36, 64)[12:24],)], tmp_path)
    (name, shapes), = _regions(tmp_path).items()
    (entry,) = shapes["points"][0]["inputs"]
    offset = entry[2]["off"]
    assert offset in shapes["symbols"], entry
    base = shapes["points"][0]["bindings"]
    rows = _check(tmp_path, name, [base, {**base, "s0": 5, offset: 100}])
    assert [r["numerics"] for r in rows] == ["pass", "pass"], rows


def test_aliased_views_replay_at_a_larger_binding(tmp_path):
    def f(q, k):
        return (q * k).sum(-1)

    packed = torch.randn(12, 128)
    _capture(f, [(packed[:, :64], packed[:, 64:])], tmp_path)
    (name, shapes), = _regions(tmp_path).items()
    point = shapes["points"][0]
    assert point.get("alias_group_nbytes"), point
    batch = point["inputs"][0][0][0]
    rows = _check(tmp_path, name, [point["bindings"], {**point["bindings"], batch: 40}])
    assert [r["numerics"] for r in rows] == ["pass", "pass"], rows


def test_guard_floor_of_rational_sum_is_exact():
    from dynamic_shape_replay import _exact_floors, _symint_expr_evaluator
    from input_codec import _sympify_expr

    # 200/3 + 200/3 + 200/3 is 199.99999999999997 in floats.
    text = "Eq(floor(s1/3 + s2/3 + s3/3), 200)"
    guard = _symint_expr_evaluator(_exact_floors(_sympify_expr(text)))
    assert guard({"s1": 200, "s2": 200, "s3": 200})
    assert _symint_expr_evaluator(_exact_floors(_sympify_expr("ceiling(s1/3)")))({"s1": 200}) == 67


def test_point_exclusion_inside_a_bounded_range(tmp_path):
    """Tracing a [53, s0, 256] permute against [53, 256] weights guards
    s0 != 53. The family keeps the exclusion and a bounded range (as an
    exported model's), replays inside both and rejects bindings outside."""
    def f(x, w, b):
        p = x.permute(1, 0, 2)
        y = torch.nn.functional.layer_norm(p.float(), (256,)).to(x.dtype) * w + b
        return (p * y.sigmoid()).permute(1, 0, 2)

    x = torch.randn(53, 96, 256, dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(x, 1)
    w, b = torch.randn(53, 256, dtype=torch.bfloat16), torch.randn(53, 256, dtype=torch.bfloat16)
    _capture(f, [(x, w, b)], tmp_path, dynamic=None)
    (name, shapes), = _regions(tmp_path).items()
    assert shapes["guards"] == ["Ne(s0, 53)"], shapes["guards"]
    shapes["symbols"]["s0"]["range"] = [48, 4096]
    (tmp_path / "canonical" / name / "shapes.json").write_text(json.dumps(shapes))
    rows = _check(tmp_path, name, [{"s0": 96}, {"s0": 60}])
    assert [r["numerics"] for r in rows] == ["pass", "pass"], rows
    for s0, error in [(53, "violates guard"), (40, "below range min 48"), (5000, "above range max 4096")]:
        with pytest.raises(ValueError, match=error):
            _check(tmp_path, name, [{"s0": s0}])


def test_bounded_range_tolerates_guards_the_replay_compile_adds(tmp_path):
    """The replay compile of a reduction guards its width (s1 <= 4096 on CPU);
    strict min/max marking of a bounded range rejected that at every binding."""
    _capture(lambda x: x.relu().sum(-1), [(torch.randn(12, 64),)], tmp_path)
    (name, shapes), = _regions(tmp_path).items()
    width = next(s for s, d in shapes["symbols"].items() if d["hint"] == 64)
    shapes["symbols"][width]["range"] = [2, 100000]
    (tmp_path / "canonical" / name / "shapes.json").write_text(json.dumps(shapes))
    point = shapes["points"][0]["bindings"]
    rows = _check(tmp_path, name, [point, {**point, width: 100}])
    assert [r["numerics"] for r in rows] == ["pass", "pass"], rows


def test_floor_over_a_symbolic_denominator_builds_a_spec():
    from dynamic_shape_replay import _shape_env_spec_from_contract

    # A reshape to (-1, 200, 192) of (s1, ...) rows records floor(200*s0/s1).
    contract = {
        "points": [{"inputs": [[["s0", "floor(200*s0/s1)"], "f32"], ["I", 128, "s1"]],
                    "bindings": {"s0": 128, "s1": 128}, "captured_dynamic": True}],
        "symbols": {"s0": {"range": [2, None], "hint": 128},
                    "s1": {"range": [2, None], "hint": 128}},
    }
    spec = str(_shape_env_spec_from_contract(contract)[0])
    assert re.search(r"\(\(200\*s0#\d+\)//s1#\d+\)", spec), spec


@pytest.mark.parametrize("text, want", [
    ("floor(4096/s0)", 4096 // 3), ("ceiling(4096/s0)", -(-4096 // 3)), ("floor(4096/(s0*s1))", 4096 // 21)])
def test_floor_of_a_constant_over_symbols(text, want):
    from dynamic_shape_replay import _exact_floors, _shape_env_spec_from_contract, _symint_expr_evaluator
    from input_codec import _sympify_expr

    contract = {
        "points": [{"inputs": [[["s0", text], "f32"], ["I", 7, "s1"]],
                    "bindings": {"s0": 3, "s1": 7}, "captured_dynamic": True}],
        "symbols": {"s0": {"range": [2, None], "hint": 3}, "s1": {"range": [2, None], "hint": 7}},
    }
    _shape_env_spec_from_contract(contract)
    assert _symint_expr_evaluator(_exact_floors(_sympify_expr(text)))({"s0": 3, "s1": 7}) == want


class _FloorNorm(torch.nn.Module):
    """A LayerNorm over rows reshaped by a symbol that is never a bare dim:
    x is [s0, floor(200*s1/s0), 1, 192]."""

    def forward(self, x, s0, rows, w, b):
        v = x.view(s0, rows, 192)
        mean = v.mean(-1, keepdim=True)
        var = ((v - mean) ** 2).mean(-1, keepdim=True)
        return (v - mean) * torch.rsqrt(var + 1e-5) * w + b


class _ScaledDim(torch.nn.Module):
    """x is [2, 32*s0, 128]; s0 only arrives as a SymInt input."""

    def forward(self, x, s0, w):
        return x.view(2, s0, 32, 128).mean(2) @ w.t()


class _HalfOpen(torch.nn.Module):
    """s0 has the range [3, inf), as after a guard s0 - 1 >= 2."""

    def forward(self, x):
        return (x[1:] - x[:-1]).sum(0), x * 2


def _floor_case(s0, s1):
    contract = {
        "points": [{"inputs": [
            [["s0", "floor(200*s1/s0)", 1, 192], "f32", {"st": ["192*floor(200*s1/s0)", 192, 192, 1]}],
            ["I", 128, "s0"], ["I", 200, "floor(200*s1/s0)"], [[192], "f32"], [[192], "f32"]],
            "bindings": {"s0": 128, "s1": 128}, "captured_dynamic": True}],
        "symbols": {"s0": {"range": [2, None], "hint": 128}, "s1": {"range": [2, None], "hint": 128}},
        "guards": ["Eq(Mod(200*s1, s0), 0)"],
    }
    w, b, src = torch.randn(192), torch.randn(192), torch.randn(s1, 200, 192)
    x = src.view(s0, -1, 1, 192)

    def original(a, src, w, b):
        x = src.view(a.size(0), -1, 1, 192)
        return _FloorNorm()(x, a.size(0), x.size(1), w, b)

    return (_FloorNorm(), contract, original, [torch.randn(s0), src, w, b], [(0, 0), (1, 0)],
            [x, s0, x.size(1), w, b], {"s0": s0, "s1": s1})


def _scaled_case(s0):
    contract = {
        "points": [{"inputs": [[[2, "32*s0", 128], "f32"], ["I", 64, "s0"], [[64, 128], "f32"]],
                    "bindings": {"s0": 64}, "captured_dynamic": True}],
        "symbols": {"s0": {"range": [2, None], "hint": 64}},
    }
    w, src = torch.randn(64, 128), torch.randn(2, s0, 32, 128)

    def original(src, w):
        return _ScaledDim()(src.view(2, -1, 128), src.size(1), w)

    return _ScaledDim(), contract, original, [src, w], [(0, 1)], [src.view(2, -1, 128), s0, w], {"s0": s0}


def _half_open_case(s0):
    contract = {
        "points": [{"inputs": [[["s0", 64], "f32"]], "bindings": {"s0": 64}, "captured_dynamic": True}],
        "symbols": {"s0": {"range": [3, None], "hint": 64}},
        "guards": ["s0 - 1 >= 2"],
    }
    x = torch.randn(s0, 64)

    def original(x):
        torch._check(x.size(0) >= 3)
        return _HalfOpen()(x)

    return _HalfOpen(), contract, original, [x], [(0, 0)], [x], {"s0": s0}


class _ColumnSlice(torch.nn.Module):
    """x is [13, 64] with strides [s0, 1]: a column slice of a buffer whose
    width is dynamic, so every size is static and only the stride is not."""

    def forward(self, x, y):
        return (x * 2).sum(-1) + y


def _stride_case(width):
    contract = {
        "points": [{"inputs": [[[13, 64], "f32", {"st": ["s0", 1]}], [[13], "f32"]],
                    "bindings": {"s0": 80}, "captured_dynamic": True}],
        "symbols": {"s0": {"range": [2, None], "hint": 80}},
    }
    buf, y = torch.randn(13, width), torch.randn(13)

    def original(buf, y):
        return _ColumnSlice()(buf[:, :64], y)

    return _ColumnSlice(), contract, original, [buf, y], [(0, 1)], [buf[:, :64], y], {"s0": width}


class _FloorRowsFlatten(torch.nn.Module):
    """x is [s0, floor(200*s1/s0), 64], normalized in place and then flattened
    to [s0*floor(200*s1/s0), 64]. The flatten is a view only if Inductor sees
    the expression as x's size, not a fresh symbol tied to it by a guard."""

    def forward(self, x, w, n):
        mean = x.mean(-1, keepdim=True)
        y = (x - mean) * torch.rsqrt(((x - mean) ** 2).mean(-1, keepdim=True) + 1e-5) * w
        return y.sigmoid().view(n, 64)


def _floor_reshape_case(s0, s1):
    rows = "floor(200*s1/s0)"
    contract = {
        "points": [{"inputs": [
            [["s0", rows, 64], "f32", {"st": [f"64*{rows}", 64, 1]}], [[64], "f32"], ["I", 25600, f"s0*{rows}"]],
            "bindings": {"s0": 128, "s1": 128}, "captured_dynamic": True}],
        "symbols": {"s0": {"range": [2, None], "hint": 128}, "s1": {"range": [2, None], "hint": 128}},
        "guards": ["Eq(Mod(200*s1, s0), 0)"],
    }
    w, src = torch.randn(64), torch.randn(s1, 200, 64)
    x = src.view(s0, -1, 64)

    def original(a, src, w):
        x = src.view(a.size(0), -1, 64)
        return _FloorRowsFlatten()(x, w, x.size(0) * x.size(1))

    return (_FloorRowsFlatten(), contract, original, [torch.randn(s0), src, w], [(0, 0), (1, 0)],
            [x, w, x.numel() // 64], {"s0": s0, "s1": s1})


# name -> (case factory, sizes compiled at, sizes replayed at without recompiling)
_BACKED_CASES = {
    "floor_only": (_floor_case, (64, 96), (32, 48)),
    "floor_reshape": (_floor_reshape_case, (64, 96), (32, 48)),
    "scaled_dim": (_scaled_case, (40,), (24,)),
    "half_open_range": (_half_open_case, (37,), (5,)),
    "stride_only": (_stride_case, (80,), (96,)),
}


@pytest.mark.parametrize("case", sorted(_BACKED_CASES))
def test_backed_plan_matches_a_direct_dynamic_compile(case):
    """These families fell back to unbacked ShapesSpec replay, which can
    change Inductor's fusion. The backed plan replays them with the kernel
    count of the original graph compiled with dynamic shapes, and serves a
    second binding without recompiling."""
    import torch._inductor.metrics as metrics
    from dynamic_shape_replay import _dynamic_replay_args, _prepare_dynamic_replay_from_contract

    factory, sizes, other_sizes = _BACKED_CASES[case]
    region, contract, original, original_args, marks, inputs, binding = factory(*sizes)
    prepared = _prepare_dynamic_replay_from_contract(region, contract)
    assert prepared["kind"] == "backed"

    torch._dynamo.reset()
    metrics.reset()
    for pos, dim in marks:
        torch._dynamo.mark_dynamic(original_args[pos], dim)
    reference = torch.compile(original)(*original_args)
    expected = metrics.generated_kernel_count

    torch._dynamo.reset()
    metrics.reset()
    compiled = torch.compile(prepared["module"])
    torch.testing.assert_close(compiled(*_dynamic_replay_args(inputs, binding, prepared)), reference)
    assert metrics.generated_kernel_count == expected
    *_, inputs, binding = factory(*other_sizes)
    with torch._dynamo.config.patch(error_on_recompile=True):
        compiled(*_dynamic_replay_args(inputs, binding, prepared))


def test_families_off_the_bare_dim_path_pick_the_backed_plan():
    from dynamic_shape_replay import _backed_replay_plan_from_contract

    def plan(inputs, symbols, guards=()):
        point = {"inputs": inputs, "bindings": {k: v["hint"] for k, v in symbols.items()}, "captured_dynamic": True}
        return _backed_replay_plan_from_contract({"points": [point], "symbols": symbols, "guards": list(guards)})

    rows = "floor(200*s1/s0)"
    floor_only = plan(
        [[["s0", rows, 1, 192], "f16", {"st": [f"192*{rows}", 192, 192, 1]}], ["I", 128, "s0"], ["I", 200, rows],
         [[192], "f16"], ["I", 25600, f"s0*{rows}"]],
        {"s0": {"range": [2, None], "hint": 128}, "s1": {"range": [2, None], "hint": 128}},
        ["Eq(38400*s1, 192*s0*floor(200*s1/s0))", "Eq(Mod(200*s1, s0), 0)", f"Ne({rows}, 1)",
         f"Eq(Max(1, {rows}), {rows})", f"192*Max(1, {rows}) > 192"])
    # x's size is an expression, so x is rebuilt from a 1-D base and s0 too
    # needs a carrier.
    assert floor_only is not None and floor_only["aux"] == ("s0", "s1")
    assert [spec[0] for spec in floor_only["rebased"]] == [0]
    scaled = plan([[[2, "32*s0", 128], "f16", {"st": ["4096*s0", 128, 1]}], ["I", 1190, "s0"]],
                  {"s0": {"range": [2, None], "hint": 1190}})
    assert scaled is not None and scaled["aux"] == ("s0",)
    half_open = plan([[["s0"], "i64"], [["s1", 80], "f16"], ["I", 127, "s1 - 1"]],
                     {"s0": {"range": [2, None], "hint": 128}, "s1": {"range": [3, None], "hint": 128}}, ["s1 - 1 >= 2"])
    assert half_open is not None and not half_open["aux"]
    assert "s1 >= 3" in [text for text, _ in half_open["guards"]]
    # 0/1-capable ranges still need ShapesSpec.
    assert plan([[["s0"], "f32"]], {"s0": {"range": [0, None], "hint": 4}}) is None


def test_torch_max_min_in_captured_strides_evaluate():
    from dynamic_shape_replay import _symint_expr_evaluator
    from input_codec import _sympify_expr

    # Captured exprs carry torch.utils._sympy.functions.Max/Min, not sympy's.
    stride = _symint_expr_evaluator(_sympify_expr("864*Max(1, s1)"))
    assert stride is not None and stride({"s1": 3}) == 2592
    assert _symint_expr_evaluator(_sympify_expr("Min(s0, 7)"))({"s0": 9}) == 7


def test_output_aliasing_another_output_fails_with_an_explicit_message(tmp_path):
    """y and y.permute(1, 0) are both outputs. At s0 == 512, next to the static
    512, AOTAutograd's aliasing check specializes s0. That is unsupported;
    the error must say so instead of a bare ConstraintViolationError."""
    def f(x, w):
        y = (x @ w).relu() * 2
        return y, y.permute(1, 0)

    x = torch.randn(300, 64)
    torch._dynamo.mark_dynamic(x, 0)
    _capture(f, [(x, torch.randn(64, 512))], tmp_path, dynamic=None)
    (name, shapes), = _regions(tmp_path).items()
    # This capture records Ne(s0, 512); an exported graph's family may not.
    shapes.pop("guards", None)
    (tmp_path / "canonical" / name / "shapes.json").write_text(json.dumps(shapes))
    assert [r["numerics"] for r in _check(tmp_path, name, [{"s0": 300}])] == ["pass"]
    torch._dynamo.reset()
    with pytest.raises(RuntimeError, match=r"unsupported: output 1 \(512, 512\) aliases output 0"):
        _check(tmp_path, name, [{"s0": 512}])


def test_family_identity_includes_symbolic_generator_bounds():
    from merge_captures import _hintfree_inputs_signature as sig

    def ids(gen):
        return [[["s0"], "i64", {"gen": gen}]]

    # Same family: bounds that differ only by their value at the point.
    assert sig(ids(["index", 0, 8, "s0"])) == sig(ids(["index", 0, 16, "s0"]))
    assert sig(ids(["runs", 0, 40, 9, 9, 1])) == sig(ids(["runs", 0, 57, 12, 12, 1]))
    # Different families: a batch-bounded id vs a fixed-table id, or a permutation.
    assert sig(ids(["index", 0, 8, "s0"])) != sig(ids(["index", 0, 8]))
    assert sig(ids(["perm", 8, "s0"])) != sig(ids(["index", 0, 8, "s0"]))


def test_tautological_guards_do_not_keep_stale_symbol_names():
    from merge_captures import _canonicalize_symbols

    _symbols, _inputs, guards, _ = _canonicalize_symbols(
        {"s13": {}, "s7": {}}, [[["s13", "s7"], "f32"]],
        ["Eq(((s13 + s7)//9), floor(s13/9 + s7/9))", "Eq(Mod(s13 + s7, 9), 0)"])
    assert guards == ["Eq(Mod(s0 + s1, 9), 0)"]


def test_static_and_dynamic_rows_place_views_alike(tmp_path):
    import repro_harness

    def f(x):
        return x.relu().sum(-1)

    _capture(f, [(torch.randn(36, 64)[12:24],)], tmp_path)
    (name,) = _regions(tmp_path)
    repro = str(tmp_path / "canonical" / name / "repro.py")
    module = _repro_module(repro)

    class Stop(Exception):
        pass

    seen = []
    real = repro_harness.make_inputs_from_config

    def spy(config, storage_offsets=False):
        seen.append(storage_offsets)
        config = dict(config, inputs=[dict(spec, device="cpu") for spec in config["inputs"]])
        inputs = real(config, storage_offsets=storage_offsets)
        assert inputs[0].storage_offset() == (768 if storage_offsets else 0)
        raise Stop

    for flag in ("--static-only", "--dynamic-only"):
        with mock.patch.object(repro_harness, "make_inputs_from_config", spy), pytest.raises(Stop):
            repro_harness.benchmark_repro(repro, module.Repro, module.make_inputs,
                                          args=[flag, "--no-gpu-lock"])
    assert seen == [True, True]


def test_dynamic_rows_are_checked_against_fp64(tmp_path):
    """A padded view input (stride s1 > width) made the float64 run of the
    replay wrapper fail its stride check, so dynamic rows were compared
    strictly against eager while static rows had an fp64 reference."""
    import dynamic_benchmark

    _capture(lambda x: x.float().pow(2).mean(-1).rsqrt(), [(torch.randn(12, 80, dtype=torch.bfloat16)[:, :64],)], tmp_path)
    (name, shapes), = _regions(tmp_path).items()
    assert any(isinstance(e[2], dict) and "st" in e[2] for e in shapes["points"][0]["inputs"]), shapes
    summaries = []
    real = dynamic_benchmark._dynamic_row_numerics

    def record(*args, **kwargs):
        summaries.append(real(*args, **kwargs))
        return summaries[-1]

    with mock.patch.object(dynamic_benchmark, "_dynamic_row_numerics", record):
        rows = _check(tmp_path, name, [shapes["points"][0]["bindings"]])
    assert [r["numerics"] for r in rows] == ["pass"], rows
    assert [(s["reference"], s["fp64_reference_error"]) for s in summaries] == [("fp64", None)], summaries


def test_a_failing_dynamic_half_keeps_the_static_rows(tmp_path):
    import repro_harness

    _capture(lambda x: x.relu().sum(-1), [(torch.randn(12, 64),)], tmp_path)
    (name,) = _regions(tmp_path)
    repro = str(tmp_path / "canonical" / name / "repro.py")
    real = repro_harness.make_inputs_from_config

    def on_cpu(config, storage_offsets=None):
        return real(dict(config, inputs=[dict(spec, device="cpu") for spec in config["inputs"]]), storage_offsets)

    def specialized(*args, **kwargs):
        raise RuntimeError("the replay compile specialized s0")

    (config,) = repro_harness.load_shape_configs(repro).values()
    with mock.patch.object(repro_harness, "_run_bound_benchmark", specialized), \
            mock.patch.object(repro_harness, "make_inputs_from_config", on_cpu), \
            mock.patch.object(repro_harness, "count_kernels", lambda *a, **k: (1, ["k"])):
        def bench(*flags):
            return repro_harness.benchmark_repro(
                repro, _repro_cls(repro), lambda: on_cpu(config), args=["--count-kernels-only", "--no-gpu-lock", *flags])

        rows = bench()
        assert rows.pop("::dynamic") == {"mode": "dynamic", "error": "RuntimeError: the replay compile specialized s0"}
        assert [row["n_kernels"] for row in rows.values()] == [1], rows
        with pytest.raises(RuntimeError, match="specialized"):
            bench("--dynamic-only")


def test_family_points_place_views_at_their_offset_by_default(tmp_path):
    """Callers that build a point's inputs with the default flags
    (bench_parallel's static rows) place views like repro.py's rows."""
    from repro_harness import load_shape_configs, make_inputs_from_config

    _capture(lambda x: x.relu().sum(-1), [(torch.randn(36, 64)[12:24],)], tmp_path)
    (name,) = _regions(tmp_path)
    (config,) = load_shape_configs(str(tmp_path / "canonical" / name / "repro.py")).values()
    config = dict(config, inputs=[dict(spec, device="cpu") for spec in config["inputs"]])
    assert make_inputs_from_config(config)[0].storage_offset() == 768
    static = {k: v for k, v in config.items() if k != "bindings"}
    assert make_inputs_from_config(static)[0].storage_offset() == 0


def test_padded_view_input_with_a_stride_zero_dim():
    from repro_harness import make_inputs_from_config

    config = {"inputs": [{"kind": "tensor", "shape": [4, 3], "stride": [0, 1],
                          "storage_offset": 5, "dtype": "float32", "device": "cpu"}]}
    (t,) = make_inputs_from_config(config, storage_offsets=True)
    assert (t.shape, t.stride(), t.storage_offset()) == ((4, 3), (0, 1), 5)


def _lowering_pass(fn):
    """Run ``fn(graph)`` while the model is lowered: after the capture pass,
    where Inductor adds its own guards."""
    import torch._inductor.config as inductor_config

    return inductor_config.patch(post_grad_custom_post_pass=fn)


def _first_tensor(graph):
    return next(n.meta["val"] for n in graph.find_nodes(op="placeholder")
                if isinstance(n.meta.get("val"), torch.Tensor))


def test_size_specialized_during_lowering_stays_dynamic(tmp_path):
    """A lowering that guards a size to a constant is a codegen choice of the
    model's compile; the family stays dynamic and replays at its point."""
    def specialize_batch(graph):
        assert _first_tensor(graph).shape[0] == 12

    def f(x):
        return x.relu().sum(-1)

    with _lowering_pass(specialize_batch):
        _capture(f, [(torch.randn(12, 64),)], tmp_path)
    (name, shapes), = _regions(tmp_path).items()
    assert "guards" not in shapes, shapes["guards"]
    batch = next(s for s, d in shapes["symbols"].items() if d["hint"] == 12)
    point = shapes["points"][0]["bindings"]
    rows = _check(tmp_path, name, [point, {**point, batch: 5}])
    assert [r["numerics"] for r in rows] == ["pass", "pass"], rows


def test_index_width_bound_stays_out_of_the_family(tmp_path):
    """Inductor's 32-bit indexing check over a storage size simplifies to a
    bound that is no longer INT32_MAX (4096*s0 + s1 <= 2147487742)."""
    from torch._inductor.sizevars import SizeVarAllocator

    def storage_bound(graph):
        x = _first_tensor(graph)
        rows, cols = (d.node.expr for d in x.shape)
        SizeVarAllocator(x.shape[0].node.shape_env).check_leq(4096 * (rows - 1) + cols, 2**31 - 1)

    with _lowering_pass(storage_bound):
        _capture(lambda x: x.relu() * 2, [(torch.randn(12, 64),)], tmp_path)
    (name, shapes), = _regions(tmp_path).items()
    assert "guards" not in shapes, shapes["guards"]
    assert all(d["range"] == [2, None] for d in shapes["symbols"].values()), shapes["symbols"]


def test_neighbouring_lowering_guards_do_not_split_a_family(tmp_path):
    """m2's reduction lowering guards the shared symbol (s1 <= 4096 on CPU).
    The pointwise region it shares with m1 is still one family, and the
    reduction replays past that threshold."""
    def m1(x, y):
        return (x.sin() * 3,)

    def m2(x, y):
        return x.sin() * 3, y.cos().sum(-1)

    args = (torch.randn(12, 64), torch.randn(12, 64))
    _capture(m1, [args], tmp_path, name="m1")
    _capture(m2, [args], tmp_path, name="m2")
    regions = _regions(tmp_path)
    (pointwise,) = [s for n, s in regions.items() if n.startswith("pointwise")]
    assert {m for p in pointwise["points"] for m in p["models"]} == {"toy/infer/m1", "toy/infer/m2"}
    for name, shapes in regions.items():
        assert "guards" not in shapes, (name, shapes["guards"])
        assert all(d["range"] == [2, None] for d in shapes["symbols"].values()), name
    (reduction,) = [n for n in regions if n.startswith("sum")]
    shapes = regions[reduction]
    width = next(s for s, d in shapes["symbols"].items() if d["hint"] == 64)
    point = shapes["points"][0]["bindings"]
    rows = _check(tmp_path, reduction, [point, {**point, width: 8192}])
    assert [r["numerics"] for r in rows] == ["pass", "pass"], rows


def test_size_one_dims_are_specialized(tmp_path):
    def f(x, y):
        return (x * y).sum(-1)

    _capture(f, [(torch.randn(1, 64), torch.randn(1, 64))], tmp_path)
    (shapes,) = _regions(tmp_path).values()
    # Dynamo specializes sizes 0 and 1: the batch dim stays a literal 1 and
    # only the 64 dim is a symbol.
    assert shapes["points"][0]["inputs"][0][0] == [1, "s0"]
    assert list(shapes["symbols"]) == ["s0"]


def test_region_symbols_are_scoped_to_its_inputs(tmp_path):
    def f(x, y):
        return x.relu().sum(-1), (y * 2).sigmoid()

    _capture(f, [(torch.randn(12, 64), torch.randn(5, 7))], tmp_path)
    for shapes in _regions(tmp_path).values():
        assert sorted(shapes["symbols"]) == ["s0", "s1"], shapes["symbols"]
        assert len(shapes["points"][0]["bindings"]) == 2


def test_unbacked_symbol_needs_an_explicit_binding(tmp_path):
    def f(x):
        idx = (x[:, 0] > 0).nonzero().squeeze(-1)
        return x[idx].relu().sum(-1)

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        index = _capture(f, [(torch.randn(16, 64),)], tmp_path)
    ops = {op for entry in index["captured"] for op in entry["origin_ops"]}
    # Size reads and runtime asserts are not regions of their own.
    assert not ops & {"aten.sym_size.int", "aten._assert_scalar.default"}, ops
    regions = _regions(tmp_path)
    (name, shapes), = [(n, s) for n, s in regions.items() if n.startswith("sum")]
    unbacked = [k for k, v in shapes["symbols"].items() if v.get("unbacked")]
    assert len(unbacked) == 1, shapes["symbols"]
    (point,) = shapes["points"]
    assert point["requires_binding"] == unbacked
    binding = {**point["bindings"], unbacked[0]: 5}
    rows = _check(tmp_path, name, [binding, {**binding, unbacked[0]: 9}])
    assert [r["numerics"] for r in rows] == ["pass", "pass"]


def test_permutation_inputs_are_detected_and_follow_the_batch(tmp_path):
    """index_put of arange into an empty buffer needs a true permutation;
    random ids leave slots uninitialized."""
    from full_graph_harness import prepare_full_graph_execution

    def f(perm, x):
        n = perm.shape[0]
        out = torch.empty(n, dtype=torch.int64).index_put_((perm,), torch.arange(n))
        return x[out] * 2

    _capture(f, [(torch.randperm(16), torch.randn(16, 8))], tmp_path)
    (name, shapes), = _regions(tmp_path).items()
    perm = shapes["points"][0]["inputs"][1]
    assert perm[2]["gen"] == ["perm", 16, "s0"], perm
    base = shapes["points"][0]["bindings"]
    rows = _check(tmp_path, name, [{**base, "s0": s0} for s0 in (16, 12, 40)])
    assert [r["numerics"] for r in rows] == ["pass"] * 3, rows

    (graph,) = (tmp_path / "models/toy/infer/m").glob("full_graph_*.py")
    sidecar = json.loads(graph.with_suffix(".meta.json").read_text())
    (batch,) = [s for s, d in sidecar["symbols"].items() if d["hint"] == 16]
    execution = prepare_full_graph_execution(graph, default_device="cpu")
    hints = {s: d["hint"] for s, d in sidecar["symbols"].items()}
    ids = next(a for a in execution.args({**hints, batch: 12})
               if torch.is_tensor(a) and a.dtype == torch.int64)
    assert sorted(ids.tolist()) == list(range(12))


def test_index_bound_follows_a_symbolic_table(tmp_path):
    from full_graph_harness import prepare_full_graph_execution

    def f(table, idx, w):
        return (table[idx] * w).relu().sum(-1)

    args = (torch.randn(280, 32), torch.randint(0, 280, (20,)), torch.randn(20, 32))
    _capture(f, [args], tmp_path)
    (name, shapes), = _regions(tmp_path).items()
    table, idx, _ = shapes["points"][0]["inputs"]
    # Reading the table's rows must not specialize them.
    assert table[0] == ["s0", "s1"], table
    assert idx[2]["gen"] == ["index", 0, 280, "s0"], idx
    rows = _check(tmp_path, name, [{"s0": 7, "s1": 32, "s2": 50}])
    assert [r["numerics"] for r in rows] == ["pass"], rows

    (graph,) = (tmp_path / "models/toy/infer/m").glob("full_graph_*.py")
    execution = prepare_full_graph_execution(graph, default_device="cpu")
    small = {s: 7 for s in json.loads(graph.with_suffix(".meta.json").read_text())["symbols"]}
    with torch.no_grad():
        execution.module(*execution.args(small))


def test_unbacked_column_major_table(tmp_path):
    """A nonzero() result is a column-major region input of unbacked size:
    its contiguity and the bound of indices into it are read without a hint."""
    def f(x, idx, w):
        table = (x > 0).nonzero().to(torch.float32)
        return (table[idx] * w).relu().sum(-1)

    x = torch.randn(8, 8)
    args = (x, torch.randint(0, int((x > 0).sum()), (20,)), torch.randn(20, 2))
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        index = _capture(f, [args], tmp_path)
    assert not index["dropped"], index["dropped"]
    (name, shapes), = [(n, s) for n, s in _regions(tmp_path).items() if n.startswith("sum")]
    (u,) = [k for k, v in shapes["symbols"].items() if v.get("unbacked")]
    (point,) = shapes["points"]
    table, idx, _ = point["inputs"]
    assert table[2]["st"] == [1, u], table
    assert idx[2]["gen"][3] == u, idx
    rows = _check(tmp_path, name, [{**point["bindings"], u: n} for n in (5, 30)])
    assert [r["numerics"] for r in rows] == ["pass", "pass"], rows


def test_full_graph_replays_one_artifact_at_two_bindings(mlp_root):
    from full_graph_harness import prepare_full_graph_execution

    (graph,) = sorted((mlp_root / "models/toy/infer/mlp").glob("full_graph_*.py"))
    sidecar = json.loads(graph.with_suffix(".meta.json").read_text())
    assert sidecar["symbols"], sidecar.keys()
    execution = prepare_full_graph_execution(graph, default_device="cpu")
    assert execution.dynamic
    (symbol,) = sidecar["symbols"]
    torch._dynamo.reset()
    compiled = execution.compile()
    from torch._dynamo.utils import counters
    before = counters["stats"]["unique_graphs"]
    with torch.no_grad():
        for value in (8, 24, 5):
            args = execution.args({symbol: value})
            torch.testing.assert_close(compiled(*args), execution.module(*args))
    assert counters["stats"]["unique_graphs"] - before == 1


def test_recapturing_a_dynamic_full_graph_is_a_fixed_point(mlp_root, tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import recapture_full_graphs

    models = mlp_root / "models"
    assert recapture_full_graphs.main([
        str(models), "--models-root", str(models), "--canonical-root",
        str(tmp_path), "--no-validate", "--no-isolate"]) == 0
    # The saved graph is the batch-16 compile: the same families, one point.
    first = {name: s for name, s in _regions(mlp_root).items()}
    again = _regions(tmp_path)
    assert sorted(again) == sorted(first)
    for name, shapes in again.items():
        assert [p["bindings"] for p in shapes["points"]] == [{"s0": 16}]
        assert shapes["points"][0]["inputs"] == first[name]["points"][1]["inputs"]


LEGACY_GRAPH = """
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s16)", arg1_1: "f32[64, s16][s16, 1]cpu"):
        mul: "f32[64, s16][s16, 1]cpu" = torch.ops.aten.mul.Tensor(arg1_1, 2.0)
        sum_1: "f32[64][1]cpu" = torch.ops.aten.sum.dim_IntList(mul, [1])
        return (sum_1,)
"""


@pytest.mark.parametrize("loader", ["repartition", "ingest"])
def test_legacy_symbolic_graph_retraces_at_its_recorded_sizes(tmp_path, loader):
    """A graph saved before dynamic capture names its symbols in the forward
    annotations while its sidecar holds the concrete inputs it ran with. Its
    regions take those values as their bindings (not a template value)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    graph = tmp_path / "full_graph_000.py"
    graph.write_text(LEGACY_GRAPH)
    graph.with_suffix(".meta.json").write_text(json.dumps({
        "schema_version": 2,
        "inputs": [["sym", 24], [[64, 24], "f32", {"n": "arg1_1", "dev": "cpu"}]],
    }))
    if loader == "repartition":
        from repartition_from_graphs import load_graph_module
    else:
        from ingest_tlparse import load_graph_module
    gm = load_graph_module(graph)
    vals = [n.meta["val"] for n in gm.graph.find_nodes(op="placeholder")]
    assert isinstance(vals[0], torch.SymInt)
    assert vals[1].shape[1].node.expr == vals[0].node.expr
    assert vals[0].node.hint == 24


SIZE_READ_GRAPH = """
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[s16, 64][64, 1]cpu", arg1_1: "f32[s16, 64][64, 1]cpu"):
        sym_size_int: "Sym(s16)" = torch.ops.aten.sym_size.int(arg0_1, 0)
        view: "f32[1, s16, 64][64*s16, 64, 1]cpu" = torch.ops.aten.view.default(arg0_1, [1, sym_size_int, 64])
        relu: "f32[1, s16, 64][64*s16, 64, 1]cpu" = torch.ops.aten.relu.default(view)
        cat: "f32[2*s16, 64][64, 1]cpu" = torch.ops.aten.cat.default([arg1_1, arg1_1])
        view_1: "f32[2, s16, 64][64*s16, 64, 1]cpu" = torch.ops.aten.view.default(cat, [2, sym_size_int, 64])
        sigmoid: "f32[2, s16, 64][64*s16, 64, 1]cpu" = torch.ops.aten.sigmoid.default(view_1)
        return (relu, sigmoid)
"""


def test_sizes_read_inside_a_region_stay_symbolic(tmp_path):
    """Export-style graphs read sizes off tensors inside the region; the
    canonicalizing retrace must not bake them to the hint."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import recapture_full_graphs

    models = tmp_path / "models"
    graph = models / "toy/infer/m/full_graph_000.py"
    graph.parent.mkdir(parents=True)
    graph.write_text(SIZE_READ_GRAPH)
    graph.with_suffix(".meta.json").write_text(json.dumps({
        "schema_version": 2,
        "inputs": [[[12, 64], "f32", {"n": "arg0_1", "dev": "cpu"}],
                   [[12, 64], "f32", {"n": "arg1_1", "dev": "cpu"}]],
    }))
    out = tmp_path / "out"
    assert recapture_full_graphs.main([
        str(models), "--models-root", str(models), "--canonical-root",
        str(out), "--no-validate", "--no-isolate"]) == 0
    regions = _regions(out)
    assert regions
    for name, shapes in regions.items():
        assert shapes["points"][0]["bindings"] == {"s0": 12}, (name, shapes["symbols"])
        rows = _check(out, name, [{"s0": 12}, {"s0": 5}])
        outcomes = [(r["numerics"], r["recompiled"]) for r in rows]
        assert outcomes == [("pass", False)] * 2, (name, rows)


def test_tlparse_ingest_takes_hints_from_the_runnable(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from ingest_tlparse import load_graph_module, runnable_symbol_hints

    graph = tmp_path / "inductor_post_grad_graph_0.txt"
    graph.write_text(LEGACY_GRAPH.replace("class GraphModule", "class <lambda>"))
    (tmp_path / "fx_graph_runnable_0.txt").write_text("s16 = 1811\ns99 = 3\n")
    hints = runnable_symbol_hints(graph)
    assert hints == {"s16": 1811, "s99": 3}
    gm = load_graph_module(graph, symbol_hints=hints)
    (symint, _x) = [n.meta["val"] for n in gm.graph.find_nodes(op="placeholder")]
    assert symint.node.hint == 1811


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_repro_default_bench_times_static_points_and_dynamic_artifact(tmp_path):
    from repro_harness import benchmark_repro

    root = tmp_path / "root"
    torch.manual_seed(0)
    model = MLP().eval()
    _capture(model, [_mlp_args(8)], root, name="mlp")
    _capture(model, [_mlp_args(16)], root, name="mlp")
    name = next(n for n in _regions(root) if n.startswith("var_mean"))
    repro = str(root / "canonical" / name / "repro.py")
    module = _repro_module(repro)
    results = benchmark_repro(repro, module.Repro, module.make_inputs, args=[
        "--no-gpu-lock", "--no-cd", "--n-warmup", "2", "--n-rep", "5"])
    static = {k: v for k, v in results.items() if v.get("mode") == "static"}
    dynamic = {k: v for k, v in results.items() if v.get("mode") == "dynamic"}
    assert sorted(v["binding"]["s0"] for v in static.values()) == [8, 16]
    assert sorted(v["binding"]["s0"] for v in dynamic.values()) == [8, 16]
    assert all("::" not in k for k in static)  # keyed like a static repro
    assert all(k.endswith("::dynamic") for k in dynamic)
    for row in dynamic.values():
        assert row["numerics"]["status"] == "pass", row["numerics"]
        assert row["recompiled"] is False
        assert row["compiled_us"] > 0
