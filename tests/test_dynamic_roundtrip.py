"""End-to-end dynamic capture: capture a model compiled with dynamic shapes,
merge its regions into families, and replay them at the recorded and at new
bindings against eager.

Capture and replay run on CPU (Inductor's C++ backend).
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

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


def _repro_cls(repro_file):
    spec = importlib.util.spec_from_file_location("repro_under_test", repro_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Repro


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

