"""Capture with constant_folding=True leaves out work that runtime constant
folding computes once at load time (parameter-only subgraphs, e.g. weight
quantization), and keeps everything that depends on a user input."""

import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch._dynamo
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import capture_hook
from capture_hook import install_capture_hook, runtime_constant_nodes, uninstall_capture_hook
from input_codec import spec_from_compact


class WeightPrepModel(nn.Module):
    """A parameter-only transform feeding a matmul (folded), a parameter
    transform fused with an activation (only the parameter part is folded),
    and an activation epilogue (never folded)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(32, 48))
        self.b = nn.Parameter(torch.randn(48))

    def forward(self, x):
        w = (self.w * self.w.abs().amax()).relu()
        y = x @ w
        return (y + self.b.sin()).tanh()


def _capture(tmp_path, model, *inputs, constant_folding, grad=False, backward=False):
    torch._dynamo.reset()
    output_dir = tempfile.mkdtemp(prefix="cf_", dir=tmp_path)
    graph_dir = tempfile.mkdtemp(prefix="cf_graph_", dir=tmp_path)
    install_capture_hook(output_dir, label="cf", validate=False,
                         graph_dir=graph_dir, constant_folding=constant_folding)
    try:
        compiled = torch.compile(model, backend="inductor")
        with torch.set_grad_enabled(grad):
            out = compiled(*inputs)
        if backward:
            out.sum().backward()
    finally:
        uninstall_capture_hook()
    index = json.loads(open(os.path.join(output_dir, "index.json")).read())
    sources = []
    for path in sorted(glob.glob(os.path.join(output_dir, "*.py"))):
        sources.append(open(path).read())
    meta = [json.loads(open(p).read())
            for p in sorted(glob.glob(os.path.join(graph_dir, "*.meta.json")))]
    index["graph_dir"] = graph_dir
    return index, sources, meta


def _ops(sources, op):
    return sum(src.count(op) for src in sources)


def test_parameter_only_regions_are_left_out(tmp_path):
    model, x = WeightPrepModel(), torch.randn(8, 32)
    index, sources, meta = _capture(tmp_path, model, x, constant_folding=False)
    # Without the flag, index.json and the full-graph sidecar are as before.
    assert "constant_folding" not in index
    assert "constant_folded" not in meta[0]["node_accounting"]["counts"]
    assert "constant_inputs" not in meta[0]
    assert _ops(sources, "aten.amax") == 1
    assert _ops(sources, "aten.sin") == 1

    index, sources, meta = _capture(tmp_path, model, x, constant_folding=True)
    cf = index["constant_folding"]
    assert cf["inference_graphs"] == 1
    assert cf["skipped_graphs"] == {"training": 0, "subgraph": 0, "no_metadata": 0}
    assert cf["folded_nodes"] > 0
    # The weight transform and the bias transform are folded at load time...
    assert _ops(sources, "aten.amax") == 0
    assert _ops(sources, "aten.sin") == 0
    # ...but the activation epilogue still runs every call.
    assert _ops(sources, "aten.tanh") == 1
    accounting = meta[0]["node_accounting"]
    assert accounting["counts"]["constant_folded"] == cf["folded_nodes"]
    assert sum(accounting["counts"].values()) == accounting["total_call_functions"]
    assert len(meta[0]["constant_inputs"]) == 2


def test_input_dependent_work_is_never_folded(tmp_path):
    class AllDynamic(nn.Module):
        def forward(self, x, y):
            return (x.abs().amax() * y).relu()

    index, sources, _ = _capture(tmp_path, AllDynamic(), torch.randn(16, 8), torch.randn(16, 8),
                                constant_folding=True)
    assert index["constant_folding"]["folded_nodes"] == 0
    assert _ops(sources, "aten.amax") == 1


def test_mutated_buffer_is_not_a_constant(tmp_path):
    class Stateful(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.zeros(4))

        def forward(self, x):
            self.count.add_(1)
            return (x + self.count.cos()).relu()

    index, sources, _ = _capture(tmp_path, Stateful(), torch.randn(4), constant_folding=True)
    # count changes every call, so count.cos() is per-call work.
    assert _ops(sources, "aten.cos") == 1


def test_buffer_mutated_in_an_earlier_graph_is_not_a_constant(tmp_path):
    class MutateThenRead(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.zeros(4))

        def forward(self, x):
            self.count.add_(1)
            torch._dynamo.graph_break()
            return (x + self.count.cos()).relu()

    index, sources, _ = _capture(tmp_path, MutateThenRead(), torch.randn(4), constant_folding=True)
    assert _ops(sources, "aten.cos") == 1
    assert index["constant_folding"]["inference_graphs"] == 2
    assert index["n_dropped"] == 0


def test_buffer_mutated_in_a_later_graph_is_recorded_as_a_drop(tmp_path):
    class ReadThenMutate(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.zeros(4))

        def forward(self, x):
            y = (x + self.count.cos()).relu()
            torch._dynamo.graph_break()
            self.count.add_(1)
            return y * 2

    index, _, _ = _capture(tmp_path, ReadThenMutate(), torch.randn(4), constant_folding=True)
    # The first graph was captured with count.cos() folded before the second
    # graph showed count changes; the capture must not merge silently.
    reasons = [d["reason"] for d in index["dropped"]]
    assert len(reasons) == 1
    assert "constant folding" in reasons[0] and "count" in reasons[0]


def test_buffer_mutated_through_an_argument_is_not_a_constant(tmp_path):
    class MutateViaArg(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buf", torch.zeros(4))

        def forward(self, x, state):
            state.add_(1)
            torch._dynamo.graph_break()
            return (x + self.buf.cos()).relu()

    model = MutateViaArg()
    index, sources, _ = _capture(tmp_path, model, torch.randn(4), model.buf,
                                 constant_folding=True)
    # state is model.buf under another name: buf.cos() is per-call work.
    assert _ops(sources, "aten.cos") == 1


def test_graph_with_effect_tokens_is_folded(tmp_path):
    class Effectful(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4))
            self.register_buffer("count", torch.zeros(4))

        def forward(self, x):
            torch.ops.aten._print("step")
            self.count.add_(1)
            return (x + self.count.cos() + self.w.sin()).relu()

    index, sources, _ = _capture(tmp_path, Effectful(), torch.randn(4), constant_folding=True)
    assert _ops(sources, "aten.sin") == 0
    assert _ops(sources, "aten.cos") == 1


def test_training_graphs_are_not_folded(tmp_path):
    class Train(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(8, 8))
            self.b = nn.Parameter(torch.randn(8))

        def forward(self, x):
            return ((x @ (self.w * 2).t()) + self.b.sin()).tanh()

    index, sources, _ = _capture(tmp_path, Train(), torch.randn(4, 8), constant_folding=True,
                                 grad=True, backward=True)
    cf = index["constant_folding"]
    assert cf["folded_nodes"] == 0
    assert cf["inference_graphs"] == 0
    assert cf["skipped_graphs"]["training"] == 2
    assert _ops(sources, "aten.sin") >= 1


def test_parameter_passed_as_an_argument_is_not_a_constant(tmp_path):
    class UsesArg(nn.Module):
        def forward(self, x, p):
            return (x + p.sin()).relu()

    index, sources, _ = _capture(tmp_path, UsesArg(), torch.randn(8), nn.Parameter(torch.randn(8)),
                                 constant_folding=True)
    assert index["constant_folding"]["folded_nodes"] == 0
    assert _ops(sources, "aten.sin") == 1


def test_static_address_input_is_not_a_constant(tmp_path):
    class UsesArg(nn.Module):
        def forward(self, x, s):
            return (x + s.sin()).relu()

    s = torch.randn(8)
    torch._dynamo.mark_static_address(s)
    index, sources, _ = _capture(tmp_path, UsesArg(), torch.randn(8), s, constant_folding=True)
    assert index["constant_folding"]["folded_nodes"] == 0
    assert _ops(sources, "aten.sin") == 1


def test_module_tensor_attribute_is_a_constant(tmp_path):
    # torch.export lifts a plain tensor attribute as a tensor constant, which
    # AOTInductor folds like a buffer.
    class Attr(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.randn(8)

        def forward(self, x):
            return (x + self.table.sin()).relu()

    index, sources, _ = _capture(tmp_path, Attr(), torch.randn(8), constant_folding=True)
    assert index["constant_folding"]["folded_nodes"] > 0
    assert _ops(sources, "aten.sin") == 0
    assert _ops(sources, "aten.relu") == 1


def test_constructor_and_random_consumers_stay_per_call(tmp_path):
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(8))

        def forward(self, x):
            noisy = F.dropout(self.w, p=0.5, training=True)
            return x * (self.w + torch.arange(8.0)) + noisy

    index, sources, _ = _capture(tmp_path, Mixed(), torch.randn(8), constant_folding=True)
    assert index["constant_folding"]["folded_nodes"] == 0
    assert _ops(sources, "prims.iota") == 1
    assert _ops(sources, "prims.inductor_random") == 1


def test_higher_order_op_subgraph_is_not_folded(tmp_path):
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(8))

        def forward(self, x):
            return torch.cond(x.sum() > 0, lambda x: x + self.w.sin(),
                              lambda x: x - self.w.cos(), (x,))

    index, _, _ = _capture(tmp_path, Branch(), torch.randn(8), constant_folding=True)
    assert index["constant_folding"]["skipped_graphs"]["subgraph"] == 2
    assert index["constant_folding"]["inference_graphs"] == 1


def test_get_attr_constant_is_folded():
    gm = fx.GraphModule({"table": torch.randn(8)}, fx.Graph())
    x = gm.graph.placeholder("x")
    table = gm.graph.get_attr("table")
    sin = gm.graph.call_function(torch.ops.aten.sin.default, (table,))
    add = gm.graph.call_function(torch.ops.aten.add.Tensor, (x, sin))
    gm.graph.output(add)
    gm.recompile()
    assert runtime_constant_nodes(gm, []) == {sin}


def test_graph_without_metadata_is_captured_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_hook, "graph_input_kind", lambda gm: "no_metadata")
    index, sources, _ = _capture(tmp_path, WeightPrepModel(), torch.randn(8, 32), constant_folding=True)
    assert index["constant_folding"]["skipped_graphs"]["no_metadata"] == 1
    assert index["constant_folding"]["folded_nodes"] == 0
    assert _ops(sources, "aten.amax") == 1


def test_plain_pass_chained_on_top_still_captures(tmp_path):
    # Another tool installs its own pass on top and chains to the previous
    # one the conventional way, prev(graph), with no is_inference.
    import torch._inductor.config as inductor_config

    for constant_folding in (False, True):
        torch._dynamo.reset()
        output_dir = tempfile.mkdtemp(prefix="cf_", dir=tmp_path)
        install_capture_hook(output_dir, label="cf", validate=False,
                             constant_folding=constant_folding)
        prev = inductor_config.post_grad_custom_pre_pass
        inductor_config.post_grad_custom_pre_pass = lambda graph: prev(graph)
        try:
            with torch.no_grad():
                torch.compile(WeightPrepModel())(torch.randn(8, 32))
        finally:
            uninstall_capture_hook()
        index = json.loads(open(os.path.join(output_dir, "index.json")).read())
        assert index["n_captured"] > 0 and not index["dropped"]
        if constant_folding:
            # Without is_inference the graph is not known to be inference.
            assert index["constant_folding"]["skipped_graphs"]["no_metadata"] == 1


def test_graph_without_placeholders_counts_as_inference(tmp_path):
    class NoInputs(nn.Module):
        def forward(self):
            return torch.ones(4).cos() * 2

    index, _, _ = _capture(tmp_path, NoInputs(), constant_folding=True)
    cf = index["constant_folding"]
    assert cf["inference_graphs"] == 1
    assert sum(cf["skipped_graphs"].values()) == 0


def test_model_accounting_leaves_out_folded_nodes(tmp_path):
    from model_attribution import collect_extern_points
    from model_graph_accounting import analyze_graph, trace_full_graph

    class WeightMatmul(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(16, 8))
            self.b = nn.Parameter(torch.randn(8, 16))

        def forward(self, x):
            return (x @ (self.a @ self.b).sin()).relu()

    index, _, meta = _capture(tmp_path, WeightMatmul(), torch.randn(4, 16), constant_folding=True)
    graph = Path(index["graph_dir"]) / "full_graph_000.py"
    gm = trace_full_graph(graph, device="cpu")
    acc = analyze_graph(gm, str(graph), graph.stem)
    assert (sum(acc.constant_folded_ops.values())
            == meta[0]["node_accounting"]["counts"]["constant_folded"] > 0)
    assert [op.target for op in acc.non_fusible_ops] == ["aten::mm.default"]
    assert all("aten.sin" not in o.origin_ops for o in acc.occurrences)
    # Only the per-call matmul is an extern point; the weight product is folded.
    points = collect_extern_points(gm, graph=graph)
    assert sum(p["count"] for p in points.values()) == 1

    # A tensor constant with no constant placeholder: the sidecar records
    # constant_inputs == [] and the retrace still folds the get_attr work.
    class TensorConstant(nn.Module):
        def forward(self, x):
            return (x + torch.tensor([1.0, 2.0, 3.0, 4.0]).sin()).relu()

    index, sources, meta = _capture(tmp_path, TensorConstant(), torch.randn(4),
                                    constant_folding=True)
    assert meta[0]["constant_inputs"] == []
    graph = Path(index["graph_dir"]) / "full_graph_000.py"
    acc = analyze_graph(trace_full_graph(graph, device="cpu"), str(graph), graph.stem)
    assert (sum(acc.constant_folded_ops.values())
            == meta[0]["node_accounting"]["counts"]["constant_folded"] > 0)
    assert _ops(sources, "aten.sin") == 0
    assert all("aten.sin" not in o.origin_ops for o in acc.occurrences)


def test_folded_integer_buffer_feeding_an_index_keeps_its_bound(tmp_path):
    class Lookup(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("ids", torch.randint(0, 64, (16,), dtype=torch.int32))

        def forward(self, table):
            return table[self.ids.long()].relu()

    index, sources, _ = _capture(tmp_path, Lookup(), torch.randn(64, 8), constant_folding=True)
    assert index["constant_folding"]["folded_nodes"] > 0
    entries = [e for e in index["captured"] if "aten.index.Tensor" in e["origin_ops"]]
    assert len(entries) == 1
    specs = [spec_from_compact(e) for e in entries[0]["inputs"]]
    ids = [s for s in specs if s["dtype"] == "int64"]
    assert len(ids) == 1
    assert ids[0]["gen"] == {"kind": "index", "low": 0, "high": 64}
