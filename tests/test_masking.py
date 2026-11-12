"""The shared masking scaffold: mask_reason rules, node accounting and the
full-graph sidecar's graph-level mask."""
import json
import operator

import pytest
import torch
import torch.fx as fx

import capture_hook
from capture_hook import (
    get_fusion_partitions,
    graph_mask,
    graph_node_accounting,
    mask_reason,
    mask_reasons,
    node_mask_counts,
)


def _graph_with_user_triton_kernel():
    from torch._higher_order_ops.triton_kernel_wrap import (
        triton_kernel_wrapper_functional,
    )

    graph = fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.ops.aten.mul.Tensor, (x, 2.0))
    out = graph.call_function(
        triton_kernel_wrapper_functional,
        (),
        {
            "kernel_idx": 10_000,
            "constant_args_idx": 0,
            "grid": [(1, 1, 1)],
            "tma_descriptor_metadata": {},
            "kwargs": {"x_ptr": y, "out_ptr": y},
            "tensors_to_clone": ["out_ptr"],
        },
    )
    z = graph.call_function(
        torch.ops.aten.relu.default,
        (graph.call_function(operator.getitem, (out, "out_ptr")),),
    )
    graph.output((z,))
    return fx.GraphModule(torch.nn.Module(), graph)


def _hop(gm):
    return next(n for n in gm.graph.nodes if "triton_kernel_wrapper" in str(n.target))


def test_masked_nodes_get_their_own_accounting_bucket():
    gm = _graph_with_user_triton_kernel()

    acct = graph_node_accounting(gm)

    assert acct["ops"]["masked"] == {
        "torch.ops.higher_order.triton_kernel_wrapper_functional": {
            "count": 1,
            "reasons": ["user_triton"],
        }
    }
    assert acct["counts"]["masked"] == 1
    assert "non_fusible" not in acct["ops"] or not acct["ops"]["non_fusible"]
    assert sum(acct["counts"].values()) == acct["total_call_functions"]
    assert all(_hop(gm) not in part for part in get_fusion_partitions(gm))


def test_unmasked_graph_has_no_mask_or_bucket():
    gm = fx.symbolic_trace(lambda x: (x * 2).relu())

    assert graph_mask(gm) is None
    assert "masked" not in graph_node_accounting(gm)["counts"]


def test_every_rule_runs_and_all_reasons_are_kept(monkeypatch):
    def mask_relu(node):
        return "jagged" if node.target is torch.ops.aten.relu.default else None

    def mask_hops(node):
        return "hstu" if node is _hop(gm) else None

    gm = _graph_with_user_triton_kernel()
    monkeypatch.setattr(
        capture_hook, "MASK_RULES", [*capture_hook.MASK_RULES, mask_relu, mask_hops]
    )
    relu = next(n for n in gm.graph.nodes if n.target is torch.ops.aten.relu.default)

    assert mask_reasons(_hop(gm)) == ("user_triton", "hstu")
    assert mask_reason(_hop(gm)) == "user_triton"
    assert mask_reasons(relu) == ("jagged",)
    assert not capture_hook.partition_node_is_supported(relu)
    acct = graph_node_accounting(gm)
    assert acct["ops"]["masked"][
        "torch.ops.higher_order.triton_kernel_wrapper_functional"
    ]["reasons"] == ["hstu", "user_triton"]
    assert graph_mask(gm) == {
        "reasons": ["hstu", "jagged", "user_triton"],
        "scope": "graph",
    }


def test_node_rules_cannot_return_repro_or_unknown_reasons(monkeypatch):
    monkeypatch.setattr(
        capture_hook,
        "MASK_RULES",
        [lambda node: "approx_index", lambda node: "slow"],
    )
    gm = fx.symbolic_trace(lambda x: x + 1)
    node = next(n for n in gm.graph.nodes if n.op == "call_function")

    assert mask_reasons(node) == ()
    assert graph_mask(gm) is None


def test_shipped_rules_return_only_their_declared_reasons():
    gm = _graph_with_user_triton_kernel()
    comps = [[n] for n in gm.graph.nodes]

    for node in gm.graph.nodes:
        for rule in capture_hook.MASK_RULES:
            assert rule(node) in capture_hook.NODE_MASK_REASONS | {None}
    for comp in comps:
        for rule in capture_hook.REGION_MASK_RULES:
            assert rule(comp) in capture_hook.REGION_MASK_REASONS | {None}
    assert set(capture_hook.MASK_REASONS) == (
        capture_hook.NODE_MASK_REASONS | capture_hook.REGION_MASK_REASONS
    )


def test_user_triton_launch_in_a_hop_subgraph_masks_the_graph():
    from torch._higher_order_ops.cond import cond
    from torch._higher_order_ops.triton_kernel_wrap import (
        triton_kernel_wrapper_mutation,
    )

    from replay_dependencies import (
        check_replay_dependencies,
        full_graph_calls_user_triton,
        replay_dependencies,
        user_triton_hops_in_class,
    )

    sub = fx.Graph()
    x = sub.placeholder("x")
    sub.call_function(
        triton_kernel_wrapper_mutation,
        (),
        {
            "kernel_idx": 10_000,
            "constant_args_idx": 0,
            "grid": [(1, 1, 1)],
            "tma_descriptor_metadata": {},
            "kwargs": {"x_ptr": x},
        },
    )
    sub.output((x,))
    root = torch.nn.Module()
    root.true_graph_0 = fx.GraphModule(torch.nn.Module(), sub)
    root.false_graph_0 = fx.GraphModule(torch.nn.Module(), sub)
    graph = fx.Graph()
    pred, a = graph.placeholder("pred"), graph.placeholder("a")
    out = graph.call_function(
        cond,
        (pred, graph.get_attr("true_graph_0"), graph.get_attr("false_graph_0"), (a,)),
    )
    graph.output(out)
    gm = fx.GraphModule(root, graph)

    assert graph_mask(gm) == {"reasons": ["user_triton"], "scope": "graph"}
    assert [b["category"] for b in check_replay_dependencies(
        replay_dependencies(gm))["blocked"]] == ["user_triton_kernel"]

    namespace = {"torch": torch}
    exec(gm.print_readable(print_output=False), namespace)
    graph_cls = namespace["GraphModule"]
    assert user_triton_hops_in_class(graph_cls) == ["triton_kernel_wrapper_mutation"]
    assert not user_triton_hops_in_class(type(fx.symbolic_trace(lambda x: x + 1)))

    # load_full_graph and the preflight use the same check on the loaded class.
    assert full_graph_calls_user_triton(graph_cls, {})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="validates the repro on CUDA")
def test_region_rule_marks_repro_without_dropping_it(tmp_path, monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx

    from merge_captures import merge_one_capture

    def mask_relu_regions(comp):
        has_relu = any(n.target is torch.ops.aten.relu.default for n in comp)
        return "approx_index" if has_relu else None

    monkeypatch.setattr(capture_hook, "REGION_MASK_RULES", [mask_relu_regions])
    gm = make_fx(lambda x, w: (x @ w).relu() + 1)(torch.randn(8, 8), torch.randn(8, 8))
    state = capture_hook._CaptureState(str(tmp_path / "capture"), label="m")

    state.process_graph(gm)
    state.finalize()

    assert state.dropped == []
    assert [entry.get("masked_reason") for entry in state.captured] == ["approx_index"]
    merge_one_capture(tmp_path / "capture", tmp_path / "corpus", "hf/infer/bbtest_model")
    (meta_path,) = (tmp_path / "corpus" / "canonical").glob("*/meta.json")
    assert json.loads(meta_path.read_text())["masked_reason"] == "approx_index"


def _unbacked(x, idx):
    u = torch.unique(idx)
    n = idx.sum().item()
    return (x * 2.0 + 1.0).relu(), u.float() * 3.0, torch.ones(n, device=x.device) + 1.0


def test_nodes_with_unbacked_sizes_or_values_are_masked():
    from torch.fx.experimental.proxy_tensor import make_fx

    gm = make_fx(_unbacked, tracing_mode="symbolic")(
        torch.randn(8, 16), torch.randint(0, 5, (32,))
    )
    masked = {
        str(n.target) for n in gm.graph.nodes if "unbacked_shape" in mask_reasons(n)
    }

    assert {
        "aten._unique2.default",
        "aten._local_scalar_dense.default",
        "aten.ones.default",
        "aten._to_copy.default",
    } <= masked
    assert "aten.relu.default" not in masked
    assert "aten.sum.default" not in masked
    assert graph_mask(gm) == {"reasons": ["unbacked_shape"], "scope": "graph"}
    counts = node_mask_counts(gm)
    assert set(counts) == {"unbacked_shape"}
    assert counts["unbacked_shape"] == sum(
        1 for n in gm.graph.nodes if "unbacked_shape" in mask_reasons(n)
    )
    assert counts["unbacked_shape"] >= 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="validates the repro on CUDA")
def test_unbacked_regions_are_masked_not_dropped(tmp_path):
    from torch.fx.experimental.proxy_tensor import make_fx

    from merge_captures import merge_one_capture

    gm = make_fx(_unbacked, tracing_mode="symbolic")(
        torch.randn(8, 16, device="cuda"), torch.randint(0, 5, (32,), device="cuda")
    )
    state = capture_hook._CaptureState(
        str(tmp_path / "capture"), label="m", graph_dir=str(tmp_path / "graphs")
    )

    state.process_graph(gm)
    state.finalize()

    assert state.dropped == []
    assert sorted(tuple(e["origin_ops"]) for e in state.captured) == [
        ("aten.add.Tensor", "aten.mul.Tensor", "aten.relu.default"),
        ("aten.sum.default",),
    ]
    sidecar = json.loads((tmp_path / "graphs" / "full_graph_000.meta.json").read_text())
    assert sidecar["masked"] == {"reasons": ["unbacked_shape"], "scope": "graph"}
    assert sidecar["node_mask_counts"]["unbacked_shape"] >= 4
    assert sidecar["roundtrip"] == "ok"
    merge_one_capture(tmp_path / "capture", tmp_path / "corpus", "hf/infer/bbtest_model")
    assert len(list((tmp_path / "corpus" / "canonical").glob("*/repro.py"))) == 2


def test_symbolic_slice_bounds_partition_like_a_static_trace():
    """Ingest traces int inputs as unbacked symbols, so a slice whose bounds
    come from an input gets a fresh unbacked storage offset. That symbol is a
    function of the inputs, not data-dependent: the symbolic trace must
    partition like the static re-trace the round-trip gate compares it to,
    and its runtime asserts must not become partitions."""
    from torch.fx.experimental.dynamic_spec import IntVar, STATIC, TensorSpec
    from torch.fx.experimental.proxy_tensor import make_fx

    from roundtrip_validation import nontransparent_key

    def f(x, start):
        a = (x * 2.0).sum(1, keepdim=True)
        y = x[:, start : start + 64].reshape(8, 2, 64).clone().reshape(16, 64)
        return a, (y + 1.0).sum(1)

    x = torch.randn(16, 128)
    symbolic = make_fx(
        f,
        tracing_mode="fake",
        dynamic_shapes={"x": TensorSpec([STATIC, STATIC]), "start": IntVar("start")},
    )(x, 32)
    static = make_fx(f, tracing_mode="fake")(x, 32)

    def keys(gm):
        return sorted(
            nontransparent_key([str(n.target) for n in comp])
            for comp in get_fusion_partitions(gm)
        )

    assert not any("unbacked_shape" in mask_reasons(n) for n in symbolic.graph.nodes)
    assert keys(symbolic) == keys(static)


def _slice_by_input(x, start):
    a = (x * 2.0).sum(1, keepdim=True)
    y = x[:, start : start + 64].reshape(8, 2, 64).clone().reshape(16, 64)
    return a, (y + 1.0).sum(1)


def _traced_slice_graph(tracing: str):
    from torch.fx.experimental.dynamic_spec import IntVar, STATIC, TensorSpec
    from torch.fx.experimental.proxy_tensor import make_fx

    x = torch.randn(16, 128, device="cuda")
    if tracing == "unbacked":
        # Ingest's ShapesSpec path: int inputs become unbacked symbols, and
        # the slice's offset gets runtime asserts built from
        # aten.sym_storage_offset.
        gm = make_fx(
            _slice_by_input,
            tracing_mode="fake",
            dynamic_shapes={"x": TensorSpec([STATIC, STATIC]), "start": IntVar("start")},
        )(x, 32)
    else:
        gm = make_fx(_slice_by_input, tracing_mode=tracing)(x, 32)
    # The ingest loader records the concrete input values.
    gm.meta["better_benchmark_input_specs"] = [
        {"kind": "tensor", "shape": [16, 128]},
        {"kind": "symint", "value": 32},
    ]
    return gm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runs the written graph on CUDA")
@pytest.mark.parametrize("tracing", ["symbolic", "unbacked"])
def test_written_full_graph_compiles_with_dynamo(tracing, tmp_path):
    """The driver benchmarks unmasked full graphs with torch.compile, so the
    written source must trace under Dynamo, whatever trace produced it."""
    from full_graph_harness import (
        instantiate_full_graph,
        load_full_graph_definition,
        make_inputs_from_full_graph_specs,
    )

    state = capture_hook._CaptureState(
        str(tmp_path / "capture"), label="m", graph_dir=str(tmp_path / "graphs")
    )

    state.process_graph(_traced_slice_graph(tracing))

    path = tmp_path / "graphs" / "full_graph_000.py"
    assert "masked" not in json.loads(path.with_suffix(".meta.json").read_text())
    definition = load_full_graph_definition(path)
    module = instantiate_full_graph(definition, default_device="cuda")
    inputs = make_inputs_from_full_graph_specs(definition.input_specs, default_device="cuda")
    torch._dynamo.reset()
    torch.compile(module, fullgraph=True, backend="eager")(*inputs)


@torch.library.custom_op("bbtest::positive_values", mutates_args=())
def _positive_values(x: torch.Tensor) -> torch.Tensor:
    return x[x > 0].clone()


@_positive_values.register_fake
def _(x):
    return x.new_empty(torch.library.get_ctx().new_dynamic_size())


def test_custom_op_with_a_data_dependent_size_is_masked():
    """A custom op whose fake impl calls new_dynamic_size carries no aten
    tags (the jagged ops do this), but its output size still comes from
    tensor data: its consumers must stay unbacked_shape-masked."""
    from torch.fx.experimental.proxy_tensor import make_fx

    def f(x):
        return (_positive_values(x) * 2.0).sum(), (x + 1.0).sum()

    gm = make_fx(f, tracing_mode="symbolic")(torch.randn(64))
    masked = {
        str(n.target) for n in gm.graph.nodes if "unbacked_shape" in mask_reasons(n)
    }

    assert {"aten.mul.Tensor", "bbtest.positive_values.default"} <= masked
    assert "aten.add.Tensor" not in masked
    assert sorted(
        [str(n.target) for n in comp] for comp in get_fusion_partitions(gm)
    ) == [["aten.add.Tensor", "aten.sum.default"]]


def test_stripping_runtime_asserts_keeps_the_scalar_they_check():
    """``.item()`` is a device-to-host sync the full graph must still time,
    even when only runtime asserts read its value."""
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts

    def f(x):
        n = (x > 0).sum().item()
        torch._check(n >= 0)
        torch._check(n <= 64)
        return x * 2.0

    gm = make_fx(f, tracing_mode="symbolic")(torch.randn(64))
    shape_env = next(
        n.meta["val"].node.shape_env
        for n in gm.graph.nodes
        if isinstance(n.meta.get("val"), torch.SymInt)
    )
    insert_deferred_runtime_asserts(gm, shape_env, "test")
    gm.recompile()
    output = next(n for n in gm.graph.nodes if n.op == "output")
    output.meta["marker"] = True

    class _CodeGen(torch.fx.graph.CodeGen):
        pass

    gm.graph.set_codegen(_CodeGen())

    stripped = capture_hook._without_runtime_asserts(gm)

    targets = {n.target for n in stripped.graph.nodes}
    assert torch.ops.aten._local_scalar_dense.default in targets
    assert torch.ops.aten._assert_scalar.default not in targets
    assert next(n for n in stripped.graph.nodes if n.op == "output").meta["marker"]
    assert isinstance(stripped.graph._codegen, _CodeGen)
    # The input graph is left as it was.
    assert torch.ops.aten._assert_scalar.default in {n.target for n in gm.graph.nodes}
