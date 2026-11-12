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
