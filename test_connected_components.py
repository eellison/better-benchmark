"""
Test: connected-component splitting in capture_hook.

Verifies that CapabilityBasedPartitioner partitions that contain
multiple independent data-flow chains get split into separate repros.
"""
import copy
import os
import shutil
import tempfile

import torch
import torch.fx as fx
import torch._inductor.config as inductor_config

inductor_config.force_disable_caches = True


def _build_two_independent_chains_graph():
    """Build an FX graph with two independent pointwise chains, each with real compute.

    Chain 1: input_a -> mul -> add (pointwise compute on A)
    Chain 2: input_b -> mul -> sub (pointwise compute on B, independent of chain 1)

    These should NOT be in the same repro.
    """
    graph = fx.Graph()

    # Inputs
    a = graph.placeholder("input_a")
    a.meta = {"val": torch.randn(2, 8, 64, 64, device="cpu")}

    b = graph.placeholder("input_b")
    b.meta = {"val": torch.randn(512, 512, device="cpu")}

    # Chain 1: input_a -> mul -> add
    mul1 = graph.call_function(torch.ops.aten.mul.Tensor, args=(a, 2.0))
    mul1.meta = {"val": torch.randn(2, 8, 64, 64, device="cpu")}

    add1 = graph.call_function(torch.ops.aten.add.Tensor, args=(mul1, 1.0))
    add1.meta = {"val": torch.randn(2, 8, 64, 64, device="cpu")}

    # Chain 2: input_b -> mul -> sub (INDEPENDENT of chain 1!)
    mul2 = graph.call_function(torch.ops.aten.mul.Tensor, args=(b, 3.0))
    mul2.meta = {"val": torch.randn(512, 512, device="cpu")}

    sub2 = graph.call_function(torch.ops.aten.sub.Tensor, args=(mul2, 0.5))
    sub2.meta = {"val": torch.randn(512, 512, device="cpu")}

    # Output both chains
    graph.output((add1, sub2))

    gm = fx.GraphModule(torch.nn.Module(), graph)
    return gm


def _build_var_mean_chain_graph():
    """Build an FX graph with a single data-dependent chain.

    input -> var_mean -> getitem(0) -> getitem(1) -> mul -> add
    All connected via data flow. Should stay as ONE repro.
    """
    graph = fx.Graph()

    x = graph.placeholder("x")
    x.meta = {"val": torch.randn(8, 512, 768, device="cpu")}

    # var_mean returns a tuple
    var_mean = graph.call_function(
        torch.ops.aten.var_mean.correction,
        args=(x,),
        kwargs={"dim": [-1], "correction": 0, "keepdim": True},
    )
    var_mean.meta = {"val": (torch.randn(8, 512, 1, device="cpu"), torch.randn(8, 512, 1, device="cpu"))}

    import operator
    getitem0 = graph.call_function(operator.getitem, args=(var_mean, 0))
    getitem0.meta = {"val": torch.randn(8, 512, 1, device="cpu")}

    getitem1 = graph.call_function(operator.getitem, args=(var_mean, 1))
    getitem1.meta = {"val": torch.randn(8, 512, 1, device="cpu")}

    # Subtract mean from x
    sub = graph.call_function(torch.ops.aten.sub.Tensor, args=(x, getitem1))
    sub.meta = {"val": torch.randn(8, 512, 768, device="cpu")}

    # Add epsilon to var
    add_eps = graph.call_function(torch.ops.aten.add.Tensor, args=(getitem0, 1e-5))
    add_eps.meta = {"val": torch.randn(8, 512, 1, device="cpu")}

    # rsqrt
    rsqrt = graph.call_function(torch.ops.aten.rsqrt.default, args=(add_eps,))
    rsqrt.meta = {"val": torch.randn(8, 512, 1, device="cpu")}

    # mul: (x - mean) * rsqrt(var + eps)
    mul = graph.call_function(torch.ops.aten.mul.Tensor, args=(sub, rsqrt))
    mul.meta = {"val": torch.randn(8, 512, 768, device="cpu")}

    graph.output(mul)
    gm = fx.GraphModule(torch.nn.Module(), graph)
    return gm


def _build_chain_plus_independent_scalar():
    """Build an FX graph with a chain + an independent scalar op.

    Chain 1: x -> mul -> add (connected)
    Chain 2: y -> abs (independent scalar op)

    Should produce TWO separate captures.
    """
    graph = fx.Graph()

    x = graph.placeholder("x")
    x.meta = {"val": torch.randn(8, 512, device="cpu")}

    y = graph.placeholder("y")
    y.meta = {"val": torch.randn(4, 4, device="cpu")}

    # Chain 1: x -> mul -> add
    mul = graph.call_function(torch.ops.aten.mul.Tensor, args=(x, 2.0))
    mul.meta = {"val": torch.randn(8, 512, device="cpu")}

    add = graph.call_function(torch.ops.aten.add.Tensor, args=(mul, 1.0))
    add.meta = {"val": torch.randn(8, 512, device="cpu")}

    # Chain 2: y -> abs (independent)
    abs_op = graph.call_function(torch.ops.aten.abs.default, args=(y,))
    abs_op.meta = {"val": torch.randn(4, 4, device="cpu")}

    graph.output((add, abs_op))
    gm = fx.GraphModule(torch.nn.Module(), graph)
    return gm


def test_two_independent_chains():
    """Two independent chains should produce two separate captures."""
    from capture_hook import _CaptureState

    output_dir = tempfile.mkdtemp(prefix="test_cc_")
    try:
        state = _CaptureState(output_dir, label="test", validate=False)
        gm = _build_two_independent_chains_graph()
        state.process_graph(gm)

        # Should have 2 captures (one per independent chain)
        assert len(state.captured) == 2, (
            f"Expected 2 captures for two independent chains, got {len(state.captured)}. "
            f"Captured: {[c.get('kind') for c in state.captured]}"
        )
        print("PASS: Two independent chains -> 2 separate captures")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_var_mean_single_chain():
    """var_mean -> getitem -> pointwise should stay as one capture."""
    from capture_hook import _CaptureState

    output_dir = tempfile.mkdtemp(prefix="test_cc_")
    try:
        state = _CaptureState(output_dir, label="test", validate=False)
        gm = _build_var_mean_chain_graph()
        state.process_graph(gm)

        # Should have 1 capture (all nodes are data-connected)
        assert len(state.captured) == 1, (
            f"Expected 1 capture for var_mean chain, got {len(state.captured)}. "
            f"Captured: {[c.get('kind') for c in state.captured]}"
        )
        print("PASS: var_mean -> getitem -> pointwise -> 1 capture (data-dependent)")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_chain_plus_independent_scalar():
    """A chain + an independent scalar op -> separate captures."""
    from capture_hook import _CaptureState

    output_dir = tempfile.mkdtemp(prefix="test_cc_")
    try:
        state = _CaptureState(output_dir, label="test", validate=False)
        gm = _build_chain_plus_independent_scalar()
        state.process_graph(gm)

        # Should have 2 captures (chain and independent scalar)
        assert len(state.captured) == 2, (
            f"Expected 2 captures for chain + independent scalar, got {len(state.captured)}. "
            f"Captured: {[c.get('kind') for c in state.captured]}"
        )
        print("PASS: chain + independent scalar -> 2 separate captures")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    test_two_independent_chains()
    test_var_mean_single_chain()
    test_chain_plus_independent_scalar()
    print("\nAll tests passed!")
