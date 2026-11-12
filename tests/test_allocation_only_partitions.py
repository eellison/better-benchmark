from __future__ import annotations

import pytest
import torch
import torch.fx as fx

import capture_hook


def _empty(graph: fx.Graph) -> fx.Node:
    return graph.call_function(
        torch.ops.aten.empty.memory_format,
        args=([4],),
        kwargs={
            "dtype": torch.float32,
            "layout": torch.strided,
            "device": torch.device("cpu"),
            "pin_memory": False,
        },
    )


def test_extraction_omits_standalone_allocation_only_component(monkeypatch):
    graph = fx.Graph()
    allocation = _empty(graph)
    graph.output(allocation)
    module = fx.GraphModule({}, graph)
    monkeypatch.setattr(
        capture_hook,
        "partition_node_is_supported",
        lambda node: node is allocation,
    )

    assert capture_hook.partition_is_standalone_allocation_only([allocation])
    assert capture_hook.get_fusion_partitions(module) == []


def test_extraction_retains_empty_consumed_by_index_put(monkeypatch):
    graph = fx.Graph()
    indices = graph.placeholder("indices")
    values = graph.placeholder("values")
    allocation = _empty(graph)
    writer = graph.call_function(
        torch.ops.aten.index_put.default,
        args=(allocation, [indices], values),
    )
    graph.output(writer)
    module = fx.GraphModule({}, graph)
    supported = {allocation, writer}
    monkeypatch.setattr(
        capture_hook,
        "partition_node_is_supported",
        lambda node: node in supported,
    )

    assert not capture_hook.partition_is_standalone_allocation_only(
        [allocation, writer]
    )
    partitions = capture_hook.get_fusion_partitions(module)
    assert len(partitions) == 1
    assert set(partitions[0]) == supported


def test_extraction_omits_empty_written_by_external_kernel(monkeypatch):
    """An output buffer filled by a non-fusible writer (e.g. a user Triton
    kernel) extracts to a bare allocation, whose repro returns uninitialized
    memory; the writer is outside every fusible partition."""

    def custom_writer(buffer):
        return buffer

    graph = fx.Graph()
    allocation = _empty(graph)
    writer = graph.call_function(custom_writer, args=(allocation,))
    graph.output(writer)
    module = fx.GraphModule({}, graph)
    monkeypatch.setattr(
        capture_hook,
        "partition_node_is_supported",
        lambda node: node is allocation,
    )

    assert capture_hook.partition_is_standalone_allocation_only([allocation])
    assert capture_hook.get_fusion_partitions(module) == []


def test_extraction_omits_viewed_empty_written_by_external_kernel(monkeypatch):
    def custom_writer(buffer):
        return buffer

    graph = fx.Graph()
    allocation = _empty(graph)
    view = graph.call_function(torch.ops.aten.view.default, args=(allocation, [2, 2]))
    writer = graph.call_function(custom_writer, args=(view,))
    graph.output(writer)
    module = fx.GraphModule({}, graph)
    supported = {allocation, view}
    monkeypatch.setattr(
        capture_hook,
        "partition_node_is_supported",
        lambda node: node in supported,
    )

    assert capture_hook.partition_is_standalone_allocation_only([allocation, view])
    assert capture_hook.get_fusion_partitions(module) == []


def _captured_outputs_with_nan_primed_allocator(gm, tmp_path):
    """Capture ``gm``'s regions and run each repro after filling the caching
    allocator's free blocks with NaN, so an uninitialized read shows up."""
    import importlib.util

    from input_codec import spec_from_compact
    from repro_harness import make_inputs_from_config

    state = capture_hook._CaptureState(str(tmp_path), label="m")
    state.process_graph(gm)
    assert state.dropped == []
    outputs = []
    for entry in state.captured:
        spec = importlib.util.spec_from_file_location("region", entry["file"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = {"inputs": [spec_from_compact(e) for e in entry["inputs"]]}
        inputs = make_inputs_from_config(config)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        for numel in (64 * 128, 1 << 20, 1 << 22):
            torch.full((numel,), float("nan"), dtype=torch.float16, device="cuda")
        outputs.append(module.Repro()(*inputs))
    return state.captured, outputs


def _graph_with_external_writer(writer: str) -> fx.GraphModule:
    """``empty`` -> an in-place writer whose result nobody reads -> a view of
    the allocation -> a reduction, as Dynamo graphs order them."""
    from torch._higher_order_ops.triton_kernel_wrap import (
        triton_kernel_wrapper_mutation,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    def f(x):
        buffer = torch.empty(64, 1, 128, dtype=torch.float16, device="cuda")
        y = (x + buffer.view(64, 128)).float()
        return y.pow(2).mean(1)

    gm = make_fx(f, tracing_mode="fake")(
        torch.randn(64, 128, dtype=torch.float16, device="cuda")
    )
    allocation = next(
        n for n in gm.graph.nodes if n.target is torch.ops.aten.empty.memory_format
    )
    with gm.graph.inserting_after(allocation):
        if writer == "user_triton":
            gm.graph.call_function(
                triton_kernel_wrapper_mutation,
                (),
                {
                    "kernel_idx": 10_000,
                    "constant_args_idx": 0,
                    "grid": [(8, 1, 1)],
                    "tma_descriptor_metadata": {},
                    "kwargs": {"out_ptr": allocation},
                },
            )
        else:
            node = gm.graph.call_function(
                torch.ops.aten.fill_.Scalar, (allocation, 1.0)
            )
            node.meta["val"] = allocation.meta["val"]
    gm.recompile()
    return gm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="runs repros on CUDA")
@pytest.mark.parametrize("writer", ["user_triton", "aten_inplace"])
def test_buffer_written_in_place_enters_region_as_input(writer, tmp_path):
    gm = _graph_with_external_writer(writer)

    captured, outputs = _captured_outputs_with_nan_primed_allocator(gm, tmp_path)

    assert captured
    for out in outputs:
        for leaf in out if isinstance(out, tuple) else (out,):
            assert torch.isfinite(leaf).all()
    assert not any("empty" in op for e in captured for op in e["origin_ops"])


def _launch_writing(gm: fx.GraphModule, after: fx.Node, buffer: fx.Node, **reads) -> None:
    from torch._higher_order_ops.triton_kernel_wrap import (
        triton_kernel_wrapper_mutation,
    )

    with gm.graph.inserting_after(after):
        gm.graph.call_function(
            triton_kernel_wrapper_mutation,
            (),
            {
                "kernel_idx": 10_000,
                "constant_args_idx": 0,
                "grid": [(8, 1, 1)],
                "tma_descriptor_metadata": {},
                "kwargs": {**reads, "out_ptr": buffer},
            },
        )
    gm.recompile()


_WRITTEN_BUFFERS = {
    # A reinplaced accumulator (post-grad zeros).
    "full": lambda x: torch.full((64, 128), 0.0),
    # decompose_triton_kernel_wrapper_functional's copy of a kernel's input.
    "clone": lambda x: (x * 3.0).clone(),
    # Written through views the alias walk must follow back to the empty.
    "split_with_sizes": lambda x: torch.ops.aten.split_with_sizes.default(
        torch.empty(64, 256), [128, 128], 1
    )[0],
    "view_dtype": lambda x: torch.empty(64, 128, dtype=torch.int32).view(torch.float32),
    "alias": lambda x: torch.ops.aten.alias.default(torch.empty(64, 128)),
    "unflatten": lambda x: torch.empty(64 * 128).unflatten(0, (64, 128)),
}


def _graph_with_written_buffer(kind: str):
    """``x + buffer`` -> a reduction, where a user Triton kernel writes
    ``buffer`` in place before it is read."""
    from torch.fx.experimental.proxy_tensor import make_fx

    make = _WRITTEN_BUFFERS[kind]

    def f(x):
        buffer = torch.ops.aten.alias.default(make(x)) if kind != "alias" else make(x)
        return (x + buffer).pow(2).mean(1)

    gm = make_fx(f, tracing_mode="fake")(torch.randn(64, 128))
    # The kernel writes the value the reader reads (the last alias before it).
    reader = next(n for n in gm.graph.nodes if n.target is torch.ops.aten.add.Tensor)
    buffer = reader.args[1]
    _launch_writing(gm, buffer, buffer)
    return gm, reader, buffer


@pytest.mark.parametrize("kind", sorted(_WRITTEN_BUFFERS))
def test_buffer_written_outside_partitions_enters_region_as_input(kind):
    gm, reader, buffer = _graph_with_written_buffer(kind)

    written = capture_hook.written_allocation_aliases(gm)
    parts = capture_hook.get_fusion_partitions(gm)

    assert buffer in written
    (region,) = [p for p in parts if reader in p]
    # The region reads the written buffer as an input, not recomputed.
    assert buffer not in region
    assert not written & {n for p in parts for n in p}


def _kernel_reading_a_writing_b(read_a_after: bool):
    """``a = x * 3`` and ``b = clone(x * 5)``; a kernel launch takes ``a``
    as an input and writes ``b``; ``x + b`` is read after the launch, and
    ``x + a`` before it or, with ``read_a_after``, after it."""
    from torch.fx.experimental.proxy_tensor import make_fx

    def f(x):
        a = x * 3.0
        b = (x * 5.0).clone()
        first = (x + a).sum(1)
        return first, (x + b).pow(2).mean(1)

    gm = make_fx(f, tracing_mode="fake")(torch.randn(64, 128))
    adds = [n for n in gm.graph.nodes if n.target is torch.ops.aten.add.Tensor]
    a, b = adds[0].args[1], adds[1].args[1]
    first_sum = next(iter(adds[0].users))
    _launch_writing(gm, first_sum, b, in_ptr=a)
    if read_a_after:
        adds[1].args = (adds[1].args[0], b)
        with gm.graph.inserting_after(adds[1]):
            late = gm.graph.call_function(torch.ops.aten.add.Tensor, (adds[1], a))
        adds[1].replace_all_uses_with(late, delete_user_cb=lambda u: u is not late)
        gm.recompile()
    return gm, a, b


def test_kernel_input_read_only_before_the_launch_stays_in_its_partition():
    gm, a, b = _kernel_reading_a_writing_b(read_a_after=False)

    written = capture_hook.written_allocation_aliases(gm)
    parts = capture_hook.get_fusion_partitions(gm)

    assert b in written
    assert a not in written
    assert any(a in p for p in parts)
    assert not any(b in p for p in parts)


def test_kernel_input_read_after_the_launch_is_cut_too():
    """Which tensors a launch writes is not in the graph, so an input read
    after the launch is cut like a written buffer."""
    gm, a, b = _kernel_reading_a_writing_b(read_a_after=True)

    written = capture_hook.written_allocation_aliases(gm)

    assert {a, b} <= written
    assert not written & {n for p in capture_hook.get_fusion_partitions(gm) for n in p}


def _fake_propagated(graph: fx.Graph, *inputs) -> fx.GraphModule:
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    gm = fx.GraphModule({}, graph)
    mode = FakeTensorMode()
    FakeTensorProp(gm, mode).propagate(*[mode.from_tensor(t) for t in inputs])
    return gm


def _full(graph: fx.Graph) -> fx.Node:
    return graph.call_function(
        torch.ops.aten.full.default,
        ([64, 128], 0.0),
        {"dtype": torch.float32, "layout": torch.strided,
         "device": torch.device("cpu"), "pin_memory": False},
    )


def test_fusible_in_place_writer_shares_its_buffers_partition():
    graph = fx.Graph()
    x, idx, val = (graph.placeholder(n) for n in ("x", "idx", "val"))
    buffer = _full(graph)
    graph.call_function(torch.ops.aten.index_put_.default, (buffer, [idx], val, True))
    graph.output(graph.call_function(
        torch.ops.aten.sum.dim_IntList,
        (graph.call_function(torch.ops.aten.add.Tensor, (x, buffer)), [1])))
    gm = _fake_propagated(
        graph, torch.randn(64, 128), torch.zeros(10, dtype=torch.long), torch.randn(10, 128)
    )

    (region,) = capture_hook.get_fusion_partitions(gm)

    assert buffer in region
    assert {str(n.target) for n in region} >= {
        "aten.index_put_.default", "aten.add.Tensor"
    }


def test_buffer_leaves_a_partition_its_fusible_writer_cannot_join():
    """``index_put_(buffer, ..., mm(buffer, w))``: the matmul between the
    buffer and its writer keeps them in different partitions, so the reader
    after the write must take the buffer as an input."""
    graph = fx.Graph()
    x, w, idx = (graph.placeholder(n) for n in ("x", "w", "idx"))
    buffer = _full(graph)
    product = graph.call_function(torch.ops.aten.mm.default, (buffer, w))
    rows = graph.call_function(torch.ops.aten.slice.Tensor, (product, 0, 0, 10))
    writer = graph.call_function(
        torch.ops.aten.index_put_.default, (buffer, [idx], rows, True)
    )
    reader = graph.call_function(torch.ops.aten.add.Tensor, (x, buffer))
    graph.output(graph.call_function(torch.ops.aten.sum.dim_IntList, (reader, [1])))
    gm = _fake_propagated(
        graph, torch.randn(64, 128), torch.randn(128, 128), torch.arange(10)
    )

    parts = capture_hook.get_fusion_partitions(gm)

    (region,) = [p for p in parts if reader in p]
    assert writer not in region
    assert buffer not in region


def _layer_norm_kernel():
    """A layer norm forward shaped like a captured user kernel: it reads X, W
    and B and writes Y, Mean and Rstd."""
    import triton
    import triton.language as tl

    @triton.jit
    def layer_norm_fwd(X, Y, W, B, Mean, Rstd, N, BLOCK_D: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        diff = tl.where(mask, x - mean, 0.0)
        rstd = 1 / tl.sqrt(tl.sum(diff * diff, axis=0) / N + 1e-5)
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        tl.store(Y + row * N + cols, (diff * rstd * w + b).to(Y.dtype.element_ty), mask=mask)

    return layer_norm_fwd


def _graph_normalizing_a_value_read_again():
    """``h = a + b``; a layer-norm launch reads ``h`` as X and writes the
    allocations Y, Mean and Rstd; ``h + c`` reads ``h`` after the launch, the
    way a residual stream is read again after its norm."""
    from torch._higher_order_ops.triton_kernel_wrap import (
        triton_kernel_wrapper_mutation,
    )
    from torch.fx.experimental.proxy_tensor import make_fx

    def f(a, b, c, w, bias):
        h = a + b
        y = torch.empty_like(h)
        mean = torch.empty(64, device="cuda")
        rstd = torch.empty(64, device="cuda")
        return (h + c).pow(2).mean(1), y.float().sum(1), mean, rstd

    inputs = [torch.randn(64, 128, device="cuda") for _ in range(3)] + [
        torch.randn(128, device="cuda") for _ in range(2)
    ]
    gm = make_fx(f, tracing_mode="fake")(*inputs)
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    h = next(n for n in gm.graph.nodes if n.target is torch.ops.aten.add.Tensor)
    y = next(n for n in gm.graph.nodes if n.target is torch.ops.aten.empty_like.default)
    mean, rstd = [n for n in gm.graph.nodes if n.target is torch.ops.aten.empty.memory_format]
    with gm.graph.inserting_after(rstd):
        gm.graph.call_function(
            triton_kernel_wrapper_mutation,
            (),
            {
                "kernel_idx": 0,
                "constant_args_idx": 0,
                "grid": [(64, 1, 1)],
                "tma_descriptor_metadata": {},
                "kwargs": {
                    "X": h, "Y": y, "W": placeholders[3], "B": placeholders[4],
                    "Mean": mean, "Rstd": rstd, "N": 128,
                },
            },
        )
    gm.recompile()
    return gm, h, y


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton analyzes the kernel for CUDA")
def test_trusted_kernel_reading_a_value_leaves_it_in_its_partition():
    from user_triton import TrustedKernels, WRITE_ANALYSIS_KEY

    gm, h, y = _graph_normalizing_a_value_read_again()
    gm.meta[WRITE_ANALYSIS_KEY] = TrustedKernels({0: _layer_norm_kernel()}, {0: {"BLOCK_D": 128}})

    written = capture_hook.written_allocation_aliases(gm)

    assert y in written
    assert h not in written
    assert any(h in p for p in capture_hook.get_fusion_partitions(gm))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="traces on CUDA")
@pytest.mark.parametrize("trusted", ["none", "other_kernel_index"])
def test_untrusted_kernel_counts_as_writing_everything_it_takes(trusted):
    from user_triton import TrustedKernels, WRITE_ANALYSIS_KEY

    gm, h, y = _graph_normalizing_a_value_read_again()
    if trusted == "other_kernel_index":
        # The loader trusted no kernel at this launch's index (e.g. its
        # identity did not match the reviewed source).
        gm.meta[WRITE_ANALYSIS_KEY] = TrustedKernels({1: _layer_norm_kernel()}, {0: {"BLOCK_D": 128}})

    written = capture_hook.written_allocation_aliases(gm)

    assert {h, y} <= written
    assert not any(h in p for p in capture_hook.get_fusion_partitions(gm))


def test_views_of_a_written_buffer_no_partition_holds_stay_partitioned():
    """A matmul's output written by a kernel and sliced afterwards: no
    partition recomputes the matmul, so its views read it as an input and
    need no cut."""
    from torch.fx.experimental.proxy_tensor import make_fx

    def f(x, w):
        m = x @ w
        return (m[:, 64:] - m[:, :64]).sum(1)

    gm = make_fx(f, tracing_mode="fake")(torch.randn(64, 128), torch.randn(128, 128))
    mm = next(n for n in gm.graph.nodes if n.target is torch.ops.aten.mm.default)
    slices = [n for n in gm.graph.nodes if n.target is torch.ops.aten.slice.Tensor]
    _launch_writing(gm, mm, mm)

    assert capture_hook.written_allocation_aliases(gm) == set()
    assert any(set(slices) <= set(p) for p in capture_hook.get_fusion_partitions(gm))


def _graph_reading_a_written_constant(read_before_write: bool):
    """``x + c`` for a graph constant ``c`` that a kernel launch writes in
    place, read after the launch or, with ``read_before_write``, before it."""
    root = torch.nn.Module()
    root.register_buffer("constant", torch.randn(64, 128))
    graph = fx.Graph()
    x = graph.placeholder("x")
    c = graph.get_attr("constant")
    reader = graph.call_function(torch.ops.aten.add.Tensor, (x, c))
    squared = graph.call_function(torch.ops.aten.pow.Tensor_Scalar, (reader, 2))
    graph.output(graph.call_function(torch.ops.aten.mean.dim, (squared, [1])))
    gm = fx.GraphModule(root, graph)
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    FakeTensorProp(gm, mode).propagate(mode.from_tensor(torch.randn(64, 128)))
    _launch_writing(gm, reader if read_before_write else c, c)
    return gm, reader


@pytest.mark.parametrize("read_before_write", [False, True])
def test_region_reading_a_constant_after_a_write_is_masked(read_before_write):
    gm, reader = _graph_reading_a_written_constant(read_before_write)

    (region,) = [p for p in capture_hook.get_fusion_partitions(gm) if reader in p]

    expected = None if read_before_write else "written_constant"
    assert capture_hook.region_mask_reason(region) == expected
