from __future__ import annotations

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
