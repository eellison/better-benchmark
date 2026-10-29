"""Describe MM corpus DAGs for reporting."""
from __future__ import annotations

import itertools

import torch

from matmul_partitioner import (
    horizontal_matmul_shapes_match,
    is_computation,
    is_matmul,
    is_reduction,
    op_name,
    tensor_shape,
    value_inputs,
)


def analyze_graph(gm):
    """Report coverage from actual value dependencies and reduction axes."""
    nodes = list(gm.graph.nodes)
    dots = [n for n in nodes if is_matmul(n)]
    ancestors, nearest_mm, frontier, depth = {}, {}, {}, {}
    coverage = set()

    def pw(n):
        return (
            is_computation(n) and not is_reduction(n)
            and torch.Tag.pointwise in getattr(n.target, "tags", ())
        )

    def parallel(values):
        return any(a not in ancestors[b] and b not in ancestors[a]
                   for a, b in itertools.combinations(values, 2))

    for n in nodes:
        inputs = value_inputs(n)
        ancestors[n] = set().union(*(ancestors[a] | {a} for a in inputs))
        upstream = set().union(*(nearest_mm[a] for a in inputs))
        nearest_mm[n] = {n} if is_matmul(n) else upstream
        depth[n] = max((depth[a] for a in inputs), default=0) + int(is_reduction(n))
        previous = {}
        for a in inputs:
            for red, seen_pw in frontier[a].items():
                previous[red] = previous.get(red, False) or seen_pw
        if pw(n):
            previous = {red: True for red in previous}
            if parallel(upstream):
                coverage.add("horizontal_pw_join")
                if op_name(n) in {"aten::mul", "prims::mul"}:
                    coverage.add("gated_parallel_mm")
        if is_matmul(n):
            if upstream:
                coverage.add("vertical_mm")
            if parallel(upstream):
                coverage.add("horizontal_then_mm")
            if any(is_matmul(red) and seen_pw for red, seen_pw in previous.items()):
                coverage.add("mm_pw_mm")
            if any(pw(a) and not nearest_mm[a] for a in ancestors[n]):
                coverage.add("pw_prologue")
        if is_reduction(n):
            if not is_matmul(n):
                name = op_name(n)
                if name in {"aten::sum", "prims::sum"} and any(
                    is_matmul(red) and seen_pw for red, seen_pw in previous.items()
                ):
                    coverage.add("mm_pw_sum")
                bindings = {arg.name: value for arg, value in zip(n.target._schema.arguments, n.args)}
                bindings.update(n.kwargs)
                shape = tensor_shape(inputs[0])
                dims = bindings.get("dim", bindings.get("dims"))
                dims = [dims] if isinstance(dims, int) else dims
                dims = list(range(len(shape))) if dims is None or dims == [] else list(dims)
                reshaped = any(op_name(a) in {"aten::view", "aten::reshape", "aten::_unsafe_view"}
                               and len(tensor_shape(a) or ()) > 2 and nearest_mm[a] & upstream
                               for a in ancestors[n])
                if upstream and name in {"aten::sum", "prims::sum", "aten::amax", "prims::amax"}:
                    if len(dims) > 1 or reshaped:
                        coverage.add("mm_block_amax_sum")
            frontier[n] = {n: False}
        else:
            frontier[n] = previous

    if parallel(dots):
        coverage.add("horizontal")
        if not any(is_computation(n) for n in nodes):
            coverage.add("pure_horizontal")
    for middle in dots:
        pre = [n for n in ancestors[middle] if is_reduction(n) and not is_matmul(n)]
        post = [n for n in nodes if is_reduction(n) and not is_matmul(n) and middle in ancestors[n]]
        if pre and post:
            coverage.add("reduction_mm_reduction")
            if any(op_name(a) in {"aten::index", "aten::index_select", "aten::gather", "aten::embedding"}
                   for red in pre for a in ancestors[red]):
                coverage.add("gather_reduction_mm_reduction")

    output_node = next(n for n in nodes if n.op == "output")
    metrics = {
        "mm": len(dots),
        "compute": sum(is_matmul(n) or is_computation(n) for n in nodes),
        "nodes": sum(n.op in {"call_function", "call_method", "get_attr"} for n in nodes),
        "inputs": sum(n.op == "placeholder" and isinstance(n.meta.get("val"), torch.Tensor) for n in nodes),
        "outputs": len(output_node.all_input_nodes),
        "contraction_depth": max(depth.values(), default=0),
    }
    return {
        "coverage": sorted(coverage),
        "metrics": metrics,
        "horizontal_shapes_match": horizontal_matmul_shapes_match(
            dots, ancestors, {n: tensor_shape(n) for n in dots}
        ),
    }
