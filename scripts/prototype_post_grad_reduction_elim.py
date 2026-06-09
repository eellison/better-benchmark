#!/usr/bin/env python3
"""Prototype post-grad affine reduction extraction.

This is intentionally a small analyzer, not an Inductor pass.  It traces one of
the standalone post-grad-style repro modules, finds reductions, and tries to
push a chosen `sum(..., dims)` through affine pointwise structure:

    sum(a + b)       -> sum(a) + sum(b)
    sum(a - b)       -> sum(a) - sum(b)
    sum(k * x)       -> k * sum(x), when k is invariant over the reduced dims
    sum(broadcast(k))-> reduction_numel * k

Opaque/nonlinear producers such as `where` remain leaves and become summary
reductions.  This is the generic core that would fit naturally as a guarded
post-grad simplification before scheduling/fusion.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.fx as fx


SHAPE_RE = re.compile(r"\[[^\]]*\]")


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"{path.stem}_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_shape(annotation: object) -> tuple[int, ...] | None:
    if not isinstance(annotation, str):
        return None
    match = SHAPE_RE.search(annotation)
    if match is None:
        return None
    body = match.group(0)[1:-1].strip()
    if not body:
        return ()
    dims: list[int] = []
    for dim in body.split(","):
        dim = dim.strip()
        if not dim or not re.fullmatch(r"-?\d+", dim):
            return None
        dims.append(int(dim))
    return tuple(dims)


def _normalize_dim(dim: int, rank: int) -> int:
    return dim + rank if dim < 0 else dim


def _slice_len(length: int, start: int | None, end: int | None) -> int:
    # Match Python slicing enough for captured repro constants.
    lo = 0 if start is None else start
    hi = length if end is None or end == sys.maxsize else end
    if lo < 0:
        lo += length
    if hi < 0:
        hi += length
    lo = max(0, min(length, lo))
    hi = max(0, min(length, hi))
    return max(0, hi - lo)


def _broadcast_shape(*shapes: tuple[int, ...] | None) -> tuple[int, ...] | None:
    known = [shape for shape in shapes if shape is not None]
    if not known:
        return None
    rank = max(len(shape) for shape in known)
    result: list[int] = []
    for axis_from_right in range(1, rank + 1):
        dims = []
        for shape in known:
            dims.append(shape[-axis_from_right] if axis_from_right <= len(shape) else 1)
        non_one = {dim for dim in dims if dim != 1}
        if len(non_one) > 1:
            return None
        result.append(next(iter(non_one)) if non_one else 1)
    return tuple(reversed(result))


def infer_shapes(graph: fx.Graph) -> dict[fx.Node, tuple[int, ...] | None]:
    shapes: dict[fx.Node, tuple[int, ...] | None] = {}
    for node in graph.nodes:
        shape: tuple[int, ...] | None = None
        if node.op == "placeholder":
            shape = _parse_shape(node.type)
        elif node.op == "call_function":
            target = node.target
            args = node.args
            if target == torch.ops.aten.slice.Tensor:
                base = shapes.get(args[0])
                if base is not None:
                    dim = _normalize_dim(int(args[1]), len(base))
                    out = list(base)
                    out[dim] = _slice_len(base[dim], int(args[2]), int(args[3]))
                    shape = tuple(out)
            elif target == torch.ops.aten.unsqueeze.default:
                base = shapes.get(args[0])
                if base is not None:
                    dim = _normalize_dim(int(args[1]), len(base) + 1)
                    out = list(base)
                    out.insert(dim, 1)
                    shape = tuple(out)
            elif target == torch.ops.aten.sum.dim_IntList:
                base = shapes.get(args[0])
                if base is not None:
                    dims = {_normalize_dim(int(dim), len(base)) for dim in args[1]}
                    shape = tuple(dim for idx, dim in enumerate(base) if idx not in dims)
            elif target in {
                torch.ops.aten.add.Tensor,
                torch.ops.aten.sub.Tensor,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.where.self,
                torch.ops.aten.le.Scalar,
            }:
                shape = _broadcast_shape(
                    *(shapes.get(arg) for arg in args if isinstance(arg, fx.Node))
                )
        shapes[node] = shape
    return shapes


def _node_name(value: object) -> str:
    return value.name if isinstance(value, fx.Node) else repr(value)


def _strip_unsqueeze(node: fx.Node) -> fx.Node:
    while (
        isinstance(node, fx.Node)
        and node.op == "call_function"
        and node.target == torch.ops.aten.unsqueeze.default
    ):
        node = node.args[0]
    return node


def _format_value(value: object) -> str:
    if isinstance(value, fx.Node):
        stripped = _strip_unsqueeze(value)
        if stripped is not value:
            return stripped.name
        return value.name
    return repr(value)


def _format_invariant(value: object) -> str:
    if not isinstance(value, fx.Node):
        return repr(value)
    stripped = _strip_unsqueeze(value)
    if stripped is not value:
        return _format_invariant(stripped)
    if value.op != "call_function":
        return value.name

    target = value.target
    args = value.args
    if target == torch.ops.aten.mul.Tensor:
        return f"({_format_invariant(args[0])} * {_format_invariant(args[1])})"
    if target == torch.ops.aten.add.Tensor:
        return f"({_format_invariant(args[0])} + {_format_invariant(args[1])})"
    if target == torch.ops.aten.sub.Tensor:
        return f"({_format_invariant(args[0])} - {_format_invariant(args[1])})"
    return value.name


@dataclass(frozen=True)
class Expr:
    text: str

    def __add__(self, other: "Expr") -> "Expr":
        return Expr(f"({self.text} + {other.text})")

    def __sub__(self, other: "Expr") -> "Expr":
        return Expr(f"({self.text} - {other.text})")

    def __mul__(self, other: "Expr") -> "Expr":
        return Expr(f"({self.text} * {other.text})")


def _as_expr(value: object) -> Expr:
    if isinstance(value, (int, float)):
        return Expr(repr(value))
    return Expr(_format_value(value))


def _broadcast_dim(shape: tuple[int, ...], out_rank: int, dim: int) -> int:
    offset = out_rank - len(shape)
    local_dim = dim - offset
    if local_dim < 0:
        return 1
    return shape[local_dim]


def is_invariant(
    value: object,
    reduce_dims: Iterable[int],
    out_shape: tuple[int, ...],
    shapes: dict[fx.Node, tuple[int, ...] | None],
) -> bool:
    if not isinstance(value, fx.Node):
        return True
    shape = shapes.get(value)
    if shape is None:
        return False
    out_rank = len(out_shape)
    for dim in reduce_dims:
        if _broadcast_dim(shape, out_rank, dim) != 1:
            return False
    return True


def reduction_count(out_shape: tuple[int, ...], reduce_dims: Iterable[int]) -> int:
    count = 1
    for dim in reduce_dims:
        count *= out_shape[dim]
    return count


def summarize(
    value: object,
    reduce_dims: tuple[int, ...],
    out_shape: tuple[int, ...],
    shapes: dict[fx.Node, tuple[int, ...] | None],
    existing_reductions: dict[tuple[fx.Node, tuple[int, ...]], fx.Node],
) -> Expr:
    if is_invariant(value, reduce_dims, out_shape, shapes):
        return Expr(f"{reduction_count(out_shape, reduce_dims)} * {_format_invariant(value)}")
    if not isinstance(value, fx.Node) or value.op != "call_function":
        return Expr(f"sum({_format_value(value)})")

    existing = existing_reductions.get((value, reduce_dims))
    if existing is not None:
        return Expr(existing.name)

    target = value.target
    lhs = value.args[0] if value.args else None
    rhs = value.args[1] if len(value.args) > 1 else None

    if target == torch.ops.aten.add.Tensor:
        return summarize(lhs, reduce_dims, out_shape, shapes, existing_reductions) + summarize(
            rhs, reduce_dims, out_shape, shapes, existing_reductions
        )
    if target == torch.ops.aten.sub.Tensor:
        return summarize(lhs, reduce_dims, out_shape, shapes, existing_reductions) - summarize(
            rhs, reduce_dims, out_shape, shapes, existing_reductions
        )
    if target == torch.ops.aten.mul.Tensor:
        lhs_inv = is_invariant(lhs, reduce_dims, out_shape, shapes)
        rhs_inv = is_invariant(rhs, reduce_dims, out_shape, shapes)
        if lhs_inv and not rhs_inv:
            return Expr(_format_invariant(lhs)) * summarize(
                rhs, reduce_dims, out_shape, shapes, existing_reductions
            )
        if rhs_inv and not lhs_inv:
            return Expr(_format_invariant(rhs)) * summarize(
                lhs, reduce_dims, out_shape, shapes, existing_reductions
            )
        if lhs_inv and rhs_inv:
            return Expr(f"{reduction_count(out_shape, reduce_dims)} * {_format_invariant(value)}")

    return Expr(f"sum({_format_value(value)})")


def find_reductions(graph: fx.Graph) -> list[fx.Node]:
    return [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.sum.dim_IntList
    ]


def existing_reduction_map(
    reductions: Iterable[fx.Node],
    *,
    exclude: fx.Node | None = None,
) -> dict[tuple[fx.Node, tuple[int, ...]], fx.Node]:
    result: dict[tuple[fx.Node, tuple[int, ...]], fx.Node] = {}
    for node in reductions:
        if node is exclude:
            continue
        producer = node.args[0]
        if isinstance(producer, fx.Node):
            result[(producer, tuple(int(dim) for dim in node.args[1]))] = node
    return result


def _maybe_reciprocal(count: int, value: float) -> bool:
    return math.isclose(count * value, 1.0, rel_tol=0.0, abs_tol=0.0)


def print_repro_specific_notes(reductions: list[fx.Node], count: int) -> None:
    # This is not part of the generic extraction.  It explains the extra
    # simplification visible in this repro once extraction has exposed it.
    for node in reductions:
        producer = node.args[0]
        if not isinstance(producer, fx.Node):
            continue
        for user in list(node.users):
            if (
                user.op == "call_function"
                and user.target == torch.ops.aten.mul.Tensor
                and len(user.args) == 2
            ):
                const = user.args[0] if not isinstance(user.args[0], fx.Node) else user.args[1]
                if isinstance(const, float) and _maybe_reciprocal(count, const):
                    print(
                        f"  {count} * {user.name} == {node.name} "
                        f"because {const!r} is 1/{count}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repro",
        type=Path,
        nargs="?",
        default=Path("repros/canonical/sum_sum_sum_d8e421a07bf7/repro.py"),
    )
    parser.add_argument(
        "--reduction",
        default=None,
        help="FX node name to analyze; default is the last sum.dim_IntList node",
    )
    args = parser.parse_args()

    module = _load_module(args.repro)
    graph_module = fx.symbolic_trace(module.Repro())
    shapes = infer_shapes(graph_module.graph)
    reductions = find_reductions(graph_module.graph)
    if not reductions:
        raise SystemExit("no reductions found")

    if args.reduction is None:
        reduction = reductions[-1]
    else:
        by_name = {node.name: node for node in reductions}
        reduction = by_name[args.reduction]

    reductions_by_producer = existing_reduction_map(reductions, exclude=reduction)

    input_node = reduction.args[0]
    input_shape = shapes[input_node]
    if input_shape is None:
        raise SystemExit(f"could not infer input shape for {reduction.name}")
    reduce_dims = tuple(_normalize_dim(int(dim), len(input_shape)) for dim in reduction.args[1])

    print(f"repro: {args.repro}")
    print(f"candidate: {reduction.name} = sum({input_node.name}, dims={list(reduction.args[1])})")
    print(f"input_shape: {input_shape}")
    print(f"reduction_count: {reduction_count(input_shape, reduce_dims)}")
    print("\naffine extraction:")
    extracted = summarize(
        input_node,
        reduce_dims,
        input_shape,
        shapes,
        reductions_by_producer,
    )
    print(f"  {reduction.name} := {extracted.text}")

    print("\nreciprocal-count simplifications:")
    print_repro_specific_notes(reductions, reduction_count(input_shape, reduce_dims))

    print("\nsummary reductions available/introduced:")
    seen: set[str] = set()
    for node in reductions:
        producer = node.args[0]
        text = f"sum({_node_name(producer)}, dims={list(node.args[1])}) -> {node.name}"
        if text not in seen:
            seen.add(text)
            print(f"  {text}")


if __name__ == "__main__":
    main()
