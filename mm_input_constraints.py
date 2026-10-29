"""Input domains needed to replay extracted MM regions with useful values."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import fx

_SQRT_OPS = {
    "aten.sqrt.default",
    "aten.rsqrt.default",
    "prims.sqrt.default",
    "prims.rsqrt.default",
}
_CAST_OPS = {
    "prims.convert_element_type.default",
    "aten._to_copy.default",
    "aten.to.dtype",
    "aten.to.device",
}
_VALUE_VIEWS = {
    "aten.alias.default",
    "aten.as_strided.default",
    "aten.clone.default",
    "aten.detach.default",
    "aten.expand.default",
    "aten.permute.default",
    "aten.reshape.default",
    "aten.select.int",
    "aten.slice.Tensor",
    "aten.squeeze.default",
    "aten.squeeze.dim",
    "aten.squeeze.dims",
    "aten.t.default",
    "aten.transpose.int",
    "aten.unsqueeze.default",
    "aten.view.default",
    "aten._unsafe_view.default",
}
_ADD_OPS = {"aten.add.Tensor", "aten.add.Scalar", "prims.add.default"}


def annotate_external_sqrt_inputs(
    gm: fx.GraphModule,
    placeholder_info: dict[str, dict[str, Any]],
    *,
    minimum: float = 0.1,
) -> list[str]:
    """Give external sqrt/rsqrt statistics a positive input generator.

    Only follow value-preserving views, floating casts and nonnegative scalar
    epsilon additions to one floating placeholder. Computed variance and
    arbitrary arithmetic already define their own domains and are excluded.
    The capture caller opts in for MM regions; other corpus generators keep
    their existing behavior. Existing explicit generation specs are respected.

    Returns the annotated placeholder names in graph order. Does not change
    the graph or its input values.
    """
    if not 0.0 < minimum < math.inf:
        raise ValueError("positive input minimum must be finite and greater than zero")

    def scalar(value: Any) -> int | float | None:
        if isinstance(value, fx.Node) and value.op == "placeholder":
            value = value.meta.get("val")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        return None

    roots: dict[fx.Node, fx.Node | None] = {}

    def root(node: fx.Node) -> fx.Node | None:
        if node in roots:
            return roots[node]
        found = None
        value = node.meta.get("val")
        if node.op == "placeholder":
            if (
                node.name in placeholder_info
                and isinstance(value, torch.Tensor)
                and value.is_floating_point()
            ):
                found = node
        elif node.op == "call_function":
            target = str(node.target)
            if target in _VALUE_VIEWS | _CAST_OPS:
                if (
                    node.args
                    and isinstance(node.args[0], fx.Node)
                    and (
                        target not in _CAST_OPS
                        or isinstance(value, torch.Tensor)
                        and value.is_floating_point()
                    )
                ):
                    found = root(node.args[0])
            elif target in _ADD_OPS and len(node.args) >= 2:
                left, right = node.args[:2]
                left_scalar, right_scalar = scalar(left), scalar(right)
                alpha = scalar(node.kwargs.get("alpha", 1))
                if alpha == 1:
                    if (
                        isinstance(left, fx.Node)
                        and left_scalar is None
                        and right_scalar is not None
                        and math.isfinite(right_scalar)
                        and right_scalar >= 0
                    ):
                        found = root(left)
                    elif (
                        isinstance(right, fx.Node)
                        and right_scalar is None
                        and left_scalar is not None
                        and math.isfinite(left_scalar)
                        and left_scalar >= 0
                    ):
                        found = root(right)
        roots[node] = found
        return found

    required = set()
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and str(node.target) in _SQRT_OPS
            and node.args
            and isinstance(node.args[0], fx.Node)
        ):
            source = root(node.args[0])
            if source is not None:
                required.add(source)

    annotated = []
    for node in gm.graph.nodes:
        if node not in required:
            continue
        info = placeholder_info[node.name]
        gen = info.get("gen") or info.get("generator")
        if (
            info.get("exact")
            or gen
            and gen.get("kind") not in (None, "randn", "positive")
        ):
            continue
        info["constraint"] = "nonnegative"
        if not gen or gen.get("kind") != "positive":
            info["gen"] = {"kind": "positive", "minimum": minimum}
        annotated.append(node.name)
    return annotated
