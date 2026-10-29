"""Separate simple MM alignment/input-packing controls from richer fusion DAGs.

This labels benchmark purpose, not mathematical equivalence or removability.
In particular, a BMM equality-to-infinity mask can be true for nonfinite inputs
or finite values that overflow. Controls remain runnable benchmark cases.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.fx import GraphModule, Node

from matmul_partitioner import (
    is_computation,
    is_matmul,
    op_name,
    tensor_shape,
    value_inputs,
)

_VALUE_WRAPPERS = {
    "aten::view",
    "aten::reshape",
    "aten::_unsafe_view",
    "aten::permute",
    "aten::transpose",
    "aten::t",
    "aten::squeeze",
    "aten::unsqueeze",
    "aten::expand",
    "aten::clone",
    "aten::contiguous",
    "aten::detach",
    "aten::_to_copy",
    "prims::convert_element_type",
}
_CROP_WRAPPERS = {
    "aten::clone",
    "aten::contiguous",
    "aten::detach",
    "aten::_to_copy",
    "prims::convert_element_type",
    "aten::slice",
}
_ZERO_ALLOCATIONS = {"aten::zeros", "aten::zeros_like", "aten::new_zeros"}
_AXIS_PRESERVERS = {
    "aten::clone",
    "aten::contiguous",
    "aten::detach",
    "aten::_to_copy",
    "prims::convert_element_type",
}


def _literal_metadata(value) -> bool:
    if value is None or type(value) in (int, float, bool, str):
        return True
    return isinstance(value, (list, tuple)) and all(_literal_metadata(x) for x in value)


def _concrete_value(value):
    """Resolve lifted argument metadata, never tensor contents or estimates."""
    if isinstance(value, Node):
        if "val" in value.meta and _literal_metadata(value.meta["val"]):
            return value.meta["val"]
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_concrete_value(x) for x in value)
    return value


def _zero_scalar(value) -> bool:
    return type(value) in (int, float, bool) and value == 0


def _binding(node: Node, name: str, default=None):
    if name in node.kwargs:
        return _concrete_value(node.kwargs[name])
    schema = getattr(node.target, "_schema", None)
    if schema is not None:
        for index, arg in enumerate(schema.arguments):
            if arg.name == name and index < len(node.args):
                return _concrete_value(node.args[index])
    return default


def _tensor(node: Node):
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _contiguous(sizes, strides):
    expected = 1
    for size, stride in zip(reversed(sizes), reversed(strides)):
        if size > 1 and stride != expected:
            return False
        expected *= size
    return True


def _known_zero(node: Node, memo: dict[Node, bool], included: set[Node]) -> bool:
    # An external producer is an opaque input, even if it happens to be full(0).
    if node not in included:
        return False
    if node in memo:
        return memo[node]
    name = op_name(node)
    zero = name in _ZERO_ALLOCATIONS
    if name in {"aten::full", "aten::full_like"}:
        zero = _zero_scalar(_binding(node, "fill_value"))
    elif name in _VALUE_WRAPPERS and node.args and isinstance(node.args[0], Node):
        zero = _known_zero(node.args[0], memo, included)
    elif name == "aten::cat":
        zero = all(_known_zero(n, memo, included) for n in node.args[0])
    memo[node] = zero
    return zero


def _padding(node: Node, zero_memo: dict[Node, bool], included: set[Node]):
    """Return positive, zero-filled extensions, without reading tensor data."""
    shape = tensor_shape(node)
    if shape is None:
        return None
    name = op_name(node)
    if name == "aten::constant_pad_nd":
        pads = _binding(node, "pad")
        old_shape = tensor_shape(node.args[0])
        if old_shape is None or not _zero_scalar(_binding(node, "value", 0)):
            return None
        if not isinstance(pads, (tuple, list)) or len(pads) % 2:
            return None
        if any(type(p) is not int or p < 0 for p in pads):
            return None
        extensions = []
        for i in range(len(pads) // 2):
            before, after = pads[2 * i : 2 * i + 2]
            if before or after:
                dim = len(shape) - i - 1
                if dim < 0 or shape[dim] != old_shape[dim] + before + after:
                    return None
                extensions.append((dim, before, after, old_shape[dim]))
        return extensions or None
    if name == "aten::cat":
        tensors = node.args[0]
        real = [n for n in tensors if not _known_zero(n, zero_memo, included)]
        if len(real) != 1 or len(tensors) == 1:
            return None
        dim = _binding(node, "dim", 0)
        if type(dim) is not int:
            return None
        dim %= len(shape)
        old_shape = tensor_shape(real[0])
        if old_shape is None:
            return None
        index = tensors.index(real[0])
        before = sum(tensor_shape(n)[dim] for n in tensors[:index])
        after = sum(tensor_shape(n)[dim] for n in tensors[index + 1 :])
        return [(dim, before, after, old_shape[dim])] if before + after else None
    return None


def _slice_range(node: Node, size: int):
    start = _binding(node, "start", 0)
    end = _binding(node, "end")
    step = _binding(node, "step", 1)
    if any(v is not None and type(v) is not int for v in (start, end, step)):
        return None
    if step == 0:
        return None
    return slice(start, end, step).indices(size)


def _full_overwrite(node: Node) -> bool:
    name = op_name(node)
    if name not in {
        "aten::select_scatter",
        "aten::slice_scatter",
        "aten::as_strided_scatter",
    }:
        return False
    base, source = node.args[:2]
    b, s = _tensor(base), _tensor(source)
    if b is None or s is None or b.numel() != s.numel():
        return False
    if name == "aten::select_scatter":
        dim = _binding(node, "dim")
        if type(dim) is not int:
            return False
        dim %= b.ndim
        return b.shape[dim] == 1
    if name == "aten::slice_scatter":
        dim = _binding(node, "dim", 0)
        if type(dim) is not int:
            return False
        dim %= b.ndim
        bounds = _slice_range(node, b.shape[dim])
        if bounds is None:
            return False
        start, stop, step = bounds
        return start == 0 and stop == b.shape[dim] and step == 1
    sizes = _binding(node, "size")
    strides = _binding(node, "stride")
    offset = _binding(node, "storage_offset", 0)
    if not isinstance(sizes, (tuple, list)) or not isinstance(strides, (tuple, list)):
        return False
    if len(sizes) != len(strides) or any(
        type(d) is not int for d in (*sizes, *strides)
    ):
        return False
    if offset is None:
        offset = b.storage_offset()
    if type(offset) is not int:
        return False
    return (
        offset == 0
        and b.storage_offset() == 0
        and _contiguous(b.shape, b.stride())
        and _contiguous(sizes, strides)
        and math.prod(sizes) == b.numel()
    )


def _same_axis_crop(
    mm: Node, node: Node, dim: int, before: int, old_size: int, included: set[Node]
):
    """Prove that a downstream slice discards exactly the padded M/N lanes."""
    if op_name(node) != "aten::slice":
        return False
    current = node.args[0]
    while current is not mm:
        if current not in included or op_name(current) not in _CROP_WRAPPERS:
            return False
        current = current.args[0]
        if not isinstance(current, Node):
            return False
    shape = tensor_shape(node.args[0])
    if shape is None or len(shape) != len(tensor_shape(mm)):
        return False
    axis = _binding(node, "dim", 0)
    if type(axis) is not int or axis % len(shape) != dim:
        return False
    bounds = _slice_range(node, shape[dim])
    if bounds is None:
        return False
    start, stop, step = bounds
    return start == before and stop == before + old_size and step == 1


def _unwrapped_operand(node: Node, included: set[Node]) -> Node:
    while (
        node in included
        and op_name(node) in _AXIS_PRESERVERS
        and isinstance(node.args[0], Node)
    ):
        parent = node.args[0]
        if parent not in included:
            break
        if tensor_shape(parent) != tensor_shape(node):
            break
        node = parent
    return node


def _classify_nodes(nodes: list[Node]) -> dict:
    included = set(nodes)
    mms = [n for n in nodes if is_matmul(n)]
    core = {
        "category": "core",
        "control_types": [],
        "reasons": [],
        "finite_input_only": False,
    }
    if len(mms) != 1:
        return core
    mm = mms[0]
    if tensor_shape(mm) is None:
        return core
    ancestors = {}
    for n in nodes:
        ancestors[n] = set().union(
            *(ancestors.get(p, set()) | {p} for p in value_inputs(n))
        )
    substantive = [
        n for n in nodes if is_computation(n) and op_name(n) not in _ZERO_ALLOCATIONS
    ]
    if not substantive:
        return core
    if len(substantive) == 1 and op_name(mm) == "aten::bmm":
        eq = substantive[0]
        if op_name(eq) == "aten::eq" and mm in ancestors[eq]:
            resolved = [_concrete_value(a) for a in eq.args]
            literal = [a for a in resolved if type(a) in (float, int)]
            if literal == [-math.inf]:
                return {
                    "category": "control",
                    "control_types": ["bmm_infinity_mask"],
                    "reasons": [
                        "The only extra computation is BMM == -inf; useful as a mask control, usually false for bounded finite sampled inputs. Nonfinite inputs or finite overflow can make it true."
                    ],
                    "finite_input_only": True,
                }

    start = 1 if op_name(mm) in {"aten::addmm", "aten::baddbmm", "aten::addmv"} else 0
    operands = mm.args[start : start + 2]
    if len(operands) != 2 or any(tensor_shape(p) is None for p in operands):
        return core
    bias = mm.args[0] if start else None
    zero_memo = {}
    types, reasons = set(), []
    finite = False
    for n in substantive:
        if n not in ancestors[mm]:
            return core
        if _full_overwrite(n):
            types.add("identity_input_copy")
            reasons.append(
                f"{n.name} copies a source over the entire destination before MM; base values never survive."
            )
            continue
        if (
            op_name(n) == "aten::cat"
            and bias is not None
            and (n is bias or n in ancestors.get(bias, set()))
            and not any(n is p or n in ancestors.get(p, set()) for p in operands)
        ):
            types.add("bias_packing")
            reasons.append(
                f"{n.name} only packs the intrinsic additive MM bias; it does not combine MM input/weight streams."
            )
            continue
        extensions = _padding(n, zero_memo, included)
        if extensions is None:
            return core
        matching = [
            i for i, p in enumerate(operands) if n is _unwrapped_operand(p, included)
        ]
        if not matching:
            # Bias padding/zero concatenation is checked through the bias role.
            if bias is not None and (n is bias or n in ancestors.get(bias, set())):
                types.add("bias_packing")
                reasons.append(f"{n.name} zero-pads the intrinsic additive MM bias.")
                continue
            return core
        for i in matching:
            shape = tensor_shape(operands[i])
            kdim = len(shape) - (1 if i == 0 else 2)
            output_dim = len(shape) - (2 if i == 0 else 1)
            for dim, before, after, old_size in extensions:
                if dim == kdim:
                    finite = True
                elif dim == output_dim:
                    if not any(
                        _same_axis_crop(mm, q, dim, before, old_size, included)
                        for q in nodes
                    ):
                        return core
                else:
                    return core
        types.add("zero_padding")
        reasons.append(
            f"{n.name} adds only known zero lanes; M/N padding is cropped back to the original domain, or padding is confined to K."
        )
    if not types:
        return core
    if finite:
        reasons.append(
            "K padding is an alignment control, not a universal identity: zero multiplied by nonfinite values and a changed floating-point reduction schedule can affect results."
        )
    return {
        "category": "control",
        "control_types": sorted(types),
        "reasons": reasons,
        "finite_input_only": finite,
    }


def classify_region_quality(region: Iterable[Node]) -> dict:
    """Classify an FX node subset in place, treating external nodes as inputs.

    No subgraph construction, tracing, canonicalization or capture_hook import
    is needed. Producer operations outside the region are opaque boundaries;
    their tensor metadata can establish shapes but never zero tensor values.
    Concrete non-tensor shape/scalar parameter metadata remains usable.
    """
    included = set(region)
    ordered, visited = [], set()

    def visit(node: Node):
        if node in visited:
            return
        visited.add(node)
        for parent in value_inputs(node):
            if parent in included:
                visit(parent)
        ordered.append(node)

    for node in sorted(included, key=lambda n: n.name):
        visit(node)
    return _classify_nodes(ordered)


def classify_graph_quality(gm: GraphModule) -> dict:
    """Return the same shape-specific core/control label for a whole graph.

    There are no model/hash allowlists, rankings or corpus count limits.
    """
    return _classify_nodes(list(gm.graph.nodes))
