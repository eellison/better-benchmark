"""Valid synthetic lengths and offsets for the inputs of jagged ops.

A jagged op reads segments of a values tensor through a lengths or offsets
input. Uniform random integers are not valid there: offsets must start at 0,
never decrease and end at the values length, and lengths must add up to it.
``jagged_input_specs`` finds, from the graph alone, which graph inputs feed
which jagged op and with what total, and returns a generator spec for each.
A jagged op is covered when every lengths/offsets operand it reads traces
back (through views, casts and complete cumsums) to a graph input whose
constraints all agree; a graph whose jagged nodes are all covered no longer
needs to be held at its production time.

The values are synthetic: segment lengths are drawn from a registered
distribution ("uniform" by default), so the kernels run valid work, but not
production's length distribution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

LENGTHS_KIND = "jagged_lengths"
OFFSETS_KIND = "jagged_offsets"
JAGGED_KINDS = frozenset({LENGTHS_KIND, OFFSETS_KIND})
DEFAULT_DISTRIBUTION = "uniform"


def _uniform_lengths(count: int, total: int, params: dict) -> Any:
    """Segment lengths of ``count`` segments adding up to ``total``: the gaps
    between sorted uniform cut points in [0, total]."""
    import torch

    cuts = torch.randint(0, total + 1, (max(count - 1, 0),), dtype=torch.int64).sort().values
    bounds = torch.cat([torch.zeros(1, dtype=torch.int64), cuts,
                        torch.full((1,), total, dtype=torch.int64)])
    return bounds.diff()


# family -> fn(count, total, params) -> int64 lengths (CPU, length ``count``,
# adding up to ``total``). A generator's "distribution" is a family name or
# {"family": name, **params}. Draw from the global torch RNG so seeded runs
# repeat.
LENGTHS_DISTRIBUTIONS: dict[str, Callable[[int, int, dict], Any]] = {
    DEFAULT_DISTRIBUTION: _uniform_lengths,
}


def register_lengths_distribution(name: str, fn: Callable[[int, int, dict], Any]) -> None:
    LENGTHS_DISTRIBUTIONS[name] = fn


def _cap(lengths: Any, max_len: int) -> Any:
    """Move lengths above ``max_len`` into segments with room, keeping the
    total. Callers check that count * max_len >= total."""
    import torch

    over = (lengths - max_len).clamp(min=0)
    lengths = lengths - over
    excess = int(over.sum())
    if excess:
        order = torch.randperm(lengths.numel())
        room = (max_len - lengths)[order]
        before = room.cumsum(0) - room
        lengths[order] += (excess - before).clamp(min=0).minimum(room)
    return lengths


def make_lengths(count: int, total: int, max_len: int | None = None,
                 distribution: str | dict = DEFAULT_DISTRIBUTION) -> Any:
    """int64 CPU lengths: ``count`` non-negative values adding up to
    ``total``, none above ``max_len`` when that is feasible."""
    import torch

    if count <= 0:
        if total:
            raise ValueError(f"no segments to hold a total of {total}")
        return torch.zeros(0, dtype=torch.int64)
    params = dict(distribution) if isinstance(distribution, dict) else {"family": distribution}
    family = params.pop("family")
    lengths = LENGTHS_DISTRIBUTIONS[family](count, total, params).to(torch.int64)
    if max_len is not None and count * max_len >= total:
        lengths = _cap(lengths, max_len)
    if lengths.numel() != count or int(lengths.sum()) != total or bool((lengths < 0).any()):
        raise ValueError(f"distribution {family!r} returned invalid lengths")
    return lengths


def resolve_total(gen: dict, specs: list[dict]) -> int:
    """The concrete total of a jagged generator, read from the other input
    specs of the same graph (after any symbolic point is instantiated), or
    the captured value when they are not given."""
    ref = gen["total"]
    by_name = {spec.get("name"): spec for spec in specs}
    if "sym" in ref and ref["sym"] in by_name:
        return int(by_name[ref["sym"]]["value"])
    if "size" in ref and ref["size"][0] in by_name:
        name, dim = ref["size"]
        return int(by_name[name]["shape"][dim])
    return int(ref["value"])


def make_jagged_tensor(gen: dict, shape: tuple, dtype: Any, device: Any) -> Any:
    """Lengths or offsets for a generator; its total is ``total_value`` when
    resolved against the graph's other inputs, else the captured one."""
    import torch
    from torch._guards import detect_fake_mode

    if len(shape) != 1:
        raise ValueError(f"{gen['kind']} inputs must be 1-D, got shape {shape}")
    if detect_fake_mode() is not None:
        return torch.empty(shape, dtype=dtype, device=device)
    count = shape[0] if gen["kind"] == LENGTHS_KIND else shape[0] - 1
    total = gen["total_value"] if "total_value" in gen else resolve_total(gen, [])
    lengths = make_lengths(count, int(total), gen.get("max_len"),
                           gen.get("distribution", DEFAULT_DISTRIBUTION))
    if gen["kind"] == OFFSETS_KIND:
        lengths = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    return lengths.to(dtype=dtype, device=device)


# Graph analysis.

# (namespace, op name) -> fn(node) -> [(operand, kind, total, max_len)] or
# None when the node's jagged inputs cannot be constrained. ``operand`` is
# an fx.Node, ``kind`` is "lengths" or "offsets", ``total`` is an fx.Node /
# SymInt / int, ``max_len`` an int or None.
JaggedRule = Callable[[Any], "list[tuple[Any, str, Any, Any]] | None"]


def _arg(node, index, name, default=None):
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def _rows(node):
    val = getattr(node, "meta", {}).get("val")
    return val.shape[0] if val is not None and len(val.shape) else None


def _dim(node, d):
    val = getattr(node, "meta", {}).get("val")
    return val.shape[d] if val is not None and len(val.shape) > d else None


def _single(offsets):
    return offsets[0] if isinstance(offsets, (list, tuple)) and len(offsets) == 1 else None


def _repeat_interleave(node):
    size = node.kwargs.get("output_size")
    repeats = node.args[0] if node.args else None
    if size is None or repeats is None:
        return None
    return [(repeats, "lengths", size, None)]


def _pack_segments(node):
    values, lengths, max_length = (_arg(node, i, n) for i, n in
                                   ((0, "t_in"), (1, "lengths"), (2, "max_length")))
    return [(lengths, "lengths", _rows(values), max_length)]


def _jagged_to_padded_dense(node):
    values, offsets, max_lengths = (_arg(node, i, n) for i, n in
                                    ((0, "values"), (1, "offsets"), (2, "max_lengths")))
    offsets = _single(offsets)
    max_len = _single(max_lengths)
    if offsets is None or max_len is None:
        return None
    return [(offsets, "offsets", _rows(values), max_len)]


def _jagged_to_padded_dense_backward(node):
    grad, offsets, total = (_arg(node, i, n) for i, n in
                            ((0, "grad_output"), (1, "offsets"), (2, "total_L")))
    offsets = _single(offsets)
    if offsets is None or total is None:
        return None
    return [(offsets, "offsets", total, _dim(grad, 1))]


def _dense_to_jagged(node):
    dense, offsets, total = (_arg(node, i, n) for i, n in
                             ((0, "dense"), (1, "offsets"), (2, "total_L")))
    offsets = _single(offsets)
    if offsets is None or total is None:
        return None
    return [(offsets, "offsets", total, _dim(dense, 1))]


def _cumsum(node):
    # Valid for any input; a consumer that reads its output as offsets
    # constrains its input through _source.
    return []


JAGGED_RULES: dict[tuple[str, str], JaggedRule] = {
    ("aten", "repeat_interleave"): _repeat_interleave,
    ("fbgemm", "asynchronous_complete_cumsum"): _cumsum,
    ("fbgemm", "asynchronous_exclusive_cumsum"): _cumsum,
    ("fbgemm", "dense_to_jagged"): _dense_to_jagged,
    ("fbgemm", "dense_to_jagged_forward"): _dense_to_jagged,
    ("fbgemm", "jagged_to_padded_dense"): _jagged_to_padded_dense,
    ("fbgemm", "jagged_to_padded_dense_backward"): _jagged_to_padded_dense_backward,
    ("fbgemm", "jagged_to_padded_dense_forward"): _jagged_to_padded_dense,
    ("fbgemm", "pack_segments"): _pack_segments,
}

_PASSTHROUGH = frozenset({
    "view", "reshape", "_to_copy", "clone", "alias", "contiguous", "squeeze",
    "unsqueeze", "convert_element_type", "flatten",
})
_COMPLETE_CUMSUM = ("fbgemm", "asynchronous_complete_cumsum")


def _op(node):
    target = getattr(node, "target", None)
    namespace = getattr(target, "namespace", None)
    packet = getattr(target, "overloadpacket", None)
    name = getattr(packet, "__name__", "").split(".")[-1]
    return (namespace, name) if namespace else None


def _source(node, kind):
    """(placeholder, kind) that ``node`` reads as ``kind``, following views,
    casts and complete cumsums (offsets = complete_cumsum(lengths)); None
    when it is computed in the graph."""
    while getattr(node, "op", None) == "call_function":
        op = _op(node)
        if op is None:
            return None
        if op[1] in _PASSTHROUGH and op[0] in ("aten", "prims"):
            node = node.args[0]
        elif op == _COMPLETE_CUMSUM and kind == "offsets":
            node, kind = node.args[0], "lengths"
        else:
            return None
    if getattr(node, "op", None) != "placeholder":
        return None
    return node, kind


def _hint(value):
    if isinstance(value, int):
        return value
    import torch

    val = value.meta.get("val") if hasattr(value, "meta") else value
    if isinstance(val, int):
        return val
    if isinstance(val, torch.SymInt):
        expr = val.node.expr
        return int(expr) if expr.is_number else val.node.hint
    return None


def _total_ref(total, placeholders):
    """JSON reference to ``total``: a symbolic input or a dimension of a
    tensor input when it is one, and always its captured value."""
    import torch

    hint = _hint(total)
    if hint is None:
        return None
    ref = {"value": int(hint)}
    val = total.meta.get("val") if hasattr(total, "meta") else total
    if isinstance(val, torch.SymInt) and not val.node.expr.is_number:
        expr = val.node.expr
        for node in placeholders:
            pval = node.meta.get("val")
            if isinstance(pval, torch.SymInt) and pval.node.expr == expr:
                return {"sym": node.name, **ref}
        for node in placeholders:
            pval = node.meta.get("val")
            if isinstance(pval, torch.Tensor):
                for d, size in enumerate(pval.shape):
                    if isinstance(size, torch.SymInt) and size.node.expr == expr:
                        return {"size": [node.name, d], **ref}
    return ref


@dataclass
class JaggedInputs:
    specs: dict[str, dict] = field(default_factory=dict)
    covered: set = field(default_factory=set)
    uncovered: set = field(default_factory=set)


def jagged_input_specs(gm, jagged_nodes, *, distribution=DEFAULT_DISTRIBUTION,
                       grouped_inputs=frozenset()) -> JaggedInputs:
    """Generator specs for the graph inputs that ``jagged_nodes`` read as
    lengths/offsets, and which of those nodes they cover. Inputs in
    ``grouped_inputs`` (views of one storage) are never constrained."""
    import torch

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    result = JaggedInputs()
    wanted: dict[Any, list] = {}
    for node in jagged_nodes:
        rule = JAGGED_RULES.get(_op(node))
        constraints = rule(node) if rule is not None else None
        if constraints is None:
            result.uncovered.add(node)
            continue
        resolved = []
        for operand, kind, total, max_len in constraints:
            source = _source(operand, kind)
            ref = _total_ref(total, placeholders) if total is not None else None
            if source is None or ref is None:
                resolved = None
                break
            resolved.append((source, ref, _hint(total), _hint(max_len) if max_len is not None else None))
        if resolved is None:
            result.uncovered.add(node)
            continue
        for (placeholder, kind), ref, hint, max_len in resolved:
            wanted.setdefault(placeholder, []).append((node, kind, ref, hint, max_len))
        result.covered.add(node)

    for placeholder, uses in wanted.items():
        val = placeholder.meta.get("val")
        kinds = {kind for _, kind, *_ in uses}
        hints = {hint for *_, hint, _ in uses}
        ok = (
            isinstance(val, torch.Tensor)
            and val.dim() == 1
            and not val.dtype.is_floating_point
            and val.dtype != torch.bool
            and tuple(val.stride()) == (1,)
            and placeholder.name not in grouped_inputs
            and len(kinds) == 1
            and len(hints) == 1
            and None not in hints
        )
        if not ok:
            for node, *_ in uses:
                result.covered.discard(node)
                result.uncovered.add(node)
            continue
        caps = [m for *_, m in uses if m is not None]
        result.specs[placeholder.name] = {
            "kind": LENGTHS_KIND if kinds == {"lengths"} else OFFSETS_KIND,
            "total": uses[0][2],
            "max_len": min(caps) if caps else None,
            "distribution": distribution,
        }
    # A node is covered only if every input it constrains was kept.
    for placeholder, uses in wanted.items():
        if placeholder.name not in result.specs:
            for node, *_ in uses:
                result.covered.discard(node)
                result.uncovered.add(node)
    result.covered -= result.uncovered
    return result

