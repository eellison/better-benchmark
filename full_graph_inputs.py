"""Generators for the integer inputs of a saved full graph that its capture could not constrain, inferred from how
the graph consumes them.

Captures without real input values (TORCH_TRACE ingests) mark integer inputs ``default_unobserved`` and
``make_tensor_from_spec`` fills them with uniform ids in ``[0, max(shape))``. Captures that did observe values keep
only min/max. Both are wrong for the jagged-tensor metadata of recommendation models: lengths must sum to the jagged
dimension, offsets must be a cumulative sum ending at it, and table ids must stay below the table size. fbgemm kernels,
``repeat_interleave(output_size=...)`` and user Triton kernels then read out of bounds (illegal memory access,
device-side asserts).

``refine_unobserved_inputs`` rebuilds the FX graph from the graph source (the forward's annotations give every value's
shape), infers from each such input's consumers what it is, and replaces its generator:

  * lengths summing to N (``{"kind": "lengths", "total": N, "max": M}``): the repeats of
    ``repeat_interleave(output_size=N)``; the lengths of ``pack_segments`` and ``pack_segments_backward``; lengths
    whose ``asynchronous_complete_cumsum`` (or ``cat([0], cumsum)``) feeds an offsets consumer below; lengths whose
    (exclusive) cumsum indexes rows. Slices of a larger lengths tensor (the features of a KJT) get their own totals
    (``"segments": [[start, stop, total], ...]``; elements outside every slice are 0).
  * offsets ending at N (``{"kind": "offsets", "total": N, "max": M}``): offsets of fbgemm jagged ops
    (``jagged_to_padded_dense*``, ``dense_to_jagged*``, other ``jagged_*`` ops taking ``(values, [offsets], ...)``),
    ``X_offsets``/``OffsetsX`` kwargs of user Triton kernels next to a tensor kwarg ``X`` (or ``OutX``), and offsets
    whose adjacent differences are read as lengths.
  * ids below a bound (``{"kind": "index"}``): ``infer_index_bounds_from_gm`` on the rebuilt graph, the key arguments
    of ``KEY_BOUND_OPS``, and user Triton kernels' ``indices*`` kwargs (below the first dim of the kwarg they are
    named after).
  * sorted values (``{"kind": "sorted"}``) for bucketize boundaries and searchsorted's sorted sequence, integer or
    float: a binary search over unsorted values returns implementation-defined buckets.

Segments are balanced (lengths differ by at most one), so a segment never exceeds ``max`` when ``N <= B * max``;
otherwise the total wins over ``max``.
"""

from __future__ import annotations

import ast
import operator
import re
import types
from typing import Any

_ANNOTATION = re.compile(r"^(\w+)\[([^\]]*)\](?:\[[^\]]*\])?")
_PASSTHROUGH = {
    "prims.convert_element_type.default", "aten._to_copy.default", "aten.clone.default", "aten.view.default",
    "aten.reshape.default", "aten.unsqueeze.default", "aten.squeeze.default", "aten.squeeze.dim",
    "aten.alias.default", "aten.detach.default", "aten.lift_fresh_copy.default", "aten.contiguous.default",
}
# The passthroughs that keep the shape, so a slice of their output is the same slice of their input.
_SAME_SHAPE = {
    "prims.convert_element_type.default", "aten._to_copy.default", "aten.clone.default", "aten.alias.default",
    "aten.detach.default", "aten.lift_fresh_copy.default", "aten.contiguous.default",
}
_CUMSUM_OFFSETS = {"fbgemm.asynchronous_complete_cumsum.default"}
# op name -> (position of an ids argument, position of the int bound they stay below); for custom ops of other
# libraries, which register them here
KEY_BOUND_OPS: dict[str, tuple[int, int]] = {}


class _Unresolved:
    """Target of a call whose op this process does not have (e.g. a custom op without its library)."""

    def __init__(self, name: str):
        self.__name__ = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"{self.__name__} is not available")

    def __str__(self):
        return self.__name__

    __repr__ = __str__


def _meta_value(annotation: str):
    """A meta tensor for an FX annotation like ``f32[16, 8][8, 1]cuda:0``, else None."""
    import torch

    from input_codec import _long_dtype

    m = _ANNOTATION.match(annotation)
    if not m:
        return None
    try:
        dtype = getattr(torch, _long_dtype(m.group(1)))
        shape = [int(d) for d in m.group(2).split(",") if d.strip()]
    except (AttributeError, ValueError):
        return None
    if not isinstance(dtype, torch.dtype):
        return None
    return torch.empty(shape, dtype=dtype, device="meta")


def _qualname(value: Any) -> str:
    return getattr(value, "__name__", None) or str(value)


def graph_from_source(content: str):
    """The FX graph a print_readable full graph module was printed from, with ``meta["val"]`` meta tensors from
    its annotations. Calls to ops this process lacks get an ``_Unresolved`` target named like the op."""
    import math

    import torch
    import torch.fx as fx

    tree = ast.parse(content)
    forward = next((f for c in tree.body if isinstance(c, ast.ClassDef) for f in c.body
                    if isinstance(f, ast.FunctionDef) and f.name == "forward"), None)
    if forward is None:
        raise ValueError("no forward() in the graph source")
    graph = fx.Graph()
    env: dict[str, Any] = {}
    names = {"torch": torch, "operator": operator, "inf": math.inf, "nan": math.nan, "device": torch.device}
    self_marker = object()
    op_types = (torch._ops.OpOverload, torch._ops.OpOverloadPacket, torch._ops.HigherOrderOperator)

    def annotate(node, annotation):
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            node.meta["val"] = _meta_value(annotation.value)

    def has_node(value):
        if isinstance(value, fx.Node):
            return True
        if isinstance(value, (list, tuple)):
            return any(has_node(v) for v in value)
        if isinstance(value, dict):
            return any(has_node(v) for v in value.values())
        return False

    def ev(e):
        if isinstance(e, ast.Name):
            if e.id in env:
                return env[e.id]
            if e.id == "self":
                return self_marker
            return names[e.id]
        if isinstance(e, ast.Constant):
            return e.value
        if isinstance(e, (ast.List, ast.Tuple)):
            items = [ev(x) for x in e.elts]
            return items if isinstance(e, ast.List) else tuple(items)
        if isinstance(e, ast.Dict):
            return {ev(k): ev(v) for k, v in zip(e.keys, e.values)}
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub):
            return -ev(e.operand)
        if isinstance(e, ast.Attribute):
            base = ev(e.value)
            if base is self_marker:
                return graph.get_attr(e.attr)
            if isinstance(base, _Unresolved):
                return _Unresolved(f"{base}.{e.attr}")
            try:
                return getattr(base, e.attr)
            except (AttributeError, RuntimeError):
                qualified = getattr(base, "_qualified_op_name", None)  # an OpOverloadPacket: "aten::add"
                prefix = qualified.replace("::", ".") if qualified else _qualname(base).removeprefix("torch.ops.")
                return _Unresolved(f"{prefix}.{e.attr}")
        if isinstance(e, ast.Subscript):
            base = ev(e.value)
            index = ev(e.slice)
            if isinstance(base, fx.Node):
                return graph.call_function(operator.getitem, (base, index))
            return base[index]
        if isinstance(e, ast.Call):
            fn = ev(e.func)
            args = tuple(ev(a) for a in e.args)
            kwargs = {k.arg: ev(k.value) for k in e.keywords}
            if has_node(args) or has_node(kwargs) or isinstance(fn, (_Unresolved, *op_types)):
                return graph.call_function(fn, args, kwargs)
            return fn(*args, **kwargs)
        raise ValueError(f"unsupported expression {ast.dump(e)[:200]}")

    for arg in forward.args.args[1:]:
        node = graph.placeholder(arg.arg)
        annotate(node, arg.annotation)
        env[arg.arg] = node
    for stmt in forward.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            value = ev(stmt.value)
            if isinstance(value, fx.Node):
                value.name = stmt.target.id
                annotate(value, stmt.annotation)
            env[stmt.target.id] = value
        elif isinstance(stmt, ast.Assign):
            if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                continue  # `x = y = None`: the printed graph freeing values
            value = ev(stmt.value)
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    if isinstance(value, fx.Node):
                        value.name = t.id
                    env[t.id] = value
        elif isinstance(stmt, ast.Return):
            graph.output(ev(stmt.value) if stmt.value is not None else None)
        elif isinstance(stmt, ast.Expr):
            ev(stmt.value)
    return graph


def _target(node) -> str:
    return str(node.target) if node.op == "call_function" else ""


def _shape(node) -> list[int] | None:
    import torch

    val = node.meta.get("val") if hasattr(node, "meta") else None
    return list(val.shape) if isinstance(val, torch.Tensor) else None


def _int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _source(node):
    """The placeholder ``node``'s values come from unchanged (through casts, views, clones), else None."""
    for _ in range(16):
        if not hasattr(node, "op"):
            return None
        if node.op == "placeholder":
            return node
        if _target(node) not in _PASSTHROUGH or not node.args:
            return None
        node = node.args[0]
    return None


def _offsets_of_lengths(node):
    """``o`` when ``node`` is ``o[1:] - o[:-1]`` (lengths from offsets), else None."""
    if _target(node) != "aten.sub.Tensor" or len(node.args) < 2:
        return None
    hi, lo = node.args[0], node.args[1]
    if _target(hi) != "aten.slice.Tensor" or _target(lo) != "aten.slice.Tensor":
        return None
    if hi.args[0] is not lo.args[0] or hi.args[1:3] != (0, 1) or lo.args[1:3] != (0, 0):
        return None
    # the whole tensor: o[1:] - o[:-1], not a part of it
    size = (_shape(hi.args[0]) or [None])[0]
    hi_end = hi.args[3] if len(hi.args) > 3 else None
    lo_end = lo.args[3] if len(lo.args) > 3 else None
    if size is None or not (hi_end is None or (_int(hi_end) is not None and hi_end >= size)) \
            or lo_end not in (-1, size - 1):
        return None
    return hi.args[0]


# A total is (value, max segment length or None, exact): exact totals are what a consumer reads all of (the rows of
# the jagged values, repeat_interleave's output_size); the others are upper bounds (row ids that must stay in range).


def _offsets_totals(o) -> list[tuple[int, int | None, bool]]:
    """Totals of every consumer of offsets value ``o`` that pins them."""
    found = []
    for user in o.users:
        name = _target(user)
        args, kwargs = user.args, user.kwargs
        if user.op != "call_function":
            continue
        if name.startswith("fbgemm.") and "jagged" in name and len(args) >= 2 and isinstance(args[1], (list, tuple)) \
                and len(args[1]) == 1 and args[1][0] is o:
            values, cap = args[0], None
            if name.startswith(("fbgemm.dense_to_jagged", "fbgemm.jagged_to_padded_dense_backward")):
                total = _int(args[2] if len(args) > 2 else kwargs.get("total_L"))
                if total is None and name.startswith("fbgemm.dense_to_jagged"):
                    # without total_L, the jagged output's rows
                    out = next((u for u in user.users if _target(u) == "<built-in function getitem>"
                                and u.args[1] == 0), user)
                    total = (_shape(out) or [None])[0]
                dense = _shape(values)  # (B, max_len, D)
                cap = dense[1] if dense and len(dense) > 1 else None
            else:
                total = (_shape(values) or [None])[0]
                max_lengths = args[2] if len(args) > 2 else kwargs.get("max_lengths")
                if name.startswith("fbgemm.jagged_to_padded_dense") and isinstance(max_lengths, (list, tuple)) \
                        and len(max_lengths) == 1:
                    cap = _int(max_lengths[0])
            if total is not None:
                found.append((total, cap, True))
        elif "triton_kernel_wrapper" in name:
            kw = kwargs.get("kwargs") or {}
            for key, value in kw.items():
                if value is not o or "offsets" not in key.lower():
                    continue
                # the jagged tensor they index: X_offsets -> X, OffsetsB -> B or OutB, x_offsets_ptr -> x_ptr; a bare
                # offsets_ptr names no tensor
                stem = re.sub("_?ptr$", "", key)
                base = re.sub("offsets", "", stem, flags=re.IGNORECASE).strip("_")
                if not base:
                    continue
                for cand in (base, f"{base}_ptr", f"Out{base}", f"out_{base}"):
                    shape = _shape(kw[cand]) if cand != key and hasattr(kw.get(cand), "meta") else None
                    if shape:
                        found.append((shape[0], None, False))
                        break
        elif name == "aten.slice.Tensor":
            for sub in user.users:
                if _offsets_of_lengths(sub) is o:
                    found += [(t, None, exact) for t, _, exact in _lengths_totals(sub)]
    return found


def _lengths_totals(n) -> list[tuple[int, int | None, bool]]:
    """Totals of every consumer of lengths value ``n`` that pins them."""
    found = []
    for user in n.users:
        name = _target(user)
        args, kwargs = user.args, user.kwargs
        if name in _PASSTHROUGH:
            found += _lengths_totals(user)
        elif name == "aten.repeat_interleave.Tensor" and args and args[0] is n:
            total = _int(kwargs.get("output_size"))
            if total is not None:
                found.append((total, None, True))
        elif name == "fbgemm.pack_segments.default" and len(args) >= 2 and args[1] is n:
            shape = _shape(args[0])
            cap = _int(args[2] if len(args) > 2 else kwargs.get("max_length"))
            if shape:
                found.append((shape[0], cap, True))
        elif name == "fbgemm.pack_segments_backward.default" and len(args) >= 3 and args[1] is n:
            total = _int(args[2])
            cap = _int(args[3]) if len(args) > 3 else None
            if total is not None:
                found.append((total, cap, True))
        elif name in _CUMSUM_OFFSETS and args and args[0] is n:
            found += _offsets_totals(user)
        elif name in ("fbgemm.asynchronous_exclusive_cumsum.default", "aten.cumsum.default") and args \
                and args[0] is n and (name != "aten.cumsum.default" or args[1] in (0, -1)):
            # Segment starts (exclusive cumsum, at most the total) or ends (cumsum, the last one the total) used as
            # row ids: the total is bounded by the largest id the indexed dim takes.
            found += [(bound, None, False) for bound in _index_value_bounds(user)]
            if name == "aten.cumsum.default":  # offsets as cat([zeros(1), cumsum(lengths)])
                for cat in user.users:
                    parts = cat.args[0] if _target(cat) == "aten.cat.default" else None
                    if parts and len(parts) == 2 and parts[1] is user and _target(parts[0]) == "aten.full.default" \
                            and parts[0].args[1] == 0:
                        found += _offsets_totals(cat)
    return found


def _slice_totals(n) -> list[list[int]]:
    """[start, stop, total] for each dim-0 slice of lengths value ``n`` (one feature of a KJT) whose lengths a
    consumer of the slice reads in full (exact totals that agree)."""
    found = []
    shape = _shape(n)
    for user in n.users:
        args = list(user.args) + [None, 0, None, None, 1][len(user.args):]  # slice(self, dim=0, start, end, step=1)
        if _target(user) in _SAME_SHAPE:
            found += _slice_totals(user)
        elif _target(user) == "aten.slice.Tensor" and args[0] is n and args[1] == 0 and args[4] == 1 and shape \
                and _int(args[2]) is not None and _int(args[3]) is not None:
            start, stop = (min(max(i + shape[0] if i < 0 else i, 0), shape[0]) for i in args[2:4])
            exact = {t for t, _, is_exact in _lengths_totals(user) if is_exact}
            if len(exact) == 1 and stop > start:
                found.append([start, stop, exact.pop()])
    return found


def _index_value_bounds(n, shift=0, hops=4):
    """Upper bounds on the values of ``n`` from the dims it indexes through ``aten.index.Tensor`` (size - 1), also
    after adding/subtracting constants (``shift`` is what was added on the way)."""
    bounds = []
    for user in n.users:
        name = _target(user)
        if name == "aten.index.Tensor" and isinstance(user.args[1], (list, tuple)):
            shape = _shape(user.args[0])
            dims = [d for d, i in enumerate(user.args[1]) if i is n]
            if shape and dims and dims[0] < len(shape) and shape[dims[0]] - 1 - shift >= 0:
                bounds.append(shape[dims[0]] - 1 - shift)
        elif hops and name in ("aten.sub.Tensor", "aten.add.Tensor") and user.args[0] is n \
                and _int(user.args[1]) is not None:
            step = user.args[1] if name == "aten.add.Tensor" else -user.args[1]
            bounds += _index_value_bounds(user, shift + step, hops - 1)
        elif hops and name in _PASSTHROUGH:
            bounds += _index_value_bounds(user, shift, hops - 1)
    return bounds


def _pick_total(totals) -> tuple[int, int | None] | None:
    """The total all consumers accept: the one exact total (consumers that read all of it must agree) within every
    upper bound, else the smallest upper bound. None when they conflict."""
    exact = {t for t, _, is_exact in totals if is_exact}
    bounds = [t for t, _, is_exact in totals if not is_exact]
    caps = [c for _, c, _ in totals if c]
    if len(exact) > 1 or (exact and bounds and min(exact) > min(bounds)):
        return None
    return (exact.pop() if exact else min(bounds)), (min(caps) if caps else None)


def _with_remainder(segments, whole, size):
    """Slices that do not overlap, plus, when the whole tensor has a total too, the rest of it spread over the
    largest gap between them. None when the slices overlap or cannot fit the whole tensor's total."""
    if any(a[1] > b[0] for a, b in zip(segments, segments[1:])):
        return None
    if whole is None:
        return segments
    remainder = whole[0] - sum(t for _, _, t in segments)
    gaps = [(b - a, a, b) for a, b in zip([0] + [s[1] for s in segments], [s[0] for s in segments] + [size]) if b > a]
    if remainder < 0 or (remainder and not gaps):
        return None
    if remainder:
        _, start, stop = max(gaps)
        segments = sorted([*segments, (start, stop, remainder)])
    return segments


def infer_jagged_inputs_from_gm(graph) -> dict[str, dict[str, Any]]:
    """{placeholder name: lengths/offsets generator} for the 1-D integer placeholders the graph uses as jagged
    metadata (see the module docstring). Inputs whose consumers disagree are left out."""
    import torch

    out: dict[str, dict[str, Any]] = {}
    for node in graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.dtype.is_floating_point or val.dtype == torch.bool \
                or val.dim() != 1:
            continue
        offsets = _offsets_totals(node)
        for user in node.users:
            if _target(user) in _PASSTHROUGH:
                offsets += _offsets_totals(user)
        lengths = _lengths_totals(node)
        segments = sorted({tuple(seg) for seg in _slice_totals(node)})
        if segments:
            whole = _pick_total(lengths) if lengths else None
            segments = None if lengths and whole is None else _with_remainder(segments, whole, val.shape[0])
            if segments is not None:
                out[node.name] = {"kind": "lengths", "total": sum(t for _, _, t in segments),
                                  "segments": [list(seg) for seg in segments]}
            continue
        for kind, totals in (("lengths", lengths), ("offsets", offsets)):
            if totals:
                picked = _pick_total(totals)
                if picked is not None:
                    total, cap = picked
                    out[node.name] = {"kind": kind, "total": int(total), **({"max": int(cap)} if cap else {})}
                break
    return out


_SORTED_ARG = {"aten.bucketize.Tensor": 1, "aten.searchsorted.Tensor": 0, "aten.searchsorted.Scalar": 0}


def _sorted_inputs(graph) -> set[str]:
    """Placeholders used (through casts and views) as bucketize boundaries or searchsorted's sorted sequence: binary
    searches over unsorted values return implementation-defined buckets, which differ between eager and compiled."""
    found = set()
    for node in graph.nodes:
        pos = _SORTED_ARG.get(_target(node))
        if pos is not None and len(node.args) > pos:
            src = _source(node.args[pos])
            if src is not None:
                found.add(src.name)
    return found


def _key_bounds(graph) -> dict[str, int]:
    bounds = {}
    for node in graph.nodes:
        positions = KEY_BOUND_OPS.get(_target(node))
        if positions and len(node.args) > max(positions):
            src = _source(node.args[positions[0]])
            bound = _int(node.args[positions[1]])
            if src is not None and bound:
                bounds[src.name] = min(bound, bounds.get(src.name, bound))
    return bounds


def _triton_index_bounds(graph) -> dict[str, int]:
    """User Triton kernels name row ids after the tensor they index: ``indices_group_ptrsK`` -> ``input_group_ptrsK``
    (fbgemm's group_index_select) or ``d_in_ptrsK`` (its backward, which scatter-adds into the table gradients),
    ``X_indices`` -> ``X``. The ids stay below that tensor's first dim."""
    bounds = {}
    for node in graph.nodes:
        if "triton_kernel_wrapper" not in _target(node):
            continue
        kw = node.kwargs.get("kwargs") or {}
        for key, value in kw.items():
            if "indices" not in key or not hasattr(value, "op"):
                continue
            src = _source(value)
            cands = (key.replace("indices", "input", 1), key.replace("indices_group_ptrs", "d_in_ptrs"),
                     key.replace("_indices", ""), key.replace("indices_", ""))
            for cand in cands:
                paired = kw.get(cand)
                shape = _shape(paired) if cand != key and hasattr(paired, "op") else None
                if src is not None and shape:
                    bounds[src.name] = min(shape[0], bounds.get(src.name, shape[0]))
                    break
    return bounds


def _is_integer(spec: dict) -> bool:
    """An integer tensor input whose values are not already pinned (exact data, or lengths/offsets generators)."""
    gen = spec.get("gen") or spec.get("generator") or {}
    return spec.get("kind") == "tensor" and str(spec.get("dtype", "")).startswith(("int", "uint")) \
        and not spec.get("exact") and not is_jagged(gen)


def _is_unobserved_integer(spec: dict) -> bool:
    return _is_integer(spec) and spec.get("constraint_source") in (None, "default_unobserved", "annotation_default")


def _is_unconstrained_float(spec: dict) -> bool:
    gen = spec.get("gen") or spec.get("generator") or {"kind": "randn"}
    return spec.get("kind") == "tensor" and "float" in str(spec.get("dtype", "")) and not spec.get("exact") \
        and gen.get("kind") in (None, "randn")


def refine_unobserved_inputs(content: str, input_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``input_specs`` with inferred generators (``constraint_source: graph_inference``) for the integer inputs the
    capture left unconstrained, for observed ones the graph uses as jagged lengths/offsets, and for unconstrained
    inputs used as sorted sequences. Inputs the capture constrained, and inputs that alias others (their generators
    must agree), are returned unchanged; so is everything when the source or its analysis is not covered."""
    searches = "bucketize" in content or "searchsorted" in content  # floats only change as sorted sequences
    if not any(_is_unobserved_integer(s) or (searches and _is_unconstrained_float(s))
               or (_is_integer(s) and s.get("constraint_source") == "observed") for s in input_specs):
        return input_specs
    from full_graph_harness import infer_index_bounds_from_gm, placeholder_info_from_gm

    try:  # best effort: a graph source this parser or analysis does not cover keeps its specs
        graph = graph_from_source(content)
        gens = infer_jagged_inputs_from_gm(graph)
        bounds = infer_index_bounds_from_gm(types.SimpleNamespace(graph=graph),
                                            placeholder_info_from_gm(types.SimpleNamespace(graph=graph)))
        for found in (_key_bounds(graph), _triton_index_bounds(graph)):
            for name, bound in found.items():
                bounds[name] = min(bound, bounds.get(name, bound))
        sorted_inputs = _sorted_inputs(graph)
    except Exception:
        return input_specs
    out = []
    for spec in input_specs:
        name = spec.get("name")
        if spec.get("alias_group") is not None:
            out.append(spec)
            continue
        # An observed min/max bound loses the structure of jagged metadata (sorted offsets, lengths' sum).
        gen = gens.get(name) if _is_integer(spec) and spec.get("constraint_source") != "graph_inference" else None
        if gen is not None and spec.get("stride") not in (None, [1], (1,)):
            gen = None  # jagged generators make contiguous tensors
        # Offsets are sorted already; other sorted sequences (searchsorted over offsets is common) come after them.
        if gen is None and name in sorted_inputs and (_is_unobserved_integer(spec) or _is_unconstrained_float(spec)):
            gen = {"kind": "sorted"}
            if _is_integer(spec):
                import torch

                from full_graph_harness import _torch_dtype

                high = min(max(_numel(spec.get("shape") or []), 1), torch.iinfo(_torch_dtype(spec["dtype"])).max)
                gen.update(low=0, high=min(high, max(int(bounds.get(name, high)), 1)))
        if gen is None and _is_unobserved_integer(spec) and name in bounds:
            gen = {"kind": "index", "low": 0, "high": max(int(bounds[name]), 1)}
        if gen is not None:
            spec = {**spec, "gen": gen, "generator": gen, "constraint_source": "graph_inference"}
        out.append(spec)
    return out


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def balanced_lengths(n: int, total: int) -> list[int]:
    """``n`` non-negative lengths summing to ``total`` that differ by at most one.

    They stay within a generator's ``max`` whenever ``total <= n * max``; when they cannot, the total wins (the ops
    that take a max length truncate segments, while a short total leaves rows the graph reads unset)."""
    if n <= 0:
        return []
    base, extra = divmod(max(int(total), 0), n)
    return [base + (1 if i < extra else 0) for i in range(n)]


def make_jagged_tensor(spec: dict[str, Any], gen: dict[str, Any], default_device: str = "cuda"):
    """A tensor for a ``lengths``/``offsets`` generator, in the spec's dtype, shape and strides."""
    import torch

    from full_graph_harness import _dense_storage_size, _resolve_device, _torch_dtype, _view_storage

    dtype = _torch_dtype(spec.get("dtype", "int64"))
    shape = tuple(spec.get("shape", ()))
    device = _resolve_device(spec.get("device"), default_device)
    numel = _numel(shape)
    is_offsets = gen["kind"] == "offsets"
    count = numel - 1 if is_offsets else numel
    if gen.get("segments"):  # slices with their own totals, the rest stays empty
        lengths = [0] * count
        for start, stop, total in gen["segments"]:
            lengths[start:stop] = balanced_lengths(stop - start, total)
    else:
        lengths = balanced_lengths(count, gen["total"])
    values = torch.tensor(lengths, dtype=torch.int64, device=device)
    if is_offsets:
        values = torch.cat([values.new_zeros(1), values.cumsum(0)])[:numel]
    values = values.to(dtype).reshape(shape)
    stride, offset = spec.get("stride"), int(spec.get("storage_offset", 0) or 0)
    if stride is None and offset == 0:
        return values
    storage = torch.zeros((_dense_storage_size(shape, stride, storage_offset=offset),), dtype=dtype, device=device)
    return _view_storage(storage, shape, stride, offset).copy_(values)


def is_jagged(gen: dict[str, Any]) -> bool:
    """A generator of this module (the repro capture's own ``offsets`` generator has a ``high`` instead)."""
    return gen.get("kind") == "lengths" or (gen.get("kind") == "offsets" and "total" in gen)
