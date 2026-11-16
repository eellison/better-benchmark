"""Numerics helpers for the oracle gate and the full-graph check.

Both checks compare float outputs against a float64 reference and record one
entry per output in the same format (:func:`anchored_gate`). The float64 cast
mode and RNG detection are used by the full-graph check.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


def _storage_key(value: Any) -> int | None:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        return None
    return value.untyped_storage()._cdata


def written_args(func: Any, args: Any, kwargs: dict[str, Any]) -> list[Any]:
    """Flattened arguments that ``func``'s schema marks as written."""
    written = []
    for index, arg in enumerate(func._schema.arguments):
        if arg.alias_info is None or not arg.alias_info.is_write:
            continue
        if arg.kwarg_only or index >= len(args):
            value = kwargs.get(arg.name)
        else:
            value = args[index]
        written.extend(tree_flatten(value)[0])
    return written


# ---------------------------------------------------------------------------
# Float64 reference
# ---------------------------------------------------------------------------


def _attention_probs(query, key, is_causal, scale):
    """softmax(q k^T * scale) and its logsumexp, the math behind the flash kernels (causal: top-left aligned)."""
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    scores = query @ key.transpose(-2, -1) * scale
    if is_causal:
        keep = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~keep, -torch.inf)
    lse = scores.logsumexp(-1)
    return (scores - lse.unsqueeze(-1)).exp(), lse, scale


def _flash_attention_math(query, key, value, dropout_p=0.0, is_causal=False, return_debug_mask=False, *,
                          scale=None):
    """aten._scaled_dot_product_flash_attention for float32/float64 (the kernel only takes fp16/bf16). Dropout is
    not reproduced: such outputs depend on RNG and are skipped by RNG detection."""
    probs, lse, _ = _attention_probs(query, key, is_causal, scale)
    empty = query.new_empty(0)
    # the query's strides, as the kernel returns without head-dim padding (graphs view it by its strides)
    out = torch.empty_like(query).copy_(probs @ value)
    return (out, lse, empty, empty, query.shape[-2], key.shape[-2], empty, empty, empty)


def _flash_attention_backward_math(grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, max_q,
                                   max_k, dropout_p, is_causal, philox_seed, philox_offset, *, scale=None):
    probs, _, scale = _attention_probs(query, key, is_causal, scale)
    grad_probs = grad_out @ value.transpose(-2, -1)
    grad_scores = probs * (grad_probs - (grad_out * out).sum(-1, keepdim=True))
    grads = (grad_scores @ key * scale, grad_scores.transpose(-2, -1) @ query * scale,
             probs.transpose(-2, -1) @ grad_out)
    return tuple(torch.empty_like(x).copy_(g) for x, g in zip((query, key, value), grads))


_FLASH_ATTENTION_MATH = {
    torch.ops.aten._scaled_dot_product_flash_attention.default: _flash_attention_math,
    torch.ops.aten._scaled_dot_product_flash_attention_backward.default: _flash_attention_backward_math,
}


def narrower_float(dtype: Any, target: torch.dtype) -> bool:
    """Whether ``dtype`` is a float dtype narrower than ``target`` (what the reference upcasts)."""
    return isinstance(dtype, torch.dtype) and dtype.is_floating_point and dtype.itemsize < target.itemsize


def _flash_math_applies(func: Any, args: Any, kwargs: dict[str, Any]) -> bool:
    """The math matches the kernel: no dropout (the kernel's RNG is not reproduced; the backward is not tagged as
    an RNG op) and, if causal, square attention (the math is top-left aligned)."""
    names = [a.name for a in func._schema.arguments]
    bound = {**dict(zip(names, args)), **kwargs}
    query, key = bound["query"], bound["key"]
    causal_ok = not bound.get("is_causal", False) or query.shape[-2] == key.shape[-2]
    return not bound.get("dropout_p", 0.0) and causal_ok


def float64_casts_mode(target: torch.dtype = torch.float64) -> TorchDispatchMode:
    """Dispatch mode that turns float dtypes requested inside the graph into float64 (or ``target``: float32 for a
    reference when the float64 one does not fit in memory).

    Captured graphs cast explicitly (``convert_element_type(x, bfloat16)``) and
    create float tensors with explicit dtypes (``full``/``empty``/
    ``scalar_tensor(..., dtype=bfloat16)``, ``sum(..., dtype=float32)``), so
    upcasting only parameters and inputs would mix dtypes and fail (``mm`` of
    float64 and bfloat16) or round the reference back to low precision. Flash
    attention, which has no float32 or float64 kernel, runs as its math
    definition; custom ops without a float64 kernel that do not write their
    arguments run on float32 copies. A bitcast (``view(dtype)``) of a float
    tensor fails the reference run.
    """
    bitcast = torch.ops.aten.view.dtype

    class Float64Casts(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if func in _FLASH_ATTENTION_MATH and args[0].dtype == target and _flash_math_applies(func, args, kwargs):
                return _FLASH_ATTENTION_MATH[func](*args, **kwargs)
            if func is bitcast:
                # Reinterpreting bits has no upcast equivalent: an integer source keeps its bitcast (its float
                # result is upcast where it is used), an upcast float source cannot be reinterpreted faithfully.
                if args[0].is_floating_point():
                    raise RuntimeError(f"bitcast of an upcast {args[0].dtype} tensor to {args[1]}")
                return func(*args, **kwargs)
            # every float `dtype` argument narrower than target, by keyword or position (convert_element_type,
            # _assert_tensor_metadata); a float32 reference keeps float64 where the graph asks for it
            names = [a.name for a in func._schema.arguments]
            if "dtype" in names and names.index("dtype") < len(args):
                i = names.index("dtype")
                if narrower_float(args[i], target):
                    args = (*args[:i], target, *args[i + 1:])
            if narrower_float(kwargs.get("dtype"), target):
                kwargs["dtype"] = target
            mutates = any(a.alias_info is not None and a.alias_info.is_write for a in func._schema.arguments)
            if func.namespace in ("aten", "prims") or mutates:
                return func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            except torch.OutOfMemoryError:
                raise
            except (NotImplementedError, RuntimeError):
                # custom ops without float64 kernels (fbgemm jagged ops): float32 in, float64 out. Not for ops that
                # write their arguments, whose writes would land in the float32 copies.
                return _widen(func(*_narrow(args), **_narrow(kwargs)))

    return Float64Casts()


def _narrow(value: Any) -> Any:
    from torch.utils._pytree import tree_map

    return tree_map(lambda t: t.float() if isinstance(t, torch.Tensor) and t.dtype == torch.float64 else t, value)


def _widen(value: Any) -> Any:
    from torch.utils._pytree import tree_map

    return tree_map(lambda t: t.double() if isinstance(t, torch.Tensor) and t.dtype == torch.float32 else t, value)


# ---------------------------------------------------------------------------
# RNG detection
# ---------------------------------------------------------------------------

# RNG prims without the ``nondeterministic_seeded`` tag.
_RNG_OP_NAMES = frozenset(
    {"inductor_random", "inductor_randint", "inductor_seed", "inductor_seeds", "philox_rand"}
)
# Tagged ops (attention, RNNs, dropout) that only draw random numbers when a
# dropout probability is nonzero or training is on.
_RNG_PROBABILITY_ARGS = ("p", "dropout_p", "dropout")
_RNG_TRAINING_ARGS = ("train", "training")


def _rng_disabled_by_args(func: Any, args: Any, kwargs: dict[str, Any]) -> bool:
    for index, arg in enumerate(func._schema.arguments):
        if arg.name not in _RNG_PROBABILITY_ARGS + _RNG_TRAINING_ARGS:
            continue
        if not arg.kwarg_only and index < len(args):
            value = args[index]
        elif arg.name in kwargs:
            value = kwargs[arg.name]
        elif arg.has_default_value():
            value = arg.default_value
        else:
            continue
        if arg.name in _RNG_TRAINING_ARGS and value is False:
            return True
        if (
            arg.name in _RNG_PROBABILITY_ARGS
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 0
        ):
            return True
    return False


def is_rng_call(func: Any, args: Any = (), kwargs: dict[str, Any] | None = None) -> bool:
    """True if this dispatcher call draws from the RNG."""
    packet = getattr(func, "overloadpacket", None)
    name = getattr(packet if packet is not None else func, "__name__", "")
    tags = getattr(func, "tags", ())
    if name not in _RNG_OP_NAMES and torch.Tag.nondeterministic_seeded not in tags:
        return False
    if not hasattr(func, "_schema"):
        return True
    return not _rng_disabled_by_args(func, args, kwargs or {})


# Attention ops that always return their dropout RNG state (philox seed/offset), at these output positions, even
# with dropout off: the state differs between eager and compiled runs but carries no values.
_RNG_STATE_OUTPUTS = {
    "aten::_scaled_dot_product_flash_attention": (6, 7),
    "aten::_scaled_dot_product_cudnn_attention": (6, 7),
    "aten::_scaled_dot_product_efficient_attention": (2, 3),
    "aten::_efficient_attention_forward": (2, 3),
    "aten::_flash_attention_forward": (2, 3),
}
# Graph outputs allocated by these and not written by a dispatched op hold whatever memory they got: eager and
# compiled cannot be compared there.
_UNINITIALIZED_ALLOCATIONS = frozenset({
    "aten::empty", "aten::empty_like", "aten::empty_strided", "aten::empty_permuted", "aten::new_empty",
    "aten::new_empty_strided", "prims::empty_strided",
})


def scan_rng_dependence(
    fn: Callable[..., Any], inputs: Any, rng_state: set[int] | None = None
) -> set[int] | None:
    """Flattened indices of the outputs of ``fn(*inputs)`` that depend on an RNG draw.

    Taint is tracked per storage during that one run, so views taken before an
    in-place RNG write and buffers written from RNG values count. Returns
    ``None`` if the run fails. Outputs that are an attention op's RNG state
    (``_RNG_STATE_OUTPUTS``, also added to ``rng_state`` when given: they
    carry no values) or an allocation no dispatched op wrote
    (``_UNINITIALIZED_ALLOCATIONS``; a write the dispatcher does not see,
    e.g. a Triton kernel launched directly, cannot be told apart) are in the
    result too. Consuming them (attention backward) does not taint.
    """
    states: dict[int, Any] = {}
    # storage key -> storage of allocations not written yet (a partial write counts as initialized)
    uninitialized: dict[int, Any] = {}
    # Tainted storages stay referenced for the whole scan so their keys cannot
    # be reused by later allocations.
    tainted: dict[int, Any] = {}

    def is_tainted(value: Any) -> bool:
        if isinstance(value, torch.Tensor) and value.layout != torch.strided:
            return id(value) in tainted
        return _storage_key(value) in tainted

    def taint(value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            return
        key = _storage_key(value)
        if key is None:
            tainted[id(value)] = value
        else:
            tainted.setdefault(key, value.untyped_storage())

    class _RngTaintMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            out = func(*args, **kwargs)
            name = func._schema.name
            for index in _RNG_STATE_OUTPUTS.get(name, ()):
                if isinstance(out, tuple) and isinstance(out[index], torch.Tensor):
                    states[id(out[index])] = out[index]
            if name in _UNINITIALIZED_ALLOCATIONS and _storage_key(out) is not None:
                uninitialized[_storage_key(out)] = out.untyped_storage()
            elif uninitialized:
                for value in written_args(func, args, kwargs):
                    uninitialized.pop(_storage_key(value), None)
            if not is_rng_call(func, args, kwargs) and not any(
                is_tainted(value) for value in tree_flatten((args, kwargs))[0]
            ):
                return out
            for value in written_args(func, args, kwargs):
                taint(value)
            for value in tree_flatten(out)[0]:
                taint(value)
            return out

    try:
        with _RngTaintMode(), torch.no_grad():
            out = fn(*inputs)
    except Exception:
        return None
    leaves = tree_flatten(out)[0]
    state = {i for i, leaf in enumerate(leaves) if id(leaf) in states}
    if rng_state is not None:
        rng_state.update(state)
    unwritten = {i for i, leaf in enumerate(leaves) if _storage_key(leaf) in uninitialized and leaf.numel()}
    return {i for i, leaf in enumerate(leaves) if is_tainted(leaf)} | state | unwritten


# ---------------------------------------------------------------------------
# Anchored comparison
# ---------------------------------------------------------------------------

LeafCheck = Callable[[int, Any, Any, Any], "tuple[dict[str, Any], float | None]"]


def anchored_gate(
    candidate: list[Any],
    baseline: list[Any],
    reference: list[Any],
    skip: set[int] | frozenset[int],
    *,
    leaf_check: LeafCheck,
    ref_precision: str,
) -> dict[str, Any]:
    """Judge each candidate output against ``baseline``, anchored at ``reference``.

    ``leaf_check(idx, candidate, baseline, reference)`` returns the fields of
    the output's record and, for a failing output, how far over its bound it is
    (used to pick ``worst_output_idx``). A record holds ``idx`` and either
    ``skip`` (reason) or ``pass``. Outputs in ``skip`` are recorded as
    ``skip: "stochastic"`` without being checked.
    """
    result: dict[str, Any] = {
        "pass": True,
        "per_output": [],
        "worst_output_idx": None,
        "ref_precision": ref_precision,
    }
    worst_ratio = 0.0
    for i, (cand, base, ref) in enumerate(zip(candidate, baseline, reference)):
        entry: dict[str, Any] = {"idx": i}
        if i in skip:
            entry["skip"] = "stochastic"
            result["per_output"].append(entry)
            continue
        fields, ratio = leaf_check(i, cand, base, ref)
        entry.update(fields)
        if entry.get("pass") is False:
            result["pass"] = False
            if ratio is not None and ratio > worst_ratio:
                worst_ratio = ratio
                result["worst_output_idx"] = i
        result["per_output"].append(entry)
    return result
