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


def float64_casts_mode() -> TorchDispatchMode:
    """Dispatch mode that turns float-to-float dtype casts into float64 casts.

    Captured graphs cast explicitly (``convert_element_type(x, bfloat16)``), so
    upcasting only parameters and inputs would mix dtypes and fail.
    """
    convert = torch.ops.prims.convert_element_type.default
    to_copy = torch.ops.aten._to_copy.default

    class Float64Casts(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if (
                func is convert
                and args[0].is_floating_point()
                and args[1].is_floating_point
            ):
                args = (args[0], torch.float64)
            elif (
                func is to_copy
                and args[0].is_floating_point()
                and kwargs.get("dtype") is not None
                and kwargs["dtype"].is_floating_point
            ):
                kwargs["dtype"] = torch.float64
            return func(*args, **kwargs)

    return Float64Casts()


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


def scan_rng_dependence(fn: Callable[..., Any], inputs: Any) -> set[int] | None:
    """Flattened indices of the outputs of ``fn(*inputs)`` that depend on an RNG draw.

    Taint is tracked per storage during that one run, so views taken before an
    in-place RNG write and buffers written from RNG values count. Returns
    ``None`` if the run fails.
    """
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
    return {i for i, leaf in enumerate(tree_flatten(out)[0]) if is_tainted(leaf)}


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
