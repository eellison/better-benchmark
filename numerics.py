"""Numerics helpers for the oracle gate and the full-graph check.

The oracle gate compares float outputs against a float64 reference and
records one entry per output (:func:`anchored_gate`). The repro checker and
the full-graph bench gate label each output with :func:`judge_float_leaf`
against an upcast reference run (:func:`upcast_casts_mode`).
"""

from __future__ import annotations

import math
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
# Upcast reference
# ---------------------------------------------------------------------------


def _quantizes(dtype: Any) -> bool:
    """True for float dtypes narrower than 16 bits (float8), whose casts quantize."""
    return isinstance(dtype, torch.dtype) and dtype.is_floating_point and dtype.itemsize < 2


def _narrower(dtype: Any, target: torch.dtype) -> bool:
    return (
        isinstance(dtype, torch.dtype)
        and dtype.is_floating_point
        and dtype.itemsize < target.itemsize
    )


def upcast_casts_mode(target: torch.dtype) -> TorchDispatchMode:
    """Dispatch mode that turns float casts to dtypes narrower than ``target`` into ``target`` casts.

    Captured graphs cast explicitly (``convert_element_type(x, bfloat16)``), so
    upcasting only parameters and inputs would mix dtypes and fail. A cast to
    float8 keeps its quantization: the value goes through float8 and comes
    back as ``target``, so the reference does not skip the graph's own rounding.
    """
    convert = torch.ops.prims.convert_element_type.default
    to_copy = torch.ops.aten._to_copy.default

    class UpcastCasts(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if func is convert and _quantizes(args[1]):
                return func(*args, **kwargs).to(target)
            if func is to_copy and _quantizes(kwargs.get("dtype")):
                return func(*args, **kwargs).to(target)
            if (
                func is convert
                and args[0].is_floating_point()
                and _narrower(args[1], target)
            ):
                args = (args[0], target)
            elif (
                func is to_copy
                and args[0].is_floating_point()
                and _narrower(kwargs.get("dtype"), target)
            ):
                kwargs["dtype"] = target
            return func(*args, **kwargs)

    return UpcastCasts()


def upcast_factory_dtypes_mode(target: torch.dtype) -> TorchDispatchMode:
    """Dispatch mode that gives ops with a float ``dtype`` kwarg narrower than ``target`` that dtype.

    Factory ops (``aten.full(..., dtype=bf16)``) otherwise create buffers that
    meet ``target`` values later (``index_put``) and fail the reference run.
    Build inputs outside this mode: under it, ``randn(dtype=bf16)`` draws
    different values, not the same ones. Casts to float8 are left to
    :func:`upcast_casts_mode`, which keeps their quantization.
    """
    to_copy = torch.ops.aten._to_copy.default

    class UpcastFactoryDtypes(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            dtype = kwargs.get("dtype")
            if _narrower(dtype, target) and not (func is to_copy and _quantizes(dtype)):
                kwargs["dtype"] = target
            return func(*args, **kwargs)

    return UpcastFactoryDtypes()


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


# ---------------------------------------------------------------------------
# Compiled-vs-eager judgement (repro checker and bench gate)
# ---------------------------------------------------------------------------

# atol = rtol for float32 (and wider) outputs vs eager.
FP32_TOLERANCE = 1e-4
# pass_precision: no element's relative error may exceed ELEMENT_CAP times
# eager's largest relative error (plus rtol).
ELEMENT_CAP = 4.0


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return dtype.to_real() if dtype.is_complex else dtype


def is_low_precision(dtype: torch.dtype) -> bool:
    """bf16, fp16 and float8 (and their complex forms)."""
    return _real_dtype(dtype).itemsize <= 2


def precision_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """``(rtol, atol)`` for outputs of ``dtype``: a few ulps of the dtype.

    bf16 and fp16 use ``torch.testing``'s defaults; float8 gets two ulps and
    its smallest subnormal; wider dtypes get :data:`FP32_TOLERANCE`.
    """
    real = _real_dtype(dtype)
    if real == torch.bfloat16:
        return 1.6e-2, 1e-5
    if real == torch.float16:
        return 1e-3, 1e-5
    if real.itemsize == 1:
        finfo = torch.finfo(real)
        return 2 * finfo.eps, finfo.smallest_normal * finfo.eps
    return FP32_TOLERANCE, FP32_TOLERANCE


def reference_dtype(dtype: torch.dtype) -> torch.dtype:
    """Precision of the reference run for outputs of ``dtype``: float32 for
    low-precision outputs, float64 (tiebreak only) for the rest."""
    return torch.float32 if is_low_precision(dtype) else torch.float64


def real_float64(value: torch.Tensor) -> torch.Tensor:
    """``value`` as float64 on CPU; complex values become (real, imag) pairs."""
    value = value.detach()
    if value.is_complex():
        value = torch.view_as_real(value.resolve_conj())
    return value.to("cpu", torch.float64)


def _same_nonfinite(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise: equal, with NaN equal to NaN."""
    return (a == b) | (a.isnan() & b.isnan())


def _close(
    reference: torch.Tensor, compiled: torch.Tensor, dtype: torch.dtype
) -> dict[str, Any]:
    """Is ``compiled`` within ``precision_tolerance(dtype)`` of ``reference``
    at every element? Where ``reference`` is nonfinite in ``dtype``, compiled
    must equal it there; elsewhere compiled must be finite."""
    rtol, atol = precision_tolerance(dtype)
    ref, cand = real_float64(reference), real_float64(compiled)
    ref_out = ref.to(dtype).double()
    finite = ref_out.isfinite()
    if (finite & ~cand.isfinite()).any():
        return {"pass": False, "ratio": math.inf, "detail": "compiled is nonfinite where the reference is finite"}
    if not _same_nonfinite(ref_out[~finite], cand[~finite]).all():
        return {"pass": False, "ratio": math.inf, "detail": "nonfinite values differ from the reference"}
    err = (cand - ref)[finite].abs()
    bound = atol + rtol * ref[finite].abs()
    ratio = float((err / bound).max()) if err.numel() else 0.0
    return {"pass": ratio <= 1.0, "ratio": ratio}


def _no_worse_than_eager(
    reference: torch.Tensor,
    eager: torch.Tensor,
    compiled: torch.Tensor,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Is ``compiled`` no further from ``reference`` than eager is?

    Compiled's RMSE must be at most eager's times ``1 + 3 / sqrt(2 n)`` (three
    standard errors of an RMSE over ``n`` elements, so equally accurate
    outputs do not flip on sampling noise; at least 1%), and
    each element's relative error ``|c - ref| / (|ref| + eager_rmse + atol)``
    at most :data:`ELEMENT_CAP` times eager's largest relative error plus
    rtol: a few wrong small elements cannot hide under eager's error at large
    ones (eager's RMSE is the noise floor for elements near zero). Where eager is
    nonfinite and the reference finite, eager's error counts as 0. Where the
    reference is nonfinite in ``dtype``, compiled must equal eager or the
    reference; elsewhere compiled must be finite.
    """
    rtol, atol = precision_tolerance(dtype)
    ref, base, cand = (real_float64(t) for t in (reference, eager, compiled))
    ref_out = ref.to(dtype).double()
    finite = ref_out.isfinite()
    if (finite & ~cand.isfinite()).any():
        return {"pass": False, "detail": "compiled is nonfinite where the reference is finite"}
    nf = ~finite
    if not (_same_nonfinite(base[nf], cand[nf]) | _same_nonfinite(ref_out[nf], cand[nf])).all():
        return {"pass": False, "detail": "nonfinite values differ from eager and the reference"}
    ref, base, cand = ref[finite], base[finite], cand[finite]
    err_c = (cand - ref).abs()
    err_e = (base - ref).abs()
    err_e = err_e.masked_fill(~err_e.isfinite(), 0)
    if not err_c.numel():
        return {"pass": True, "rmse_compiled": 0.0, "rmse_eager": 0.0, "element_ratio": 0.0}
    rmse_c = float(err_c.square().mean().sqrt())
    rmse_e = float(err_e.square().mean().sqrt())
    scale = ref.abs() + rmse_e + atol
    cap = ELEMENT_CAP * float((err_e / scale).max()) + rtol
    element_ratio = float((err_c / scale).max()) / cap
    rmse_slack = 1 + max(0.01, 3 / math.sqrt(2 * err_c.numel()))
    rmse_ok = rmse_c <= rmse_slack * rmse_e + atol
    result = {
        "pass": rmse_ok and element_ratio <= 1.0,
        "rmse_compiled": rmse_c,
        "rmse_eager": rmse_e,
        "element_ratio": element_ratio,
    }
    if not result["pass"]:
        result["detail"] = (
            "compiled RMSE above eager's"
            if not rmse_ok
            else "an element's relative error exceeds eager's cap"
        )
    return result


def judge_float_leaf(
    eager: torch.Tensor,
    compiled: torch.Tensor,
    reference: torch.Tensor | Callable[[], torch.Tensor | None] | None,
) -> dict[str, Any]:
    """Label one float output ``pass``, ``pass_precision`` or ``fail``.

    ``pass``: compiled is within ``precision_tolerance`` of eager at every
    element, or (low-precision outputs) of the float32 reference.
    ``pass_precision``: it is not, but it is no less accurate than eager vs
    the reference (:func:`_no_worse_than_eager`): a different precision or
    reduction order, not a wrong answer. ``fail``: otherwise, including a
    nonfinite value where the reference is finite and differing NaNs.

    ``reference`` (or a callable returning it, called only when compiled
    misses eager) is eager run at ``reference_dtype``: float32 for
    low-precision outputs, float64 for the rest. Without one, an output that
    misses eager fails with ``needs_reference`` set. The record carries the
    evidence: ``reference`` (``"eager"``, ``"fp32"`` or ``"fp64"``),
    ``max_ratio`` (largest error / tolerance vs that reference), and for
    ``pass_precision`` or ``fail`` both RMSEs and ``element_ratio``.
    """
    dtype = _real_dtype(eager.dtype)
    rtol, atol = precision_tolerance(dtype)
    record: dict[str, Any] = {"rtol": rtol, "atol": atol}
    vs_eager = _close(eager, compiled, dtype)
    record["eager_ratio"] = vs_eager["ratio"]
    if vs_eager["pass"]:
        return {**record, "label": "pass", "reference": "eager", "max_ratio": vs_eager["ratio"]}
    ref = reference() if callable(reference) else reference
    if ref is None:
        return {
            **record,
            "label": "fail",
            "reference": "eager",
            "max_ratio": vs_eager["ratio"],
            "detail": vs_eager.get("detail", "outside tolerance of eager; no reference"),
            "needs_reference": True,
        }
    record["reference"] = "fp32" if ref.dtype in (torch.float32, torch.complex64) else "fp64"
    if ref.shape != eager.shape:
        return {**record, "label": "fail", "max_ratio": math.inf, "detail": "reference shape differs"}
    vs_ref = _close(ref, compiled, dtype)
    record["max_ratio"] = vs_ref["ratio"]
    if is_low_precision(dtype) and vs_ref["pass"]:
        return {**record, "label": "pass"}
    verdict = _no_worse_than_eager(ref, eager, compiled, dtype)
    record.update({k: v for k, v in verdict.items() if k != "pass"})
    record["label"] = "pass_precision" if verdict["pass"] else "fail"
    return record
