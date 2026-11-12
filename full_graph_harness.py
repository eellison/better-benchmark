"""Helpers for executing saved full_graph_*.py artifacts.

The saved full graphs are print_readable-style Python modules.  They contain a
torch.nn.Module subclass with tensor metadata in forward annotations, but no
canonical make_inputs() helper.  This module reconstructs runnable inputs from
those annotations so the benchmark runner can time the whole graph directly.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from jagged_inputs import JAGGED_KINDS, make_jagged_tensor, resolve_total


FULL_GRAPH_NUMERICS_SEED = 0
NUMERICS_METHOD = "numerics.judge_float_leaf"
# Per output dtype (numerics.precision_tolerance), as in the repro checker.
NUMERICS_TOLERANCE = "numerics.precision_tolerance"
_MAX_NUMERICS_DETAILS = 8


class UserTritonKernelReplayError(RuntimeError):
    """A saved full graph calls user-defined Triton kernels it cannot resolve."""


def full_graph_calls_user_triton(definition: "FullGraphDefinition") -> bool:
    from replay_dependencies import full_graph_calls_user_triton as _calls

    return _calls(
        definition.graph_cls, (definition.metadata or {}).get("sidecar") or {}
    )


class FullGraphNumericsError(RuntimeError):
    """Raised when compiled outputs do not match eager outputs."""

    def __init__(
        self,
        message: str,
        *,
        classification: str | None = None,
        seed_record: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.classification = classification
        self.seed_record = seed_record
        self.numerics: dict[str, Any] | None = None
        self.failed_variant: str | None = None


def _upcast_inputs(inputs: Any, target: Any) -> Any:
    """Copies of the float tensors in ``inputs`` narrower than ``target``, as
    ``target``; repeated tensors stay shared."""
    import torch
    from torch.utils._pytree import tree_map

    copies: dict[int, Any] = {}

    def upcast(value: Any) -> Any:
        if not (
            isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and value.itemsize < target.itemsize
        ):
            return value
        if id(value) not in copies:
            copies[id(value)] = value.to(target, copy=True)
        return copies[id(value)]

    return tree_map(upcast, inputs)


def _run_seeded(fn: Any, make_inputs: Any, seed: int) -> Any:
    """Run ``fn`` on fresh inputs from ``make_inputs`` and return a copy of its outputs.

    Inputs and the run are both seeded with ``seed``. Outputs are cloned so a
    later run cannot overwrite them through state they alias.
    """
    import torch
    from torch.utils._pytree import tree_map

    torch.manual_seed(seed)
    inputs = make_inputs()
    torch.manual_seed(seed)
    with torch.no_grad():
        out = fn(*inputs)
    return tree_map(
        lambda v: v.clone() if isinstance(v, torch.Tensor) else v, out
    )


def rng_dependent_output_leaves(
    fn: Any, make_inputs: Any, seed: int = FULL_GRAPH_NUMERICS_SEED
) -> set[int] | None:
    """Return flattened output leaf indices that depend on an RNG op.

    Eager and compiled code use different RNG implementations even for the
    same seed, so these leaves cannot be compared. Dependence is tracked
    dynamically during one eager run because FX tracing constant-folds RNG
    ops whose arguments are not graph inputs. Returns ``None`` when the
    detection run fails.
    """
    import torch

    from numerics import scan_rng_dependence

    torch.manual_seed(seed)
    inputs = make_inputs()
    torch.manual_seed(seed)
    return scan_rng_dependence(fn, inputs)


def upcast_reference(
    fn: Any,
    make_inputs: Any,
    seed: int,
    target: Any,
    errors: list[str] | None = None,
) -> Any | None:
    """Run a copy of ``fn`` with float inputs, parameters, casts and factory
    dtypes narrower than ``target`` upcast to ``target``.

    Casts to float8 keep their quantization. Returns ``None`` when the run
    fails (unsupported dtype, OOM, ...), appending the error to ``errors``.
    """
    import copy

    import torch

    from numerics import upcast_casts_mode, upcast_factory_dtypes_mode

    def upcast(t: Any) -> Any:
        if t.is_floating_point() and t.itemsize < target.itemsize:
            return t.to(target)
        return t

    try:
        fn_up = copy.deepcopy(fn)
        if isinstance(fn_up, torch.nn.Module):
            fn_up = fn_up._apply(upcast)
        torch.manual_seed(seed)
        inputs = _upcast_inputs(make_inputs(), target)
        with upcast_casts_mode(target), upcast_factory_dtypes_mode(target):
            return _run_seeded(fn_up, lambda: inputs, seed)
    except Exception as error:  # noqa: BLE001 - no reference; callers fall back
        if errors is not None:
            errors.append(f"{type(error).__name__}: {error}"[:300])
        return None


def _reference_lookup(reference: Any, baseline_spec: Any):
    """``lookup(index, dtype)``: the reference leaf for output ``index`` of
    ``dtype``, from ``reference`` (a tree, or a callable taking the reference
    dtype, called at most once per dtype), or ``None``."""
    import torch
    from torch.utils._pytree import tree_flatten

    from numerics import reference_dtype

    cache: dict[Any, list[Any] | None] = {}

    def leaves(target: Any) -> list[Any] | None:
        if target not in cache:
            tree = reference(target) if callable(reference) else reference
            flat, spec = tree_flatten(tree) if tree is not None else (None, None)
            cache[target] = flat if spec == baseline_spec else None
        return cache[target]

    def lookup(index: int, dtype: Any) -> Any | None:
        if reference is None:
            return None
        target = reference_dtype(dtype)
        flat = leaves(target)
        if flat is None and target == torch.float64 and callable(reference):
            # No float64 run (e.g. user Triton kernels have no float64 MMA):
            # the float32 run still removes the graph's low-precision casts.
            flat = leaves(torch.float32)
        ref = flat[index] if flat is not None else None
        return ref if isinstance(ref, torch.Tensor) else None

    return lookup


def _full_graph_leaf_check(skip_leaves: set[int] | frozenset[int], lookup: Any):
    """Per-leaf rule for :func:`numerics.anchored_gate` (candidate = compiled)."""
    import torch

    from numerics import judge_float_leaf

    def fail(detail: str, fields: dict[str, Any] | None = None, ratio=math.inf):
        return {**(fields or {}), "pass": False, "detail": detail}, ratio

    def check(index: int, got: Any, expected: Any, _ref: Any):
        if isinstance(expected, torch.Tensor) != isinstance(got, torch.Tensor):
            return fail("tensor/non-tensor mismatch")
        if not isinstance(expected, torch.Tensor):
            both_nan = (
                isinstance(expected, float)
                and isinstance(got, float)
                and math.isnan(expected)
                and math.isnan(got)
            )
            if expected != got and not both_nan:
                return fail(f"{got!r} != {expected!r}")
            return {"pass": True}, None
        if expected.shape != got.shape:
            return fail(f"shape {list(got.shape)} != {list(expected.shape)}")
        if expected.dtype != got.dtype:
            return fail(f"dtype {got.dtype} != {expected.dtype}")
        if index in skip_leaves:
            return {"skip": "stochastic"}, None
        if not (expected.is_floating_point() or expected.is_complex()):
            if not torch.equal(expected, got):
                return fail(f"{expected.dtype} values differ")
            return {"pass": True}, None
        def reference() -> Any | None:
            ref = lookup(index, expected.dtype)
            # A same-precision fallback run identical to eager (nothing was
            # upcast) carries no information: judge against eager alone.
            if ref is not None and ref.dtype == expected.dtype and torch.equal(ref, expected):
                return None
            return ref

        record = judge_float_leaf(expected, got, reference)
        if record["label"] != "fail":
            return {**record, "pass": True}, None
        return fail(
            f"{record.get('detail', 'outside tolerance')} vs {record['reference']} "
            f"(max_ratio={record['max_ratio']:.3g}) shape={list(expected.shape)} "
            f"dtype={expected.dtype}",
            record,
            record["max_ratio"],
        )

    return check


def _reference_kind(per_output: list[dict[str, Any]]) -> str:
    """``"eager"`` when an output that missed eager had no reference to judge
    it against, ``"none"`` when no output needed one, else the reference runs
    used (``"fp32"``, ``"fp64"`` or ``"fp32+fp64"``)."""
    if any(e.get("needs_reference") for e in per_output):
        return "eager"
    kinds = sorted({e["reference"] for e in per_output if e.get("reference") not in (None, "eager")})
    return "+".join(kinds) or "none"


def compare_output_records(
    reference: Any | None,
    baseline: Any,
    candidate: Any,
    *,
    skip_leaves: set[int] | frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Per-leaf records for :func:`compare_outputs`, in the oracle gate's format.

    Returns :func:`numerics.anchored_gate`'s dict (``pass``, ``per_output``,
    ``worst_output_idx``, ``ref_precision``) plus ``reference``
    (:func:`_reference_kind`). Float records carry
    :func:`numerics.judge_float_leaf`'s fields (``label``, ``reference``,
    ``max_ratio``, RMSEs); failing records also a ``detail`` message. A
    structure mismatch has no per-leaf records and a top-level ``detail``.
    """
    from torch.utils._pytree import tree_flatten

    from numerics import anchored_gate

    baseline_leaves, baseline_spec = tree_flatten(baseline)
    candidate_leaves, candidate_spec = tree_flatten(candidate)
    if candidate_spec != baseline_spec:
        return {
            "pass": False,
            "per_output": [],
            "worst_output_idx": None,
            "ref_precision": "none",
            "reference": "none",
            "detail": f"output structure differs: {str(candidate_spec)[:500]}",
        }
    records = anchored_gate(
        candidate_leaves,
        baseline_leaves,
        [None] * len(baseline_leaves),
        frozenset(),
        leaf_check=_full_graph_leaf_check(
            skip_leaves, _reference_lookup(reference, baseline_spec)
        ),
        ref_precision="eager",
    )
    records["reference"] = records["ref_precision"] = _reference_kind(
        records["per_output"]
    )
    return records


def _details(records: dict[str, Any]) -> list[str]:
    details = [records["detail"]] if "detail" in records else []
    details += [
        f"leaf {entry['idx']}: {entry['detail']}"
        for entry in records["per_output"]
        if entry.get("pass") is False
    ]
    if len(details) > _MAX_NUMERICS_DETAILS:
        extra = len(details) - _MAX_NUMERICS_DETAILS
        details = details[:_MAX_NUMERICS_DETAILS] + [f"... {extra} more"]
    return details


def _failing_records(records: dict[str, Any]) -> list[dict[str, Any]]:
    failing = [e for e in records["per_output"] if e.get("pass") is False]
    return failing[:_MAX_NUMERICS_DETAILS]


def compare_outputs(
    reference: Any | None,
    baseline: Any,
    candidate: Any,
    *,
    skip_leaves: set[int] | frozenset[int] = frozenset(),
) -> tuple[bool, list[str]]:
    """Compare nested outputs with the repro checker's rule.

    ``baseline`` is the eager run and ``candidate`` the compiled run. Float
    leaves pass unless :func:`numerics.judge_float_leaf` labels them
    ``fail`` (``pass_precision`` passes). ``reference`` is the upcast eager
    run: a tree, ``None`` (eager only), or a callable taking the reference
    dtype (``numerics.reference_dtype``) and returning that run or ``None``.
    Non-float leaves must equal eager exactly. Structure, shape and dtype must
    match eager. Leaf indices in ``skip_leaves`` are only checked for shape
    and dtype.

    Returns ``(matches, details)`` where ``details`` is bounded to a few
    messages so failure records stay small for large graphs.
    """
    records = compare_output_records(
        reference, baseline, candidate, skip_leaves=skip_leaves
    )
    return records["pass"], _details(records)


def validate_full_graph_numerics(
    eager_fn: Any,
    compiled_fn: Any,
    make_inputs: Any,
    *,
    label: str = "compiled",
    seed: int = FULL_GRAPH_NUMERICS_SEED,
    skip_output_leaves: set[int] | frozenset[int] = frozenset(),
    info: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Compare the outputs of one seeded eager/compiled sample.

    Every run (eager, compiled, upcast references, eager rerun) gets fresh
    inputs from ``make_inputs`` under ``seed``. Outputs are judged with
    :func:`compare_outputs`; the float32 / float64 reference runs happen only
    when an output misses eager. On mismatch, eager is rerun to distinguish
    eager nondeterminism from a stable compiled mismatch. ``skip_output_leaves``
    indexes the flattened outputs, e.g. from RNG detection.

    Returns ``(eager_output, reference)`` where ``reference`` is
    :func:`_reference_kind`'s. ``info`` gets ``label``, the worst
    float-output label of a passing sample (``pass`` or ``pass_precision``),
    and ``ref_num_stages`` = 1 when a replayed user-Triton kernel ran the
    upcast reference with one pipeline stage (its captured configs did not
    fit in shared memory at the reference precision).
    """
    import functools

    eager = _run_seeded(eager_fn, make_inputs, seed)
    skip = frozenset(skip_output_leaves)

    from user_triton import SINGLE_STAGE_RETRIES

    @functools.cache
    def reference(target: Any) -> Any | None:
        retries = len(SINGLE_STAGE_RETRIES)
        ref = upcast_reference(eager_fn, make_inputs, seed, target)
        if info is not None and len(SINGLE_STAGE_RETRIES) > retries:
            info["ref_num_stages"] = 1
        return ref

    compiled = _run_seeded(compiled_fn, make_inputs, seed)
    records = compare_output_records(reference, eager, compiled, skip_leaves=skip)
    reference_kind = records["reference"]
    if info is not None and records["pass"]:
        info["label"] = (
            "pass_precision"
            if any(e.get("label") == "pass_precision" for e in records["per_output"])
            else "pass"
        )
    if not records["pass"]:
        eager_repeat = _run_seeded(eager_fn, make_inputs, seed)
        eager_stable, _ = compare_outputs(
            reference, eager, eager_repeat, skip_leaves=skip
        )
        classification = (
            "stable_eager_vs_compiled_mismatch"
            if eager_stable
            else "eager_self_instability"
        )
        details = _details(records)
        raise FullGraphNumericsError(
            f"{label} output mismatch (seed={seed}, reference "
            f"{reference_kind}, {classification}): {'; '.join(details)}",
            classification=classification,
            seed_record={
                "seed": seed,
                "pass": False,
                "reference": reference_kind,
                "failure_classification": classification,
                "eager_self_check": {"pass": eager_stable},
                "diagnostics": details,
                "worst_output_idx": records["worst_output_idx"],
                "per_output": _failing_records(records),
            },
        )
    return eager, reference_kind


def run_numerics_gate(
    eager_fn: Any,
    compiled_fn: Any,
    make_inputs: Any,
    *,
    label: str,
    variant: str,
    seeds: int,
    non_semantic_output_leaves: set[int] | frozenset[int] = frozenset(),
) -> tuple[dict[str, Any], Any]:
    """Validate ``compiled_fn`` against eager (:func:`compare_outputs`) for seeds ``0..seeds-1``.

    ``make_inputs`` is called, after seeding the global RNG, for every run.
    Every seed runs even after a failure so the summary always holds one
    record per seed. Returns ``(summary, first_eager_output)``; on failure
    raises the first ``FullGraphNumericsError`` with ``numerics`` and
    ``failed_variant`` attached.

    ``reference`` in the summary records which reference runs were used
    (:func:`_reference_kind`; ``eager`` when an output needed the upcast
    reference and that run failed, so it was judged against eager alone) and
    ``label`` the worst of ``pass`` / ``pass_precision`` / ``fail``. Graphs
    with user-defined Triton kernels get the same rule: the kernels run in
    the float32 reference too. Status is ``unanchored_user_kernel`` rather
    than ``fail`` only when such a graph has no usable reference at all: no
    upcast run, and eager does not reproduce itself.

    Status is ``numerics_unavailable`` rather than ``pass`` when RNG-dependent
    output leaves are not listed in ``non_semantic_output_leaves`` (an
    explicit output contract) or RNG detection failed.
    """
    records: list[dict[str, Any]] = []
    first_error: FullGraphNumericsError | None = None
    first_eager_out = None
    rng_leaves = rng_dependent_output_leaves(eager_fn, make_inputs)
    reference_kinds: set[str] = set()
    labels: list[str] = []
    ref_num_stages = None
    for seed in range(seeds):
        info: dict[str, Any] = {}
        try:
            eager_out, reference_kind = validate_full_graph_numerics(
                eager_fn,
                compiled_fn,
                make_inputs,
                label=label,
                seed=seed,
                skip_output_leaves=rng_leaves or frozenset(),
                info=info,
            )
        except FullGraphNumericsError as exc:
            ref_num_stages = ref_num_stages or info.get("ref_num_stages")
            records.append(exc.seed_record)
            reference_kinds.add(exc.seed_record["reference"])
            first_error = first_error or exc
            continue
        ref_num_stages = ref_num_stages or info.get("ref_num_stages")
        labels.append(info["label"])
        reference_kinds.add(reference_kind)
        if first_eager_out is None:
            first_eager_out = eager_out
        del eager_out
        records.append(
            {
                "seed": seed,
                "pass": True,
                "reference": reference_kind,
                "failure_classification": None,
                "eager_self_check": None,
                "diagnostics": [],
                "worst_output_idx": None,
                "per_output": [],
            }
        )
    skipped = sorted(rng_leaves or ())
    unjustified = [leaf for leaf in skipped if leaf not in non_semantic_output_leaves]
    from user_triton import launches_user_triton

    unanchored = (
        first_error is not None
        and launches_user_triton(eager_fn)
        and all(
            r["reference"] == "eager" and not r["eager_self_check"]["pass"]
            for r in records
            if not r["pass"]
        )
    )
    if unanchored:
        status = "unanchored_user_kernel"
        if first_eager_out is None:
            first_eager_out = _run_seeded(eager_fn, make_inputs, 0)
    elif first_error:
        status = "fail"
    elif unjustified or rng_leaves is None:
        status = "numerics_unavailable"
    else:
        status = "pass"
    summary = {
        "status": status,
        "seeds": records,
        "method": NUMERICS_METHOD,
        "tolerance": NUMERICS_TOLERANCE,
        "reference": (
            next(iter(reference_kinds)) if len(reference_kinds) == 1 else "mixed"
        ),
        "reference_available": "eager" not in reference_kinds,
        "label": (
            "fail"
            if first_error is not None
            else "pass_precision" if "pass_precision" in labels else "pass"
        ),
        **({"ref_num_stages": ref_num_stages} if ref_num_stages else {}),
        "rng_detection_ok": rng_leaves is not None,
        "skipped_rng_output_leaves": skipped,
        "unjustified_skipped_output_leaves": unjustified,
    }
    if unanchored:
        summary["label"] = None
        return summary, first_eager_out
    if first_error is not None:
        first_error.numerics = summary
        first_error.failed_variant = variant
        raise first_error
    return summary, first_eager_out


@dataclass(frozen=True)
class FullGraphDefinition:
    path: Path
    graph_cls: type
    input_specs: list[dict[str, Any]]
    tensor_attrs: dict[str, dict[str, Any]]
    forward_takes_no_inputs: bool
    metadata: dict[str, Any]


_DTYPE_SHORT_NAMES = {
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "i64": "int64",
    "i32": "int32",
    "i16": "int16",
    "i8": "int8",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "b8": "bool",
    "f64": "float64",
    "c64": "complex64",
    "c128": "complex128",
    "f8e4m3fn": "float8_e4m3fn",
    "f8e4m3fnuz": "float8_e4m3fnuz",
    "f8e4m3b11fnuz": "float8_e4m3b11fnuz",
    "f8e5m2": "float8_e5m2",
    "f8e5m2fnuz": "float8_e5m2fnuz",
    "f8e8m0fnu": "float8_e8m0fnu",
}


def _torch_dtype(dtype_name: str):
    import torch

    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {dtype_name}")
    return dtype


def _dtype_name(dtype: Any) -> str:
    text = str(dtype)
    return text.removeprefix("torch.")


def _default_generator_for_dtype(
    dtype_name: str,
    shape: tuple[int, ...] | list[int] = (),
) -> dict[str, Any]:
    if dtype_name == "bool":
        return {"kind": "randint", "low": 0, "high": 2}
    if dtype_name.startswith("int") or dtype_name.startswith("uint"):
        high = max([int(dim) for dim in shape], default=100)
        return {"kind": "index", "low": 0, "high": max(high, 1)}
    return {"kind": "randn"}


def _is_integer_or_bool_dtype_name(dtype_name: str) -> bool:
    return (
        dtype_name == "bool"
        or dtype_name.startswith("int")
        or dtype_name.startswith("uint")
    )


def _default_generator_for_attr_dtype(dtype_name: str) -> dict[str, Any]:
    # Annotation-only tensor constants have no payload. Zero is safer than
    # random bounded indices for constants that feed indexing/scatter paths.
    if _is_integer_or_bool_dtype_name(dtype_name):
        return {"kind": "constant", "value": 0}
    return {"kind": "randn"}


def _concrete_int(value: Any, *, default: int = 32) -> int:
    try:
        if hasattr(value, "node") and hasattr(value.node, "hint"):
            return int(value.node.hint)
        return int(value)
    except Exception:
        return default


def placeholder_info_from_gm(gm: Any) -> dict[str, dict]:
    """Build the placeholder_info shape/dtype map used for input generators."""
    import torch

    placeholder_info = {}
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = (node.meta or {}).get("val")
        if torch.is_tensor(val):
            placeholder_info[node.name] = {
                "shape": [_concrete_int(s) for s in val.shape],
                "stride": [_concrete_int(s) for s in val.stride()]
                if not val.is_contiguous()
                else [],
                "dtype": str(val.dtype),
                "device": str(val.device),
            }
        elif isinstance(val, (torch.SymInt, torch.SymFloat)):
            placeholder_info[node.name] = {
                "shape": [],
                "stride": [],
                "dtype": "symint",
                "device": "cpu",
                "hint": _concrete_int(val),
            }
    return placeholder_info


# Bound for an int64 input that any value is safe for; values over the whole
# range keep every bit field of a hashed id uniform.
_ANY_INDEX = 2**62


def _masked_to_index_range(node: Any, passthrough_ops: set) -> bool:
    """True when every use of an integer input reaches indexing only through
    a non-negative constant mask (``bitwise_and``) that fits the indexed dim.

    ``x & m`` lies in [0, m] for every int64 ``x``, negative or not, so such
    an input (e.g. a hashed id split into bit fields that each index a table)
    is safe for any value. Before the mask only constant shifts and offsets
    are followed (comparisons and graph outputs are harmless uses); after it
    only value-preserving ops, ``cat`` (paired with the ``select`` that reads
    the value's slice back) and index consumers.
    Any other use, or a path not fully explored, gives False.
    """
    import torch

    aten = torch.ops.aten
    pre_mask_ops = {
        aten.add.Tensor,
        aten.sub.Tensor,
        aten.__rshift__.Scalar,
        aten.bitwise_right_shift.Tensor_Scalar,
    }
    masks = {aten.bitwise_and.Scalar, aten.__and__.Scalar}
    # A comparison yields booleans whatever the input's range.
    comparisons = {
        aten.eq.Scalar, aten.ne.Scalar, aten.lt.Scalar,
        aten.le.Scalar, aten.gt.Scalar, aten.ge.Scalar,
    }
    value_ops = passthrough_ops - {aten.where.self, aten._to_copy.default}

    def dim_size(value, dim):
        if not isinstance(value, torch.Tensor) or not -value.dim() <= dim < value.dim():
            return None
        size = value.shape[dim]
        return int(size) if isinstance(size, int) else None

    def cat_slot(cat, n):
        tensors = cat.args[0]
        dim = cat.args[1] if len(cat.args) > 1 else cat.kwargs.get("dim", 0)
        if not isinstance(dim, int) or sum(t is n for t in tensors) != 1:
            return None
        start = 0
        for t in tensors:
            size = dim_size(getattr(t, "meta", {}).get("val"), dim)
            if size is None:
                return None
            if t is n:
                return dim, start, start + size
            start += size
        return None

    def index_fits(user, n, high):
        """n (values in [0, high]) is an index operand of user that fits."""
        if user.target == aten.embedding.default:
            return (
                len(user.args) > 1 and user.args[1] is n
                and user.args[0] is not n
                and (dim_size(user.args[0].meta.get("val"), 0) or 0) > high
            )
        if user.target in (aten.index.Tensor, aten._unsafe_index.Tensor):
            indices = user.args[1] if len(user.args) > 1 else None
            if user.args[0] is n or not isinstance(indices, (list, tuple)):
                return False
            # A bool/uint8 mask index spans several dims, so list positions
            # no longer map to dims.
            if any(
                index is not None
                and getattr(getattr(index, "meta", {}).get("val"), "dtype", None)
                in (torch.bool, torch.uint8)
                for index in indices
            ):
                return False
            value = user.args[0].meta.get("val")
            return all(
                (dim_size(value, dim) or 0) > high
                for dim, index in enumerate(indices)
                if index is n
            ) and sum(index is n for index in indices) > 0
        return False

    budget = [256]
    indexed = [False]

    def after_mask(n, high, slot=None) -> bool:
        budget[0] -= 1
        if budget[0] < 0:
            return False
        for user in n.users:
            if user.op == "output":
                continue
            if user.op != "call_function":
                return False
            target = user.target
            if slot is not None and target == aten.select.int and len(user.args) > 2:
                dim, index = user.args[1], user.args[2]
                size = dim_size(n.meta.get("val"), slot[0])
                if not isinstance(dim, int) or not isinstance(index, int) or size is None:
                    return False
                value = n.meta["val"]
                if dim % value.dim() == slot[0] % value.dim():
                    if not slot[1] <= index % size < slot[2]:
                        continue  # reads another input's slice
            if target in value_ops and user.args and user.args[0] is n:
                if not after_mask(user, high):
                    return False
            elif target == aten.cat.default:
                slot_of_n = cat_slot(user, n)
                if slot_of_n is None or not after_mask(user, high, slot_of_n):
                    return False
            elif index_fits(user, n, high):
                indexed[0] = True
            else:
                return False
        return True

    def before_mask(n) -> bool:
        budget[0] -= 1
        if budget[0] < 0:
            return False
        for user in n.users:
            if user.op == "output":
                continue
            if user.op != "call_function" or not user.args or user.args[0] is not n:
                return False
            if user.target in comparisons:
                continue
            other = user.args[1] if len(user.args) > 1 else None
            target = user.target
            if target in masks and type(other) is int and other >= 0:
                if not after_mask(user, other):
                    return False
            elif target in pre_mask_ops and type(other) is int:
                if not before_mask(user):
                    return False
            elif target in value_ops:
                if not before_mask(user):
                    return False
            else:
                return False
        return True

    return before_mask(node) and indexed[0]


def infer_index_bounds_from_gm(
    gm: Any,
    placeholder_info: dict[str, dict],
    constants_out: dict[str, int] | None = None,
) -> dict[str, int]:
    """Infer valid index bounds for integer placeholders by inspecting consumers.

    `constants_out` (optional dict, filled in place) receives placeholders
    whose only safe generation is a CONSTANT value rather than a range —
    e.g. maxpool offset tensors, where the window-center offset is the one
    value in-bounds for every window under padding (random offsets put edge
    windows out of range -> device-side assert in the consuming scatter).
    """
    import torch

    index_consumers = {
        torch.ops.aten.scatter.src: (0, 1),
        torch.ops.aten.scatter.value: (0, 1),
        torch.ops.aten.scatter_add.default: (0, 1),
        torch.ops.aten.scatter.reduce: (0, 1),
        torch.ops.aten.scatter.value_reduce: (0, 1),
        torch.ops.aten.scatter_reduce.two: (0, 1),
        torch.ops.aten.index_add.default: (0, 1),
        torch.ops.aten.index_copy.default: (0, 1),
        torch.ops.aten.index_fill.int_Scalar: (0, 1),
        torch.ops.aten.index_fill.int_Tensor: (0, 1),
        torch.ops.aten.gather.default: (0, 1),

        torch.ops.aten.index_select.default: (0, 1),
        torch.ops.aten.embedding.default: (0, None),
    }
    # Only the INDEX argument is bounded: a scatter_add's src (a float value
    # converted from a bf16 input) reaching the consumer must stay random.
    index_arg_pos = {
        torch.ops.aten.scatter.src: 2,
        torch.ops.aten.scatter.value: 2,
        torch.ops.aten.scatter_add.default: 2,
        torch.ops.aten.scatter.reduce: 2,
        torch.ops.aten.scatter.value_reduce: 2,
        torch.ops.aten.scatter_reduce.two: 2,
        torch.ops.aten.index_add.default: 2,
        torch.ops.aten.index_copy.default: 2,
        torch.ops.aten.index_fill.int_Scalar: 2,
        torch.ops.aten.index_fill.int_Tensor: 2,
        torch.ops.aten.gather.default: 2,
        torch.ops.aten.index_select.default: 2,
        torch.ops.aten.embedding.default: 1,
    }
    passthrough_ops = {
        torch.ops.aten.clone.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.select.int,
        torch.ops.aten.permute.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.where.self,
        # A lifted tensor constant (e.g. a small table of row ids) reaches
        # index() through lift_fresh_copy.
        torch.ops.aten.lift_fresh_copy.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
        # cat/stack keep each input's values: an index built as
        # cat(x[:, 0], x[:, 1] + 100, ...) must bound x by the tightest
        # branch (rows - 100), or the offset columns index out of bounds.
        torch.ops.aten.cat.default,
        torch.ops.aten.stack.default,
        # unbind/split + getitem hand each column to its own consumer: a
        # feature tensor whose columns index tables of different sizes must
        # be bounded by the smallest table, or the small lookup asserts OOB.
        torch.ops.aten.unbind.int,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes.default,
        operator.getitem,
        # An index cast to another integer dtype keeps its values.
        torch.ops.aten._to_copy.default,
    }

    bounds = {}
    constants = constants_out if constants_out is not None else {}
    ph_nodes = {n.name: n for n in gm.graph.nodes if n.op == "placeholder"}

    def _node_shape(n):
        if not hasattr(n, "meta"):
            return None
        val = n.meta.get("val")
        if isinstance(val, torch.Tensor):
            return list(val.shape)
        if n.op == "call_function" and n.target in {
            torch.ops.aten.empty.memory_format,
            torch.ops.aten.full.default,
        }:
            if n.args:
                shape_arg = n.args[0]
                if isinstance(shape_arg, (list, tuple)):
                    return list(shape_arg)
                # Canonical (lifted) graphs replace the literal shape list with
                # a _shape_param_N placeholder NODE whose meta['val'] holds the
                # concrete list (capture_hook.lift_shape_params). Resolve it —
                # otherwise index_put(full(_shape_param), [None,None,i,j], ..)
                # loses its target shape and the index args fall through to the
                # degenerate [0,1) default (the sum_sum_sum pytorch_unet class:
                # arg3/arg6 should be [0,320), arg4/arg5 [0,479), not [0,1)).
                if hasattr(shape_arg, "meta"):
                    mv = shape_arg.meta.get("val")
                    if isinstance(mv, (list, tuple)):
                        return list(mv)
        return None

    for name, node in ph_nodes.items():
        info = placeholder_info.get(name, {})
        dtype = info.get("dtype", "")
        is_float = "float" in dtype or "bfloat" in dtype
        if "int" not in dtype and not is_float:
            continue

        if "int8" in dtype:
            # Maxpool OFFSET tensors (offset-within-window). Random offsets
            # are NOT uniformly valid: offsets_to_indices converts them to
            # flat indices relative to each window's position, and under
            # padding the edge windows go negative / past the dim for most
            # values (offset 0 at a padded edge -> index -113) — the
            # consuming scatter_add then device-side-asserts and poisons
            # the CUDA context (the wave-1 torchbench train assert class).
            # The WINDOW-CENTER offset is in-bounds for every window
            # (measured: 3x3 pad1 on 112x112 -> [0, 12430] vs dim 12544),
            # so emit it via constants_out as a constant-value generator.
            for user in node.users:
                if user.op != "call_function":
                    continue
                target_name = str(user.target)
                if "max_pool_offsets_to_indices" not in target_name:
                    continue
                kernel_size = user.args[1] if len(user.args) >= 2 else None
                # In canonical (lifted) graphs the kernel size is a
                # _shape_param placeholder NODE whose meta["val"] holds the
                # list — resolve it (vgg16: literal-only check fell through
                # to the 3x3 default const 4 on a 2x2 kernel -> offset 4
                # invalid -> scatter assert).
                if (kernel_size is not None
                        and not isinstance(kernel_size, (list, tuple))
                        and hasattr(kernel_size, "meta")):
                    mv = kernel_size.meta.get("val")
                    if isinstance(mv, (list, tuple)):
                        kernel_size = mv
                if isinstance(kernel_size, (list, tuple)) and kernel_size:
                    if len(kernel_size) >= 2:
                        kh, kw = int(kernel_size[0]), int(kernel_size[1])
                        center = (kh // 2) * kw + (kw // 2)
                    else:
                        center = int(kernel_size[0]) // 2
                    constants[name] = center
                    break
            if name not in constants:
                # Offset 0 (first element of the window) is valid for any
                # kernel WITHOUT padding; under padding only interior
                # offsets are — but an unknown kernel size gives no better
                # choice, and 0 is right far more often than a 3x3 center.
                constants[name] = 0
            continue

        def _find_bound(start_node, max_hops=16):
            # A cross-entropy label (f64 -> nan_to_num where-chain -> int64 ->
            # ignore_index where -> unsqueeze -> gather) is 9 hops deep.
            # Collect EVERY bound the value reaches and return the MIN: a
            # value consumed by several index ops must satisfy all of them
            # (Electra: ids flow through gather into the 2-row token-type
            # embedding — first-found returned the 512 gather bound and the
            # 2-row table asserted OOB, poisoning the CUDA context).
            #
            # The walk follows VALUE flow through int arithmetic too, and
            # INVERTS each transform when a consumption bound is found
            # (Longformer: position ids = (ids * mask + 1) indexes a
            # 4098-row table — the leaf bound is isqrt(4097), not 4098):
            #   add/sub const c   ->  bound - |c|
            #   mul const c>0    ->  bound // c
            #   mul TENSOR       ->  isqrt(bound) (both operands generated
            #                        by us; each < sqrt(B) keeps product < B)
            #   add TENSOR       ->  bound // 2  (same reasoning)
            #   convert/clone/.. ->  unchanged
            import math as _math

            found: list[int] = []

            def _dim(v):
                """A dim's concrete size, or None for a data-dependent
                (unbacked) size: that index is left unbounded."""
                if isinstance(v, torch.SymInt):
                    hint = getattr(v.node, "hint", None)
                    if hint is not None:
                        return int(hint)
                    try:
                        return int(v)
                    except Exception:
                        return None
                return int(v)

            def _add(size, chain) -> None:
                if size is not None:
                    found.append(_invert(size, chain))

            def _invert(bound: int, chain) -> int:
                b = bound
                for kind, arg in reversed(chain):
                    if kind == "addc":
                        b = b - abs(int(arg))
                    elif kind == "mulc":
                        c = abs(int(arg))
                        b = b // c if c > 1 else b
                    elif kind == "mult":
                        b = int(_math.isqrt(max(b - 1, 1)))
                    elif kind == "addt":
                        b = b // 2
                    b = max(b, 1)
                return b

            _INT_ARITH = {
                torch.ops.aten.add.Tensor: "add",
                torch.ops.aten.sub.Tensor: "add",   # same inversion class
                torch.ops.aten.mul.Tensor: "mul",
            }
            _VALUE_PRESERVING = {
                torch.ops.prims.convert_element_type.default,
            }
            _STATIC_PASSTHROUGH = {
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten.expand.default,
                torch.ops.aten.unsqueeze.default,
                torch.ops.aten.clone.default,
                torch.ops.prims.convert_element_type.default,
            }

            def _static_max(m, depth=0):
                # Max value of an operand computed only from iota and
                # constants, else None. Inductor's split scatter indexes
                # P*N rows with idx + (iota & (P-1)) * N: idx is bounded by
                # N = rows - (P-1)*N, not rows // 2.
                if not hasattr(m, "op"):
                    return int(m) if isinstance(m, int) and m >= 0 else None
                if m.op != "call_function" or depth > 8:
                    return None
                t = m.target
                if t == torch.ops.prims.iota.default:
                    length = m.args[0]
                    start = m.kwargs.get("start", 0)
                    step = m.kwargs.get("step", 1)
                    if (isinstance(length, int) and isinstance(start, int)
                            and isinstance(step, int) and start >= 0 and step >= 0):
                        return start + max(length - 1, 0) * step
                    return None
                if t == torch.ops.aten.full.default and len(m.args) > 1:
                    v = m.args[1]
                    return int(v) if isinstance(v, int) and v >= 0 else None
                if t == torch.ops.aten.bitwise_and.Scalar:
                    mask = m.args[1]
                    if isinstance(mask, int) and mask >= 0:
                        inner = _static_max(m.args[0], depth + 1)
                        return mask if inner is None else min(mask, inner)
                    return None
                if t in (torch.ops.aten.mul.Tensor, torch.ops.aten.add.Tensor):
                    a = _static_max(m.args[0], depth + 1)
                    b = _static_max(m.args[1], depth + 1)
                    if a is None or b is None:
                        return None
                    return a * b if t == torch.ops.aten.mul.Tensor else a + b
                if t in _STATIC_PASSTHROUGH and m.args:
                    return _static_max(m.args[0], depth + 1)
                return None

            frontier = [(start_node, ())]
            for _hop in range(max_hops):
                next_frontier = []
                for n, chain in frontier:
                    for user in n.users:
                        if user.op != "call_function":
                            continue
                        target = user.target

                        if target == torch.ops.aten.embedding.default:
                            # only when n is the INDICES arg (args[1])
                            if len(user.args) > 1 and user.args[1] is n:
                                weight_arg = user.args[0] if user.args else None
                                if weight_arg and hasattr(weight_arg, "meta"):
                                    val = weight_arg.meta.get("val")
                                    if isinstance(val, torch.Tensor) and len(val.shape) > 0:
                                        _add(_dim(val.shape[0]), chain)
                                        continue

                        if (target == torch.ops.aten.embedding_dense_backward.default
                                and len(user.args) > 2 and user.args[1] is n
                                and isinstance(user.args[2], int)):
                            found.append(_invert(int(user.args[2]), chain))
                            continue

                        # The masked variants take (self, mask, indices, ...).
                        indices_pos = (
                            2 if target in (
                                torch.ops.aten._unsafe_masked_index.default,
                                torch.ops.aten._unsafe_masked_index_put_accumulate.default,
                            ) else 1
                        )
                        if (target in (torch.ops.aten.index.Tensor,
                                       torch.ops.aten.index_put.default,
                                       torch.ops.aten.index_put_.default,
                                       torch.ops.aten._unsafe_index.Tensor,
                                       torch.ops.aten._unsafe_index_put.default,
                                       torch.ops.aten._unsafe_masked_index.default,
                                       torch.ops.aten._unsafe_masked_index_put_accumulate.default)
                                and len(user.args) > indices_pos):
                            # indices list position maps to DIM, INCLUDING
                            # None slots (index_put([None, None, i, j]):
                            # i indexes dim 2, j dim 3 — Yitu: bounding j
                            # by shape[0]=32 against dim3 size 1 asserted).
                            # index_put_ (in-place) shares the schema; both
                            # take (self, indices_list, values, ...).
                            target_shape = _node_shape(user.args[0])
                            indices = user.args[indices_pos]
                            if target_shape and isinstance(indices, (list, tuple)):
                                for dim, index_node in enumerate(indices):
                                    if index_node is n and dim < len(target_shape):
                                        _add(_dim(target_shape[dim]), chain)

                        if target == torch.ops.aten.gather.default:
                            if user.args and user.args[0] is n:
                                # n is gather's DATA source: values flow on.
                                next_frontier.append((user, chain))
                                continue

                        if target in index_consumers:
                            target_arg_idx, dim_arg_idx = index_consumers[target]
                            pos = index_arg_pos[target]
                            if len(user.args) > max(target_arg_idx, pos) and user.args[pos] is n:
                                target_node = user.args[target_arg_idx]
                                target_shape = _node_shape(target_node)
                                if target_shape and target_node is not n:
                                    if dim_arg_idx is not None and len(user.args) > dim_arg_idx:
                                        dim = user.args[dim_arg_idx]
                                        if isinstance(dim, int) and dim < len(target_shape):
                                            _add(_dim(target_shape[dim]), chain)
                                    else:
                                        _add(_dim(target_shape[0]), chain)

                        if (target == torch.ops.aten._to_copy.default
                                and isinstance(user.kwargs.get("dtype"), torch.dtype)
                                and user.kwargs["dtype"].is_floating_point
                                and user.kwargs["dtype"].itemsize < 4):
                            # fp16/bf16 round large ints up (4099 -> 4100 in
                            # fp16), so a value below the bound can land on it.
                            continue
                        if target in passthrough_ops or target in _VALUE_PRESERVING:
                            next_frontier.append((user, chain))
                        elif target in _INT_ARITH:
                            kind = _INT_ARITH[target]
                            other = None
                            for a in user.args[:2]:
                                if a is not n:
                                    other = a
                            static = (_static_max(other)
                                      if kind == "add" and not isinstance(other, (int, float))
                                      else None)
                            if isinstance(other, (int, float)):
                                next_frontier.append(
                                    (user, chain + ((kind + "c", other),)))
                            elif static is not None:
                                next_frontier.append(
                                    (user, chain + (("addc", static),)))
                            else:
                                next_frontier.append(
                                    (user, chain + ((kind + "t", None),)))
                # unbind/split -> getitem -> cat/stack fan-out and fan-in reach a node along many paths;
                # identical (node, chain) entries give identical bounds.
                frontier = list(dict.fromkeys(next_frontier))
                if not frontier:
                    break
            if found:
                return min(found)
            return None

        if dtype == "torch.int64" and _masked_to_index_range(node, passthrough_ops):
            bounds[name] = _ANY_INDEX
            continue
        bound = _find_bound(node)
        if bound is not None:
            if is_float:
                # FLOAT placeholder whose VALUES flow (via convert chains)
                # into index consumption — OPT: position ids derive from a
                # float attention mask, (mask - 1).to(int64) + 2 indexes a
                # 2050-row table. randn would go negative/OOB; generate
                # bounded non-negative values instead.
                info.setdefault("gen", {"kind": "index", "low": 0,
                                        "high": bound})
            bounds[name] = bound

    return bounds


def infer_sorted_inputs_from_gm(gm: Any, placeholder_info: dict[str, dict]) -> None:
    """Attach a sorted generator to float placeholders used as bucketize
    boundaries or searchsorted's sorted sequence. Unsorted random boundaries
    send most values to a few buckets, so a downstream index_put(accumulate)
    serializes on atomics the model never contends on."""
    import torch

    sorted_arg = {
        torch.ops.aten.bucketize.Tensor: 1,
        torch.ops.aten.searchsorted.Tensor: 0,
        torch.ops.aten.searchsorted.Scalar: 0,
    }
    views = {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.clone.default,
        torch.ops.prims.convert_element_type.default,
    }
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        info = placeholder_info.get(node.name)
        dtype = (info or {}).get("dtype", "")
        if info is None or not ("float" in dtype or "bfloat" in dtype):
            continue
        frontier = [node]
        for n in frontier:
            for user in n.users:
                if user.op != "call_function":
                    continue
                pos = sorted_arg.get(user.target)
                if pos is not None and len(user.args) > pos and user.args[pos] is n:
                    info.setdefault("gen", {"kind": "sorted"})
                elif user.target in views and user.args and user.args[0] is n:
                    frontier.append(user)


def infer_permutation_indices_from_gm(
    gm: Any,
    placeholder_info: dict[str, dict],
) -> dict[str, int]:
    """Find integer placeholders that must be valid permutations."""
    import torch

    permutation_sizes = {}
    ph_nodes = {n.name: n for n in gm.graph.nodes if n.op == "placeholder"}

    def _shape_from_alloc(n):
        if n.op != "call_function":
            return None
        if n.target not in {
            torch.ops.aten.empty.memory_format,
            torch.ops.aten.full.default,
        }:
            return None
        if not n.args:
            return None
        shape_arg = n.args[0]
        # Canonical graphs lift alloc shapes to _shape_param placeholder
        # NODES — resolve via meta['val'] (gpt-oss MoE: empty(_shape_param)
        # blinded the permutation detection; the scatter ordering indices
        # then generated as random Index -> duplicate positions -> assert).
        if not isinstance(shape_arg, (list, tuple)) and hasattr(shape_arg, "meta"):
            mv = shape_arg.meta.get("val")
            if isinstance(mv, (list, tuple)):
                shape_arg = mv
        if not isinstance(shape_arg, (list, tuple)):
            return None
        return list(shape_arg)

    iota_passthrough_ops = {
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
    }

    def _iota_size(n, max_hops=4):
        if n.op != "call_function":
            return None
        if n.target != torch.ops.prims.iota.default:
            if max_hops > 0 and n.target in iota_passthrough_ops and n.args:
                return _iota_size(n.args[0], max_hops - 1)
            return None
        if not n.args:
            return None
        length = n.args[0]
        # lifted/symbolic length: resolve via meta['val'] or SymInt hint
        if not isinstance(length, int) and hasattr(length, "meta"):
            mv = length.meta.get("val")
            if isinstance(mv, int):
                length = mv
            elif hasattr(mv, "node") and hasattr(mv.node, "hint"):
                length = int(mv.node.hint)
        if isinstance(length, int):
            return length
        return None

    for name, node in ph_nodes.items():
        info = placeholder_info.get(name, {})
        dtype = info.get("dtype", "")
        if "int" not in dtype:
            continue
        for user in node.users:
            if user.op != "call_function" or user.target != torch.ops.aten.index_put.default:
                continue
            if len(user.args) < 3:
                continue
            indices = user.args[1]
            if not isinstance(indices, (list, tuple)) or node not in indices:
                continue
            target_shape = _shape_from_alloc(user.args[0])
            iota_size = _iota_size(user.args[2])
            if target_shape and iota_size and target_shape[0] == iota_size:
                permutation_sizes[name] = iota_size

        for user in node.users:
            if user.op != "call_function" or user.target != torch.ops.aten.scatter.src:
                continue
            if len(user.args) < 4 or user.args[2] is not node:
                continue
            target_shape = _shape_from_alloc(user.args[0])
            iota_size = _iota_size(user.args[3])
            dim = user.args[1]
            if target_shape and iota_size and isinstance(dim, int):
                if dim < 0:
                    dim += len(target_shape)
                if 0 <= dim < len(target_shape) and target_shape[dim] == iota_size:
                    permutation_sizes[name] = iota_size
    return permutation_sizes


def _tensor_spec_from_value(
    name: str,
    value: Any,
    *,
    gen_override: dict[str, Any] | None = None,
    include_exact: bool = False,
) -> dict[str, Any]:
    dtype_name = _dtype_name(value.dtype)
    shape = [_concrete_int(dim) for dim in value.shape]
    generator = gen_override or _default_generator_for_dtype(dtype_name, shape)
    spec = {
        "kind": "tensor",
        "name": name,
        "shape": shape,
        "dtype": dtype_name,
        "stride": [_concrete_int(dim) for dim in value.stride()],
        "device": "cuda" if str(value.device).startswith("cuda") else str(value.device),
        "storage_offset": _concrete_int(value.storage_offset(), default=0),
        "generator": generator,
    }
    if generator.get("kind") in {"index", "permutation", "runs", *JAGGED_KINDS}:
        spec["gen"] = generator
        spec["constraint_source"] = "graph_inference" if gen_override else "dtype_shape_default"
    exact = _exact_tensor_payload(value) if include_exact else None
    if exact is not None:
        spec.update(exact)
    else:
        spec["exact"] = False
        if include_exact and _is_integer_or_bool_dtype_name(dtype_name):
            spec["requires_exact"] = True
    return spec


def _is_fake_or_meta_tensor(value: Any) -> bool:
    if getattr(value, "is_meta", False):
        return True
    try:
        from torch._subclasses.fake_tensor import FakeTensor

        if isinstance(value, FakeTensor):
            return True
    except Exception:
        pass
    return "fake_tensor" in type(value).__module__


def _exact_tensor_payload(value: Any, *, max_numel: int = 4096) -> dict[str, Any] | None:
    if _is_fake_or_meta_tensor(value):
        return None
    dtype_name = _dtype_name(value.dtype)
    if value.numel() > max_numel:
        return None
    if not (
        dtype_name == "bool"
        or dtype_name.startswith("int")
        or dtype_name.startswith("uint")
        or value.numel() <= 16
    ):
        return None
    try:
        detached = value.detach().cpu().contiguous()
    except Exception:
        return None
    return {
        "exact": True,
        "data": detached.reshape(-1).tolist(),
    }


def _scalar_spec_from_value(name: str, value: Any) -> dict[str, Any]:
    import torch

    if isinstance(value, torch.Generator):
        device = value.device
        return {
            "kind": "generator",
            "name": name,
            "device_type": device.type,
            "device_index": device.index or 0,
        }
    if isinstance(value, (torch.SymInt, torch.SymFloat)):
        return {"kind": "symint", "name": name, "value": _concrete_int(value)}
    if isinstance(value, (int, float, bool)):
        return {"kind": "scalar", "name": name, "value": value}
    return {"kind": "scalar", "name": name, "value": 1}


def _index_gen_for_placeholder(bound: int, value: Any, observed: dict | None):
    """Index generator for an integer placeholder: ``runs`` when the observed
    real values are run-structured (input_codec.index_gen_from_observed),
    else uniform ids in [0, bound). Float placeholders feeding index math
    keep the uniform spec."""
    from input_codec import index_gen_from_observed

    if value.dtype.is_floating_point or _dtype_name(value.dtype) == "bool":
        return {"kind": "index", "low": 0, "high": bound}
    return index_gen_from_observed(bound, observed)


def graph_constraints_from_gm(
    gm: Any,
    *,
    index_bounds: dict[str, int] | None = None,
    permutation_indices: dict[str, int] | None = None,
    observed_stats: dict[str, dict] | None = None,
    jagged_specs: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Extract replay constraints from an FX GraphModule.

    The exporter writes this next to full_graph_*.py so newly added model
    graphs do not depend solely on parsing print_readable annotations.

    Bound hierarchy (settled 2026-06-10):
      1. graph_inference: known consumer patterns (embedding, gather, scatter)
      2. observed: fallback from real execution stats (high = observed.max + 1)
      3. default_unobserved: ONLY when observation was impossible (should not
         occur for new captures)
    """
    import torch

    index_bounds = index_bounds or {}
    permutation_indices = permutation_indices or {}
    observed_stats = observed_stats or {}
    jagged_specs = jagged_specs or {}
    inputs = []
    outputs = []
    tensor_attrs = {}
    for node in gm.graph.nodes:
        value = (node.meta or {}).get("val")
        if node.op == "placeholder":
            if torch.is_tensor(value):
                gen_override = None
                constraint_source = None
                if node.name in jagged_specs:
                    gen_override = jagged_specs[node.name]
                    constraint_source = "jagged_inputs"
                elif node.name in permutation_indices:
                    gen_override = {
                        "kind": "permutation",
                        "size": int(permutation_indices[node.name]),
                    }
                    constraint_source = "graph_inference"
                elif node.name in index_bounds:
                    # Same rule as region capture, so region and full-graph
                    # benchmarks feed this input the same id distribution.
                    gen_override = _index_gen_for_placeholder(
                        int(index_bounds[node.name]), value,
                        observed_stats.get(node.name))
                    constraint_source = "graph_inference"
                elif node.name in observed_stats:
                    # Observed-value fallback: use observed max + 1 as bound
                    obs = observed_stats[node.name]
                    dtype_name = _dtype_name(value.dtype)
                    if _is_integer_or_bool_dtype_name(dtype_name):
                        gen_override = _index_gen_for_placeholder(
                            int(obs["max"]) + 1, value, obs)
                        constraint_source = "observed"
                else:
                    # No inference bound and no observation
                    dtype_name = _dtype_name(value.dtype)
                    if _is_integer_or_bool_dtype_name(dtype_name):
                        constraint_source = "default_unobserved"

                spec = _tensor_spec_from_value(
                    node.name,
                    value,
                    gen_override=gen_override,
                )
                # Override constraint_source with the hierarchy decision
                if constraint_source is not None:
                    spec["constraint_source"] = constraint_source
                # Attach observed stats alongside (regardless of which source won)
                if node.name in observed_stats:
                    spec["observed"] = observed_stats[node.name]
                    # Note potential permutation: n_unique == numel on 1-D int tensor
                    obs = observed_stats[node.name]
                    shape = spec.get("shape", [])
                    if (
                        len(shape) == 1
                        and shape[0] > 0
                        and obs.get("n_unique") == shape[0]
                        and _is_integer_or_bool_dtype_name(spec.get("dtype", ""))
                        and spec.get("dtype", "") != "bool"
                    ):
                        spec["maybe_permutation"] = True
                inputs.append(spec)
            else:
                inputs.append(_scalar_spec_from_value(node.name, value))
        elif node.op == "get_attr":
            attr_value = _fetch_attr(gm, node.target)
            if torch.is_tensor(attr_value):
                tensor_attrs[str(node.target)] = _tensor_spec_from_value(
                    str(node.target),
                    attr_value,
                    include_exact=True,
                )
        elif node.op == "output":
            for idx, leaf in enumerate(_iter_tree_leaves(node.args)):
                if isinstance(leaf, torch.fx.Node):
                    leaf_value = (leaf.meta or {}).get("val")
                    if torch.is_tensor(leaf_value):
                        outputs.append(_tensor_spec_from_value(f"output_{idx}", leaf_value))
                    elif leaf_value is not None:
                        outputs.append(_scalar_spec_from_value(f"output_{idx}", leaf_value))

    payload = {
        "schema_version": 1,
        "inputs": inputs,
        "outputs": outputs,
        "tensor_attrs": tensor_attrs,
    }
    storage_groups = _placeholder_storage_groups(gm)
    if storage_groups:
        # Inputs that are VIEWS OF ONE STORAGE (packed-qkv saved views in
        # backward graphs etc). Regenerating them as independent storages
        # changes footprint/locality and is wrong under mutation — replay
        # must allocate one buffer per group and as_strided each member.
        # Detectable ONLY at capture (live fake vals share a storage; any
        # later retrace re-fabricates inputs and the identity is gone).
        payload["storage_groups"] = storage_groups
    return payload


def _merge_source_input_constraints(
    generated: list[dict[str, Any]],
    source: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    if len(generated) != len(source):
        raise ValueError(
            "source input count does not match captured graph placeholders: "
            f"{len(source)} != {len(generated)}"
        )

    merged = []
    captured_dynamic = False
    for index, (generated_spec, source_spec) in enumerate(zip(generated, source)):
        fake_generator = (
            generated_spec.get("kind") == "scalar"
            and source_spec.get("kind") == "generator"
        )
        if generated_spec.get("kind") != source_spec.get("kind") and not fake_generator:
            raise ValueError(
                f"source input kind does not match captured placeholder {index}"
            )
        spec = dict(generated_spec)
        if fake_generator:
            spec.pop("value", None)
            spec["kind"] = "generator"
        if spec.get("kind") in {"scalar", "symint"} and "value" in source_spec:
            spec["value"] = source_spec["value"]
        if spec.get("kind") == "generator":
            spec["device_type"] = source_spec.get("device_type", "cpu")
            spec["device_index"] = int(source_spec.get("device_index", 0))
        for key in ("shape", "stride", "device", "storage_offset", "alias_group"):
            if source_spec.get(key) is not None:
                spec[key] = source_spec[key]
        for key in (
            "symbolic_expr",
            "symbolic_dims",
            "symbolic_stride_dims",
            "symbolic_storage_offset",
        ):
            if key in source_spec:
                spec[key] = source_spec[key]
                captured_dynamic = True
        merged.append(spec)
    return merged, captured_dynamic


def _placeholder_storage_groups(gm: Any) -> list[list[str]]:
    """Group placeholder names whose meta vals share one untyped storage."""
    import torch

    groups: dict[int, list[str]] = {}
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = (node.meta or {}).get("val")
        if not torch.is_tensor(val):
            continue
        try:
            key = id(val.untyped_storage())
        except Exception:
            continue
        groups.setdefault(key, []).append(node.name)
    return [names for names in groups.values() if len(names) > 1]


def _fetch_attr(module: Any, target: Any) -> Any:
    value = module
    for atom in str(target).split("."):
        value = getattr(value, atom)
    return value


def _iter_tree_leaves(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tree_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tree_leaves(item)
    else:
        yield value


def write_full_graph_metadata(
    graph_path: str | Path,
    gm: Any,
    *,
    extra: dict[str, Any] | None = None,
    index_bounds: dict[str, int] | None = None,
    permutation_indices: dict[str, int] | None = None,
    observed_stats: dict[str, dict] | None = None,
    source_input_specs: list[dict[str, Any]] | None = None,
    jagged_specs: dict[str, dict] | None = None,
) -> Path:
    graph_path = Path(graph_path)
    payload = graph_constraints_from_gm(
        gm,
        index_bounds=index_bounds,
        permutation_indices=permutation_indices,
        observed_stats=observed_stats,
        jagged_specs=jagged_specs,
    )
    if source_input_specs is not None:
        payload["inputs"], captured_dynamic = _merge_source_input_constraints(
            payload["inputs"], source_input_specs
        )
        payload["captured_dynamic"] = captured_dynamic
        if (gm.meta or {}).get("better_benchmark_source_code") is not None:
            renamed = {}
            for spec, source_spec in zip(payload["inputs"], source_input_specs):
                if source_spec.get("name"):
                    renamed[spec["name"]] = source_spec["name"]
                    spec["name"] = source_spec["name"]
            if payload.get("storage_groups"):
                payload["storage_groups"] = [
                    [renamed.get(name, name) for name in group]
                    for group in payload["storage_groups"]
                ]
            for spec in payload["inputs"]:
                gen = spec.get("gen") or {}
                total = gen.get("total") if gen.get("kind") in JAGGED_KINDS else {}
                if "sym" in total:
                    total["sym"] = renamed.get(total["sym"], total["sym"])
                elif "size" in total:
                    total["size"][0] = renamed.get(total["size"][0], total["size"][0])
    payload["graph"] = graph_path.name
    if extra:
        payload.update(extra)
    # Serialize inputs/outputs/tensor_attrs in the compact shared encoding
    # (input_codec) — schema_version 2. The verbose dict-per-tensor form
    # remains the in-memory representation; loaders inflate on read.
    from input_codec import compact_from_spec

    payload["schema_version"] = 2
    for key in ("inputs", "outputs"):
        if payload.get(key):
            payload[key] = [compact_from_spec(s, include_name=True)
                            for s in payload[key]]
    if payload.get("tensor_attrs"):
        payload["tensor_attrs"] = {
            name: compact_from_spec(s)
            for name, s in payload["tensor_attrs"].items()
        }
    meta_path = graph_path.with_suffix(".meta.json")
    meta_path.write_text(_dumps_compact_entries(payload) + "\n")
    return meta_path


class _OneLine:
    """Marker: serialize this value on a single line inside indented JSON."""

    def __init__(self, value):
        self.value = value


def dumps_with_onelines(obj) -> str:
    """json.dumps(indent=2), but any _OneLine-wrapped value is emitted on a
    single line (a compact input entry exploded across 12 indented lines
    would re-create the verbosity the compact encoding exists to remove)."""
    placeholders: list[str] = []

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, _OneLine):
                placeholders.append(
                    json.dumps(o.value, separators=(",", ":")))
                return f"@@ONELINE{len(placeholders) - 1}@@"
            return super().default(o)

    text = json.dumps(obj, indent=2, cls=_Enc)
    for i, body in enumerate(placeholders):
        text = text.replace(f'"@@ONELINE{i}@@"', body)
    return text


def _dumps_compact_entries(payload: dict) -> str:
    """Sidecar serializer: inputs/outputs/tensor_attrs entries one-lined."""
    marked = dict(payload)
    for key in ("inputs", "outputs"):
        if marked.get(key):
            marked[key] = [_OneLine(e) for e in marked[key]]
    if marked.get("tensor_attrs"):
        marked["tensor_attrs"] = {
            k: _OneLine(v) for k, v in marked["tensor_attrs"].items()
        }
    return dumps_with_onelines(marked)


def _split_signature_args(sig: str) -> list[str]:
    args = []
    start = 0
    in_quote = False
    quote = ""
    for i, ch in enumerate(sig):
        if ch in {"'", '"'}:
            if in_quote and ch == quote:
                in_quote = False
                quote = ""
            elif not in_quote:
                in_quote = True
                quote = ch
        elif ch == "," and not in_quote:
            part = sig[start:i].strip()
            if part:
                args.append(part)
            start = i + 1
    part = sig[start:].strip()
    if part:
        args.append(part)
    return args


def _parse_intish(value: str, *, default: int = 32) -> int:
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        pass
    sym_match = re.fullmatch(r"Sym\(([^)]*)\)", value)
    if sym_match:
        return _parse_intish(sym_match.group(1), default=default)
    if re.fullmatch(r"[A-Za-z_]\w*", value):
        return default
    match = re.search(r"-?\d+", value)
    if match:
        return int(match.group(0))
    return default


def _split_dim_exprs(value: str) -> list[str]:
    if not value.strip():
        return []
    parsed = ast.parse(f"({value},)", mode="eval").body
    if not isinstance(parsed, ast.Tuple):
        raise ValueError(f"expected dimension sequence: {value!r}")
    return [ast.unparse(item) for item in parsed.elts]


def _symbolic_dims(shape_str: str) -> list[dict[str, Any]]:
    symbols = []
    for idx, dim in enumerate(_split_dim_exprs(shape_str)):
        try:
            int(dim)
        except ValueError:
            symbols.append({"dim": idx, "symbol": dim, "default": _parse_intish(dim)})
    return symbols


def _symbolic_stride_dims(stride_str: str | None) -> list[dict[str, Any]]:
    if stride_str is None:
        return []
    symbolic = []
    for idx, dim in enumerate(_split_dim_exprs(stride_str)):
        try:
            int(dim)
        except ValueError:
            symbolic.append({"dim": idx, "expr": dim})
    return symbolic


def _parse_shape(shape_str: str) -> tuple[int, ...]:
    return tuple(_parse_intish(dim) for dim in _split_dim_exprs(shape_str))


def _parse_stride(stride_str: str | None) -> tuple[int, ...] | None:
    if stride_str is None:
        return None
    return tuple(_parse_intish(dim) for dim in _split_dim_exprs(stride_str))


def _parse_device(device_str: str | None) -> str | None:
    if not device_str:
        return None
    device_str = device_str.strip()
    if not device_str:
        return None
    if device_str.startswith("cuda"):
        return "cuda"
    if device_str.startswith("cpu"):
        return "cpu"
    return None


def _forward_signature(content: str) -> str | None:
    match = re.search(r"def forward\(self,?\s*(.*?)\):", content, re.DOTALL)
    return match.group(1) if match else None


def forward_takes_no_inputs(content: str) -> bool:
    sig = _forward_signature(content)
    return sig is not None and not sig.strip()


def parse_full_graph_inputs(content: str) -> list[dict[str, Any]]:
    """Parse forward annotations into input specs.

    Examples handled:
      x: "f32[2, 3][3, 1]cuda:0"
      s0: "Sym(s0)"
      unannotated scalar parameters
    """
    sig = _forward_signature(content)
    if sig is None or not sig.strip():
        return []

    specs = []
    for arg in _split_signature_args(sig):
        name_match = re.match(r"(\w+)", arg)
        name = name_match.group(1) if name_match else f"arg{len(specs)}"
        annotation_match = re.match(r'\w+\s*:\s*"([^"]*)"', arg)
        if annotation_match is None:
            specs.append({"kind": "scalar", "name": name, "value": 1})
            continue

        annotation = annotation_match.group(1)
        if annotation.startswith("Sym("):
            specs.append({
                "kind": "symint",
                "name": name,
                "value": _parse_intish(annotation),
                "symbolic_expr": annotation.removeprefix("Sym(").removesuffix(")"),
            })
            continue

        tensor_match = re.match(
            r"(\w+)\[([^\]]*)\](?:\[([^\]]*)\])?(.*)",
            annotation,
        )
        if tensor_match is None:
            specs.append({"kind": "scalar", "name": name, "value": 1})
            continue

        dtype_token, shape_str, stride_str, device_str = tensor_match.groups()
        if dtype_token not in _DTYPE_SHORT_NAMES:
            raise ValueError(f"unsupported full graph dtype: {dtype_token}")
        dtype_name = _DTYPE_SHORT_NAMES[dtype_token]
        shape = _parse_shape(shape_str)
        generator = _default_generator_for_dtype(dtype_name, shape)
        spec = {
            "kind": "tensor",
            "name": name,
            "shape": shape,
            "dtype": dtype_name,
            "stride": _parse_stride(stride_str),
            "device": _parse_device(device_str),
            # print_readable annotations don't carry storage_offset: None =
            # unknown (validator skips), never a false claim of 0 — packed
            # qkv views (offset 192/384/...) are real and sidecar-recorded.
            "storage_offset": None,
            "generator": generator,
        }
        if generator.get("kind") == "index":
            spec["gen"] = generator
            spec["constraint_source"] = "annotation_default"
        symbolic_dims = _symbolic_dims(shape_str)
        if symbolic_dims:
            spec["symbolic_dims"] = symbolic_dims
        symbolic_stride_dims = _symbolic_stride_dims(stride_str)
        if symbolic_stride_dims:
            spec["symbolic_stride_dims"] = symbolic_stride_dims
        specs.append({
            **spec,
        })

    return specs


def parse_full_graph_tensor_attrs(content: str) -> dict[str, dict[str, Any]]:
    """Find saved self._frozen_param* and self._tensor_constant* references."""
    attrs: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r'\w+:\s*"(\w+)\[([^\]]*)\](?:\[([^\]]*)\])?([^"]*)"\s*=\s*'
        r"self\.(_(?:frozen_param|tensor_constant)\w*)"
    )
    for line in content.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        dtype_token, shape_str, stride_str, device_str, attr_name = match.groups()
        if dtype_token not in _DTYPE_SHORT_NAMES:
            raise ValueError(f"unsupported full graph dtype: {dtype_token}")
        dtype_name = _DTYPE_SHORT_NAMES[dtype_token]
        shape = _parse_shape(shape_str)
        generator = _default_generator_for_attr_dtype(dtype_name)
        attrs[attr_name] = {
            "kind": "tensor",
            "name": attr_name,
            "shape": shape,
            "dtype": dtype_name,
            "stride": _parse_stride(stride_str),
            "device": _parse_device(device_str),
            "storage_offset": 0,
            "generator": generator,
        }
        if _is_integer_or_bool_dtype_name(dtype_name):
            attrs[attr_name]["requires_exact"] = True
        if generator.get("kind") == "index":
            attrs[attr_name]["gen"] = generator
    return attrs


def full_graph_meta_path(graph_path: str | Path) -> Path:
    return Path(graph_path).with_suffix(".meta.json")


def load_full_graph_sidecar(graph_path: str | Path) -> dict[str, Any]:
    meta_path = full_graph_meta_path(graph_path)
    if not meta_path.exists():
        return {}
    sidecar = json.loads(meta_path.read_text())
    # schema v2: inputs/outputs/tensor_attrs stored in the compact shared
    # encoding (input_codec); inflate to the verbose in-memory spec dicts
    # so every downstream consumer is format-agnostic.
    if sidecar.get("schema_version", 1) >= 2:
        from input_codec import spec_from_compact

        for key in ("inputs", "outputs"):
            if sidecar.get(key):
                sidecar[key] = [spec_from_compact(e) for e in sidecar[key]]
        if sidecar.get("tensor_attrs"):
            sidecar["tensor_attrs"] = {
                name: spec_from_compact(e)
                for name, e in sidecar["tensor_attrs"].items()
            }
    return sidecar


def _normalize_dtype_name(dtype_name: Any) -> str:
    return str(dtype_name).removeprefix("torch.")


def _rank(spec: dict[str, Any]) -> int | None:
    shape = spec.get("shape")
    if shape is None:
        return None
    return len(shape)


def _normalize_device_name(device: Any) -> str | None:
    if device is None:
        return None
    text = str(device)
    if not text:
        return None
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("cpu"):
        return "cpu"
    return text


def _resolve_annotation_expr(
    expr_text: str,
    bindings: dict[str, int],
    *,
    label: str,
    dim: int,
    kind: str,
) -> int:
    import sympy

    normalized = expr_text.replace("math.ceil", "ceiling").replace(
        "math.floor", "floor"
    )
    expr = sympy.sympify(
        normalized,
        locals={
            "CeilToInt": sympy.ceiling,
            "FloorDiv": lambda left, right: sympy.floor(left / right),
            "IntTrueDiv": lambda left, right: left / right,
        },
        rational=False,
    )
    missing = sorted(
        str(symbol)
        for symbol in expr.free_symbols
        if str(symbol) not in bindings
    )
    if missing:
        raise ValueError(
            f"full graph annotation {kind} expression has unbound "
            f"symbols for {label} dim {dim}: {missing}"
        )
    expected = expr.subs(
        {sympy.Symbol(name): value for name, value in bindings.items()}
    )
    if not expected.is_Integer:
        raise ValueError(
            f"full graph annotation {kind} expression is not an "
            f"integer for {label} dim {dim}: {expected}"
        )
    return int(expected)


def _validate_tensor_sidecar_spec(
    sidecar_spec: dict[str, Any],
    annotation_spec: dict[str, Any],
    label: str,
    symbol_bindings: dict[str, int] | None = None,
) -> None:
    sidecar_dtype = _normalize_dtype_name(sidecar_spec.get("dtype", "float32"))
    annotation_dtype = _normalize_dtype_name(annotation_spec.get("dtype", "float32"))
    if sidecar_dtype != annotation_dtype:
        raise ValueError(
            "full graph sidecar tensor dtype does not match forward annotations "
            f"for {label}: {sidecar_dtype!r} != {annotation_dtype!r}"
        )

    sidecar_shape = sidecar_spec.get("shape")
    annotation_shape = annotation_spec.get("shape")
    if sidecar_shape is not None and annotation_shape is not None:
        if len(sidecar_shape) != len(annotation_shape):
            raise ValueError(
                "full graph sidecar tensor rank does not match forward annotations "
                f"for {label}: {len(sidecar_shape)} != {len(annotation_shape)}"
            )
        symbolic_shapes = {
            int(item["dim"]): item["symbol"]
            for item in annotation_spec.get("symbolic_dims", [])
        }
        bindings = dict(symbol_bindings or {})
        for idx, (sidecar_dim, annotation_dim) in enumerate(zip(sidecar_shape, annotation_shape)):
            expected = annotation_dim
            if idx in symbolic_shapes:
                expected = _resolve_annotation_expr(
                    symbolic_shapes[idx],
                    bindings,
                    label=label,
                    dim=idx,
                    kind="shape",
                )
            if int(sidecar_dim) != int(expected):
                raise ValueError(
                    "full graph sidecar tensor shape does not match forward annotations "
                    f"for {label} dim {idx}: {sidecar_dim} != {expected}"
                )

    sidecar_stride = sidecar_spec.get("stride")
    annotation_stride = annotation_spec.get("stride")
    if sidecar_stride is not None and annotation_stride is not None:
        if len(sidecar_stride) != len(annotation_stride):
            raise ValueError(
                "full graph sidecar tensor stride rank does not match forward annotations "
                f"for {label}: {len(sidecar_stride)} != {len(annotation_stride)}"
            )
        bindings = dict(symbol_bindings or {})
        for item in annotation_spec.get("symbolic_dims", []):
            symbol = item.get("symbol")
            dim = int(item["dim"])
            if (
                isinstance(symbol, str)
                and re.fullmatch(r"[A-Za-z_]\w*", symbol)
                and sidecar_shape is not None
                and dim < len(sidecar_shape)
            ):
                bindings.setdefault(symbol, int(sidecar_shape[dim]))
        symbolic_strides = {
            int(item["dim"]): item["expr"]
            for item in annotation_spec.get("symbolic_stride_dims", [])
        }
        for idx, (sidecar_dim, annotation_dim) in enumerate(zip(sidecar_stride, annotation_stride)):
            expected = annotation_dim
            if idx in symbolic_strides:
                expected = _resolve_annotation_expr(
                    symbolic_strides[idx],
                    bindings,
                    label=label,
                    dim=idx,
                    kind="stride",
                )
            if int(sidecar_dim) != int(expected):
                raise ValueError(
                    "full graph sidecar tensor stride does not match forward annotations "
                    f"for {label} dim {idx}: {sidecar_dim} != {expected}"
                )

    sidecar_device = _normalize_device_name(sidecar_spec.get("device"))
    annotation_device = _normalize_device_name(annotation_spec.get("device"))
    if sidecar_device and annotation_device and sidecar_device != annotation_device:
        raise ValueError(
            "full graph sidecar tensor device does not match forward annotations "
            f"for {label}: {sidecar_device!r} != {annotation_device!r}"
        )

    sidecar_offset = sidecar_spec.get("storage_offset")
    annotation_offset = annotation_spec.get("storage_offset")
    if (
        sidecar_offset is not None
        and annotation_offset is not None
        and int(sidecar_offset) != int(annotation_offset)
    ):
        raise ValueError(
            "full graph sidecar tensor storage_offset does not match forward annotations "
            f"for {label}: {sidecar_offset} != {annotation_offset}"
        )


def _validate_sidecar_inputs(
    sidecar_inputs: list[dict[str, Any]] | None,
    annotation_inputs: list[dict[str, Any]],
) -> None:
    if sidecar_inputs is None:
        return
    if annotation_inputs and len(sidecar_inputs) != len(annotation_inputs):
        raise ValueError(
            "full graph sidecar input count does not match forward annotations: "
            f"{len(sidecar_inputs)} != {len(annotation_inputs)}"
        )
    symbol_bindings = {}
    explicit_symbols = set()

    def bind_symbol(symbol: str, value: int) -> None:
        if symbol in symbol_bindings and symbol_bindings[symbol] != value:
            raise ValueError(
                "full graph sidecar has conflicting values for symbolic "
                f"dimension {symbol}: {symbol_bindings[symbol]} != {value}"
            )
        symbol_bindings[symbol] = value

    for sidecar_spec, annotation_spec in zip(sidecar_inputs, annotation_inputs):
        expr = annotation_spec.get("symbolic_expr")
        if (
            annotation_spec.get("kind") == "symint"
            and isinstance(expr, str)
            and re.fullmatch(r"[A-Za-z_]\w*", expr)
        ):
            bind_symbol(
                expr,
                int(
                    sidecar_spec.get("value", annotation_spec.get("value", 1))
                ),
            )
            explicit_symbols.add(expr)
    for sidecar_spec, annotation_spec in zip(sidecar_inputs, annotation_inputs):
        if (
            sidecar_spec.get("kind") != "tensor"
            or annotation_spec.get("kind") != "tensor"
        ):
            continue
        sidecar_shape = sidecar_spec.get("shape")
        if sidecar_shape is None:
            continue
        for item in annotation_spec.get("symbolic_dims", []):
            symbol = item.get("symbol")
            dim = int(item["dim"])
            if (
                isinstance(symbol, str)
                and re.fullmatch(r"[A-Za-z_]\w*", symbol)
                and dim < len(sidecar_shape)
            ):
                value = int(sidecar_shape[dim])
                if symbol not in explicit_symbols:
                    bind_symbol(symbol, value)

    for idx, sidecar_spec in enumerate(sidecar_inputs):
        if idx >= len(annotation_inputs):
            break
        annotation_spec = annotation_inputs[idx]
        sidecar_kind = sidecar_spec.get("kind")
        annotation_kind = annotation_spec.get("kind")
        if sidecar_kind != annotation_kind and not (
            sidecar_kind == "generator" and annotation_kind == "scalar"
        ):
            raise ValueError(
                "full graph sidecar input kind does not match forward annotations "
                f"for input {idx}: {sidecar_kind!r} != {annotation_kind!r}"
            )
        sidecar_name = sidecar_spec.get("name")
        annotation_name = annotation_spec.get("name")
        if sidecar_name and annotation_name and sidecar_name != annotation_name:
            raise ValueError(
                "full graph sidecar input name does not match forward annotations "
                f"for input {idx}: {sidecar_name!r} != {annotation_name!r}"
            )
        if sidecar_kind != "tensor":
            continue
        _validate_tensor_sidecar_spec(
            sidecar_spec,
            annotation_spec,
            str(sidecar_name or idx),
            symbol_bindings,
        )


def _validate_sidecar_tensor_attrs(
    sidecar_attrs: dict[str, dict[str, Any]] | None,
    annotation_attrs: dict[str, dict[str, Any]],
) -> None:
    if sidecar_attrs is None:
        return
    for attr_name, sidecar_spec in sidecar_attrs.items():
        annotation_spec = annotation_attrs.get(attr_name)
        if annotation_spec is None:
            continue
        _validate_tensor_sidecar_spec(sidecar_spec, annotation_spec, attr_name)


_MODE_NAMES = {"infer", "train"}


def infer_full_graph_source(path: str | Path) -> dict[str, Any]:
    """Infer model/microbenchmark identity from repros/models layout."""
    path = Path(path)
    parts = path.parts
    try:
        models_idx = next(
            i for i in range(len(parts) - 1)
            if parts[i] == "repros" and parts[i + 1] == "models"
        )
        rel_parts = parts[models_idx + 2:]
    except StopIteration:
        rel_parts = parts

    if len(rel_parts) < 3:
        return {
            "kind": "unknown",
            "suite": None,
            "mode": None,
            "model": path.parent.name,
            "graph": path.name,
        }

    suite = rel_parts[0]
    label_parts = list(rel_parts[1:-1])
    mode = None
    if label_parts and label_parts[0] in _MODE_NAMES:
        mode = label_parts.pop(0)
    model = Path(*label_parts).as_posix() if label_parts else path.parent.name
    return {
        "kind": "microbenchmark" if suite == "genai" else "model",
        "suite": suite,
        "mode": mode,
        "model": model,
        "graph": path.name,
    }


def result_metadata(definition: FullGraphDefinition) -> dict[str, Any]:
    sidecar = definition.metadata.get("sidecar", {})
    constraints = {
        "inputs": definition.input_specs,
        "tensor_attrs": definition.tensor_attrs,
    }
    if sidecar.get("outputs") is not None:
        constraints["outputs"] = sidecar["outputs"]

    return {
        "schema_version": 1,
        "path": str(definition.path),
        "source": definition.metadata["source"],
        "constraints": constraints,
        "constraints_source": "sidecar" if sidecar else "annotations",
    }


def _strip_runnable_boilerplate(content: str) -> str:
    content = content.replace("class <lambda>", "class Repro")
    content = re.sub(
        r"^if __name__\s*==\s*['\"]__main__['\"]:\s*\n(?:(?:[ \t]+.*)?\n)*",
        "",
        content,
        flags=re.MULTILINE,
    )
    return re.sub(r"^mod\s*=\s*Repro\(\)\s*$", "", content, flags=re.MULTILINE)


def load_graph_class_from_source(
    content: str,
    path: str | Path,
    *,
    custom_ops: frozenset[str] = frozenset(),
    user_triton: Any = None,
):
    import builtins
    import torch
    import torch.fx as fx
    import torch.nn
    from replay_dependencies import BUILTIN_OP_NAMESPACES, COLLECTIVE_OP_NAMESPACES

    tree = ast.parse(content, filename=str(path))
    graph_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(ast.unparse(base) in {"Module", "torch.nn.Module"} for base in node.bases)
    ]
    if len(graph_classes) != 1:
        raise ValueError(
            f"{path} must define exactly one torch.nn.Module graph class"
        )
    graph_class = graph_classes[0]
    # With a resolved bundle (user_triton.ResolvedBundle) covering every
    # user-Triton HOP call, those calls launch the resolved kernels instead;
    # otherwise they stay, and full_graph_calls_user_triton refuses the graph.
    user_triton_launcher = None
    if user_triton is not None:
        from user_triton import LAUNCHER_NAME, make_launcher, rewrite_hop_calls

        if rewrite_hop_calls(graph_class, user_triton):
            user_triton_launcher = make_launcher(user_triton)
    if (
        len(graph_class.bases) != 1
        or ast.unparse(graph_class.bases[0]) not in {"Module", "torch.nn.Module"}
        or graph_class.decorator_list
        or graph_class.keywords
    ):
        raise ValueError(
            f"{path} graph class must have one Module base and no decorators or keywords"
        )

    allowed_calls = {
        "device",
        "self.register_buffer",
        "super",
        "super().__init__",
        "tensor",
        "torch.sym_float",
        "torch.sym_max",
        "torch.sym_min",
        "torch.sym_not",
        "torch.sym_sum",
    }
    if user_triton_launcher is not None:
        allowed_calls.add(LAUNCHER_NAME)
    allowed_operator_namespaces = BUILTIN_OP_NAMESPACES | COLLECTIVE_OP_NAMESPACES | {
        "ads_mkl",
        "aten",
        "efficient_module_suite",
        "ems_mxfp8",
        "fb",
        "fbgemm",
        "flash_attn",
        "higher_order",
        "prims",
    }
    registered_ops = set()
    # Operators such as aten::__rshift__ have dunder names; only these are
    # exempt from the dunder-attribute check, only as torch.ops.<ns>.<op>, and
    # only when that resolves to the operator (not a namespace object method).
    dunder_operators = set()
    for op in torch._C._dispatch_get_all_op_names():
        namespace, _, operator = op.partition("::")
        if namespace in allowed_operator_namespaces:
            packet_name = operator.split(".")[0]
            if packet_name.startswith("__") and isinstance(
                getattr(getattr(torch.ops, namespace), packet_name, None),
                torch._ops.OpOverloadPacket,
            ):
                dunder_operators.add(f"torch.ops.{namespace}.{packet_name}")
            continue
        name = f"torch.ops.{namespace}.{operator}"
        registered_ops.add(name)
        registered_ops.add(f"torch.ops.{namespace}.{operator.split('.')[0]}")
        if "." not in operator:
            registered_ops.add(name + ".default")
    custom_ops = custom_ops | frozenset(registered_ops)
    custom_namespaces = {op.split(".")[2] for op in custom_ops}

    def is_operator(parts: list[str]) -> bool:
        return (
            4 <= len(parts) <= 5
            and parts[:2] == ["torch", "ops"]
            and (parts[2] in allowed_operator_namespaces or ".".join(parts) in custom_ops)
        )
    allowed_torch_attributes = {
        *(f"torch.{name}" for name in _DTYPE_SHORT_NAMES.values()),
        "torch.contiguous_format",
        "torch.ops",
        "torch.strided",
        "torch.sym_float",
        "torch.sym_max",
        "torch.sym_min",
        "torch.sym_not",
        "torch.sym_sum",
        "torch.uint8",
    }
    forbidden_nodes = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.DictComp,
        ast.For,
        ast.FunctionDef,
        ast.GeneratorExp,
        ast.Global,
        ast.If,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.ListComp,
        ast.Match,
        ast.Nonlocal,
        ast.SetComp,
        ast.Try,
        ast.While,
        ast.With,
        ast.Yield,
        ast.YieldFrom,
    )
    # Device types the graph's __init__ may first use (see the warm-up below).
    device_types = {"cpu"}
    for member in graph_class.body:
        if isinstance(member, ast.Expr) and isinstance(member.value, ast.Constant):
            continue
        if not isinstance(member, ast.FunctionDef):
            raise ValueError(f"{path} graph class contains executable class-level code")
        if member.decorator_list:
            raise ValueError(f"{path} graph methods cannot use decorators")
        defaults = [*member.args.defaults, *member.args.kw_defaults]
        if any(
            default is not None and not isinstance(default, ast.Constant)
            for default in defaults
        ):
            raise ValueError(f"{path} graph methods require literal defaults")
        local_names = {
            arg.arg
            for arg in (
                *member.args.posonlyargs,
                *member.args.args,
                *member.args.kwonlyargs,
            )
        }
        if member.args.vararg is not None:
            local_names.add(member.args.vararg.arg)
        if member.args.kwarg is not None:
            local_names.add(member.args.kwarg.arg)
        local_names.update(
            node.id
            for node in ast.walk(member)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        parents = {
            child: parent
            for parent in ast.walk(member)
            for child in ast.iter_child_nodes(parent)
        }
        # fx_graph_runnable writes buffers whose values were not saved as
        # self.register_buffer(name, torch.randint(...)) or torch.randn(...)
        # (size and dtype only), with .to("<device>") for non-CPU buffers;
        # allow exactly that form.
        buffer_factory_nodes: set[ast.AST] = set()
        for node in ast.walk(member):
            if not (
                member.name == "__init__"
                and isinstance(node, ast.Call)
                and ast.unparse(node.func) == "self.register_buffer"
                and len(node.args) == 2
            ):
                continue
            value = node.args[1]
            to_device = None
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "to"
                and len(value.args) == 1
                and not value.keywords
                and isinstance(value.args[0], ast.Constant)
                and isinstance(value.args[0].value, str)
            ):
                to_device = value
                value = value.func.value
            if not (
                isinstance(value, ast.Call)
                and ast.unparse(value.func) in {"torch.randint", "torch.randn"}
                and "torch" not in local_names
                and not any(isinstance(arg, ast.Starred) for arg in value.args)
                and all(keyword.arg in {"size", "dtype"} for keyword in value.keywords)
            ):
                continue
            buffer_factory_nodes.update((value, value.func))
            if to_device is not None:
                try:
                    device_type = torch.device(to_device.args[0].value).type
                except RuntimeError as e:
                    raise ValueError(f"{path} graph buffer names an invalid device") from e
                if device_type not in {"cpu", "cuda"}:
                    raise ValueError(
                        f"{path} graph buffer names unsupported device type "
                        f"{device_type!r}"
                    )
                buffer_factory_nodes.update((to_device, to_device.func))
                device_types.add(device_type)
        for node in ast.walk(member):
            if node is not member and isinstance(node, forbidden_nodes):
                raise ValueError(
                    f"{path} graph method contains unsupported {type(node).__name__}"
                )
            if isinstance(node, ast.Name) and node.id in {
                "__builtins__",
                "__import__",
                "compile",
                "eval",
                "exec",
                "globals",
                "locals",
                "open",
            }:
                raise ValueError(f"{path} graph method references unsafe name {node.id!r}")
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id not in local_names
                and node.id
                not in {"device", "inf", "nan", "super", "tensor", "torch"}
                and not (user_triton_launcher is not None and node.id == LAUNCHER_NAME)
            ):
                raise ValueError(
                    f"{path} graph method references unsupported global {node.id!r}"
                )
            if (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("__")
                and ast.unparse(node) != "super().__init__"
                and not (
                    ast.unparse(node) in dunder_operators
                    and "torch" not in local_names
                )
            ):
                raise ValueError(
                    f"{path} graph method references unsafe attribute {node.attr!r}"
                )
            if isinstance(node, ast.Attribute) and node not in buffer_factory_nodes:
                attribute_name = ast.unparse(node)
                parts = attribute_name.split(".")
                is_operator_attribute = (
                    len(parts) >= 2
                    and parts[:2] == ["torch", "ops"]
                    and (
                        len(parts) == 2
                        or (
                            3 <= len(parts) <= 5
                            and (
                                parts[2] in allowed_operator_namespaces
                                or len(parts) == 3 and parts[2] in custom_namespaces
                                or is_operator(parts)
                            )
                        )
                    )
                )
                if (
                    attribute_name.startswith("torch.")
                    and attribute_name not in allowed_torch_attributes
                    and not is_operator_attribute
                ):
                    raise ValueError(
                        f"{path} graph method references unsupported torch "
                        f"attribute {attribute_name!r}"
                    )
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id == "torch"
            ):
                outermost: ast.AST = node
                while (
                    isinstance(parents.get(outermost), ast.Attribute)
                    and parents[outermost].value is outermost
                ):
                    outermost = parents[outermost]
                torch_name = ast.unparse(outermost)
                parent = parents.get(outermost)
                is_call_target = (
                    isinstance(parent, ast.Call) and parent.func is outermost
                )
                call_parts = torch_name.split(".")
                is_operator_call = (
                    is_call_target
                    and is_operator(call_parts)
                )
                is_operator_argument = (
                    not is_call_target
                    and is_operator(call_parts)
                    and isinstance(parent, ast.Call)
                    and (
                        outermost in parent.args
                        or any(
                            keyword.value is outermost for keyword in parent.keywords
                        )
                    )
                )
                if (
                    torch_name not in allowed_torch_attributes
                    or torch_name
                    in {"torch.ops", "torch.sym_float", "torch.sym_not", "torch.sym_sum"}
                    and not is_call_target
                ) and not (
                    is_operator_call
                    or is_operator_argument
                    or outermost in buffer_factory_nodes
                ):
                    raise ValueError(
                        f"{path} graph method references unsupported torch value "
                        f"{torch_name!r}"
                    )
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                raise ValueError(
                    f"{path} graph method cannot mutate attributes or subscripts"
                )
            if isinstance(node, ast.Call):
                call_name = ast.unparse(node.func)
                call_parts = call_name.split(".")
                if (
                    call_name not in allowed_calls
                    and not is_operator(call_parts)
                    and node not in buffer_factory_nodes
                ):
                    raise ValueError(
                        f"{path} graph method contains unsupported call {call_name!r}"
                    )

    exec_globals: dict[str, Any] = {
        "__builtins__": {
            "__build_class__": builtins.__build_class__,
            "super": builtins.super,
        },
        "__name__": "better_benchmark_graph",
        "device": torch.device,
        "fx": fx,
        "inf": math.inf,
        "nan": float("nan"),
        "tensor": torch.tensor,
        "torch": torch,
    }
    if user_triton_launcher is not None:
        exec_globals[LAUNCHER_NAME] = user_triton_launcher
    for attr in dir(torch.nn):
        if not attr.startswith("_"):
            exec_globals[attr] = getattr(torch.nn, attr)
    class_module = ast.fix_missing_locations(
        ast.Module(body=[graph_class], type_ignores=[])
    )
    # The first torch.tensor call on a device type imports modules (torch's
    # storage types, CUDA's lazy init), which the restricted builtins below
    # cannot do; graph __init__s may create buffers on any device they name.
    # Only __init__ runs at load: warming up a device forward() merely names
    # would initialize CUDA on CPU-only hosts and in parents that later fork.
    device_types |= {
        keyword.value.value
        for member in graph_class.body
        if isinstance(member, ast.FunctionDef) and member.name == "__init__"
        for node in ast.walk(member)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "device"
        for keyword in node.keywords
        if keyword.arg == "type" and isinstance(keyword.value, ast.Constant)
    }
    for device_type in sorted(device_types):
        try:
            torch.tensor(0, device=device_type)
        except (AssertionError, RuntimeError):
            # No such device here; __init__ fails on it either way.
            pass
    exec(compile(class_module, str(path), "exec"), exec_globals)
    return exec_globals[graph_class.name]


def load_full_graph_definition(graph_path: str | Path) -> FullGraphDefinition:
    import torch
    import torch.fx as fx
    import torch.nn
    import torch._inductor.inductor_prims  # noqa: F401 - registers inductor prims.

    path = Path(graph_path)
    raw_content = path.read_text()
    content = _strip_runnable_boilerplate(raw_content)
    sidecar = load_full_graph_sidecar(path)
    annotation_inputs = parse_full_graph_inputs(raw_content)
    annotation_attrs = parse_full_graph_tensor_attrs(raw_content)
    _validate_sidecar_inputs(sidecar.get("inputs"), annotation_inputs)
    _validate_sidecar_tensor_attrs(sidecar.get("tensor_attrs"), annotation_attrs)

    custom_ops = set()
    for entry in (sidecar.get("replay_dependencies") or {}).get("custom_ops", []):
        op = entry["op"]
        if not isinstance(op, str) or op.count("::") != 1:
            raise ValueError("invalid replay dependency operator name")
        name = "torch.ops." + op.replace("::", ".")
        if len(name.split(".")) not in {4, 5}:
            raise ValueError("invalid replay dependency operator name")
        custom_ops.add(name)
        custom_ops.add(".".join(name.split(".")[:4]))
    from user_triton import resolve_bundle

    graph_cls = load_graph_class_from_source(
        content,
        path,
        custom_ops=frozenset(custom_ops),
        user_triton=resolve_bundle(sidecar, scope=str(path.resolve())),
    )

    return FullGraphDefinition(
        path=path,
        graph_cls=graph_cls,
        input_specs=sidecar.get("inputs") or annotation_inputs,
        tensor_attrs=sidecar.get("tensor_attrs") or annotation_attrs,
        forward_takes_no_inputs=forward_takes_no_inputs(raw_content),
        metadata={
            "source": sidecar.get("source") or infer_full_graph_source(path),
            "sidecar": sidecar,
        },
    )


def _resolve_device(requested: str | None, default_device: str):
    import torch

    target = requested or default_device
    if target.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda:0")
    if target.startswith("cpu") or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(default_device)


def _is_integer_dtype(dtype) -> bool:
    import torch

    integer_dtypes = [
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
    ]
    for name in ("uint16", "uint32", "uint64"):
        if hasattr(torch, name):
            integer_dtypes.append(getattr(torch, name))
    return dtype in tuple(integer_dtypes)


def _is_float8_dtype(dtype) -> bool:
    import torch

    return dtype in tuple(
        getattr(torch, name)
        for name in (
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e4m3b11fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
            "float8_e8m0fnu",
        )
        if hasattr(torch, name)
    )


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= max(int(dim), 1)
    return tuple(reversed(stride))


def _effective_stride(
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    if stride is not None and len(stride) == len(shape):
        return tuple(int(dim) for dim in stride)
    return _contiguous_stride(shape)


def _validate_tensor_layout(
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
    storage_offset: int,
) -> None:
    if any(dim < 0 for dim in shape):
        raise ValueError("input shape dimensions must be non-negative")
    if storage_offset < 0:
        raise ValueError("input storage offset must be non-negative")
    if stride is not None:
        if len(stride) != len(shape):
            raise ValueError("input stride rank does not match shape rank")
        if any(step < 0 for step in stride):
            raise ValueError("input strides must be non-negative")


def _dense_storage_size(
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
    *,
    storage_offset: int = 0,
) -> int:
    storage_offset = int(storage_offset)
    if not shape:
        return storage_offset + 1
    stride = _effective_stride(shape, stride)
    return max(
        storage_offset + sum(s * (d - 1) for s, d in zip(stride, shape) if d > 1) + 1,
        1,
    )


def _view_storage(
    storage,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
    storage_offset: int,
):
    if not shape:
        return storage.as_strided((), (), storage_offset=storage_offset)
    return storage.as_strided(
        shape,
        _effective_stride(shape, stride),
        storage_offset=storage_offset,
    )


def _generation_spec(spec: dict[str, Any]) -> dict[str, Any] | None:
    gen = spec.get("gen")
    if gen is not None:
        return gen
    gen = spec.get("generator")
    if gen is not None:
        return gen
    return None


def _make_permutation_tensor(
    shape: tuple[int, ...],
    dtype,
    device,
    *,
    stride: tuple[int, ...] | list[int] | None = None,
    storage_offset: int = 0,
    size: int | None = None,
):
    import torch

    logical_numel = _numel(shape)
    size = int(size if size is not None else logical_numel)
    if size == logical_numel:
        values = torch.randperm(size, dtype=torch.int64, device=device).reshape(shape)
    elif logical_numel % size == 0:
        rows = logical_numel // size
        values = torch.stack(
            [torch.randperm(size, dtype=torch.int64, device=device) for _ in range(rows)],
            dim=0,
        ).reshape(shape)
    else:
        raise ValueError(
            "permutation generator needs size == numel or size to divide numel, "
            f"got size={size}, numel={logical_numel}"
        )
    values = values.to(dtype)
    if stride is None and storage_offset == 0:
        return values
    storage = torch.empty(
        (_dense_storage_size(shape, stride, storage_offset=storage_offset),),
        dtype=dtype,
        device=device,
    )
    out = _view_storage(storage, shape, stride, storage_offset)
    out.copy_(values)
    return out


def _make_exact_tensor_from_spec(
    spec: dict[str, Any],
    *,
    dtype,
    device,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
    storage_offset: int,
):
    import torch

    if not spec.get("exact") or "data" not in spec:
        return None
    data = spec.get("data") or []
    source_dtype = torch.float32 if _is_float8_dtype(dtype) else dtype
    logical = torch.tensor(data, dtype=source_dtype, device=device).to(dtype)
    logical = logical.reshape(shape if shape else ())
    if stride is None and storage_offset == 0:
        return logical
    storage = torch.empty(
        (_dense_storage_size(shape, stride, storage_offset=storage_offset),),
        dtype=dtype,
        device=device,
    )
    out = _view_storage(storage, shape, stride, storage_offset)
    out.copy_(logical)
    return out


def _dtype_iinfo_max(dtype) -> int | None:
    """Return the max representable value for an integer dtype, or None for non-int."""
    import torch

    try:
        return torch.iinfo(dtype).max
    except (TypeError, RuntimeError):
        return None


def _randint_storage(
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
    storage_offset: int,
    dtype,
    device,
    low: int,
    high: int,
):
    import torch

    # Clamp high to dtype max+1 to avoid overflow (e.g. int8 max is 127)
    dtype_max = _dtype_iinfo_max(dtype)
    if dtype_max is not None and high > dtype_max + 1:
        high = dtype_max + 1
    if high <= low:
        high = low + 1
    storage_size = _dense_storage_size(shape, stride, storage_offset=storage_offset)
    storage = torch.randint(low, high, (storage_size,), dtype=dtype, device=device)
    return _view_storage(storage, shape, stride, storage_offset)


def _randn_storage(
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
    storage_offset: int,
    dtype,
    device,
):
    import torch

    storage_size = _dense_storage_size(shape, stride, storage_offset=storage_offset)
    if _is_float8_dtype(dtype):
        storage = torch.randn((storage_size,), dtype=torch.float32, device=device).to(dtype)
    else:
        storage = torch.randn((storage_size,), dtype=dtype, device=device)
    return _view_storage(storage, shape, stride, storage_offset)


def make_tensor_from_spec(spec: dict[str, Any], *, default_device: str = "cuda"):
    import torch

    dtype = _torch_dtype(spec.get("dtype", "float32"))
    shape = tuple(spec.get("shape", ()))
    stride = spec.get("stride")
    dev = _resolve_device(spec.get("device"), default_device)
    storage_offset = int(spec.get("storage_offset", 0) or 0)
    _validate_tensor_layout(shape, stride, storage_offset)
    generator = _generation_spec(spec) or _default_generator_for_dtype(
        _normalize_dtype_name(spec.get("dtype", "float32")),
        shape,
    )

    exact = _make_exact_tensor_from_spec(
        spec,
        dtype=dtype,
        device=dev,
        shape=shape,
        stride=stride,
        storage_offset=storage_offset,
    )
    if exact is not None:
        return exact
    if generator.get("kind") in JAGGED_KINDS:
        if stride is not None and tuple(stride) not in ((), (1,)):
            raise ValueError(f"{generator['kind']} inputs must be contiguous, got stride {stride}")
        values = make_jagged_tensor(generator, shape, dtype, dev)
        if not storage_offset:
            return values
        base = torch.zeros((storage_offset + values.numel(),), dtype=dtype, device=dev)
        base[storage_offset:] = values
        return _view_storage(base, shape, (1,), storage_offset)
    if spec.get("requires_exact"):
        # Integer tensor constants without exact data: use zeros as a safe
        # fallback. Zero is always a valid index for indexing ops.
        import warnings

        name = spec.get("name", "<unnamed>")
        warnings.warn(
            f"Full graph tensor attribute {name!r} requires exact data but none "
            "available; using zeros as fallback. Results may not be numerically "
            "correct but the graph will be runnable for benchmarking.",
            stacklevel=2,
        )
        storage_size = _dense_storage_size(shape, stride, storage_offset=storage_offset)
        base = torch.zeros((storage_size,), dtype=dtype, device=dev)
        return _view_storage(base, shape, stride, storage_offset)

    if generator.get("kind") == "constant":
        value = generator.get("value", 0)
        storage_size = _dense_storage_size(shape, stride, storage_offset=storage_offset)
        base = torch.full((storage_size,), value, dtype=dtype, device=dev)
        return _view_storage(base, shape, stride, storage_offset)
    if generator.get("kind") == "permutation":
        return _make_permutation_tensor(
            shape,
            dtype,
            dev,
            stride=stride,
            storage_offset=storage_offset,
            size=generator.get("size"),
        )
    if generator.get("kind") == "runs":
        from repro_harness import _make_runs_input

        return _make_runs_input(shape, _effective_stride(shape, stride), generator,
                                dtype, dev, storage_offset=storage_offset)
    if generator.get("kind") == "index":
        low = int(generator.get("low", 0))
        high = int(generator.get("high", 100))
        if _is_integer_dtype(dtype):
            return _randint_storage(
                shape=shape,
                stride=stride,
                storage_offset=storage_offset,
                dtype=dtype,
                device=dev,
                low=low,
                high=high,
            )
        if high <= low:
            high = low + 1
        storage_size = _dense_storage_size(shape, stride, storage_offset=storage_offset)
        storage = torch.randint(
            low,
            high,
            (storage_size,),
            dtype=torch.int64,
            device=dev,
        ).to(dtype)
        return _view_storage(storage, shape, stride, storage_offset)
    elif dtype == torch.bool:
        return _randint_storage(
            shape=shape,
            stride=stride,
            storage_offset=storage_offset,
            dtype=torch.bool,
            device=dev,
            low=0,
            high=2,
        )
    elif _is_integer_dtype(dtype):
        high = max([int(dim) for dim in shape], default=100)
        return _randint_storage(
            shape=shape,
            stride=stride,
            storage_offset=storage_offset,
            dtype=dtype,
            device=dev,
            low=0,
            high=max(high, 1),
        )
    return _randn_storage(
        shape=shape,
        stride=stride,
        storage_offset=storage_offset,
        dtype=dtype,
        device=dev,
    )


def make_inputs_from_full_graph_specs(
    input_specs: list[dict[str, Any]],
    *,
    default_device: str = "cuda",
) -> tuple[Any, ...]:
    inputs: list[Any] = [None] * len(input_specs)
    alias_groups: dict[Any, list[int]] = {}
    for index, spec in enumerate(input_specs):
        if spec.get("kind") == "tensor":
            _validate_tensor_layout(
                tuple(spec.get("shape", ())),
                spec.get("stride"),
                int(spec.get("storage_offset", 0) or 0),
            )
            if spec.get("alias_group") is not None:
                alias_groups.setdefault(spec["alias_group"], []).append(index)

    for indices in alias_groups.values():
        if len(indices) == 1:
            continue
        if any(input_specs[index].get("exact") for index in indices):
            raise ValueError("exact values for aliased full graph inputs are unsupported")
        first = input_specs[indices[0]]
        dtype = _normalize_dtype_name(first.get("dtype", "float32"))
        device = _normalize_device_name(first.get("device"))
        storage_size = 1
        for index in indices:
            spec = input_specs[index]
            if _normalize_dtype_name(spec.get("dtype", "float32")) != dtype:
                raise ValueError("aliased full graph inputs must have the same dtype")
            if _normalize_device_name(spec.get("device")) != device:
                raise ValueError("aliased full graph inputs must have the same device")
            storage_size = max(
                storage_size,
                _dense_storage_size(
                    tuple(spec.get("shape", ())),
                    spec.get("stride"),
                    storage_offset=int(spec.get("storage_offset", 0) or 0),
                ),
            )
        gens = [
            _generation_spec(input_specs[index])
            or _default_generator_for_dtype(dtype, tuple(input_specs[index].get("shape", ())))
            for index in indices
        ]
        gen = gens[0]
        if all(g.get("kind") in {"index", "randint", "constant"} for g in gens):
            ranges = [g for g in gens if g.get("kind") != "constant"]
            constants = [g.get("value", 0) for g in gens if g.get("kind") == "constant"]
            low = max((int(g.get("low", 0)) for g in ranges), default=0)
            high = min((int(g.get("high", 100)) for g in ranges), default=1)
            if constants:
                value = constants[0]
                if any(c != value for c in constants) or ranges and not low <= value < high:
                    raise ValueError("aliased full graph inputs have incompatible generators")
                gen = {"kind": "constant", "value": value}
            else:
                if high <= low:
                    raise ValueError("aliased full graph inputs have no common generation range")
                gen = {"kind": "index", "low": low, "high": high}
        elif any(g != gen for g in gens[1:]):
            raise ValueError("aliased full graph inputs have incompatible generators")
        storage_spec = dict(first)
        storage_spec.update({"alias_group": None, "exact": False, "gen": gen})
        if gen.get("kind") == "permutation":
            layouts = []
            for index in indices:
                spec = input_specs[index]
                shape = tuple(spec.get("shape", ()))
                layouts.append((shape, _effective_stride(shape, spec.get("stride")),
                                int(spec.get("storage_offset", 0) or 0)))
            if any(layout != layouts[0] for layout in layouts[1:]):
                raise ValueError("aliased permutation inputs with different layouts are unsupported")
        else:
            storage_spec.update({"shape": (storage_size,), "storage_offset": 0, "stride": (1,)})
        storage_spec.pop("data", None)
        storage = make_tensor_from_spec(
            storage_spec,
            default_device=default_device,
        )
        for index in indices:
            spec = input_specs[index]
            inputs[index] = _view_storage(
                storage,
                tuple(spec.get("shape", ())),
                spec.get("stride"),
                int(spec.get("storage_offset", 0) or 0),
            )

    for index, spec in enumerate(input_specs):
        if inputs[index] is not None:
            continue
        kind = spec.get("kind")
        gen = _generation_spec(spec) or {}
        if kind == "tensor" and gen.get("kind") in JAGGED_KINDS:
            spec = {**spec, "gen": {**gen, "total_value": resolve_total(gen, input_specs)}}
        if kind == "symint":
            inputs[index] = int(spec.get("value", 32))
        elif kind == "scalar":
            inputs[index] = spec.get("value", 1)
        elif kind == "generator":
            import torch

            device_type = spec.get("device_type", "cpu")
            device_index = int(spec.get("device_index", 0))
            if device_type == "cuda" and not torch.cuda.is_available():
                device_type = "cpu"
                device_index = 0
            inputs[index] = torch.Generator(
                device=torch.device(device_type, device_index)
            )
        elif kind == "tensor":
            inputs[index] = make_tensor_from_spec(
                spec,
                default_device=default_device,
            )
        else:
            inputs[index] = 1
    return tuple(inputs)


def instantiate_full_graph(
    definition: FullGraphDefinition,
    *,
    default_device: str = "cuda",
):
    instance = definition.graph_cls()
    for attr_name, spec in definition.tensor_attrs.items():
        if hasattr(instance, attr_name):
            continue
        instance.register_buffer(
            attr_name,
            make_tensor_from_spec(spec, default_device=default_device),
        )
    return instance


def load_full_graph(
    graph_path_or_definition: str | Path | FullGraphDefinition,
    *,
    default_device: str = "cuda",
    allow_user_triton: bool = False,
):
    """Instantiate a saved full graph with inputs built from its specs.

    A graph that launches user-defined Triton kernels is refused unless
    ``allow_user_triton`` is set; set it only to trace the graph in fake mode,
    which never resolves the kernels."""
    definition = (
        graph_path_or_definition
        if isinstance(graph_path_or_definition, FullGraphDefinition)
        else load_full_graph_definition(graph_path_or_definition)
    )
    if not definition.input_specs and not definition.forward_takes_no_inputs:
        raise RuntimeError(f"{definition.path} has no parseable forward inputs")
    if not allow_user_triton and full_graph_calls_user_triton(definition):
        from user_triton import BUNDLE_KEY, RESOLVER_ENV

        unresolved = ""
        if (definition.metadata or {}).get("sidecar", {}).get(BUNDLE_KEY):
            unresolved = (
                f" (its kernel bundle did not resolve here: {RESOLVER_ENV} is "
                "unset or a kernel changed since capture)"
            )
        raise UserTritonKernelReplayError(
            f"{definition.path} launches user-defined Triton kernels by "
            "kernel_side_table index, which only the capturing process can "
            f"resolve; replaying it here would fail or run other kernels{unresolved}"
        )
    instance = instantiate_full_graph(definition, default_device=default_device)
    inputs = make_inputs_from_full_graph_specs(
        definition.input_specs,
        default_device=default_device,
    )
    return instance, inputs, definition


def tensor_bytes(value: Any) -> int:
    import torch

    seen: set[tuple[int, int, int]] = set()

    def visit(obj: Any) -> int:
        if torch.is_tensor(obj):
            key = (
                obj.untyped_storage().data_ptr(),
                obj.untyped_storage().nbytes(),
                obj.storage_offset(),
            )
            if key in seen:
                return 0
            seen.add(key)
            return obj.nelement() * obj.element_size()
        if isinstance(obj, dict):
            return sum(visit(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(visit(v) for v in obj)
        return 0

    return visit(value)
