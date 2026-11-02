"""Single-MM replay preserves alias constraints, offsets and source devices.

Simple constraints are intersected for overlapping storage spans; disjoint spans
are generated independently. Span overlap is conservative for gapped views, so
some satisfiable combinations reject instead of silently violating a constraint.
Permutation/runs aliases require identical non-overlapping layouts and generators.
Exact payloads are supported up to 4096 logical elements per input. Legacy replay
is unchanged; the optional benchmark adapter patches it only for its own call.
"""

from __future__ import annotations

from itertools import product
import json
import math
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

from input_codec import spec_from_compact
import repro_harness as _harness

_BASE_MAKE_INPUTS = _harness.make_inputs_from_config
_SIMPLE = {"index", "randint", "constant", "positive"}
_COMPLEX = {"permutation", "runs"}


def decode_point(point: dict) -> dict:
    return {
        **point,
        "inputs": [
            spec_from_compact(s) if isinstance(s, list) else s for s in point["inputs"]
        ],
    }


def _int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a concrete integer")
    return value


def _stride(shape):
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * max(shape[i + 1], 1)
    return st


def _layout(spec):
    shape = [_int(d, "shape dimension") for d in spec["shape"]]
    st = [_int(s, "stride") for s in (spec.get("stride") or _stride(shape))]
    off = _int(spec.get("storage_offset", 0), "storage_offset")
    if (
        len(st) != len(shape)
        or off < 0
        or any(d < 0 for d in shape)
        or any(s < 0 for s in st)
    ):
        raise ValueError("invalid shape, stride or storage offset")
    span = (
        None
        if 0 in shape
        else (off, off + 1 + sum((d - 1) * s for d, s in zip(shape, st)))
    )
    return shape, st, off, span


def _size(spec):
    _shape, _st, off, span = _layout(spec)
    return max(1, span[1] if span is not None else off + 1)


def _dtype(spec):
    return getattr(torch, spec["dtype"].removeprefix("torch."))


def _generator(spec):
    gen = _harness._generation_spec(spec) or {}
    if not isinstance(gen, dict) or gen.get("kind") not in _SIMPLE | _COMPLEX | {
        None,
        "randn",
    }:
        raise ValueError("unsupported replay generator")
    if spec.get("exact") and "data" not in spec:
        raise ValueError("exact input has no payload")
    return gen


def _cast_scalar(value, dtype):
    """Validate payloads using tiny CPU scalars, including under FakeTensorMode."""
    if not isinstance(value, (bool, int, float, complex)):
        raise ValueError("payload/constant must be numeric")
    try:
        with unset_fake_temporarily():
            tensor = torch.tensor(value, dtype=dtype, device="cpu")
            return tensor.item(), tuple(tensor.reshape(1).view(torch.uint8).tolist())
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "payload/constant cannot be represented by its dtype"
        ) from error


def _integer_bounds(dtype):
    if dtype == torch.bool:
        return 0, 2
    try:
        info = torch.iinfo(dtype)
    except TypeError:
        return None
    return info.min, info.max + 1


def _bounds(gen, dtype):
    lo, hi = _int(gen.get("low", 0), "low"), _int(gen.get("high", 100), "high")
    if hi <= lo:
        raise ValueError("empty index/randint range")
    domain = _integer_bounds(dtype)
    if domain is not None:
        lo, hi = max(lo, domain[0]), min(hi, domain[1])
    elif dtype.is_floating_point:
        # Casting a sampled integer must not round it outside the requested range.
        exact_limit = int(2 / torch.finfo(dtype).eps)
        if lo < -exact_limit or hi - 1 > exact_limit:
            raise ValueError(
                "floating index range exceeds exactly representable integers"
            )
    else:
        raise ValueError("index/randint requires a real numeric dtype")
    # Sampling is performed in int64 before conversion to the recorded dtype.
    i64 = torch.iinfo(torch.int64)
    if hi <= lo or lo < i64.min or hi > i64.max:
        raise ValueError("index/randint range is not representable")
    return lo, hi


def _positive_floor(gen, dtype):
    value = gen.get("minimum", 0.1)
    if (
        not dtype.is_floating_point
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            "positive generation needs a finite positive minimum and floating dtype"
        )
    floor, _key = _cast_scalar(value, dtype)
    try:
        if floor < value:
            with unset_fake_temporarily():
                scalar = torch.tensor(floor, dtype=dtype, device="cpu")
                floor = torch.nextafter(
                    scalar, torch.full_like(scalar, math.inf)
                ).item()
    except RuntimeError as error:
        raise ValueError(
            "positive minimum cannot be rounded safely for this dtype"
        ) from error
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("positive minimum is not representable")
    return floor


def _flatten(value):
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _flatten(child)]
    return [value]


def _exact_slots(spec, dtype):
    shape, st, off, _span = _layout(spec)
    if math.prod(shape) > 4096:
        raise ValueError("exact payload exceeds 4096 logical elements")
    values = _flatten(spec["data"])
    if len(values) != math.prod(shape):
        raise ValueError("exact payload size mismatch")
    indices = product(*(range(d) for d in shape)) if shape else [()]
    slots = {}
    for index, value in zip(indices, values):
        address = off + sum(i * s for i, s in zip(index, st))
        scalar = _cast_scalar(value, dtype)
        if address in slots and slots[address][1] != scalar[1]:
            raise ValueError("exact payload conflicts in an overlapping view")
        slots[address] = scalar
    return slots


def _write_exact(storage, slots):
    if slots:
        indices = torch.tensor(list(slots), dtype=torch.int64, device=storage.device)
        values = torch.tensor(
            [value for value, _key in slots.values()],
            dtype=storage.dtype,
            device=storage.device,
        )
        storage.index_copy_(0, indices, values)


def _components(members):
    """Connected bounding spans of constrained views; unconstrained views do not bridge them."""
    spans = []
    for spec in members:
        gen = _generator(spec)
        span = _layout(spec)[3]
        if span is not None and (gen.get("kind") in _SIMPLE or "data" in spec):
            spans.append((*span, spec))
    spans.sort(key=lambda item: (item[0], item[1]))
    result = []
    for start, stop, spec in spans:
        if result and start < result[-1][1]:
            result[-1][1] = max(stop, result[-1][1])
            result[-1][2].append(spec)
        else:
            result.append([start, stop, [spec]])
    return result


def _contract(members, dtype):
    bounds, floors, constants, exact = [], [], [], {}
    for spec in members:
        gen = _generator(spec)
        kind = gen.get("kind")
        if kind in {"index", "randint"}:
            bounds.append(_bounds(gen, dtype))
        elif kind == "positive":
            floors.append(_positive_floor(gen, dtype))
        elif kind == "constant":
            value = gen.get("value", 0)
            scalar = _cast_scalar(value, dtype)
            if _integer_bounds(dtype) is not None and scalar[0] != value:
                raise ValueError("constant is not representable without truncation")
            constants.append(scalar)
        if "data" in spec:
            for address, scalar in _exact_slots(spec, dtype).items():
                if address in exact and exact[address][1] != scalar[1]:
                    raise ValueError("incompatible exact alias payloads")
                exact[address] = scalar
    floor = max(floors) if floors else None
    lo = max(a for a, _b in bounds) if bounds else None
    hi = min(b for _a, b in bounds) if bounds else None
    if lo is not None and floor is not None:
        lo = max(lo, math.ceil(floor))
    if lo is not None and hi <= lo:
        raise ValueError("incompatible or unsupported overlapping generator bounds")
    if constants and any(key != constants[0][1] for _value, key in constants):
        raise ValueError("incompatible or unsupported overlapping constants")

    def check(scalar):
        value, key = scalar
        if lo is not None and (
            isinstance(value, complex) or not lo <= value < hi or value != int(value)
        ):
            raise ValueError("constant/exact payload violates index/randint bounds")
        if floor is not None and (
            isinstance(value, complex) or not math.isfinite(value) or value < floor
        ):
            raise ValueError("constant/exact payload violates positive minimum")
        if constants and key != constants[0][1]:
            raise ValueError("exact payload violates constant constraint")

    for scalar in constants + list(exact.values()):
        check(scalar)
    return lo, hi, floor, constants[0][0] if constants else None, bool(constants), exact


def _non_overlapping(shape, st):
    extent = 1
    for stride, dim in sorted((s, d) for d, s in zip(shape, st) if d > 1):
        if stride < extent:
            return False
        extent += (dim - 1) * stride
    return True


def _complex_generator(spec, dtype):
    gen = _generator(spec)
    count = math.prod(spec["shape"])
    domain = _integer_bounds(dtype)
    if domain is None:
        raise ValueError("permutation/runs generation requires an integer dtype")
    if gen["kind"] == "permutation":
        size = gen.get("size")
        size = count if size is None else _int(size, "permutation size")
        if (
            dtype == torch.bool
            or size < 0
            or (count and (size == 0 or count % size))
            or (not count and size)
        ):
            raise ValueError("invalid permutation size/dtype")
        if size and (domain[0] > 0 or size > domain[1]):
            raise ValueError("permutation values do not fit the dtype")
        return {"kind": "permutation", "size": size}
    requested_lo = _int(gen.get("low", 0), "low")
    requested_hi = _int(gen.get("high", requested_lo + 1), "high")
    lo, hi = _bounds({**gen, "low": requested_lo, "high": requested_hi}, dtype)
    if (lo, hi) != (requested_lo, requested_hi):
        raise ValueError("runs bounds do not fit the dtype")
    unique = _int(gen.get("n_unique", 1), "n_unique")
    runs = _int(gen.get("n_runs", unique), "n_runs")
    sorted_ = gen.get("sorted", False)
    if (
        not isinstance(sorted_, bool)
        or not 1 <= unique <= min(count, hi - lo)
        or not unique <= runs <= count
    ):
        raise ValueError("invalid runs cardinality")
    if sorted_ and runs != unique:
        raise ValueError("sorted runs require n_runs == n_unique")
    return {
        "kind": "runs",
        "low": lo,
        "high": hi,
        "n_unique": unique,
        "n_runs": runs,
        "sorted": sorted_,
    }


def _random_storage(typed):
    if typed.dtype.is_floating_point or typed.dtype.is_complex:
        typed.copy_(torch.randn(typed.shape, dtype=typed.dtype, device=typed.device))
    else:
        domain = _integer_bounds(typed.dtype)
        if domain is None:
            raise ValueError("unsupported replay dtype")
        typed.copy_(
            torch.randint(
                0,
                min(100, domain[1]),
                typed.shape,
                dtype=torch.int64,
                device=typed.device,
            ).to(typed.dtype)
        )


def _alias_storage(members, nbytes):
    nbytes = _int(nbytes, "alias allocation size")
    if nbytes <= 0:
        raise ValueError("missing alias allocation size")
    devices = {str(torch.device(s.get("device", "cuda"))) for s in members}
    if len(devices) != 1:
        raise ValueError("alias group spans devices")
    dtypes = {_dtype(s) for s in members}
    generators = [_generator(s) for s in members]
    constrained = any(
        g.get("kind") in _SIMPLE | _COMPLEX or "data" in s
        for g, s in zip(generators, members)
    )
    if constrained and len(dtypes) != 1:
        raise ValueError("constrained mixed-dtype aliases are unsupported")
    layouts = [_layout(s) for s in members]
    for spec, (_shape, _st, _off, span) in zip(members, layouts):
        dtype = _dtype(spec)
        if nbytes % dtype.itemsize:
            raise ValueError("alias allocation is not dtype-aligned")
        if span is not None and span[1] * dtype.itemsize > nbytes:
            raise ValueError("alias view exceeds the recorded allocation")
        if span is None and "data" in spec:
            _exact_slots(spec, dtype)
    dtype = _dtype(members[0])
    complex_ = [s for s, g in zip(members, generators) if g.get("kind") in _COMPLEX]
    contracts = []
    complex_gen = None
    if complex_:
        if any("data" in s for s in members) or any(
            g.get("kind") in _SIMPLE for g in generators
        ):
            raise ValueError(
                "exact/simple constraints with a complex alias generator are unsupported"
            )
        if any(layout[:3] != layouts[0][:3] for layout in layouts):
            raise ValueError("complex aliases require identical layouts")
        if not _non_overlapping(*layouts[0][:2]):
            raise ValueError("complex generator on overlapping/unsupported layout")
        normalized = [_complex_generator(s, dtype) for s in complex_]
        if any(g != normalized[0] for g in normalized):
            raise ValueError("incompatible complex alias generators")
        complex_gen = normalized[0]
    else:
        contracts = [
            (start, stop, _contract(specs, dtype))
            for start, stop, specs in _components(members)
        ]
    raw = torch.empty(nbytes, dtype=torch.uint8, device=next(iter(devices)))
    if len(dtypes) != 1:
        # Zero bytes are finite under every supported reinterpretation.
        raw.zero_()
        return raw
    typed = raw.view(dtype)
    _random_storage(typed)
    if complex_gen is not None:
        shape, st, off, _span = layouts[0]
        spec = {
            "kind": "tensor",
            "shape": shape,
            "stride": st,
            "dtype": str(dtype),
            "device": str(typed.device),
            "gen": complex_gen,
        }
        values = _BASE_MAKE_INPUTS({"inputs": [spec]})[0]
        typed.as_strided(shape, st, off).copy_(values)
    for start, stop, (lo, hi, floor, constant, has_constant, exact) in contracts:
        view = typed[start:stop]
        if has_constant:
            view.fill_(constant)
        elif lo is not None:
            view.copy_(
                torch.randint(
                    lo, hi, view.shape, dtype=torch.int64, device=view.device
                ).to(dtype)
            )
        elif floor is not None:
            view.copy_(view.abs().add(floor).clamp(max=torch.finfo(dtype).max))
        _write_exact(typed, exact)
    return raw


def tensor_from_exact_spec(spec):
    if "data" not in spec:
        raise ValueError("exact input has no payload")
    raw = _alias_storage([spec], _size(spec) * _dtype(spec).itemsize)
    shape, st, off, _span = _layout(spec)
    return raw.view(_dtype(spec)).as_strided(shape, st, off)


def make_inputs_from_config(point):
    config = decode_point(point)
    groups = {}
    for spec in config["inputs"]:
        if spec.get("alias_group") is not None:
            group = _int(spec["alias_group"], "alias group")
            if group < 0 or spec.get("kind", "tensor") != "tensor":
                raise ValueError("invalid alias group/member")
            groups.setdefault(group, []).append(spec)
    allocations = {}
    sizes = config.get("alias_group_nbytes") or []
    for group, members in groups.items():
        if group >= len(sizes):
            raise ValueError("missing alias allocation size")
        allocations[group] = _alias_storage(members, sizes[group])
    result = []
    for spec in config["inputs"]:
        kind = spec.get("kind", "tensor")
        if kind == "shape":
            result.append(list(spec["dims"]))
        elif kind in ("symint", "scalar"):
            result.append(spec["value"])
        elif kind == "tensor":
            raw = (
                allocations[spec["alias_group"]]
                if spec.get("alias_group") is not None
                else _alias_storage([spec], _size(spec) * _dtype(spec).itemsize)
            )
            shape, st, off, _span = _layout(spec)
            result.append(raw.view(_dtype(spec)).as_strided(shape, st, off))
        else:
            raise ValueError(f"unsupported input kind: {kind}")
    return result


def benchmark_repro_safe(repro_file, repro_cls, make_inputs_fn):
    """Optional standard runner with a call-scoped constraint-aware generator."""
    with patch.object(_harness, "make_inputs_from_config", make_inputs_from_config):
        return _harness.benchmark_repro(repro_file, repro_cls, make_inputs_fn)


def execution_hash(gm):
    """Conservative ordered FX identity: literals, input positions and output tree."""
    import hashlib
    import torch.fx as fx

    ids = {node: i for i, node in enumerate(gm.graph.nodes)}

    def encode(value):
        if isinstance(value, fx.Node):
            return ["node", ids[value]]
        if value is None:
            return ["none"]
        if type(value) in (bool, int, str):
            return [type(value).__name__, value]
        if isinstance(value, float):
            return ["float", value.hex()]
        if isinstance(value, (tuple, list)):
            return [type(value).__name__, [encode(v) for v in value]]
        if isinstance(value, dict):
            return ["dict", [[encode(k), encode(v)] for k, v in value.items()]]
        if isinstance(value, slice):
            return [
                "slice",
                encode(value.start),
                encode(value.stop),
                encode(value.step),
            ]
        if isinstance(
            value, (torch.dtype, torch.device, torch.layout, torch.memory_format)
        ):
            return [type(value).__name__, str(value)]
        if isinstance(value, torch.SymInt) and value.node.has_hint():
            return ["symint", int(value.node.hint)]
        raise TypeError(f"Unsupported FX literal: {type(value).__name__}")

    nodes = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            nodes.append(["placeholder"])
        else:
            target = (
                str(node.target)
                if isinstance(node.target, (str, torch._ops.OpOverload))
                else (node.target.__module__ + "." + node.target.__qualname__)
            )
            nodes.append([node.op, target, encode(node.args), encode(node.kwargs)])
    return hashlib.sha256(
        json.dumps(nodes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
