"""
Shared benchmark harness for canonical repros.

Each canonical repro.py imports from here instead of inlining benchmark logic.
Supports: multi-shape configs, SOL measurement, CD tuning, CUDA graph timing,
alias-aware byte accounting, Triton kernel counting, and adjusted byte counting
via dispatch-mode tracing.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

from gpu_kind import device_kind_from_name
from dynamic_shape_replay import _symbols_and_guards_for_repro


UNVERSIONED_REPRO_VERSION = 0
# v3: default inputs load from the sibling shapes.json; the inline
# `_shapes_config` string was retired (capture_hook template + the entire
# canonical corpus emit v3). Bumped here to match what capture_hook.py
# actually writes -- the v3 migration (76c4fb8c8) changed the template and
# the corpus but left this constant at 2.
CURRENT_REPRO_VERSION = 3
_REPRO_VERSION_RE = re.compile(
    r"^_repro_version\s*=\s*(\d+)\s*(?:#.*)?$",
)
_REPRO_VERSION_ASSIGN_RE = re.compile(r"^_repro_version\s*=.*$", re.MULTILINE)
_SUPPORTED_COMPILE_POLICY_KEYS = frozenset({"inductor"})
_SUPPORTED_COMPILE_POLICY_INDUCTOR_OPTIONS = frozenset(
    {"emulate_precision_casts"}
)


def parse_repro_version(source: str) -> int:
    """Return the repro format version, or 0 for legacy unversioned source."""
    assignments = _REPRO_VERSION_ASSIGN_RE.findall(source)
    if not assignments:
        return UNVERSIONED_REPRO_VERSION
    if len(assignments) > 1:
        raise ValueError("_repro_version must appear exactly once")

    match = _REPRO_VERSION_RE.match(assignments[0])
    if match is None:
        raise ValueError("_repro_version must be a top-level integer assignment")
    return int(match.group(1))


def read_repro_version(repro_path: str | Path) -> int:
    """Read the repro format version from a repro.py path."""
    return parse_repro_version(Path(repro_path).read_text())


def normalize_compile_policy(
    policy: Any,
    *,
    source: str = "compile policy",
) -> dict[str, Any]:
    """Validate and normalize one JSON-serializable compile policy."""
    if policy is None:
        return {}
    if not isinstance(policy, Mapping):
        raise TypeError(f"{source} must be a mapping")
    if not all(isinstance(key, str) for key in policy):
        raise TypeError(f"{source} keys must be strings")

    unknown_policy = set(policy) - _SUPPORTED_COMPILE_POLICY_KEYS
    if unknown_policy:
        raise ValueError(
            f"unsupported {source} keys: "
            + ", ".join(sorted(unknown_policy))
        )

    options = policy.get("inductor", {})
    if not isinstance(options, Mapping):
        raise TypeError(f"{source}['inductor'] must be a mapping")
    if not all(isinstance(key, str) for key in options):
        raise TypeError(f"{source}['inductor'] keys must be strings")
    unknown_options = (
        set(options) - _SUPPORTED_COMPILE_POLICY_INDUCTOR_OPTIONS
    )
    if unknown_options:
        raise ValueError(
            f"unsupported per-point Inductor options in {source}: "
            + ", ".join(sorted(unknown_options))
        )
    if (
        "emulate_precision_casts" in options
        and not isinstance(options["emulate_precision_casts"], bool)
    ):
        raise TypeError(
            "per-point emulate_precision_casts policy must be a bool"
        )
    return {"inductor": dict(options)} if options else {}


def compile_policy_from_config(
    shape_config: Mapping | None,
) -> dict[str, Any]:
    """Return the validated compile policy carried by one input point."""
    if shape_config is None:
        return {}
    if not isinstance(shape_config, Mapping):
        raise TypeError("shape config must be a mapping")
    return normalize_compile_policy(
        shape_config.get("compile_policy"),
        source="shape config compile_policy",
    )


def default_shape_config(repro_file: str | Path) -> dict | None:
    """Return the first input point config used by generated ``make_inputs``."""
    return next(iter(load_shape_configs(str(repro_file)).values()), None)


def compile_repro(
    repro: Any,
    *,
    compile_policy: Mapping | None = None,
    **compile_kwargs,
):
    """Compile a repro with an explicit, point-local compile policy."""
    normalized_policy = normalize_compile_policy(
        compile_policy,
        source="compile_policy",
    )
    policy_options = normalized_policy.get("inductor", {})
    backend = compile_kwargs.get("backend")
    if policy_options and backend not in (None, "inductor"):
        raise ValueError(
            "compile_policy declares Inductor options but "
            f"torch.compile backend is {backend!r}"
        )

    explicit_options = compile_kwargs.pop("options", None)
    if explicit_options is not None and not isinstance(explicit_options, Mapping):
        raise TypeError("torch.compile options must be a mapping or None")
    options = {}
    for raw_key, value in (explicit_options or {}).items():
        if not isinstance(raw_key, str):
            raise TypeError("torch.compile option names must be strings")
        # PyTorch accepts CLI-style hyphenated aliases by normalizing them
        # before applying options.  Normalize here too so an alias cannot
        # evade the policy conflict check below.
        key = raw_key.replace("-", "_")
        if key in options and options[key] != value:
            raise ValueError(
                f"conflicting torch.compile option aliases for {key!r}"
            )
        options[key] = value
    for key, value in policy_options.items():
        if key in options and options[key] != value:
            raise ValueError(
                f"compile option {key!r} conflicts with compile_policy"
            )
        options[key] = value
    if options:
        compile_kwargs["options"] = options
    return torch.compile(repro, **compile_kwargs)


@contextmanager
def preserve_compile_environment():
    """Restore process state that lazy Inductor compilation may mutate.

    ``torch.compile(options={"emulate_precision_casts": True})`` scopes the
    Inductor config correctly, but Triton's lazy compiler also selects CUDA's
    libdevice by mutating both ``TRITON_LIBDEVICE_PATH`` and a process-global
    Triton knob.  Persistent benchmark workers must restore those values at
    the workload boundary or the next repro can inherit eager-numerics state.
    """
    missing = object()
    env_name = "TRITON_LIBDEVICE_PATH"
    prior_env = os.environ.get(env_name, missing)

    triton_knobs = None
    prior_knob = missing
    try:
        from triton import knobs

        triton_knobs = knobs
        prior_knob = knobs.nvidia.libdevice_path
    except (AttributeError, ImportError):
        pass

    try:
        yield
    finally:
        if prior_env is missing:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = prior_env
        if triton_knobs is not None and prior_knob is not missing:
            triton_knobs.nvidia.libdevice_path = prior_knob


def load_shape_configs(repro_file: str, symbol_bindings: dict | None = None) -> dict:
    """Load shape configs for a canonical repro.

    Preference order:
      1. shapes.json (new format with points array) — used for new captures
      2. shapes.txt (compact T()/S() format) — the existing 1482 repros

    symbol_bindings: for DYNAMIC repros (shapes.json with a "symbols"
    table), instantiate every symbolic point at these bindings instead of
    each point's recorded ones (e.g. {"s16": 24}). Bindings are validated
    against symbol ranges + guards; violations raise. Static repros ignore
    this parameter (no symbols to bind).
    """
    repro_dir = Path(repro_file).parent

    # Prefer shapes.json (new format from wave-1 onward)
    shapes_json = repro_dir / "shapes.json"
    if shapes_json.exists():
        return _parse_shapes_json(shapes_json, symbol_bindings=symbol_bindings)

    # Fallback: shapes.txt for existing corpus
    shapes_txt = repro_dir / "shapes.txt"
    if shapes_txt.exists():
        if symbol_bindings:
            # shapes.txt has no symbols table — silently ignoring the
            # caller's bindings would bench the WRONG point.
            raise ValueError(
                f"symbol_bindings={symbol_bindings} given but {repro_dir} "
                "has only shapes.txt (no symbols table); bindings require "
                "a shapes.json with symbolic entries")
        return _parse_shapes_txt(shapes_txt)

    return {}


def _parse_shapes_json(shapes_path: Path,
                       symbol_bindings: dict | None = None) -> dict:
    """Parse shapes.json (new format with points array).

    Each point has a shape_hash, signature (T()/S() string), and models dict.
    Returns the same dict format as _parse_shapes_txt: {label: {"inputs": [specs]}}.
    Label is constructed from shape_hash + first model key for readability.
    """
    with open(shapes_path) as f:
        data = json.load(f)

    # Legacy shapes.json format (pre-points schema)
    if "configs" in data and "points" not in data:
        return data.get("configs", {})

    points = data.get("points", [])
    if not points:
        return {}

    configs = {}
    for point in points:
        signature = point.get("signature", "")
        shape_hash = point.get("shape_hash", "unknown")
        models = point.get("models", {})

        # Build label: shape_hash_first_model_key (sanitized)
        first_model = next(iter(models), "")
        # Use last component of model path for brevity
        model_short = first_model.rsplit("/", 1)[-1] if first_model else ""
        label = f"{model_short}_{shape_hash}" if model_short else shape_hash
        point_policy = (
            normalize_compile_policy(
                point["compile_policy"],
                source=f"{shapes_path} point {shape_hash} compile_policy",
            )
            if "compile_policy" in point
            else None
        )

        # Prefer the structured compact entries (input_codec) — the data.
        # Signature-eval is the fallback for points written before the
        # compact encoding existed; the string is documentation otherwise.
        compact = point.get("inputs")
        symbols = data.get("symbols") or {}
        if symbol_bindings is None and point.get("requires_binding"):
            continue
        if compact is not None and (symbols or symbol_bindings):
            # Dynamic repro: instantiate symbolic entries (at the point's
            # recorded bindings, or the caller's). Range/guard violations
            # raise loudly inside instantiate_point.
            from input_codec import instantiate_point
            compact = instantiate_point(
                point, symbols, bindings=symbol_bindings,
                guards=data.get("guards"))
        if compact is not None:  # [] is a VALID zero-input config
            # NO silent fallback: data-only points have no signature to
            # fall back to, so a decode failure here must be LOUD. (The
            # old except-pass masked input_codec missing from the editable
            # package — every standalone repro run silently saw 0 configs.)
            from input_codec import spec_from_compact
            specs = [spec_from_compact(e) for e in compact]
            cfg = {"inputs": specs}
            if symbols:
                from input_codec import _symbol_point_value
                cfg["bindings"] = dict(
                    symbol_bindings
                    or point.get("bindings")
                    or {
                        name: value
                        for name, meta in symbols.items()
                        if (value := _symbol_point_value(meta)) is not None
                    }
                )
            # The point's shape_hash travels with the config so the oracle
            # bench loop can dispatch by hash (a DYNAMIC point's shape is
            # symbolic and won't shape-match concrete inputs — the hash is the
            # exact dispatch key). The label embeds the hash for readability;
            # this is the structured copy callers should thread, not parse.
            cfg["shape_hash"] = shape_hash
            # alias_group_nbytes must travel with the config — specs
            # carrying alias_group crash generation without it
            # (adversarial review bug #1).
            if point.get("alias_group_nbytes"):
                cfg["alias_group_nbytes"] = point["alias_group_nbytes"]
            if point_policy is not None:
                cfg["compile_policy"] = point_policy
            configs[label] = cfg
            continue
        try:
            inputs = _eval_signature(signature)
            if inputs:
                cfg = {"inputs": inputs, "shape_hash": shape_hash}
                if point_policy is not None:
                    cfg["compile_policy"] = point_policy
                configs[label] = cfg
        except Exception:
            continue

    return configs


def _make_shape_eval_ns():
    """Build the eval namespace for T()/S() signature parsing."""
    _DTYPE_MAP = {
        "f32": "torch.float32", "f16": "torch.float16",
        "bf16": "torch.bfloat16", "f64": "torch.float64",
        "i64": "torch.int64", "i32": "torch.int32",
        "i16": "torch.int16", "i8": "torch.int8",
        "b8": "torch.bool", "u8": "torch.uint8",
    }

    def Index(high, low=0):
        return {"kind": "index", "low": int(low), "high": int(high)}

    def Perm(size=None):
        return {"kind": "permutation", "size": None if size is None else int(size)}

    def Runs(high, low=0, unique=1, runs=1, sorted=False):
        return {"kind": "runs", "low": int(low), "high": int(high),
                "n_unique": int(unique), "n_runs": int(runs),
                "sorted": bool(sorted)}

    def Sorted():
        return {"kind": "sorted"}

    def T(shape, dtype, stride=None, max=None, gen=None, constraint=None, index=None):
        spec = {
            "kind": "tensor",
            "shape": shape,
            "dtype": _DTYPE_MAP.get(dtype, f"torch.{dtype}"),
            "stride": list(stride) if stride else None,
            "device": "cuda",
        }
        if constraint is None:
            constraint = index
        if gen is not None:
            spec["gen"] = gen
        elif constraint == "permutation":
            spec["gen"] = Perm(max)
        elif max is not None:
            spec["gen"] = Index(max)
        if constraint is not None:
            spec["constraint"] = constraint
        if max is not None:
            spec["max_val"] = int(max)
        return spec

    def S(dims):
        return {"kind": "shape", "dims": dims}

    return {"__builtins__": {}, "T": T, "S": S, "Index": Index, "Perm": Perm, "Runs": Runs,
            "Sorted": Sorted,
            "f32": "f32", "f16": "f16", "bf16": "bf16", "f64": "f64",
            "i64": "i64", "i32": "i32", "i16": "i16", "i8": "i8",
            "b8": "b8", "u8": "u8"}


def _eval_signature(expr: str) -> list:
    """Eval a T()/S() signature string and return a list of input specs."""
    from input_codec import _assert_safe_shape_config
    ns = _make_shape_eval_ns()
    _assert_safe_shape_config(expr)  # code-injection boundary: expr is DATA
    inputs = eval(expr, ns)  # noqa: S307
    if isinstance(inputs, tuple):
        inputs = list(inputs)
    elif isinstance(inputs, dict):
        inputs = [inputs]
    return inputs if inputs else []


def _parse_shapes_txt(shapes_path: Path) -> dict:
    """Parse shapes.txt by eval'ing each line with T() and S() as constructors."""
    from input_codec import _assert_safe_shape_config
    _eval_ns = _make_shape_eval_ns()

    configs = {}
    for line in shapes_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        colon = line.find(":")
        if colon < 0:
            continue
        label = line[:colon].strip()
        expr = line[colon + 1:].strip()

        try:
            _assert_safe_shape_config(expr)  # code-injection boundary: expr is DATA
            inputs = eval(expr, _eval_ns)
            if isinstance(inputs, tuple):
                inputs = list(inputs)
            elif isinstance(inputs, dict):
                inputs = [inputs]
            if inputs:
                configs[label] = {"inputs": inputs}
        except Exception:
            continue

    return configs


def parse_shapes_config(config_str: str) -> list:
    """Parse a _shapes_config string and generate inputs. Used by repro._default_make_inputs()."""
    _DTYPE_MAP = {
        "f32": "torch.float32", "f16": "torch.float16",
        "bf16": "torch.bfloat16", "f64": "torch.float64",
        "i64": "torch.int64", "i32": "torch.int32",
        "i16": "torch.int16", "i8": "torch.int8",
        "b8": "torch.bool", "u8": "torch.uint8",
    }

    def Index(high, low=0):
        return {"kind": "index", "low": int(low), "high": int(high)}

    def Perm(size=None):
        return {"kind": "permutation", "size": None if size is None else int(size)}

    def Runs(high, low=0, unique=1, runs=1, sorted=False):
        return {"kind": "runs", "low": int(low), "high": int(high),
                "n_unique": int(unique), "n_runs": int(runs),
                "sorted": bool(sorted)}

    def Sorted():
        return {"kind": "sorted"}

    def T(shape, dtype, stride=None, max=None, gen=None, constraint=None, index=None):
        spec = {
            "kind": "tensor",
            "shape": shape,
            "dtype": _DTYPE_MAP.get(dtype, f"torch.{dtype}"),
            "stride": list(stride) if stride else None,
            "device": "cuda",
        }
        if constraint is None:
            constraint = index
        if gen is not None:
            spec["gen"] = gen
        elif constraint == "permutation":
            spec["gen"] = Perm(max)
        elif max is not None:
            spec["gen"] = Index(max)
        if constraint is not None:
            spec["constraint"] = constraint
        if max is not None:
            spec["max_val"] = int(max)
        return spec

    def S(dims):
        return {"kind": "shape", "dims": dims}

    _ns = {"__builtins__": {}, "T": T, "S": S, "Index": Index, "Perm": Perm, "Runs": Runs,
            "Sorted": Sorted,
            "torch": torch,
            "f32": "f32", "f16": "f16", "bf16": "bf16", "f64": "f64",
            "i64": "i64", "i32": "i32", "i16": "i16", "i8": "i8",
            "b8": "b8", "u8": "u8"}
    from input_codec import _assert_safe_shape_config
    _assert_safe_shape_config(config_str)  # code-injection boundary: config_str is DATA
    inputs = eval(config_str, _ns)
    if isinstance(inputs, dict):
        inputs = [inputs]
    elif isinstance(inputs, tuple):
        inputs = list(inputs)
    if not inputs and config_str.strip() not in ("()", ""):
        raise ValueError(f"parse_shapes_config produced empty result for non-empty config: {config_str[:80]}")
    return make_inputs_from_config({"inputs": inputs})


def _storage_size_for_strided(shape: list[int], stride: list[int]) -> int:
    """Compute minimum storage size needed for a strided tensor."""
    if not shape:
        return 1
    return sum((s - 1) * st for s, st in zip(shape, stride) if s > 1) + 1


def _numel(shape: list[int]) -> int:
    """Compute number of elements from a shape."""
    result = 1
    for dim in shape:
        result *= dim
    return result


def _sample_unique(low: int, high: int, k: int) -> torch.Tensor:
    """``k`` distinct int64 values in [low, high), sorted: ``k`` sorted draws
    from [0, span - k] plus 0..k-1 are strictly increasing and below span.
    Output shapes never depend on the drawn values, so this also runs under
    FakeTensorMode (the capture write gate rebuilds inputs there)."""
    draws = torch.randint(0, high - low - k + 1, (k,)).sort().values
    return draws + torch.arange(k) + low


def _make_runs_tensor(n: int, gen: dict, dtype, device) -> torch.Tensor:
    """``n`` integer ids in [low, high) with the recorded structure: ``n_unique``
    distinct values in ``n_runs`` runs, non-decreasing if ``sorted`` (one run
    per value), else with adjacent runs differing and every value appearing
    when ``n_runs >= n_unique``. Run lengths are near-equal: the counts are
    reproduced, not the run-length distribution."""
    low = int(gen.get("low", 0))
    high = max(int(gen.get("high", low + 1)), low + 1)
    if not dtype.is_floating_point and dtype != torch.bool:
        info = torch.iinfo(dtype)
        low = max(low, info.min)
        high = max(min(high, info.max + 1), low + 1)
    k = max(1, min(int(gen.get("n_unique", 1)), high - low, n))
    values = _sample_unique(low, high, k)
    if gen.get("sorted"):
        run_values = values
    else:
        runs = max(1, min(int(gen.get("n_runs", k)), n))
        if k == 1:
            picks = torch.zeros(runs, dtype=torch.int64)
        else:
            # Every value appears (a permutation first, adjacent picks
            # distinct by construction), then a walk that never repeats the
            # previous value.
            first = torch.randperm(k)[:runs]
            rest = runs - first.numel()
            if rest > 0:
                steps = torch.randint(1, k, (rest,))
                walk = (first[-1] + torch.cumsum(steps, 0)) % k
                picks = torch.cat([first, walk])
            else:
                picks = first
        run_values = values[picks]
    # The first n % runs runs are one longer. Integer repeats keep the output
    # shape data-independent (see _sample_unique).
    longer, length = n % run_values.numel(), n // run_values.numel()
    ids = torch.cat([run_values[:longer].repeat_interleave(length + 1),
                     run_values[longer:].repeat_interleave(length)])
    return ids.to(dtype=dtype, device=device)


def _layout_overlaps(shape, stride) -> bool:
    """Whether distinct logical elements of a strided layout share memory
    (e.g. a stride-0 expanded dim)."""
    dims = sorted((st, sz) for sz, st in zip(shape, stride) if sz > 1)
    extent = 1
    for st, sz in dims:
        if st < extent:
            return True
        extent = st * sz
    return False


def _make_runs_input(shape, stride, gen: dict, dtype, device,
                     storage_offset: int = 0) -> torch.Tensor:
    """A ``runs`` tensor whose structure holds in logical (row-major) order,
    the order compute_observed_stats measured it in. Overlapping layouts
    (expanded dims) can't hold arbitrary logical values, so the storage the
    view reaches is generated directly (an outer-dim expansion then repeats
    the inner runs). The ``storage_offset`` elements before the view are
    padding."""
    shape = list(shape)
    if stride and _layout_overlaps(shape, stride):
        reached = _make_runs_tensor(_storage_size_for_strided(shape, stride),
                                    gen, dtype, device)
    else:
        values = _make_runs_tensor(_numel(shape), gen, dtype, device).reshape(shape)
        if not stride and not storage_offset:
            return values
        stride = stride or values.stride()
        reached = torch.empty(_storage_size_for_strided(shape, stride),
                              dtype=dtype, device=device)
        reached.as_strided(shape, stride).copy_(values)
    if storage_offset:
        reached = torch.cat([reached.new_zeros(storage_offset), reached])
    return reached.as_strided(shape, stride, storage_offset)


def _generation_spec(spec: dict) -> dict | None:
    """Extract generation spec from a tensor spec dict.

    Handles both new-style 'gen' field and legacy 'max_val' field.
    """
    gen = spec.get("gen")
    if gen is not None:
        return gen

    if spec.get("constraint") == "permutation":
        return {"kind": "permutation", "size": spec.get("max_val")}
    if spec.get("max_val") is not None:
        return {"kind": "index", "low": 0, "high": int(spec["max_val"])}
    return None


def _make_permutation_tensor(shape, dtype, device, stride=None, size=None):
    """Create a tensor containing random permutations of [0, size).

    If `size` is smaller than the tensor numel and divides it evenly, generate
    independent row-wise permutations. This covers inverse-permutation index
    tensors with shape like [batch, heads, N] and gen=Perm(N).
    """
    logical_numel = _numel(shape)
    size = int(size if size is not None else logical_numel)
    if size == logical_numel:
        values = torch.randperm(size, dtype=dtype, device=device).reshape(shape)
    elif logical_numel % size == 0:
        rows = logical_numel // size
        values = torch.stack(
            [torch.randperm(size, dtype=dtype, device=device) for _ in range(rows)],
            dim=0,
        ).reshape(shape)
    else:
        raise ValueError(
            "permutation generator needs size == numel or size to divide numel, "
            f"got size={size}, numel={logical_numel}"
        )
    if stride:
        out = torch.empty_strided(shape, stride, dtype=dtype, device=device)
        out.copy_(values)
        return out
    return values


def _randn(size, dtype, device) -> torch.Tensor:
    # randn has no fp8 kernels; draw in fp32 and round (normal values sit well
    # inside every fp8 format's range).
    if dtype.is_floating_point and dtype.itemsize == 1:
        return torch.randn(size, dtype=torch.float32, device=device).to(dtype)
    return torch.randn(size, dtype=dtype, device=device)


def make_inputs_from_config(config: dict) -> list:
    """Create inputs from a shape config. Returns mix of tensors and shape-param lists.

    Alias groups: specs carrying "alias_group" are views of ONE storage
    (packed-qkv saved views). One buffer is allocated per group — sized by
    config["alias_group_nbytes"][g], the true capture-time allocation —
    and each member is as_strided at its own (shape, stride, offset).
    Footprint and locality then match the model instead of giving every
    view a private storage.
    """
    group_nbytes = config.get("alias_group_nbytes")
    group_storage: dict[int, tuple[torch.Tensor, torch.device]] = {}

    def _group_buffer(g: int, device, dtype=None) -> tuple[torch.Tensor, int]:
        if (not isinstance(g, int) or isinstance(g, bool) or g < 0):
            raise ValueError(f"invalid alias_group {g!r}")
        if group_nbytes is None or g >= len(group_nbytes):
            raise ValueError(
                f"alias_group {g} has no recorded nbytes "
                f"(alias_group_nbytes={group_nbytes})")
        nbytes = group_nbytes[g]
        if (not isinstance(nbytes, int) or isinstance(nbytes, bool)
                or nbytes < 0):
            raise ValueError(
                f"alias_group {g} has invalid recorded nbytes "
                f"{nbytes!r}")
        requested_device = torch.device(device)
        if g not in group_storage:
            buf = torch.empty(nbytes, dtype=torch.uint8, device=requested_device)
            if dtype is not None and dtype.is_floating_point and nbytes % dtype.itemsize == 0:
                # Raw byte noise viewed as a float dtype includes NaN/Inf bit patterns.
                typed = buf.view(dtype)
                typed.copy_(_randn(typed.shape, dtype, requested_device))
            else:
                buf.random_(0, 255)  # bit-pattern noise; views reinterpret dtype
            group_storage[g] = (buf, requested_device)
        buf, captured_device = group_storage[g]
        if requested_device != captured_device:
            raise ValueError(
                f"alias_group {g} members disagree on device: "
                f"{captured_device} != {requested_device}")
        return buf, nbytes

    def _contiguous_stride_local(shape):
        st = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            st[i] = st[i + 1] * max(int(shape[i + 1]), 1)
        return st

    result = []
    for spec in config["inputs"]:
        if spec.get("kind") == "symfloat":
            result.append(float(spec["value"]))
            continue
        if spec.get("kind") == "shape":
            result.append(spec["dims"])
            continue

        if spec.get("kind") == "symint":
            # A live symint input (dynamic-shape graph): forward() takes a
            # plain Python int, exactly as a lifted shape param takes a
            # plain list. Without this branch the spec falls through to
            # spec["shape"] and KeyErrors — the silent-drop that excluded
            # every dynamic-compilation region from the corpus.
            result.append(int(spec["value"]))
            continue

        if spec.get("kind") == "scalar":
            result.append(spec["value"])
            continue

        if spec.get("alias_group") is not None:
            shape = spec["shape"]
            dtype = getattr(torch, spec["dtype"].replace("torch.", ""))
            stride = spec.get("stride") or _contiguous_stride_local(shape)
            device = spec.get("device", "cuda")
            offset = int(spec.get("storage_offset", 0))
            buf, nbytes = _group_buffer(spec["alias_group"], device, dtype)
            element_size = torch.empty((), dtype=dtype).element_size()
            if nbytes % element_size:
                raise ValueError(
                    f"alias_group {spec['alias_group']} capacity {nbytes} "
                    f"is not divisible by {dtype} element size "
                    f"{element_size}")
            from full_graph_harness import _dense_storage_size
            required = _dense_storage_size(
                tuple(shape), stride, storage_offset=offset) * element_size
            if required > nbytes:
                raise ValueError(
                    f"alias_group {spec['alias_group']} view requires "
                    f"{required} bytes but capture recorded {nbytes}")
            typed = buf.view(dtype) if dtype != torch.uint8 else buf
            result.append(typed.as_strided(shape, stride, offset))
            continue

        shape = spec["shape"]
        dtype_str = spec["dtype"].replace("torch.", "")
        dtype = getattr(torch, dtype_str)
        stride = spec.get("stride")
        device = spec.get("device", "cuda")
        gen = _generation_spec(spec)

        if gen and gen.get("kind") == "runs":
            # Any integer dtype (incl. unsigned): ids are drawn in int64 and
            # clamped to the dtype's range.
            result.append(_make_runs_input(shape, stride, gen, dtype, device))
            continue

        if dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
            if gen and gen.get("kind") == "constant":
                # Single safe value (maxpool window-center offsets etc).
                if stride:
                    numel = _storage_size_for_strided(shape, stride)
                    storage = torch.full((numel,), int(gen.get("value", 0)),
                                         dtype=dtype, device=device)
                    result.append(storage.as_strided(shape, stride))
                else:
                    result.append(torch.full(shape, int(gen.get("value", 0)),
                                             dtype=dtype, device=device))
                continue
            if gen and gen.get("kind") == "permutation":
                result.append(
                    _make_permutation_tensor(
                        shape,
                        dtype=dtype,
                        device=device,
                        stride=stride,
                        size=gen.get("size"),
                    )
                )
                continue

            if gen and gen.get("kind") == "offsets":
                # Segment offsets (e.g. _embedding_bag): non-decreasing,
                # first element 0, bounded by high (= len(indices)).
                hi = max(int(gen.get("high", 1)), 1)
                vals, _ = torch.sort(
                    torch.randint(0, hi, shape, dtype=dtype, device=device))
                flat = vals.reshape(-1)
                if flat.numel():
                    flat[0] = 0
                result.append(vals)
                continue

            low = int(gen.get("low", 0)) if gen else 0
            hi = int(gen.get("high", 100)) if gen else 100
            if hi <= low:
                hi = low + 1
            if stride:
                numel = _storage_size_for_strided(shape, stride)
                storage = torch.randint(low, hi, (numel,), dtype=dtype, device=device)
                result.append(storage.as_strided(shape, stride))
            else:
                result.append(torch.randint(low, hi, shape, dtype=dtype, device=device))
        elif dtype == torch.bool:
            if stride:
                numel = _storage_size_for_strided(shape, stride)
                storage = torch.randint(0, 2, (numel,), dtype=torch.bool, device=device)
                result.append(storage.as_strided(shape, stride))
            else:
                result.append(torch.randint(0, 2, shape, dtype=torch.bool, device=device))
        else:
            if gen and gen.get("kind") == "index":
                low = int(gen.get("low", 0))
                hi = int(gen.get("high", 100))
                if hi <= low:
                    hi = low + 1
                if stride:
                    numel = _storage_size_for_strided(shape, stride)
                    storage = torch.randint(low, hi, (numel,), dtype=torch.int64, device=device)
                    result.append(storage.to(dtype).as_strided(shape, stride))
                else:
                    result.append(torch.randint(low, hi, shape, dtype=torch.int64, device=device).to(dtype))
            elif gen and gen.get("kind") == "sorted":
                # Sort the flat storage: the input may be reshaped before bucketize, and a
                # non-decreasing storage stays non-decreasing along every dim with stride >= 0.
                numel = _storage_size_for_strided(shape, stride) if stride else _numel(shape)
                storage = _randn(numel, torch.float32, device).sort().values.to(dtype)
                result.append(storage.as_strided(shape, stride) if stride else storage.view(shape))
            elif stride:
                numel = _storage_size_for_strided(shape, stride)
                storage = _randn(numel, dtype, device)
                result.append(storage.as_strided(shape, stride))
            else:
                result.append(_randn(shape, dtype, device))
    return result


def _merge_default_shape_params(inputs: list, default_inputs: list) -> list:
    """Fill shape params (plain lists/ints) from default inputs when a shape
    config carries only tensors but forward() also expects lifted params."""
    if len(inputs) >= len(default_inputs):
        return inputs
    merged = []
    config_idx = 0
    for di in default_inputs:
        if isinstance(di, (list, int)) and not isinstance(di, torch.Tensor):
            # Shape param or scalar — take from default
            merged.append(di)
        else:
            # Tensor — take from config if available
            if config_idx < len(inputs):
                merged.append(inputs[config_idx])
                config_idx += 1
            else:
                merged.append(di)
    return merged


def _randn_with_bool(old_randn):
    """Wrapper around torch.randn that handles bool and fp8 dtypes (legacy repro compat)."""
    def randn(*args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype is not None and dtype.is_floating_point and dtype.itemsize == 1:
            kwargs = dict(kwargs, dtype=torch.float32)
            return old_randn(*args, **kwargs).to(dtype)
        if dtype is torch.bool:
            kwargs = dict(kwargs)
            kwargs.pop("dtype")
            if "size" in kwargs:
                size = kwargs.pop("size")
            elif len(args) == 1 and isinstance(args[0], (list, tuple, torch.Size)):
                size = tuple(args[0])
            else:
                size = args
            return torch.randint(0, 2, size, dtype=torch.bool, **kwargs)
        return old_randn(*args, **kwargs)

    return randn


def make_inputs_safely(make_inputs_fn, *args, **kwargs):
    """Run generated make_inputs while tolerating legacy bool randn repros."""
    old_randn = torch.randn
    torch.randn = _randn_with_bool(old_randn)
    try:
        return make_inputs_fn(*args, **kwargs)
    finally:
        torch.randn = old_randn


def count_bytes_naive(inputs, outputs) -> int:
    from byte_accounting import count_bytes_naive as _count_bytes_naive

    return _count_bytes_naive(inputs, outputs)


def count_bytes_adjusted(mod, inputs) -> int:
    from byte_accounting import count_bytes_effective

    return count_bytes_effective(mod, inputs)


def analyze_bytes_adjusted(mod, inputs):
    from byte_accounting import analyze_bytes_effective

    return analyze_bytes_effective(mod, inputs)


def count_kernels(
    mod,
    inputs,
    dynamic: bool | None = False,
    compile_policy: Mapping | None = None,
    second_inputs=None,
) -> tuple[int, list[str]]:
    """Compile and count how many Triton kernels Inductor generates.

    dynamic: True -> blanket dynamic compile; False -> static; None -> honor
    whatever the inputs are marked (torch._dynamo.mark_dynamic). NOTE: resets
    dynamo — never call while a compile-once artifact is live.

    second_inputs: when the inputs carry mark_dynamic dims, dynamo's 0/1/many
    rule specializes on the FIRST shape; invoking a second (differently-bound,
    same-marked) input forces generalization so the count reflects the
    DYNAMIC artifact the bench measures, not the static-specialized one.
    The kernel set is read from the LAST compile.
    """
    from torch._inductor.utils import fresh_inductor_cache
    from torch._inductor.codecache import cache_dir

    torch._dynamo.reset()
    with fresh_inductor_cache():
        if dynamic is None:
            # honor mark_dynamic on inputs
            compiled = compile_repro(mod, compile_policy=compile_policy)
        elif dynamic:
            compiled = compile_repro(
                mod,
                dynamic=True,
                compile_policy=compile_policy,
            )
        else:
            compiled = compile_repro(mod, compile_policy=compile_policy)
        with torch.no_grad():
            compiled(*inputs)
            if second_inputs is not None:
                compiled(*second_inputs)         # force past 0/1/many
            torch.cuda.synchronize()
        cd = cache_dir()
        py_files = sorted(glob.glob(os.path.join(cd, "**", "*.py"), recursive=True), key=os.path.getmtime)
        for f in reversed(py_files):
            with open(f) as fh:
                content = fh.read()
            if 'def call(' in content and '.run(' in content:
                runs = [l for l in content.split('\n') if '.run(' in l and not l.strip().startswith('#')]
                names = []
                for r in runs:
                    name = r.strip().split('.run(')[0]
                    names.append(name)
                return len(names), names
    return 0, []


def load_perf(repro_file: str) -> dict:
    """Load perf.json for a canonical repro. Returns {hardware: {shape: results}}."""
    perf_path = Path(repro_file).parent / "perf.json"
    if not perf_path.exists():
        return {}
    with open(perf_path) as f:
        return json.load(f)


def _save_perf(repro_file: str, hardware: str, shape_label: str, result: dict):
    """Append a benchmark result to perf.json, keyed by hardware + shape."""
    perf_path = Path(repro_file).parent / "perf.json"
    if perf_path.exists():
        with open(perf_path) as f:
            perf = json.load(f)
    else:
        perf = {}

    if hardware not in perf:
        perf[hardware] = {}

    normalize_compile_policy(
        result.get("compile_policy"),
        source="result compile_policy",
    )

    perf[hardware][shape_label] = result

    with open(perf_path, "w") as f:
        json.dump(perf, f, indent=2)


def _detect_hardware() -> str:
    """Auto-detect the process-visible CUDA hardware kind."""
    try:
        name = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        return "unknown"
    return device_kind_from_name(name)


def _bench_cudagraph_min_us(call_fn, inputs, n_warmup: int, n_rep: int) -> float:
    """Warm up, capture one CUDAGraph of call_fn(*inputs), and time replay.

    This is THE timing methodology for repros (3 warmup calls, CUDAGraph
    capture, do_bench return_mode='min') — both the static path and the
    --dynamic path go through here so their numbers are comparable.
    """
    from triton.testing import do_bench

    with torch.no_grad():
        for _ in range(3):
            call_fn(*inputs)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call_fn(*inputs)
        torch.cuda.synchronize()
    ms = do_bench(lambda: g.replay(), warmup=n_warmup, rep=n_rep,
                  return_mode="min")
    return ms * 1000


def timed_min_us(fn, *, warmup: int = 25, rep: int = 100) -> float:
    """``_bench_cudagraph_min_us`` for a zero-arg callable."""
    return _bench_cudagraph_min_us(lambda: fn(), [], warmup, rep)


def _unique_graph_count() -> int:
    """Cumulative count of graphs dynamo has compiled (survives reset —
    deltas of this counter detect recompiles between measurements)."""
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


def _run_bound_benchmark(repro_file, repro_cls, make_inputs_fn, parsed):
    from dynamic_benchmark import _run_bound_benchmark as run
    return run(repro_file, repro_cls, make_inputs_fn, parsed)


@preserve_compile_environment()
def benchmark_repro(repro_file: str, repro_cls, make_inputs_fn, args=None):
    """Full benchmark entry point for canonical repros.

    Supports --shape, --all-shapes, --hardware, --no-cd, --output,
    --count-kernels-only, --update-perf, --gpu, --device-kind.
    """
    from triton.testing import do_bench

    parser = argparse.ArgumentParser(description="Benchmark canonical repro")
    parser.add_argument("--shape", type=str, default=None,
                        help="Named shape config from shapes.json")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark all shape configs")
    parser.add_argument("--hardware", type=str, default=None,
                        help="Hardware label for perf.json (e.g., H100, B200). Auto-detected if not set.")
    parser.add_argument("--update-perf", action="store_true",
                        help="Write results to perf.json (keyed by hardware + shape)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON results to file")
    parser.add_argument("--no-cd", action="store_true",
                        help="Skip coordinate descent tuning")
    parser.add_argument("--run-at", action="append", default=None,
                        dest="run_at", metavar="s16=24,s82=24",
                        help="Measure at this binding (repeatable). On a "
                             "dynamic repro the artifact is DYNAMIC by "
                             "default; add --static for a fresh fully "
                             "specialized compile per --run-at binding. "
                             "Without --run-at, the family's recorded points "
                             "are measured.")
    parser.add_argument("--bind", action="append", default=None,
                        help=argparse.SUPPRESS)   # compat alias of --run-at
    parser.add_argument("--compile-at", action="append", default=None,
                        dest="compile_at", metavar="s16=24",
                        help="Invoke the dynamic artifact at this binding "
                             "BEFORE measuring (repeatable, exact order; the "
                             "first is inductor's tuning hint). One "
                             "--compile-at A with --run-at B is the "
                             "'compile at A, run at B' experiment — a "
                             "generalization/recompile at B is reported, "
                             "never hidden by an injected auto warm shape. "
                             "Without --compile-at, an automatic two-"
                             "distinct-shape warm generalizes the artifact.")
    parser.add_argument("--freeze", action="append", default=None,
                        metavar="s0[=8]",
                        help="Freeze this symbol to a compile-time constant "
                             "(its table hint, or =N), keeping every other "
                             "symbol dynamic (repeatable). PARTIAL "
                             "specialization is represented in the native "
                             "ShapeEnv contract. A "
                             "--run-at/--compile-at binding that contradicts "
                             "a frozen value errors loudly; freezing every "
                             "symbol errors (use --static).")
    parser.add_argument("--dynamic", action="store_true",
                        help="Force the dynamic artifact (compat: this is "
                             "already the DEFAULT for a repro with a symbols "
                             "table; on a static repro it forces blanket "
                             "dynamic=True).")
    parser.add_argument("--static", action="store_true",
                        help="Fully specialized (never mark_dynamic) static "
                             "artifacts. Alone: ONE artifact with every "
                             "symbol bound to symbols[name].hint (the "
                             "all-hints shortcut; row records the actual "
                             "binding, e.g. label::s0=8::static). With "
                             "--run-at: a fresh fully specialized artifact "
                             "per requested COMPLETE binding. Requires a "
                             "dynamic repro (a symbols table). Distinct from "
                             "--all-shapes (every recorded point statically "
                             "at ITS saved bindings) and the default dynamic "
                             "replay; conflicts loudly with --dynamic, "
                             "--compile-at, --freeze, and --all-shapes.")
    parser.add_argument("--prewarm", action="append", default=None,
                        help=argparse.SUPPRESS)  # compat alias of --compile-at
    parser.add_argument("--count-kernels-only", action="store_true",
                        help="Only count generated kernels, skip timing")
    parser.add_argument("--n-warmup", type=int, default=25)
    parser.add_argument("--n-rep", type=int, default=100)
    parser.add_argument("--gpu", default=None,
                        help="GPU id to lock (default: grab any free GPU)")
    parser.add_argument("--device-kind", default=None,
                        help="GPU kind to lock, e.g. H100 or B200")
    parser.add_argument("--no-gpu-lock", action="store_true",
                        help="Disable the per-GPU benchmark lock")
    parsed = parser.parse_args(args)

    # --run-at/--compile-at are the public spellings; --bind/--prewarm are
    # hidden compat aliases. Mixing spellings for one role in
    # one invocation is rejected — a command that says both is ambiguous
    # about intent. After the merge, downstream code reads parsed.bind /
    # parsed.prewarm (the historical attribute names) with the flag the user
    # actually typed preserved for error attribution.
    if parsed.run_at and parsed.bind:
        parser.error("use --run-at or its compat alias --bind, not both")
    if parsed.compile_at and parsed.prewarm:
        parser.error("use --compile-at or its compat alias --prewarm, "
                     "not both")
    parsed._run_at_flag = "--run-at" if parsed.run_at else "--bind"
    parsed._compile_at_flag = ("--compile-at" if parsed.compile_at
                               else "--prewarm")
    parsed.bind = parsed.run_at or parsed.bind
    parsed.prewarm = parsed.compile_at or parsed.prewarm

    # Whether THIS repro is a dynamic family decides the default execution
    # mode: dynamic families run dynamically with no flags;
    # static repros keep the historical per-shape static path byte-identical.
    _repro_is_dynamic = bool(_symbols_and_guards_for_repro(repro_file)[0])

    if parsed.prewarm and not (parsed.dynamic or _repro_is_dynamic):
        # --compile-at steers the warm sequence of the ONE dynamic artifact;
        # a static repro's per-shape path compiles fresh per shape and has no
        # shared warm sequence. Silently discarding it would make
        # "--compile-at A" and "--compile-at B" produce identical numbers.
        parser.error(
            f"{parsed._compile_at_flag} only applies to the dynamic bench "
            "path; this repro has no symbols table (pass --dynamic to force "
            f"blanket dynamic, or drop {parsed._compile_at_flag}).")
    if parsed.static and parsed.dynamic:
        parser.error(
            "--static and --dynamic are contradictory: --static compiles "
            "fresh fully-specialized artifacts, --dynamic measures ONE "
            "dynamic artifact. Run them as two invocations to compare.")
    if parsed.static and parsed.prewarm:
        parser.error(
            f"--static has no warm sequence to steer (each binding is a "
            f"fresh fully specialized compile) — drop "
            f"{parsed._compile_at_flag} or drop --static.")
    if parsed.static and parsed.freeze:
        parser.error(
            "--freeze keeps the non-frozen symbols DYNAMIC; --static "
            "specializes everything. For a fully static bench use --static "
            "(optionally with --run-at); for partial specialization drop "
            "--static.")
    if parsed.static and parsed.all_shapes:
        parser.error(
            "--static compiles fully specialized artifacts at the hints or "
            "at --run-at bindings; --all-shapes statically benches EVERY "
            "recorded point at its saved bindings. Pick one.")

    def _run_benchmark():
        if (parsed.bind or parsed.dynamic or parsed.static or parsed.prewarm
                or parsed.freeze
                or (_repro_is_dynamic and not parsed.all_shapes)):
            # Family benchmarking. A DYNAMIC repro routes here by DEFAULT
            # (no flags -> the native ShapeEnv artifact over its recorded
            # points); --all-shapes keeps the
            # legacy per-point static sweep; static repros keep the legacy
            # flow below byte-identical when flags are absent.
            return _run_bound_benchmark(
                repro_file, repro_cls, make_inputs_fn, parsed)

        configs = load_shape_configs(repro_file)

        if parsed.all_shapes and configs:
            shape_names = list(configs.keys())
        elif parsed.shape and parsed.shape in configs:
            shape_names = [parsed.shape]
        else:
            shape_names = [None]

        all_results = {}

        for name in shape_names:
            if name is not None:
                shape_config = configs[name]
                inputs = make_inputs_from_config(shape_config)
                label = name
                # If shapes.json doesn't include shape params but forward() expects them,
                # merge shape params from _default_make_inputs at the correct positions
                default_inputs = make_inputs_safely(make_inputs_fn)
                if len(inputs) < len(default_inputs):
                    # Build merged list: use config tensors where available,
                    # fill shape params (plain lists/ints) from defaults
                    merged = []
                    config_idx = 0
                    for di in default_inputs:
                        if isinstance(di, (list, int)) and not isinstance(di, torch.Tensor):
                            # Shape param or scalar — take from default
                            merged.append(di)
                        else:
                            # Tensor — take from config if available
                            if config_idx < len(inputs):
                                merged.append(inputs[config_idx])
                                config_idx += 1
                            else:
                                merged.append(di)
                    inputs = merged
            else:
                shape_config = next(iter(configs.values()), None)
                inputs = make_inputs_safely(make_inputs_fn)
                label = "default"

            mod = repro_cls()
            point_policy = compile_policy_from_config(shape_config)

            with torch.no_grad():
                eager_out = mod(*inputs)

            total_bytes_naive = count_bytes_naive(inputs, eager_out)
            accounting = analyze_bytes_adjusted(mod, inputs)
            total_bytes = accounting.total_bytes
            if total_bytes_naive > total_bytes * 1.1:
                print(f"\n[{label}] Bytes adjusted: {total_bytes_naive/1e6:.1f} MB naive -> {total_bytes/1e6:.1f} MB actual")

            n_kernels, kernel_names = count_kernels(
                mod,
                inputs,
                compile_policy=point_policy,
            )
            print(f"\n[{label}] Kernels generated: {n_kernels}")
            for kn in kernel_names:
                print(f"  {kn}")

            if parsed.count_kernels_only:
                all_results[label] = {
                    "compile_policy": point_policy,
                    "n_kernels": n_kernels,
                    "kernel_names": kernel_names,
                    **accounting.to_result_fields(),
                }
                continue

            # SOL: memcopy same total bytes
            copy_elems = max(total_bytes // (2 * 4), 256)
            src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
            dst = torch.empty_like(src)
            sol_ms = do_bench(lambda: torch.add(src, 1, out=dst), warmup=parsed.n_warmup, rep=parsed.n_rep, return_mode="min")
            sol_us = sol_ms * 1000
            del src, dst

            # Compiled (default heuristics) with CUDAGraph replay
            torch._dynamo.reset()
            compiled = compile_repro(
                mod,
                compile_policy=point_policy,
            )
            with torch.no_grad():
                for _ in range(3):
                    compiled(*inputs)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    compiled(*inputs)
                torch.cuda.synchronize()
            compiled_ms = do_bench(lambda: g.replay(), warmup=parsed.n_warmup, rep=parsed.n_rep, return_mode="min")
            compiled_us = compiled_ms * 1000

            # Compiled with coordinate descent tuning
            cd_us = None
            if not parsed.no_cd:
                torch._dynamo.reset()
                compiled_cd = compile_repro(
                    mod,
                    compile_policy=point_policy,
                    options={"coordinate_descent_tuning": True},
                )
                with torch.no_grad():
                    for _ in range(3):
                        compiled_cd(*inputs)
                    torch.cuda.synchronize()
                    g_cd = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_cd):
                        compiled_cd(*inputs)
                    torch.cuda.synchronize()
                cd_ms = do_bench(lambda: g_cd.replay(), warmup=parsed.n_warmup, rep=parsed.n_rep, return_mode="min")
                cd_us = cd_ms * 1000

            print(f"\n[{label}] Kernel data: {total_bytes / 1024:.1f} KB (read+write)")
            print(f"[{label}] Memcopy SOL (same size): {sol_us:8.1f} us")
            print(f"[{label}] Compiled (default):      {compiled_us:8.1f} us")
            if cd_us is not None:
                print(f"[{label}] Compiled (coord desc):   {cd_us:8.1f} us")
            print(f"[{label}] Gap (default / SOL):     {compiled_us / sol_us:8.2f}x")
            if cd_us is not None:
                print(f"[{label}] Gap (CD / SOL):          {cd_us / sol_us:8.2f}x")

            from byte_accounting import sol_result_fields

            all_results[label] = {
                "compile_policy": point_policy,
                "compiled_us": compiled_us,
                "coord_descent_us": cd_us,
                "memcopy_sol_us": sol_us,
                "n_kernels": n_kernels,
                "kernel_names": kernel_names,
                "gap_default": compiled_us / sol_us if sol_us > 0 else None,
                "gap_cd": cd_us / sol_us if (cd_us and sol_us > 0) else None,
                **sol_result_fields(
                    accounting,
                    proxy_bytes=copy_elems * 8,
                    proxy_op="torch.add.float32_scalar_out",
                    sol_us=sol_us,
                ),
            }

        if parsed.output:
            with open(parsed.output, "w") as f:
                json.dump(all_results, f, indent=2)

        if parsed.update_perf:
            hardware = parsed.hardware or _detect_hardware()
            for shape_label, result in all_results.items():
                perf_entry = {k: v for k, v in result.items() if k != "kernel_names"}
                perf_entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                _save_perf(repro_file, hardware, shape_label, perf_entry)
            print(f"\n[perf] Saved to perf.json under hardware={hardware}")

        return all_results

    # Acquire GPU lock before any CUDA work. Since `import torch` doesn't
    # initialize CUDA, setting CUDA_VISIBLE_DEVICES here (before the first
    # .cuda() call) is safe.
    if parsed.no_gpu_lock:
        if parsed.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.gpu)
        return _run_benchmark()

    scripts_dir = Path(__file__).resolve().parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from gpu_lock import gpu_lock, gpu_lock_for_kind

    if parsed.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.gpu)
        with gpu_lock(parsed.gpu, label=f"repro {Path(repro_file).stem}") as lock_path:
            print(f"[gpu-lock] acquired {lock_path}")
            return _run_benchmark()
    else:
        with gpu_lock_for_kind(
            parsed.device_kind,
            label=f"repro {Path(repro_file).stem}",
        ) as device_info:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_info["index"]
            print(
                f"[gpu-lock] acquired {device_info['lock_path']} "
                f"({device_info['kind']} {device_info['name']})"
            )
            return _run_benchmark()
