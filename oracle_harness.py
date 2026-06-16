"""
Shared oracle infrastructure. Import this instead of inlining helpers.

Provides:
  - load_repro_module(repro_dir): load repro.py from a canonical repro directory
  - get_inputs(repro_dir): get inputs using the repro's make_inputs
  - get_repro_instance(repro_dir): create Repro() instance
  - detect_stochastic_outputs(instance, inputs): find nondeterministic outputs
  - has_stochastic_ops(repro_path): quick source-level check for stochastic ops
  - check_oracle(oracle_forward, instance, inputs, ...): correctness check
  - bench_oracle(oracle_forward, instance, inputs, repro_id, ...): benchmark
  - get_hardware_info(): get GPU hardware properties for kernel config selection
  - get_shape_key(inputs): extract hashable shape signature from inputs
  - bench_oracle_all_shapes(oracle_forward, repro_dir, repro_id, ...): benchmark across all shapes
  - OracleRegistry: per-file (hardware, shape, configs) oracle dispatch
  - register_oracle: decorator for registering oracle implementations
  - dispatch_oracle: find and call best-matching oracle for current hw/shape
"""
from __future__ import annotations

import contextlib
import copy
import fcntl
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

import torch

if __name__ == "__main__":
    sys.modules.setdefault("oracle_harness", sys.modules[__name__])


# ---------------------------------------------------------------------------
# Hardware and shape utilities
# ---------------------------------------------------------------------------

def get_hardware_info():
    """Get hardware properties for kernel config selection."""
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "sm_major": props.major,
        "sm_minor": props.minor,
        "num_sms": props.multi_processor_count,
        "shared_mem_per_sm": props.max_shared_memory_per_multiprocessor,
        "total_mem_gb": props.total_memory / 1e9,
    }


def get_shape_key(inputs):
    """Extract shape signature for config dispatch."""
    shapes = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            shapes.append(tuple(inp.shape))
    return tuple(shapes)


# ---------------------------------------------------------------------------
# Oracle dispatch: registration-based multi-implementation dispatch
# ---------------------------------------------------------------------------

def get_gpu_kind() -> str:
    """Detect GPU kind as a short string like 'B200', 'H100', 'A100'.

    Uses the same detection logic as repro_harness._detect_hardware().
    Cached after first call.
    """
    if not hasattr(get_gpu_kind, "_cached"):
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True, timeout=5
            ).strip().split("\n")[0]
            for kind in ("B200", "H200", "H100", "A100", "V100"):
                if kind in out:
                    get_gpu_kind._cached = kind
                    break
            else:
                get_gpu_kind._cached = out.replace(" ", "_")[:20]
        except Exception:
            get_gpu_kind._cached = "unknown"
    return get_gpu_kind._cached


class OracleRegistry:
    """Per-file oracle registry: explicit (hardware, shape, configs) dispatch.

    Each oracle file creates its own registry instance. Every registration
    declares the ONE concrete shape the implementation was written/tuned for —
    the shape is documentation: "this implementation was tuned for THIS shape."

    The same kernel function can be registered multiple times with different
    launch configs (different hardware or shapes need different BLOCK sizes /
    num_warps, but the kernel body is identical). `configs` is passed through
    to the implementation as keyword arguments at dispatch time.

    Usage in an oracle file:
        from oracle_harness import OracleRegistry

        registry = OracleRegistry()

        def _softmax_impl(inputs, *, BLOCK=1024, num_warps=4):
            ...one kernel body...

        registry.register(hardware="H100", shape=(32768, 1024),
                          configs={"BLOCK": 1024, "num_warps": 4})(_softmax_impl)
        registry.register(hardware="B200", shape=(32768, 1024),
                          configs={"BLOCK": 2048, "num_warps": 8})(_softmax_impl)
        registry.register(hardware="B200", shape=(8192, 262144),
                          configs={"BLOCK": 4096, "num_warps": 16})(_softmax_impl)

        @registry.register()  # unconstrained default fallback
        def oracle_default(inputs):
            ...

        def oracle_forward(inputs):
            return registry.dispatch(inputs)

    Dispatch tiers (see dispatch() for details):
      1. exact (hardware, shape) match
      2. same shape, any hardware
      3. same hardware, nearest shape
      4. any registration, nearest shape
      5. default (registered with no constraints)

    Registration is OPT-IN: oracle files that just define oracle_forward(inputs)
    with no registry keep working with the rest of the harness.
    """

    def __init__(self):
        # Each entry: dict(hardware, shape, full_sig, fn, configs, description)
        self._entries = []
        #: Info about the most recent dispatch (tier, fn, distance, ...).
        #: Benchmarks can inspect this to log "fallback dispatch, may be
        #: suboptimal" when tier > 1.
        self.last_dispatch_info = None

    # -- registration -------------------------------------------------------

    def register(self, hardware=None, shape=None, configs=None, description=None):
        """Decorator to register an oracle implementation.

        Args:
            hardware: GPU kind string (e.g. "B200", "H100") or None for any.
            shape: The CONCRETE shape this implementation was written/tuned
                for. Either a tuple of ints — the key (first tensor) input
                shape, e.g. (8192, 262144) — or a tuple of tuples covering
                ALL tensor input shapes, e.g. ((8192, 262144), (262144,)).
                None means "no shape constraint" (only sensible for defaults).
                Lambda/callable predicates are NOT supported: each oracle is
                tuned for one concrete shape, so declare it.
            configs: Optional dict of launch-config kwargs passed through to
                the implementation at dispatch time: fn(inputs, **configs).
                This lets ONE kernel body serve many (hardware, shape) points
                with different BLOCK sizes / num_warps. configs=None means the
                implementation is called plain: fn(inputs).
            description: Human-readable note (e.g. "split-K, tuned on B200").

        Returns:
            Decorator that registers the function and returns it UNCHANGED,
            so the same function can be registered repeatedly under different
            (hardware, shape, configs) combinations.
        """
        if callable(shape):
            raise TypeError(
                "shape must be a concrete tuple (key input shape) or tuple of "
                "tuples (all input shapes), not a callable predicate. Each "
                "oracle is tuned for ONE shape — declare it explicitly, e.g. "
                "shape=(8192, 262144)."
            )
        norm_shape, full_sig = self._normalize_shape(shape)
        if configs is not None and not isinstance(configs, dict):
            raise TypeError(f"configs must be a dict or None, got {type(configs).__name__}")

        def decorator(fn):
            for prev in self._entries:
                if prev["hardware"] == hardware and prev["shape"] == norm_shape:
                    print(f"WARNING: duplicate oracle registration for "
                          f"(hardware={hardware!r}, shape={norm_shape!r}): "
                          f"{prev['description']} shadows {description or fn.__name__}",
                          file=sys.stderr)
                    break
            self._entries.append({
                "hardware": hardware,
                "shape": norm_shape,        # tuple of tuples, or None
                "full_sig": full_sig,       # True if shape covers ALL inputs
                "fn": fn,
                "configs": dict(configs) if configs else None,
                "description": description or fn.__name__,
            })
            return fn
        return decorator

    @staticmethod
    def _normalize_shape(shape):
        """Normalize shape spec to (tuple_of_tuples, is_full_signature).

        - None -> (None, False): no shape constraint.
        - tuple of ints (key input shape) -> ((shape,), False)
        - tuple of tuples (all input shapes) -> (shapes, True)
        """
        if shape is None:
            return None, False
        shape = tuple(shape)
        if len(shape) == 0:
            # Empty FULL signature: the repro has zero tensor inputs (29
            # corpus repros have _shapes_config of "()" or S()-only — all
            # inputs are shape params/scalars). Matches an empty
            # get_shape_key(). (A scalar tensor registers as ((),) — a
            # 1-tuple containing the empty shape — not as ().)
            return (), True
        if all(isinstance(d, int) for d in shape):
            return (shape,), False
        if all(isinstance(s, (tuple, list, torch.Size)) for s in shape):
            return tuple(tuple(s) for s in shape), True
        raise TypeError(
            f"shape must be a tuple of ints or a tuple of tuples, got {shape!r}"
        )

    # -- dispatch ------------------------------------------------------------
    #
    # Matching is EXACT-ONLY. Every repro has a small, known, finite set of
    # shapes (its shapes.txt lines) — there is no continuous shape space to
    # interpolate over. An implementation either exactly matches the runtime
    # signature, or it declared shapes=None ("shape-general: computes its grid
    # from input dims, works at any shape"). No fuzzy nearest-shape matching:
    # silently running a shape-specific kernel at a different shape is how you
    # get garbage floor measurements.

    @staticmethod
    def _signature_matches(entry, actual_shapes):
        """True iff entry's registered signature exactly matches the inputs."""
        if entry["shape"] is None:
            return False  # shape-general entries handled separately
        if entry["full_sig"]:
            return tuple(entry["shape"]) == tuple(actual_shapes)
        # Key-shape registration: compare the first tensor input only.
        return bool(actual_shapes) and tuple(entry["shape"][0]) == tuple(actual_shapes[0])

    def select(self, inputs):
        """Select the best-matching entry for the given inputs (no call).

        Match order (first hit wins; registration order breaks ties):
          1. "hardware+shape": signature exact AND hardware == current GPU
          2. "shape":          signature exact, tuned on other/any hardware
          3. "hardware":       shape-general (shapes=None), hardware == current
          4. "any":            shape-general, no hardware constraint
        Nothing matches -> RuntimeError. A loud failure beats silently running
        a shape-specific kernel at the wrong shape.

        Returns (entry, info_dict); info_dict also stored on
        self.last_dispatch_info. info["matched"] is one of the strings above;
        info["fallback"] is True for anything but "hardware+shape".
        """
        current_hw = get_gpu_kind()
        actual_shapes = get_shape_key(inputs)

        chosen = None
        matched = None
        for entry in self._entries:  # 1: exact signature + hardware
            if self._signature_matches(entry, actual_shapes) and entry["hardware"] == current_hw:
                chosen, matched = entry, "hardware+shape"
                break
        if chosen is None:
            for entry in self._entries:  # 2: exact signature, any hardware
                if self._signature_matches(entry, actual_shapes):
                    chosen, matched = entry, "shape"
                    break
        if chosen is None:
            for entry in self._entries:  # 3: shape-general, hardware match
                if entry["shape"] is None and entry["hardware"] == current_hw:
                    chosen, matched = entry, "hardware"
                    break
        if chosen is None:
            for entry in self._entries:  # 4: shape-general, unconstrained
                if entry["shape"] is None and entry["hardware"] is None:
                    chosen, matched = entry, "any"
                    break

        if chosen is None:
            self.last_dispatch_info = None
            raise RuntimeError(
                f"No oracle registered for hardware={current_hw!r} "
                f"shapes={actual_shapes!r}. Registry has {len(self._entries)} "
                f"entries: {[e['description'] for e in self._entries]}"
            )

        info = {
            "matched": matched,
            "fallback": matched != "hardware+shape",
            "fn_name": chosen["fn"].__name__,
            "description": chosen["description"],
            "tuned_hardware": chosen["hardware"],
            "registered_shape": chosen["shape"],
            "configs": chosen["configs"],
            "current_hardware": current_hw,
            "actual_shapes": actual_shapes,
        }
        # dtype honesty: the corpus dedupes patterns across dtypes, so a
        # bf16-tuned kernel legitimately serves f32 inputs — but flag it so
        # the data can tell us if dtype ever matters for floors.
        reg_dtypes = chosen.get("dtypes")
        if reg_dtypes:
            _DT = {torch.float32: "f32", torch.float16: "f16",
                   torch.bfloat16: "bf16", torch.float64: "f64",
                   torch.int64: "i64", torch.int32: "i32",
                   torch.int16: "i16", torch.int8: "i8",
                   torch.bool: "b8", torch.uint8: "u8"}
            for _name, _tok in (("float8_e4m3fn", "f8e4m3fn"),
                                ("float8_e5m2", "f8e5m2"),
                                ("float8_e4m3fnuz", "f8e4m3fnuz"),
                                ("float8_e5m2fnuz", "f8e5m2fnuz")):
                if hasattr(torch, _name):
                    _DT[getattr(torch, _name)] = _tok
            actual_dtypes = tuple(_DT.get(t.dtype, str(t.dtype)) for t in inputs
                                  if isinstance(t, torch.Tensor))
            if (len(actual_dtypes) == len(reg_dtypes)
                    and any(r is not None and a != r
                            for a, r in zip(actual_dtypes, reg_dtypes))):
                info["dtypes_differ"] = True
                info["tuned_dtypes"] = reg_dtypes
                info["actual_dtypes"] = actual_dtypes
        self.last_dispatch_info = info
        return chosen, info

    def dispatch(self, inputs):
        """Find the best-matching oracle and call it.

        Calls fn(inputs, **configs) if the matched registration has configs,
        else fn(inputs). self.last_dispatch_info afterwards describes the
        match; fallback=True means the impl was tuned on other hardware (or is
        a shape-general default), so the measurement may be a soft floor.
        """
        entry, _info = self.select(inputs)
        if entry["configs"]:
            return entry["fn"](inputs, **entry["configs"])
        return entry["fn"](inputs)

    # -- introspection -------------------------------------------------------

    def list_entries(self):
        """List registered entries for debugging / CI."""
        return [
            {
                "hardware": e["hardware"],
                "shape": e["shape"][0] if (e["shape"] is not None and not e["full_sig"]) else e["shape"],
                "full_signature": e["full_sig"],
                "fn_name": e["fn"].__name__,
                "configs": e["configs"],
                "description": e["description"],
            }
            for e in self._entries
        ]

    def clear(self):
        """Clear all registrations (useful for testing)."""
        self._entries.clear()
        self.last_dispatch_info = None


# ---------------------------------------------------------------------------
# oracle_impl: the standard one-line registration decorator
# ---------------------------------------------------------------------------
#
# An oracle implements the WHOLE repro (multiple input tensors, possibly
# multiple kernels), so its operating point is the complete input shape
# signature — the same string format as `_shapes_config` in each repro.py
# and each shapes.txt line. One concise decorator line documents exactly
# what the implementation was written and tuned for:
#
#     from oracle_harness import oracle_impl
#
#     @oracle_impl(hardware="H100", shapes="(T([32768, 1024], bf16),)")
#     def oracle_forward(inputs): ...
#
# Variant for other hardware — same kernel body, different launch configs:
#
#     @oracle_impl(hardware="B200", shapes="(T([32768, 1024], bf16),)",
#                  configs={"BLOCK": 2048, "num_warps": 8})
#     def _b200(inputs, *, BLOCK, num_warps): ...
#
# Matching is EXACT-ONLY (no fuzzy nearest-shape interpolation): each repro
# has a small finite set of shapes (its shapes.txt lines), so an impl either
# exactly matches the runtime signature, or it declared shapes=None
# ("shape-general: grid computed from input dims, works at any shape").
# If no registration matches the runtime inputs, dispatch raises
# OracleDispatchError — a loud failure beats a silently-wrong floor
# measurement from running a shape-specific kernel at the wrong shape.
#
# dtype note: the corpus dedupes patterns across dtypes, so the same oracle
# is called with f32 inputs even if written against bf16. dtypes in the
# signature are recorded for documentation but NOT required to match.

# Per-module registries: fn.__module__ -> OracleRegistry
_module_registries = {}


class OracleDispatchError(RuntimeError):
    """No registered oracle implementation matches the runtime inputs."""


def parse_shapes_signature(shapes, *, with_dtypes=False):
    """Parse a T()/S() signature string into a tuple of shape tuples.

    Accepts the exact `_shapes_config` / shapes.txt format, e.g.
    "(T([512, 256, 13, 13], f32), T([512, 128, 27, 27], b8), S([131072, 169]))"
    Only tensor entries (T) contribute shapes; S() entries are shape params,
    not tensors, and are skipped. dtypes are NOT used for matching (the
    corpus dedupes patterns across dtypes) but are recorded for honesty:
    with_dtypes=True returns (shape_tuples, dtype_tuple) so dispatch can
    flag when e.g. a bf16-tuned kernel is serving f32 inputs.
    Already-parsed tuples of tuples pass through unchanged.
    """
    if shapes is None:
        return (None, None) if with_dtypes else None
    if not isinstance(shapes, str):
        # programmatic: tuple of shape tuples
        parsed_prog = tuple(tuple(s) for s in shapes)
        return (parsed_prog, None) if with_dtypes else parsed_prog

    def T(shape, dtype=None, *args, **kwargs):
        return ("T", tuple(shape), dtype)

    def S(dims):
        return ("S", tuple(dims))

    def Index(high, low=0):
        return None

    def Perm(size=None):
        return None

    ns = {"__builtins__": {}, "T": T, "S": S, "Index": Index, "Perm": Perm}
    for d in ("f32", "f16", "bf16", "f64", "i64", "i32", "i16", "i8", "b8", "u8",
              "f8e4m3fn", "f8e5m2", "f8e4m3fnuz", "f8e5m2fnuz"):
        ns[d] = d
    parsed = eval(shapes, ns)  # noqa: S307 - trusted repo-internal strings

    def is_entry(x):
        return (isinstance(x, tuple) and len(x) >= 2 and x[0] in ("T", "S")
                and isinstance(x[1], tuple))

    # Single-tensor signatures without a trailing comma — "(T([8192], bf16))"
    # — eval to the entry itself rather than a 1-tuple of entries.
    if is_entry(parsed):
        parsed = (parsed,)
    elif not isinstance(parsed, tuple):
        parsed = (parsed,)
    tensors = [e for e in parsed if is_entry(e) and e[0] == "T"]
    shape_tuples = tuple(e[1] for e in tensors)
    if with_dtypes:
        dtypes = tuple((e[2] if len(e) > 2 else None) for e in tensors)
        return shape_tuples, dtypes
    return shape_tuples


def oracle_impl(hardware=None, shapes=None, point=None, description=None,
                **kwargs):
    """Register an oracle implementation: one line declaring what it was
    written for.

    Args:
        hardware: GPU kind the impl was tuned on ("H100", "B200", ...).
            None = no hardware constraint.
        point: shape_hash (8-hex) of the shapes.json point this impl covers
            — THE registration key for new-corpus oracles (settled
            2026-06-12: hash as key, zero parsing; the human-readable
            signature lives in shapes.json and may be quoted in a comment).
            Resolved against the sibling shapes.json at registration time:
            an unknown hash raises immediately (stale/typo'd registrations
            fail at import, not silently at dispatch). Mutually exclusive
            with shapes=. shapes=None AND point=None still means
            shape-general.
        shapes: FULL input signature the impl was written for, as the
            T()/S() string from the repro's `_shapes_config` (copy it
            verbatim), or a pre-parsed tuple of shape tuples.
            shapes=None means the impl is SHAPE-GENERAL: it computes its
            grid from input dims and works at any shape. There is no fuzzy
            matching in between — each repro has a small finite set of
            shapes (shapes.txt); register the impl at each shape it was
            tuned for, or declare it shape-general.
        description: optional human-readable note.
        **kwargs: any other keyword arguments are passed through to the
            implementation at dispatch time: fn(inputs, **kwargs). This is
            how one kernel body serves many (hardware, shapes) points with
            different launch parameters:

                oracle_impl(hardware="H100", shapes=SIG, BLOCK=1024, num_warps=4)(_impl)
                oracle_impl(hardware="B200", shapes=SIG, BLOCK=2048, num_warps=8)(_impl)

    Returns the function unchanged, so one body can be registered N times.
    """
    if point is not None and shapes is not None:
        raise ValueError("oracle_impl: pass point= (hash) or shapes=, not both")

    parsed, dtypes = parse_shapes_signature(shapes, with_dtypes=True)

    def decorator(fn):
        reg = _module_registries.setdefault(fn.__module__, OracleRegistry())
        entry_shapes = parsed if parsed is None else tuple(parsed)
        if point is not None:
            # Hash-keyed registration: resolve the point's input shapes
            # from the SIBLING shapes.json (the oracle file lives in the
            # canonical dir). Loud failure on unknown hash.
            import inspect
            import json as _json
            from pathlib import Path as _Path

            mod_file = inspect.getmodule(fn)
            fpath = _Path(getattr(mod_file, "__file__", None)
                          or inspect.getfile(fn))
            shapes_path = fpath.parent / "shapes.json"
            if not shapes_path.exists():
                raise OracleDispatchError(
                    f"oracle_impl(point={point!r}): no shapes.json next to "
                    f"{fpath} — hash-keyed registration requires the oracle "
                    f"to live in its canonical repro dir")
            data = _json.loads(shapes_path.read_text())
            match = next((p for p in data.get("points", [])
                          if p.get("shape_hash") == point), None)
            if match is None:
                known = [p.get("shape_hash") for p in data.get("points", [])]
                raise OracleDispatchError(
                    f"oracle_impl(point={point!r}): hash not in "
                    f"{shapes_path} (known: {known[:8]}{'...' if len(known) > 8 else ''})")
            from input_codec import spec_from_compact
            specs = [spec_from_compact(e) for e in match.get("inputs", [])]
            entry_shapes = tuple(
                tuple(s["shape"]) for s in specs if s.get("kind") == "tensor")
        dec = reg.register(hardware=hardware,
                           shape=entry_shapes,
                           configs=kwargs or None,
                           description=description)
        dec(fn)
        reg._entries[-1]["dtypes"] = dtypes
        reg._entries[-1]["point"] = point
        return fn
    return decorator


def get_module_registry(module_name):
    """Return the OracleRegistry for a module (None if no registrations)."""
    return _module_registries.get(module_name)


def resolve_oracle(oracle_forward, inputs):
    """Resolve dispatch for an oracle callable against runtime inputs.

    If the oracle's module has `oracle_impl` registrations, select the best
    match and return (callable, dispatch_info). The returned callable has
    configs already bound. If the module has no registrations, returns
    (oracle_forward, None) — unmigrated oracles behave exactly as before.

    Raises OracleDispatchError if registrations exist but none match.
    """
    reg = _module_registries.get(getattr(oracle_forward, "__module__", None))
    if reg is None or not reg._entries:
        return oracle_forward, None
    try:
        entry, info = reg.select(inputs)
    except RuntimeError as e:
        raise OracleDispatchError(str(e)) from e
    if entry["configs"]:
        fn, cfgs = entry["fn"], entry["configs"]
        return (lambda inp: fn(inp, **cfgs)), info
    return entry["fn"], info


def reset_oracle_registry(module_name=None):
    """Reset registries (all, or one module's). Useful between test loads."""
    if module_name is None:
        _module_registries.clear()
    else:
        _module_registries.pop(module_name, None)


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def load_repro_module(repro_dir: str | Path):
    """Load repro.py from a canonical repro directory."""
    repro_dir = Path(repro_dir)
    repro_path = repro_dir / "repro.py"
    repo_root = repro_dir.parents[2] if repro_dir.match("repros/canonical/*") else repro_dir.parent
    sys.path.insert(0, str(repo_root))

    repro_id = repro_dir.name
    spec = importlib.util.spec_from_file_location(f"{repro_id}_repro", repro_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load repro module from {repro_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs(repro_dir: str | Path) -> list:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    mod = load_repro_module(repro_dir)
    if hasattr(mod, "make_inputs"):
        return mod.make_inputs()
    elif hasattr(mod, "_default_make_inputs"):
        return mod._default_make_inputs()
    else:
        raise RuntimeError("Repro has no make_inputs or _default_make_inputs")


def get_repro_instance(repro_dir: str | Path):
    """Create a Repro() instance for reference comparison."""
    mod = load_repro_module(repro_dir)
    return mod.Repro()


# ---------------------------------------------------------------------------
# Stochastic detection
# ---------------------------------------------------------------------------

_STOCHASTIC_OPS = [
    "aten.dropout",
    "aten.bernoulli",
    "aten.rand",
    "aten.native_dropout",
    "aten.rrelu",
    "aten.alpha_dropout",
    "aten.feature_dropout",
    "aten.feature_alpha_dropout",
]


def has_stochastic_ops(repro_path: str | Path) -> bool:
    """Quick check if repro contains known stochastic ops."""
    content = Path(repro_path).read_text()
    return any(op in content for op in _STOCHASTIC_OPS)


def detect_stochastic_outputs(instance, inputs) -> set[int]:
    """Run instance twice with same inputs, find outputs that differ (stochastic).

    Returns a set of output indices that are nondeterministic.

    Strategy: run once with an explicit seed, then run again WITHOUT reseeding.
    Any output that differs between the two runs is nondeterministic (dropout,
    bernoulli, RNG-derived indices/masks, etc.) and is reported as stochastic so
    the correctness check skips it.

    Nondeterminism is dtype-agnostic: an RNG-derived int64 index, bool dropout
    mask, or complex tensor is just as stochastic as a float one. We therefore
    compare every output regardless of dtype, mirroring _compare_oracle_outputs:
    a tolerance (allclose) check for floats, and an exact (torch.equal) check for
    int/bool/complex. An output that differs => stochastic.
    """
    # Seeded run
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    with torch.no_grad():
        out_seeded = instance(*inputs)

    # Unseeded run (detects dropout, bernoulli, etc.)
    with torch.no_grad():
        out_unseeded = instance(*inputs)

    stochastic = set()
    seeded_list = _normalize_outputs(out_seeded)
    unseeded_list = _normalize_outputs(out_unseeded)

    for i, (o_s, o_u) in enumerate(zip(seeded_list, unseeded_list)):
        try:
            if o_s.is_floating_point():
                # Float: tolerant comparison (matches the value-check path).
                differs = not torch.allclose(
                    o_s.float(), o_u.float(), atol=1e-6, rtol=0, equal_nan=True
                )
            else:
                # int / bool / complex: exact comparison (matches the
                # torch.equal path in _compare_oracle_outputs). torch.equal
                # also returns False on a shape mismatch, so differing shapes
                # between runs are treated as nondeterministic too.
                differs = not torch.equal(o_s, o_u)
        except RuntimeError:
            # Comparison failed (e.g. shapes differ between runs) -> the output
            # is not stable across runs, so treat it as stochastic.
            differs = True
        if differs:
            stochastic.add(i)

    return stochastic


# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------

def _compare_oracle_outputs(
    eager,
    oracle_out,
    stochastic: set[int],
    *,
    atol: float,
    rtol: float,
) -> bool:
    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)

    # Scope check: output count must match
    if len(oracle_list) != len(eager_list):
        print(f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
              f"eager produces {len(eager_list)}")
        return False

    all_pass = True
    for i, (e, o) in enumerate(zip(eager_list, oracle_list)):
        if i in stochastic:
            print(f"  output {i}: SKIP (stochastic)")
            continue

        # Shape check
        if e.shape != o.shape:
            print(f"  output {i}: SCOPE_MISMATCH shape oracle={list(o.shape)} "
                  f"eager={list(e.shape)}")
            all_pass = False
            continue

        # Dtype check (warn but don't fail)
        if e.dtype != o.dtype:
            print(f"  output {i}: WARNING dtype mismatch oracle={o.dtype} eager={e.dtype}")

        if not e.is_floating_point():
            # For integer/bool tensors, require exact match
            ok = torch.equal(e, o)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={e.dtype})")
            if not ok:
                all_pass = False
            continue

        # Value check
        e_f32 = e.float()
        o_f32 = o.float()
        both_nan = torch.isnan(e_f32) & torch.isnan(o_f32)
        diff = (e_f32 - o_f32).abs()
        if both_nan.any():
            diff = torch.where(both_nan, torch.zeros_like(diff), diff)
        max_diff = diff.max().item() if diff.numel() else 0.0
        ok = torch.allclose(e_f32, o_f32, atol=atol, rtol=rtol, equal_nan=True)

        status = "PASS" if ok else "FAIL"
        print(f"  output {i}: {status} (shape={list(e.shape)} dtype={e.dtype} "
              f"max_diff={max_diff:.2e})")
        if not ok:
            all_pass = False

    return all_pass


def check_oracle(
    oracle_forward,
    instance,
    inputs,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    skip_stochastic: bool = True,
) -> bool:
    """Standard oracle correctness check with auto stochastic detection.

    Args:
        oracle_forward: callable that takes inputs and returns oracle outputs
        instance: Repro() instance (eager reference)
        inputs: list of input tensors/values
        atol: absolute tolerance for allclose
        rtol: relative tolerance for allclose
        skip_stochastic: if True, auto-detect and skip stochastic outputs

    Returns:
        True if all non-stochastic outputs pass.
    """
    # Resolve oracle_impl dispatch so --check verifies the SAME callable that
    # --bench times (e.g. a B200 kwargs variant, not the undecorated default).
    try:
        oracle_forward, _dispatch_info = resolve_oracle(oracle_forward, inputs)
    except OracleDispatchError as e:
        print(f"  NO_ORACLE_FOR_SHAPE: {str(e)[:200]}")
        return False

    stochastic = detect_stochastic_outputs(instance, inputs) if skip_stochastic else set()

    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)

    return _compare_oracle_outputs(
        eager, oracle_out, stochastic, atol=atol, rtol=rtol)


def _seed_input_generation(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_oracle_all_shapes(
    oracle_forward,
    repro_dir,
    repro_id: str,
    *,
    skip_stochastic: bool = True,
    point: str | None = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict:
    """Run check_oracle at every shape config from shapes.json/shapes.txt.

    The runner builds separate deterministic input sets for eager and oracle
    execution. This avoids mutating the oracle inputs during eager reference
    evaluation for repros with user-visible in-place writes.
    """
    from repro_harness import load_shape_configs, make_inputs_from_config

    repro_dir = Path(repro_dir)
    repro_file = repro_dir / "repro.py"
    configs = load_shape_configs(str(repro_file))

    if point is not None:
        configs = {
            label: cfg for label, cfg in configs.items()
            if label == point or label.endswith(f"_{point}")
        }
        if not configs:
            print(f"  NO_SHAPE_POINT: {point}")
            return {point: False}

    if not configs:
        inputs = get_inputs(repro_dir)
        instance = get_repro_instance(repro_dir)
        return {
            "default": check_oracle(
                oracle_forward,
                instance,
                inputs,
                atol=atol,
                rtol=rtol,
                skip_stochastic=skip_stochastic,
            )
        }

    results = {}
    for label, config in configs.items():
        print(f"Checking {repro_id}_{label}...")
        instance = get_repro_instance(repro_dir)

        _seed_input_generation(0)
        detect_inputs = make_inputs_from_config(config)
        stochastic = (
            detect_stochastic_outputs(instance, detect_inputs)
            if skip_stochastic
            else set()
        )

        _seed_input_generation(1)
        eager_inputs = make_inputs_from_config(config)
        _seed_input_generation(1)
        oracle_inputs = make_inputs_from_config(config)

        try:
            selected_oracle, _dispatch_info = resolve_oracle(
                oracle_forward, oracle_inputs)
        except OracleDispatchError as e:
            print(f"  NO_ORACLE_FOR_SHAPE: {str(e)[:200]}")
            results[label] = False
            continue

        with torch.no_grad():
            eager = instance(*eager_inputs)
            oracle_out = selected_oracle(oracle_inputs)

        results[label] = _compare_oracle_outputs(
            eager, oracle_out, stochastic, atol=atol, rtol=rtol)

    return results


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def bench_oracle(
    oracle_forward,
    instance,
    inputs,
    repro_id: str,
    *,
    warmup: int = 25,
    rep: int = 200,
    rounds: int = 5,
    _skip_numerics_gate: bool = False,
    disable_gpu_lock: bool = False,
) -> dict:
    """Standard oracle benchmark: oracle vs torch.compile.

    Delegates to bench_compare's methodology: CUDAGraph capture+replay,
    exclusive GPU lock, interleaved timing rounds. This ensures oracle
    measurements use the exact same infrastructure as corpus-wide sweeps.

    Now includes:
    - CUDAGraph warning trap: any CUDAGraph-related warning during capture
      invalidates the measurement (status=INVALID_CUDAGRAPH_WARNING).
    - FP64-anchored numerics gate: oracle must not be meaningfully less
      accurate than compiled output vs a high-precision reference
      (status=NUMERICS_WORSE_THAN_COMPILED).

    Args:
        oracle_forward: callable that takes inputs and returns oracle outputs
        instance: Repro() instance
        inputs: list of input tensors/values
        repro_id: string identifier for the repro
        warmup: warmup iterations per round
        rep: repetitions per round
        rounds: number of interleaved rounds (min-of-N)
        _skip_numerics_gate: internal flag to bypass fp64 gate (testing only)
        disable_gpu_lock: skip the exclusive GPU lock (no-op context manager).
            The lock is ON by default; an INDUCTOR_GPU_BENCH_LOCK=1 /
            TORCHINDUCTOR_GPU_BENCH_LOCK=1 env var force-enables it even when
            this is True.

    Returns:
        Dict with repro_id, oracle_us, compile_us, ratio, status — plus a
        "dispatch" field when the oracle module uses oracle_impl registration
        (tier > 1 / fallback=True means the selected implementation was tuned
        for different hardware and/or shape, so the floor may be soft).
    """
    # --- Resolve oracle_impl dispatch (no-op for unmigrated oracles) ---
    dispatch_info = None
    try:
        oracle_forward, dispatch_info = resolve_oracle(oracle_forward, inputs)
    except OracleDispatchError as e:
        result = {
            "repro_id": repro_id,
            "status": "NO_ORACLE_FOR_SHAPE",
            "error": str(e)[:300],
        }
        print(json.dumps(result))
        return result

    device = _get_device(inputs)

    if device.type != "cuda":
        return _bench_oracle_cpu(oracle_forward, instance, inputs, repro_id,
                                 warmup=warmup, rep=rep)

    from triton.testing import do_bench

    # --- Compile with coordinate_descent + capture CUDAGraph (MANDATORY) ---
    import torch._inductor.config as cfg
    cfg.coordinate_descent_tuning = True
    compiled = torch.compile(instance)
    with torch.no_grad():
        for _ in range(5):
            compiled(*inputs)
        torch.cuda.synchronize()
        g_compile, compile_warnings = _capture_graph_checked(
            lambda: compiled(*inputs))

    # Check for CUDAGraph warnings on compile capture
    if compile_warnings:
        result = {
            "repro_id": repro_id,
            "status": "INVALID_CUDAGRAPH_WARNING",
            "warning_source": "compiled",
            "warnings": compile_warnings[:5],
        }
        if dispatch_info is not None:
            result["dispatch"] = {"matched": dispatch_info["matched"]}
        print(json.dumps(result))
        return result

    # --- Warm oracle + capture CUDAGraph (MANDATORY) ---
    with torch.no_grad():
        for _ in range(3):
            oracle_forward(inputs)
        torch.cuda.synchronize()
        g_oracle, oracle_warnings = _capture_graph_checked(
            lambda: oracle_forward(inputs))

    # Check for CUDAGraph warnings on oracle capture
    if oracle_warnings:
        result = {
            "repro_id": repro_id,
            "status": "INVALID_CUDAGRAPH_WARNING",
            "warning_source": "oracle",
            "warnings": oracle_warnings[:5],
        }
        if dispatch_info is not None:
            result["dispatch"] = {"matched": dispatch_info["matched"]}
        print(json.dumps(result))
        return result

    # --- FP64-anchored numerics gate (change 3) ---
    if not _skip_numerics_gate:
        numerics_result = _run_anchored_numerics_gate(
            oracle_forward, instance, compiled, inputs)
        if numerics_result is not None and not numerics_result["pass"]:
            result = {
                "repro_id": repro_id,
                "status": "NUMERICS_WORSE_THAN_COMPILED",
                "numerics_gate": numerics_result,
            }
            if dispatch_info is not None:
                result["dispatch"] = {"matched": dispatch_info["matched"]}
            print(json.dumps(result, default=str))
            return result

    # --- Adaptive rep count based on quick estimate ---
    quick_oracle = _time_graph(g_oracle, warmup=5, rep=10)
    quick_compile = _time_graph(g_compile, warmup=5, rep=10)
    est_us = min(quick_oracle, quick_compile)
    if est_us < 50:
        rep = max(rep, 500)
    elif est_us < 200:
        rep = max(rep, 300)

    # --- Interleaved timing under EXCLUSIVE GPU LOCK (MANDATORY) ---
    best_oracle_us = math.inf
    best_compile_us = math.inf

    with _gpu_exclusive_lock(repro_id, disabled=disable_gpu_lock):
        # Warm under lock
        for _ in range(warmup):
            g_oracle.replay()
            g_compile.replay()
        torch.cuda.synchronize()

        # Interleaved rounds: oracle then compile, min-of-N
        for _ in range(rounds):
            oracle_us = _time_graph(g_oracle, warmup=warmup, rep=rep)
            compile_us = _time_graph(g_compile, warmup=warmup, rep=rep)
            best_oracle_us = min(best_oracle_us, oracle_us)
            best_compile_us = min(best_compile_us, compile_us)

    ratio = best_compile_us / best_oracle_us if best_oracle_us > 0 else 0.0
    if ratio > 1.05:
        status = "GOOD"
    elif ratio < 0.95:
        status = "BAD_ORACLE"
    else:
        status = "AT_FLOOR"

    result = {
        "repro_id": repro_id,
        "oracle_us": round(best_oracle_us, 2),
        "compile_us": round(best_compile_us, 2),
        "ratio": round(ratio, 3),
        "status": status,
    }
    # Always include numerics_gate info for visibility
    if not _skip_numerics_gate and numerics_result is not None:
        result["numerics_gate"] = numerics_result
    if dispatch_info is not None:
        result["dispatch"] = {
            "matched": dispatch_info["matched"],
            "tuned_on": dispatch_info["tuned_hardware"],
            "running_on": dispatch_info["current_hardware"],
            "fallback": dispatch_info["fallback"],
        }
        if dispatch_info.get("dtypes_differ"):
            result["dispatch"]["dtypes_differ"] = True
            result["dispatch"]["tuned_dtypes"] = list(dispatch_info["tuned_dtypes"])
            result["dispatch"]["actual_dtypes"] = list(dispatch_info["actual_dtypes"])
    print(json.dumps(result, default=str))
    return result


def _bench_oracle_cpu(oracle_forward, instance, inputs, repro_id, *, warmup, rep):
    """Fallback for CPU-only benchmarks (no CUDAGraph)."""
    with torch.no_grad():
        oracle_us = _do_bench(lambda: oracle_forward(inputs), torch.device("cpu"),
                              warmup=warmup, rep=rep)
    compiled = torch.compile(instance)
    with torch.no_grad():
        for _ in range(5):
            compiled(*inputs)
        compile_us = _do_bench(lambda: compiled(*inputs), torch.device("cpu"),
                               warmup=warmup, rep=rep)
    ratio = compile_us / oracle_us if oracle_us > 0 else 0.0
    status = "GOOD" if ratio > 1.05 else ("BAD_ORACLE" if ratio < 0.95 else "AT_FLOOR")
    result = {
        "repro_id": repro_id,
        "oracle_us": round(oracle_us, 2),
        "compile_us": round(compile_us, 2),
        "ratio": round(ratio, 3),
        "status": status,
    }
    print(json.dumps(result))
    return result


def _run_anchored_numerics_gate(oracle_forward, instance, compiled, inputs):
    """Run fp64-anchored numerics comparison.

    Builds an fp64 eager reference, computes oracle and compiled outputs,
    then runs _anchored_numerics_gate. Uses deterministic seeding to
    regenerate inputs per run (mirrors check_oracle_all_shapes discipline).

    Returns the gate result dict, or None if the gate could not be run.
    """
    # Every run gets its OWN copy of the inputs: repros with in-place
    # writes mutate their inputs, and the gate is only meaningful if the
    # reference, oracle, and compiled runs all start from the same state.
    # empty_strided+copy_ rather than clone(): clone() contiguizes
    # non-dense views, and stride is part of pattern identity here.
    def _copy_one(t):
        if any(s == 0 for s in t.stride()):
            return t.clone()  # expanded: copy_ into 0-stride is undefined
        out = torch.empty_strided(t.size(), t.stride(),
                                  dtype=t.dtype, device=t.device)
        out.copy_(t)
        return out

    def _clone_inputs(src):
        return [_copy_one(i) if isinstance(i, torch.Tensor) else i
                for i in src]

    # Detect stochastic outputs for gating
    stochastic = detect_stochastic_outputs(instance, _clone_inputs(inputs))

    # Build fp64 reference: deepcopy instance, cast module + float inputs
    ref_precision = "f64"
    try:
        fp64_instance = copy.deepcopy(instance).double()
        fp64_inputs = []
        for inp in _clone_inputs(inputs):
            if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                fp64_inputs.append(inp.double())
            else:
                fp64_inputs.append(inp)
        _seed_input_generation(1)
        with torch.no_grad():
            ref_outs = _normalize_outputs(fp64_instance(*fp64_inputs))
    except Exception:
        # fp64 run failed (e.g. ops without fp64 kernels) — fall back to f32
        ref_precision = "f32"
        try:
            # Use the instance itself as the f32 reference
            _seed_input_generation(1)
            with torch.no_grad():
                ref_outs = _normalize_outputs(instance(*_clone_inputs(inputs)))
        except Exception:
            return None

    # Get oracle outputs with seeded inputs
    _seed_input_generation(1)
    with torch.no_grad():
        oracle_outs = _normalize_outputs(oracle_forward(_clone_inputs(inputs)))

    # Get compiled outputs with seeded inputs
    _seed_input_generation(1)
    with torch.no_grad():
        compiled_outs = _normalize_outputs(compiled(*_clone_inputs(inputs)))

    # Cast ref_outs to match the comparison dtype for the gate
    # (the gate casts everything to f64 internally for comparison)
    if len(oracle_outs) != len(ref_outs) or len(compiled_outs) != len(ref_outs):
        # Scope mismatch — cannot run gate
        return None

    return _anchored_numerics_gate(
        oracle_outs, compiled_outs, ref_outs, stochastic,
        ref_precision=ref_precision)


def bench_oracle_all_shapes(oracle_forward, repro_dir, repro_id, **kwargs):
    """Benchmark oracle across all shapes (shapes.json preferred, shapes.txt fallback).

    Args:
        oracle_forward: callable that takes inputs and returns oracle outputs
        repro_dir: Path to the repro directory
        repro_id: string identifier for the repro
        **kwargs: passed through to bench_oracle (warmup, rep, etc.)

    Returns:
        List of result dicts from bench_oracle, one per shape config.
    """
    from repro_harness import load_shape_configs, make_inputs_from_config

    repro_dir = Path(repro_dir)
    # load_shape_configs expects a file path whose parent dir it inspects
    repro_file = repro_dir / "repro.py"
    configs = load_shape_configs(str(repro_file))

    if not configs:
        # No shape configs found — just run with default inputs
        inputs = get_inputs(repro_dir)
        instance = get_repro_instance(repro_dir)
        return [bench_oracle(oracle_forward, instance, inputs, repro_id, **kwargs)]

    results = []
    for label, config in configs.items():
        inputs = make_inputs_from_config(config)
        instance = get_repro_instance(repro_dir)
        result = bench_oracle(oracle_forward, instance, inputs, f"{repro_id}_{label}", **kwargs)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _capture_graph(fn):
    """Capture a CUDAGraph of fn(). Raises on failure (no silent fallback).

    CUDAGraph capture is MANDATORY for reliable benchmarking. If capture fails,
    the benchmark result would be unreliable due to kernel launch overhead, so
    we raise instead of silently degrading.
    """
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


# ---------------------------------------------------------------------------
# CUDAGraph warning trap (change 2: mechanically enforce warning=INVALID)
# ---------------------------------------------------------------------------

_CUDAGRAPH_WARNING_PATTERNS = re.compile(
    r"CUDAGraph|cuda.?graph|graph.?capture|stream.?capture",
    re.IGNORECASE,
)


def _capture_graph_checked(fn):
    """Capture a CUDAGraph, trapping any CUDAGraph-related warnings.

    Returns (graph, warning_messages: list[str]). If warning_messages is
    non-empty the measurement is INVALID.
    """
    caught_warnings = []
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()

    for w in ws:
        msg = str(w.message)
        if _CUDAGRAPH_WARNING_PATTERNS.search(msg):
            caught_warnings.append(msg)
    return g, caught_warnings


# ---------------------------------------------------------------------------
# FP64-anchored numerics gate (change 3)
# ---------------------------------------------------------------------------

def _anchored_numerics_gate(
    oracle_outs: list[torch.Tensor],
    compiled_outs: list[torch.Tensor],
    ref_outs: list[torch.Tensor],
    stochastic: set[int],
    *,
    ref_precision: str = "f64",
) -> dict:
    """Compare oracle vs compiled accuracy against a high-precision reference.

    Returns a dict with:
      - "pass": bool (True if oracle is not meaningfully worse than compiled)
      - "per_output": list of per-output dicts with err_oracle, err_compiled
      - "worst_output_idx": index of the worst-ratio output (or None)
      - "ref_precision": "f64" or "f32"

    Gate: err_oracle <= max(3 * err_compiled, 1e-5 * ref_absmax_scale)
    per non-stochastic float output. Violation means the oracle uses a
    formulation substitution (fast exp, exp2-softmax, etc.).
    """
    result = {
        "pass": True,
        "per_output": [],
        "worst_output_idx": None,
        "ref_precision": ref_precision,
    }
    worst_ratio = 0.0

    for i, (o_oracle, o_compiled, o_ref) in enumerate(
        zip(oracle_outs, compiled_outs, ref_outs)
    ):
        entry = {"idx": i}
        if i in stochastic:
            entry["skip"] = "stochastic"
            result["per_output"].append(entry)
            continue
        if not o_ref.is_floating_point():
            entry["skip"] = "non_float"
            result["per_output"].append(entry)
            continue

        # Cast to f64 for comparison
        ref_f64 = o_ref.double()
        oracle_f64 = o_oracle.double()
        compiled_f64 = o_compiled.double()

        err_oracle = (oracle_f64 - ref_f64).abs().max().item()
        err_compiled = (compiled_f64 - ref_f64).abs().max().item()
        ref_absmax = ref_f64.abs().max().item()
        ref_absmax_scale = max(ref_absmax, 1.0)  # avoid 0-scale

        threshold = max(3.0 * err_compiled, 1e-5 * ref_absmax_scale)
        passes = err_oracle <= threshold

        entry["err_oracle"] = err_oracle
        entry["err_compiled"] = err_compiled
        entry["threshold"] = threshold
        entry["ref_absmax"] = ref_absmax
        entry["pass"] = passes

        if not passes:
            result["pass"] = False
            ratio = err_oracle / max(err_compiled, 1e-30)
            if ratio > worst_ratio:
                worst_ratio = ratio
                result["worst_output_idx"] = i

        result["per_output"].append(entry)

    return result


def _time_graph(graph, warmup, rep):
    """Time a CUDAGraph replay. Returns microseconds."""
    from triton.testing import do_bench
    ms = do_bench(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="min")
    return ms * 1000.0


# ---------------------------------------------------------------------------
# GPU exclusive lock (inlined, never silently skipped)
# ---------------------------------------------------------------------------
# This implements the same locking protocol as bench_compare.py's worker script.
# The lock is controlled by the INDUCTOR_GPU_BENCH_LOCK / TORCHINDUCTOR_GPU_BENCH_LOCK
# environment variable. If the env var is set, the lock is MANDATORY and will
# block until acquired. If not set, this is a hard error — callers MUST set the
# env var to explicitly opt in or out.

_GPU_BENCH_LOCK_STATE_NAME = "_torchinductor_gpu_benchmark_lock_state"


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _safe_lock_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "unknown"


def _gpu_bench_lock_state():
    import builtins
    state = getattr(builtins, _GPU_BENCH_LOCK_STATE_NAME, None)
    if state is None:
        state = {
            "mutex": threading.RLock(),
            "local": threading.local(),
        }
        setattr(builtins, _GPU_BENCH_LOCK_STATE_NAME, state)
    return state


def _write_gpu_lock_metadata(fd, gpu, mode, label):
    try:
        os.ftruncate(fd, 0)
        os.write(fd, (
            f"pid={os.getpid()}\n"
            f"gpu={gpu}\n"
            f"mode={mode}\n"
            f"label={label}\n"
            f"acquired_unix={time.time():.0f}\n"
        ).encode())
        os.fsync(fd)
    except OSError:
        pass


def _release_fd(fd):
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _gpu_exclusive_lock(label="oracle_bench", *, disabled=False):
    """Acquire exclusive GPU lock. ON BY DEFAULT.

    The lock prevents silent measurement corruption from GPU contention, so it
    is acquired by default. Gating:
      - Default (no env var, disabled=False): acquire the lock.
      - INDUCTOR_GPU_BENCH_LOCK=1 or TORCHINDUCTOR_GPU_BENCH_LOCK=1: force the
        lock on, even if disabled=True (force-enable beats --disable-gpu-lock).
      - disabled=True (e.g. --disable-gpu-lock) and no env force: no-op
        context manager (no flock taken).

    The same env-var protocol as bench_compare is respected for back-compat.
    """
    force_enabled = (
        _env_flag_enabled("INDUCTOR_GPU_BENCH_LOCK")
        or _env_flag_enabled("TORCHINDUCTOR_GPU_BENCH_LOCK")
    )

    # Env force-enable beats an explicit disable; otherwise honor `disabled`.
    if disabled and not force_enabled:
        # No-op context manager: do not take the flock.
        yield
        return

    lock_dir = (
        os.environ.get("INDUCTOR_GPU_BENCH_LOCK_DIR")
        or os.environ.get("TORCHINDUCTOR_GPU_BENCH_LOCK_DIR")
        or os.environ.get("COMPILE_UTILS_GPU_LOCK_DIR")
        or os.path.join(tempfile.gettempdir(), "compile_utils_gpu_locks")
    )
    os.makedirs(lock_dir, exist_ok=True)

    visible = [
        d.strip()
        for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if d.strip()
    ]
    gpu = _safe_lock_component(visible[0] if visible else "0")
    lock_path = os.path.join(lock_dir, f"gpu_{gpu}.lock")
    gate_path = os.path.join(lock_dir, f"gpu_{gpu}.gate")

    state = _gpu_bench_lock_state()
    mutex = state["mutex"]
    local = state["local"]
    depth = getattr(local, "depth", 0)

    # Re-entrant: if we already hold the exclusive lock, just nest
    if depth > 0:
        current_mode = getattr(local, "mode", None)
        if current_mode == "exclusive":
            local.depth = depth + 1
            try:
                yield
            finally:
                local.depth -= 1
            return

    with mutex:
        fd = None
        gate_fd = None
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
            gate_fd = os.open(gate_path, os.O_CREAT | os.O_RDWR, 0o666)
            fcntl.flock(gate_fd, fcntl.LOCK_EX)
            fcntl.flock(fd, fcntl.LOCK_EX)
            _write_gpu_lock_metadata(fd, gpu, "exclusive", label)
            local.depth = 1
            local.mode = "exclusive"
            local.fd = fd
            try:
                yield
            finally:
                local.depth = 0
                local.mode = None
                local.fd = None
        finally:
            _release_fd(fd)
            _release_fd(gate_fd)


def _normalize_outputs(out) -> list[torch.Tensor]:
    """Normalize model outputs to a flat list of tensors."""
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result = []
        for item in out:
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, (tuple, list)):
                result.extend(_normalize_outputs(item))
        return result
    return []


def _get_device(inputs) -> torch.device:
    """Extract device from the first tensor in inputs."""
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            return inp.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _do_bench(fn, device: torch.device, warmup: int = 25, rep: int = 200) -> float:
    """Benchmark fn, returning time in microseconds (min-of-rep).

    Uses triton.testing.do_bench if available, otherwise manual timing.
    """
    try:
        import triton
        has_triton = True
    except ImportError:
        has_triton = False

    if has_triton and device.type == "cuda":
        from triton.testing import do_bench
        # do_bench returns milliseconds; convert to microseconds
        return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

    # Fallback: manual timing
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    best_us = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        best_us = min(best_us, elapsed * 1_000_000.0)
    return best_us


# ---------------------------------------------------------------------------
# Shared runner CLI: python -m oracle_harness <canonical_dir> [--check|--bench]
#
# Replaces the per-file main() of the old template: slim-format oracles are
# docstring + kernel + @oracle_impl only, and THIS is how they are checked
# and benched. Loads <dir>/oracle.py, resolves oracle_forward, and runs the
# same check/bench paths the old CLI exposed.
# ---------------------------------------------------------------------------

def _load_oracle_module(canonical_dir):
    import importlib.util
    from pathlib import Path

    d = Path(canonical_dir)
    path = d / "oracle.py"
    if not path.exists():
        candidates = sorted(d.glob("oracle_*.py"))
        if not candidates:
            raise FileNotFoundError(f"no oracle.py / oracle_*.py in {d}")
        path = candidates[0]
    spec = importlib.util.spec_from_file_location(
        f"_oracle_{d.name}", path)
    mod = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "oracle_forward", None)
    if fn is None:
        raise AttributeError(f"{path} defines no oracle_forward")
    return mod, fn, d


_INVALID_STATUSES = frozenset({
    "UNVERIFIED_NUMERICS",
    "INVALID_CUDAGRAPH_WARNING",
    "NUMERICS_WORSE_THAN_COMPILED",
    "NO_ORACLE_FOR_SHAPE",
})


def _runner_main(argv=None):
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(
        prog="python -m oracle_harness",
        description="Shared check/bench runner for slim-format oracles")
    ap.add_argument("canonical_dir",
                    help="repros/canonical/<repro_id> directory")
    ap.add_argument("--check", action="store_true",
                    help="numerics check vs Repro() at every shapes.json point")
    ap.add_argument("--no-skip-stochastic", action="store_true")
    ap.add_argument("--bench", action="store_true",
                    help="CUDAGraph bench at every point (bench_oracle). "
                    "Runs check first; refuses to report numbers for "
                    "points with failing numerics.")
    ap.add_argument("--bench-unchecked", action="store_true",
                    help="Bench WITHOUT prior check gate (escape hatch). "
                    "Prints a loud warning.")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--rep", type=int, default=100)
    ap.add_argument("--point", default=None,
                    help="restrict to one shape_hash")
    ap.add_argument("--disable-gpu-lock", action="store_true",
                    help="Skip the exclusive GPU bench lock (ON by default). "
                    "An INDUCTOR_GPU_BENCH_LOCK=1 env var force-enables the "
                    "lock even with this flag set.")
    args = ap.parse_args(argv)

    mod, fn, d = _load_oracle_module(args.canonical_dir)
    repro_id = d.name
    out = {"repro": repro_id, "oracle": getattr(mod, "__file__", "?")}

    do_bench = args.bench or args.bench_unchecked
    # --bench implies --check (bench gated on check)
    do_check = args.check or (args.bench and not args.bench_unchecked)

    if do_check:
        results = check_oracle_all_shapes(
            fn, d, repro_id,
            skip_stochastic=not args.no_skip_stochastic,
            point=args.point,
        )
        out["check"] = results

    if do_bench:
        if args.bench_unchecked:
            print("WARNING: --bench-unchecked bypasses the numerics check "
                  "gate. Results are NOT validated and should not be used "
                  "for official floor measurements.", file=sys.stderr)

        from repro_harness import load_shape_configs, make_inputs_from_config

        repro_file = d / "repro.py"
        configs = load_shape_configs(str(repro_file))

        bench_results = []
        if not configs:
            # Single default point
            check_passed = True
            if do_check and not args.bench_unchecked:
                chk = out.get("check", {})
                check_passed = all(
                    v not in ("fail", False) for v in chk.values())
            if check_passed or args.bench_unchecked:
                inputs = get_inputs(d)
                instance = get_repro_instance(d)
                bench_results.append(
                    bench_oracle(fn, instance, inputs, repro_id,
                                 warmup=args.warmup, rep=args.rep,
                                 disable_gpu_lock=args.disable_gpu_lock))
            else:
                bench_results.append({
                    "repro_id": repro_id,
                    "status": "UNVERIFIED_NUMERICS",
                })
        else:
            for label, config in configs.items():
                point_id = f"{repro_id}_{label}"
                # Gate: check must have passed for this point
                check_passed = True
                if do_check and not args.bench_unchecked:
                    chk = out.get("check", {})
                    check_passed = chk.get(label) not in ("fail", False)
                    if chk.get(label) is False:
                        check_passed = False

                if check_passed or args.bench_unchecked:
                    inputs = make_inputs_from_config(config)
                    instance = get_repro_instance(d)
                    bench_results.append(
                        bench_oracle(fn, instance, inputs, point_id,
                                     warmup=args.warmup, rep=args.rep,
                                     disable_gpu_lock=args.disable_gpu_lock))
                else:
                    bench_results.append({
                        "repro_id": point_id,
                        "status": "UNVERIFIED_NUMERICS",
                    })

        out["bench"] = bench_results

    print(_json.dumps(out, indent=1, default=str))

    # Exit code: nonzero if ANY point is failed/invalid
    failed = False
    chk = out.get("check")
    if isinstance(chk, dict):
        failed = any(v in ("fail", False) for v in chk.values())
    bench_out = out.get("bench")
    if isinstance(bench_out, list):
        for b in bench_out:
            if isinstance(b, dict):
                st = b.get("status", "")
                if st in _INVALID_STATUSES:
                    failed = True
                    break
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_runner_main())
