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
from pathlib import Path

import torch


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
            # 0-d key input shape (scalar tensor)
            return ((),), False
        if all(isinstance(d, int) for d in shape):
            return (shape,), False
        if all(isinstance(s, (tuple, list, torch.Size)) for s in shape):
            return tuple(tuple(s) for s in shape), True
        raise TypeError(
            f"shape must be a tuple of ints or a tuple of tuples, got {shape!r}"
        )

    # -- shape distance ------------------------------------------------------

    @staticmethod
    def _dim_distance(actual_dims, registered_dims):
        """Distance between two single-tensor shapes.

        Same rank required (else inf). Distance is the sum of
        |log(actual_dim / registered_dim)| across dims, so 0.0 means exact.
        """
        if len(actual_dims) != len(registered_dims):
            return math.inf
        dist = 0.0
        for a, r in zip(actual_dims, registered_dims):
            if a == r:
                continue
            if a <= 0 or r <= 0:
                return math.inf
            dist += abs(math.log(a / r))
        return dist

    @classmethod
    def _entry_distance(cls, entry, actual_shapes):
        """Distance between an entry's registered shape and actual input shapes.

        Returns math.inf for incompatible rank/arity, None for entries with no
        shape constraint, 0.0 for exact matches.
        """
        if entry["shape"] is None:
            return None
        if not actual_shapes:
            return math.inf
        if entry["full_sig"]:
            if len(entry["shape"]) != len(actual_shapes):
                return math.inf
            return sum(
                cls._dim_distance(a, r)
                for a, r in zip(actual_shapes, entry["shape"])
            )
        # Key-shape registration: compare against the first tensor input.
        return cls._dim_distance(actual_shapes[0], entry["shape"][0])

    # -- dispatch ------------------------------------------------------------

    def select(self, inputs):
        """Select the best-matching entry for the given inputs (no call).

        Tiers (first tier with a match wins; within a tier the smallest shape
        distance wins, with registration order breaking ties):
          1. exact (hardware, shape): hardware == current GPU and distance == 0
          2. same shape, any hardware: distance == 0
          3. same hardware, nearest shape: hardware == current GPU, finite
             distance (entries with hardware match but no shape constraint
             qualify after any finite-distance entry)
          4. any registration, nearest shape: finite distance on any hardware
          5. default: registered with no constraints at all

        Returns (entry, info_dict). Also stores info_dict on
        self.last_dispatch_info. Raises RuntimeError if nothing matches.
        """
        current_hw = get_gpu_kind()
        actual_shapes = get_shape_key(inputs)

        scored = []  # (entry, distance_or_None, hw_match)
        for entry in self._entries:
            hw = entry["hardware"]
            hw_match = (hw is not None and hw == current_hw)
            dist = self._entry_distance(entry, actual_shapes)
            scored.append((entry, dist, hw_match))

        def pick(candidates):
            """Pick smallest-distance candidate; ties go to registration order."""
            best = None
            best_dist = math.inf
            for entry, dist, _hw in candidates:
                d = 0.0 if dist is None else dist
                if d < best_dist:
                    best, best_dist = entry, d
            return best, best_dist

        chosen = None
        tier = None
        tier_name = None
        distance = None

        # Tier 1: exact (hardware, shape)
        t1 = [(e, d, h) for e, d, h in scored if h and d == 0.0]
        if t1:
            chosen, distance = t1[0][0], 0.0
            tier, tier_name = 1, "exact_hw_shape"

        # Tier 2: same shape, any hardware
        if chosen is None:
            t2 = [(e, d, h) for e, d, h in scored if d == 0.0]
            if t2:
                chosen, distance = t2[0][0], 0.0
                tier, tier_name = 2, "shape_any_hw"

        # Tiers 3/4 are nearest-SHAPE fallback: running an implementation at a
        # shape it wasn't written for. That is only safe for impls explicitly
        # marked shape-general (exact=False). Shape-specific impls
        # (exact=True, the default) match their declared signature only —
        # silently running them at the wrong shape is the footgun this guards.
        def _general(e):
            return not e.get("exact", True)

        # Tier 3: same hardware, nearest shape (shape-general impls only)
        if chosen is None:
            t3 = [(e, d, h) for e, d, h in scored
                  if h and d is not None and math.isfinite(d) and d > 0.0
                  and _general(e)]
            if t3:
                chosen, distance = pick(t3)
                tier, tier_name = 3, "hw_nearest_shape"
            else:
                # Hardware match with no shape constraint ("any shape on this
                # hardware") — still a hardware-correct choice.
                t3b = [(e, d, h) for e, d, h in scored if h and d is None]
                if t3b:
                    chosen, distance = t3b[0][0], None
                    tier, tier_name = 3, "hw_no_shape_constraint"

        # Tier 4: any registration, nearest shape (shape-general impls only)
        if chosen is None:
            t4 = [(e, d, h) for e, d, h in scored
                  if d is not None and math.isfinite(d) and d > 0.0
                  and _general(e)]
            if t4:
                chosen, distance = pick(t4)
                tier, tier_name = 4, "any_nearest_shape"

        # Tier 5: default (no constraints at all)
        if chosen is None:
            t5 = [(e, d, h) for e, d, h in scored
                  if e["hardware"] is None and e["shape"] is None]
            if t5:
                chosen, distance = t5[0][0], None
                tier, tier_name = 5, "default"

        if chosen is None:
            self.last_dispatch_info = None
            raise RuntimeError(
                f"No oracle registered for hardware={current_hw!r} "
                f"shapes={actual_shapes!r}. Registry has {len(self._entries)} "
                f"entries: {[e['description'] for e in self._entries]}"
            )

        info = {
            "tier": tier,
            "tier_name": tier_name,
            "fallback": tier > 1,  # tier > 1 => may be suboptimal, worth logging
            "fn_name": chosen["fn"].__name__,
            "description": chosen["description"],
            "registered_hardware": chosen["hardware"],
            "registered_shape": chosen["shape"],
            "configs": chosen["configs"],
            "distance": distance,
            "current_hardware": current_hw,
            "actual_shapes": actual_shapes,
        }
        self.last_dispatch_info = info
        return chosen, info

    def dispatch(self, inputs):
        """Find the best-matching oracle and call it.

        Calls fn(inputs, **configs) if the matched registration has configs,
        else fn(inputs). After this returns, self.last_dispatch_info describes
        which tier matched (tier > 1 means fallback dispatch — the chosen
        implementation was tuned for different hardware and/or shape and may
        be suboptimal).
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
# `exact` semantics (footgun protection):
#   - exact=True (DEFAULT): the impl has baked-in shape assumptions
#     (hardcoded BLOCK dividing rnumel, persistent kernel fitting only small
#     rnumel, unrolled loops). It matches its declared signature ONLY and is
#     never selected by nearest-shape fallback.
#   - exact=False: the impl is shape-general (grid computed from input dims).
#     It is eligible for nearest-shape dispatch at other shapes.
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


def parse_shapes_signature(shapes):
    """Parse a T()/S() signature string into a tuple of shape tuples.

    Accepts the exact `_shapes_config` / shapes.txt format, e.g.
    "(T([512, 256, 13, 13], f32), T([512, 128, 27, 27], b8), S([131072, 169]))"
    Only tensor entries (T) contribute shapes; S() entries are shape params,
    not tensors, and are skipped. dtypes are ignored for matching.
    Already-parsed tuples of tuples pass through unchanged.
    """
    if shapes is None:
        return None
    if not isinstance(shapes, str):
        # programmatic: tuple of shape tuples
        return tuple(tuple(s) for s in shapes)

    def T(shape, dtype=None, *args, **kwargs):
        return ("T", tuple(shape))

    def S(dims):
        return ("S", tuple(dims))

    def Index(high, low=0):
        return None

    def Perm(size=None):
        return None

    ns = {"__builtins__": {}, "T": T, "S": S, "Index": Index, "Perm": Perm}
    for d in ("f32", "f16", "bf16", "f64", "i64", "i32", "i16", "i8", "b8", "u8"):
        ns[d] = d
    parsed = eval(shapes, ns)  # noqa: S307 - trusted repo-internal strings

    def is_entry(x):
        return (isinstance(x, tuple) and len(x) == 2 and x[0] in ("T", "S")
                and isinstance(x[1], tuple))

    # Single-tensor signatures without a trailing comma — "(T([8192], bf16))"
    # — eval to the entry itself rather than a 1-tuple of entries.
    if is_entry(parsed):
        parsed = (parsed,)
    elif not isinstance(parsed, tuple):
        parsed = (parsed,)
    return tuple(entry[1] for entry in parsed
                 if is_entry(entry) and entry[0] == "T")


def oracle_impl(hardware=None, shapes=None, configs=None, exact=True,
                description=None):
    """Register an oracle implementation: one line declaring what it was
    written for.

    Args:
        hardware: GPU kind the impl was tuned on ("H100", "B200", ...).
            None = no hardware constraint.
        shapes: FULL input signature the impl was written for, as the
            T()/S() string from the repro's `_shapes_config` (copy it
            verbatim), or a pre-parsed tuple of shape tuples.
            None = no shape constraint (unconstrained default).
        configs: optional launch-config kwargs passed through at dispatch:
            fn(inputs, **configs). Lets one kernel body serve many
            (hardware, shapes) points.
        exact: True (default) = shape-specific, matches declared signature
            only. False = shape-general, eligible for nearest-shape fallback.
        description: optional human-readable note.

    Returns the function unchanged, so one body can be registered N times.
    """
    parsed = parse_shapes_signature(shapes)

    def decorator(fn):
        reg = _module_registries.setdefault(fn.__module__, OracleRegistry())
        entry_shapes = parsed if parsed is None else tuple(parsed)
        dec = reg.register(hardware=hardware,
                           shape=entry_shapes,
                           configs=configs,
                           description=description)
        dec(fn)
        # Stamp exactness on the entry we just added.
        reg._entries[-1]["exact"] = bool(exact)
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
    info = dict(info)
    info["exact"] = entry.get("exact", True)
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

    Strategy: run with explicit seed, then run WITHOUT seeding. Outputs that
    differ are stochastic (dropout, etc.). We also do a seeded-vs-seeded run
    to distinguish truly stochastic ops (those that ignore the seed) from
    seed-respecting randomness.
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
        if not o_s.is_floating_point():
            continue
        if not torch.allclose(o_s.float(), o_u.float(), atol=1e-6, rtol=0):
            stochastic.add(i)

    return stochastic


# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------

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
    stochastic = detect_stochastic_outputs(instance, inputs) if skip_stochastic else set()

    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)

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
        max_diff = (e_f32 - o_f32).abs().max().item()
        ok = torch.allclose(e_f32, o_f32, atol=atol, rtol=rtol)

        status = "PASS" if ok else "FAIL"
        print(f"  output {i}: {status} (shape={list(e.shape)} dtype={e.dtype} "
              f"max_diff={max_diff:.2e})")
        if not ok:
            all_pass = False

    return all_pass


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
) -> dict:
    """Standard oracle benchmark: oracle vs torch.compile.

    Delegates to bench_compare's methodology: CUDAGraph capture+replay,
    exclusive GPU lock, interleaved timing rounds. This ensures oracle
    measurements use the exact same infrastructure as corpus-wide sweeps.

    Args:
        oracle_forward: callable that takes inputs and returns oracle outputs
        instance: Repro() instance
        inputs: list of input tensors/values
        repro_id: string identifier for the repro
        warmup: warmup iterations per round
        rep: repetitions per round
        rounds: number of interleaved rounds (min-of-N)

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
        g_compile = _capture_graph(lambda: compiled(*inputs))

    # --- Warm oracle + capture CUDAGraph (MANDATORY) ---
    with torch.no_grad():
        for _ in range(3):
            oracle_forward(inputs)
        torch.cuda.synchronize()
        g_oracle = _capture_graph(lambda: oracle_forward(inputs))

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

    with _gpu_exclusive_lock(repro_id):
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
    if dispatch_info is not None:
        result["dispatch"] = {
            "tier": dispatch_info["tier"],
            "tuned_hardware": dispatch_info["registered_hardware"],
            "current_hardware": dispatch_info["current_hardware"],
            "exact": dispatch_info.get("exact", True),
            "fallback": dispatch_info["fallback"],
        }
    print(json.dumps(result))
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


def bench_oracle_all_shapes(oracle_forward, repro_dir, repro_id, **kwargs):
    """Benchmark oracle across all shapes in shapes.txt.

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
    shapes_file = repro_dir / "shapes.txt"
    if not shapes_file.exists():
        # Just run default
        inputs = get_inputs(repro_dir)
        instance = get_repro_instance(repro_dir)
        return [bench_oracle(oracle_forward, instance, inputs, repro_id, **kwargs)]

    results = []
    configs = load_shape_configs(shapes_file)
    for config in configs:
        inputs = make_inputs_from_config(config)
        instance = get_repro_instance(repro_dir)
        result = bench_oracle(oracle_forward, instance, inputs, f"{repro_id}_{config.label}", **kwargs)
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
def _gpu_exclusive_lock(label="oracle_bench"):
    """Acquire exclusive GPU lock. NEVER silently skips.

    Respects the same env-var protocol as bench_compare:
      - INDUCTOR_GPU_BENCH_LOCK=1 or TORCHINDUCTOR_GPU_BENCH_LOCK=1: acquire lock
      - Neither set: raise RuntimeError (caller must explicitly configure)

    This ensures benchmarks never run without explicit lock configuration,
    preventing silent measurement corruption from GPU contention.
    """
    lock_enabled = (
        _env_flag_enabled("INDUCTOR_GPU_BENCH_LOCK")
        or _env_flag_enabled("TORCHINDUCTOR_GPU_BENCH_LOCK")
    )

    if not lock_enabled:
        raise RuntimeError(
            "GPU bench lock is not enabled. Set INDUCTOR_GPU_BENCH_LOCK=1 (or "
            "TORCHINDUCTOR_GPU_BENCH_LOCK=1) before running oracle benchmarks. "
            "The lock is required to prevent measurement corruption from GPU "
            "contention. If you are certain no other GPU work is running, set "
            "the env var to enable the lock anyway (it will be a no-op if "
            "uncontended)."
        )

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
