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
  - OracleRegistry: per-file registration-based oracle dispatch
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
    """Per-file oracle registry supporting (hardware, shape) dispatch.

    Each oracle file creates its own registry instance. Registered oracle
    implementations are standalone callables (their own kernels, own autotune).

    Dispatch priority (first match wins):
      1. Exact: hardware matches AND shape predicate passes
      2. Shape-only: no hardware constraint, shape predicate passes
      3. Hardware-only: hardware matches, no shape constraint
      4. Default: no constraints at all

    Usage in an oracle file:
        from oracle_harness import OracleRegistry

        registry = OracleRegistry()

        @registry.register(hardware="B200", shape=lambda inputs: inputs[0].shape[-1] <= 1024)
        def oracle_persistent_small(inputs):
            ...

        @registry.register(hardware="B200", shape=lambda inputs: inputs[0].shape[-1] > 1024)
        def oracle_split_k(inputs):
            ...

        @registry.register()  # default fallback
        def oracle_default(inputs):
            ...

        def oracle_forward(inputs):
            return registry.dispatch(inputs)
    """

    def __init__(self):
        self._entries = []  # list of (hardware, shape_pred, fn, description)

    def register(self, hardware=None, shape=None, description=None):
        """Decorator to register an oracle implementation.

        Args:
            hardware: GPU kind string (e.g. "B200", "H100") or None for any.
            shape: Callable(inputs) -> bool predicate, or None for any shape.
            description: Human-readable string documenting what this variant
                targets (e.g. "persistent single-pass, inner <= 1024").

        Returns:
            Decorator that registers the function and returns it unchanged.
        """
        def decorator(fn):
            desc = description or fn.__name__
            self._entries.append((hardware, shape, fn, desc))
            return fn
        return decorator

    def dispatch(self, inputs):
        """Find best matching oracle and call it.

        Lookup order:
          1. hardware+shape match (both constraints satisfied)
          2. shape-only match (hardware=None, shape predicate passes)
          3. hardware-only match (hardware matches, shape=None)
          4. default (both hardware=None and shape=None)

        Raises RuntimeError if no matching implementation is found.
        """
        current_hw = get_gpu_kind()

        # Tier 1: exact match (hardware + shape)
        for hw, shape_pred, fn, _desc in self._entries:
            if hw is not None and hw == current_hw and shape_pred is not None:
                try:
                    if shape_pred(inputs):
                        return fn(inputs)
                except Exception:
                    continue

        # Tier 2: shape-only match (hardware=None, shape predicate passes)
        for hw, shape_pred, fn, _desc in self._entries:
            if hw is None and shape_pred is not None:
                try:
                    if shape_pred(inputs):
                        return fn(inputs)
                except Exception:
                    continue

        # Tier 3: hardware-only match (hardware matches, no shape constraint)
        for hw, shape_pred, fn, _desc in self._entries:
            if hw is not None and hw == current_hw and shape_pred is None:
                return fn(inputs)

        # Tier 4: default (no constraints)
        for hw, shape_pred, fn, _desc in self._entries:
            if hw is None and shape_pred is None:
                return fn(inputs)

        raise RuntimeError(
            f"No oracle registered for hardware={current_hw!r} with given input shapes. "
            f"Registry has {len(self._entries)} entries."
        )

    def list_entries(self):
        """List registered entries for debugging."""
        entries = []
        for hw, shape_pred, fn, desc in self._entries:
            entries.append({
                "hardware": hw,
                "has_shape_pred": shape_pred is not None,
                "fn_name": fn.__name__,
                "description": desc,
            })
        return entries

    def clear(self):
        """Clear all registrations (useful for testing)."""
        self._entries.clear()


# Module-level convenience for simple cases (single oracle file, not imported
# by multiple files simultaneously). For multi-file use, instantiate your own
# OracleRegistry per file.
_default_registry = OracleRegistry()


def register_oracle(hardware=None, shape=None, description=None):
    """Module-level decorator using the default global registry.

    For per-file isolation (recommended), use OracleRegistry() directly.
    """
    return _default_registry.register(hardware=hardware, shape=shape, description=description)


def dispatch_oracle(inputs):
    """Module-level dispatch using the default global registry."""
    return _default_registry.dispatch(inputs)


def reset_oracle_registry():
    """Reset the default global registry (call between oracle file loads)."""
    _default_registry.clear()


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
        Dict with repro_id, oracle_us, compile_us, ratio, status.
    """
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
