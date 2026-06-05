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
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import sys
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

    # --- Compile with coordinate_descent + capture CUDAGraph ---
    import torch._inductor.config as cfg
    cfg.coordinate_descent_tuning = True
    compiled = torch.compile(instance)
    with torch.no_grad():
        for _ in range(5):
            compiled(*inputs)
        torch.cuda.synchronize()
        g_compile = _try_capture_graph(lambda: compiled(*inputs))

    # --- Warm oracle + capture CUDAGraph ---
    with torch.no_grad():
        for _ in range(3):
            oracle_forward(inputs)
        torch.cuda.synchronize()
        g_oracle = _try_capture_graph(lambda: oracle_forward(inputs))

    # --- Adaptive rep count based on quick estimate ---
    quick_oracle = _time_graph_or_fn(g_oracle, lambda: oracle_forward(inputs),
                                     warmup=5, rep=10)
    quick_compile = _time_graph_or_fn(g_compile, lambda: compiled(*inputs),
                                      warmup=5, rep=10)
    est_us = min(quick_oracle, quick_compile)
    if est_us < 50:
        rep = max(rep, 500)
    elif est_us < 200:
        rep = max(rep, 300)

    # --- Interleaved timing under EXCLUSIVE GPU LOCK ---
    best_oracle_us = math.inf
    best_compile_us = math.inf

    with _gpu_exclusive_lock(repro_id):
        # Warm under lock
        for _ in range(warmup):
            if g_oracle is not None:
                g_oracle.replay()
            else:
                with torch.no_grad():
                    oracle_forward(inputs)
            if g_compile is not None:
                g_compile.replay()
            else:
                with torch.no_grad():
                    compiled(*inputs)
        torch.cuda.synchronize()

        # Interleaved rounds: oracle then compile, min-of-N
        for _ in range(rounds):
            oracle_us = _time_graph_or_fn(g_oracle, lambda: oracle_forward(inputs),
                                          warmup=warmup, rep=rep)
            compile_us = _time_graph_or_fn(g_compile, lambda: compiled(*inputs),
                                           warmup=warmup, rep=rep)
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

def _try_capture_graph(fn):
    """Capture a CUDAGraph of fn(). Returns graph or None on failure."""
    try:
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g
    except Exception as e:
        import warnings
        warnings.warn(f"CUDAGraph capture failed, falling back to direct call: {e}")
        return None


def _time_graph_or_fn(graph, fallback_fn, warmup, rep):
    """Time a CUDAGraph replay (or fallback callable). Returns microseconds."""
    from triton.testing import do_bench
    if graph is not None:
        ms = do_bench(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="min")
    else:
        ms = do_bench(fallback_fn, warmup=warmup, rep=rep, return_mode="min")
    return ms * 1000.0


@contextlib.contextmanager
def _gpu_exclusive_lock(label="oracle_bench"):
    """Acquire exclusive GPU lock using scripts/gpu_lock.py infrastructure."""
    try:
        # Try importing from scripts directory
        import importlib.util as _ilu
        _repo_root = Path(__file__).resolve().parent
        _lock_path = _repo_root / "scripts" / "gpu_lock.py"
        if _lock_path.exists():
            _spec = _ilu.spec_from_file_location("gpu_lock", _lock_path)
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            with _mod.gpu_lock(0, label=label):
                yield
            return
    except Exception:
        pass
    # Fallback: no lock available (single-GPU, no contention expected)
    yield


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
