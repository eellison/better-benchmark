"""
Oracle for <REPRO_ID>

Gap diagnosis:
  Classification: <SCHEDULER_FUSION|COOPERATIVE_SPLIT_K|SCATTER_REDUCE|ONLINE_SOFTMAX|...>
  What oracle does differently: <1 sentence>
  What Inductor change would fix: <1 sentence>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration ---
REPRO_ID = "<repro_id>"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"


# --- Load repro module ---
def _load_repro_module():
    """Load the repro.py module from this oracle's directory."""
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    mod = _load_repro_module()
    if hasattr(mod, "make_inputs"):
        return mod.make_inputs()
    elif hasattr(mod, "_default_make_inputs"):
        return mod._default_make_inputs()
    else:
        raise RuntimeError("Repro has no make_inputs or _default_make_inputs")


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    mod = _load_repro_module()
    return mod.Repro()


# --- Oracle kernel(s) ---
# Replace this section with your optimized Triton kernel(s).

if triton is not None:

    @triton.jit
    def oracle_kernel(
        # TODO: Define kernel parameters
        # input_ptr, output_ptr, ...
        # N: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """TODO: Implement optimized kernel."""
        pass


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    # TODO: Implement oracle using the Triton kernel(s) above.
    # Example:
    #   x = inputs[0]
    #   output = torch.empty_like(x)
    #   grid = (x.shape[0],)
    #   oracle_kernel[grid](x, output, N=x.shape[1], BLOCK_N=8192)
    #   return output
    raise NotImplementedError("Replace with oracle implementation")


# --- Standard interface: --check ---
def run_check(inputs, *, rtol: float = 1e-2, atol: float = 1e-2) -> bool:
    """Verify oracle produces same outputs as eager Repro.

    Enforces the scope invariant: same inputs, same output structure.
    Prints per-output PASS/FAIL with max_diff for diagnostics.

    Returns True if all outputs pass.
    """
    instance = get_repro_instance()

    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)

    # Normalize to list
    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)

    # Scope check: output count must match
    if len(oracle_list) != len(eager_list):
        print(f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
              f"eager produces {len(eager_list)}")
        return False

    all_pass = True
    for i, (e, o) in enumerate(zip(eager_list, oracle_list)):
        # Shape check
        if e.shape != o.shape:
            print(f"  output {i}: SCOPE_MISMATCH shape oracle={list(o.shape)} "
                  f"eager={list(e.shape)}")
            all_pass = False
            continue

        # Dtype check (warn but don't fail if dtypes differ - cast for comparison)
        if e.dtype != o.dtype:
            print(f"  output {i}: WARNING dtype mismatch oracle={o.dtype} eager={e.dtype}")

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


# --- Standard interface: --bench ---
def run_bench(inputs, *, warmup: int = 25, rep: int = 200) -> dict:
    """Benchmark oracle vs torch.compile.

    Returns a JSON-serializable dict with machine-parseable timing results.
    Prints a single JSON line to stdout for automated consumption.
    """
    device = _get_device(inputs)

    # Oracle timing
    with torch.no_grad():
        oracle_forward(inputs)  # warmup allocation
        if device.type == "cuda":
            torch.cuda.synchronize()

        oracle_us = _do_bench(lambda: oracle_forward(inputs), device,
                              warmup=warmup, rep=rep)

    # Compile timing
    instance = get_repro_instance()
    compiled = torch.compile(instance)
    with torch.no_grad():
        for _ in range(5):
            compiled(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()

        compile_us = _do_bench(lambda: compiled(*inputs), device,
                               warmup=warmup, rep=rep)

    ratio = compile_us / oracle_us if oracle_us > 0 else 0.0
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 2),
        "compile_us": round(compile_us, 2),
        "ratio": round(ratio, 3),
        "status": "GOOD" if ratio > 1.0 else "BAD_ORACLE",
    }
    print(json.dumps(result))
    return result


# --- Helpers ---
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
    if triton is not None and device.type == "cuda":
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


# --- CLI entry point ---
def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    args = parser.parse_args()

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = run_check(inputs, rtol=args.rtol, atol=args.atol)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        result = run_bench(inputs, warmup=args.warmup, rep=args.rep)
        if result["status"] == "BAD_ORACLE":
            print(f"WARNING: oracle is slower than compile "
                  f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
