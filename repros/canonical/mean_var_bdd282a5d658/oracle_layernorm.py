"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle expresses the input add, mean reduction, unbiased variance reduction, normalization, scale, bias, and three aliased output views as one hand-written Triton row kernel, whereas Inductor already emits a comparable single fused persistent reduction kernel for this repro; Inductor cannot materially improve this today because the remaining cost is dominated by required reads of the two full inputs plus scale/bias and the single full output write, not by a missing graph-level fusion; the fix is BANDWIDTH_BOUND: there is no scheduler-pattern fix for this repro beyond lower-level memory/reduction tuning or changing the mathematical contract."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def oracle_kernel(
        addmm_ptr,
        add_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        offsets = row * N + cols

        x = (
            tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
            + tl.load(add_ptr + offsets, mask=mask, other=0.0)
        )
        mean = tl.sum(x, axis=0) / N
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / (N - 1)
        inv_denom = 1.0 / (tl.sqrt(var) + 1.0e-6)
        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        out = scale * centered * inv_denom + bias
        tl.store(out_ptr + offsets, out, mask=mask)


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _torch_fallback(inputs):
    addmm_65, add_74, arg181_1, arg182_1, shape0, shape1, shape2, shape3 = inputs
    x = add_74 + addmm_65.view(shape0)
    mean = torch.mean(x, dim=-1, keepdim=True)
    centered = x - mean
    scaled = arg181_1 * centered
    var = torch.var(x, dim=-1, correction=1, keepdim=True)
    out3d = scaled / (torch.sqrt(var) + 1.0e-6) + arg182_1
    return (out3d.view(shape1), out3d.view(shape2), out3d.view(shape3))


@oracle_impl(hardware="H100", shapes="(T([16384, 768], f32), T([128, 128, 768], f32), T([768], f32), T([768], f32), S([128, 128, 768]), S([16384, 768]), S([16384, 768]), S([16384, 768]))")
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
    addmm_65, add_74, arg181_1, arg182_1, shape0, shape1, shape2, shape3 = inputs
    if triton is None or not addmm_65.is_cuda:
        return _torch_fallback(inputs)

    n = arg181_1.numel()
    m = addmm_65.numel() // n
    block_n = _next_power_of_2(n)
    out = torch.empty_strided(tuple(shape1), (n, 1), device=addmm_65.device, dtype=addmm_65.dtype)
    oracle_kernel[(m,)](
        addmm_65,
        add_74,
        arg181_1,
        arg182_1,
        out,
        M=m,
        N=n,
        BLOCK_N=block_n,
    )
    return (out.view(shape1), out.view(shape2), out.view(shape3))


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
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
