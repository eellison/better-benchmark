"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete NFNet exact-erf GELU scale scope in one Triton pointwise kernel, including the fp16 input load, fp32 `0.5*x*(erf(0.7071067811865476*x)+1)` computation, explicit fp16 rounding, final `1.7015043497085571` multiply, and contiguous fp16 output, whereas Inductor lowers the same full pointwise graph through its generic fused pointwise path; Inductor cannot materially improve this local scope with scheduler fusion because the captured graph has no reduction, scatter, stencil, or layout producer left to eliminate and the remaining work is the required read, transcendental math, rounding, multiply, and write; the fix is BANDWIDTH_BOUND: record this as already at the pointwise math/memory floor unless broader pointwise codegen or launch-overhead improvements move the baseline."""
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _gelu_scale_kernel(
        input_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        erf_term = tl.math.erf(x * 0.7071067811865476) + 1.0
        gelu = (x * 0.5 * erf_term).to(tl.float16)
        out = (gelu * 1.7015043497085571).to(tl.float16)
        tl.store(output_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([128, 768, 6, 6], f16))")
def oracle_forward(inputs):
    """Run the full Repro.forward exact-erf GELU-plus-scale scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gelu_scaled.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects 1 input, got {len(inputs)}")

    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor")
    if x.dtype != torch.float16:
        raise TypeError(f"{REPRO_ID} expects f16 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA input")
    if not x.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects the captured contiguous input layout")

    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float16,
    )
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _gelu_scale_kernel[grid](x, output, n_elements=n_elements)
    return output


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
