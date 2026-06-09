"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the split-indexed SwiGLU-style pointwise math with the final cat+permute layout store into one full-scope Triton kernel, whereas Inductor must schedule the arithmetic through the captured split/cat/permute view pattern; Inductor cannot reliably sink this producer into the transposed output layout for this layout_indexing_stencil_fusion case because the scheduler treats the layout reconstruction as a separate indexing concern; the fix is SCHEDULER_FUSION: teach scheduler/codegen to fuse split/cat/permute layout indexing into the producing pointwise store."""
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
    def _layout_fused_swiglu_kernel(
        arg77_ptr,
        grouped_ptr,
        out_ptr,
        total: tl.constexpr,
        d: tl.constexpr,
        arg77_s0,
        arg77_s1,
        grouped_s0,
        grouped_s1,
        out_s0,
        out_s1,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        rows = offsets // d
        cols = offsets - rows * d

        x0 = tl.load(arg77_ptr + rows * arg77_s0 + cols * arg77_s1, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(arg77_ptr + rows * arg77_s0 + (cols + d) * arg77_s1, mask=mask, other=0.0).to(tl.float32)
        grouped = tl.load(grouped_ptr + rows * grouped_s0 + cols * grouped_s1, mask=mask, other=0.0).to(tl.float32)

        exp_neg = tl.exp(-x0)
        denom = exp_neg + 1.0
        sigmoid = 1.0 / denom

        # Match the repro's explicit bf16 boundary after x * sigmoid(x).
        silu_bf16 = (x0 / denom).to(tl.bfloat16)
        gated = grouped * silu_bf16.to(tl.float32)

        # Match grouped*x1 being materialized as bf16 before the f32 epilogue.
        grouped_x1_bf16 = (grouped * x1).to(tl.bfloat16)
        grad = grouped_x1_bf16.to(tl.float32) * sigmoid * (x0 * (1.0 - sigmoid) + 1.0)

        base_out = rows * out_s1 + cols * out_s0
        tl.store(out_ptr + base_out, grad, mask=mask)
        tl.store(out_ptr + base_out + d * out_s0, gated, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([16384, 1536], bf16), T([16384, 768], bf16))")
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
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    arg77_1, grouped_mm_13 = inputs
    m = arg77_1.shape[0]
    d = grouped_mm_13.shape[1]
    if arg77_1.shape != (m, 2 * d) or grouped_mm_13.shape != (m, d):
        raise ValueError(
            f"unexpected input shapes: arg77_1={tuple(arg77_1.shape)} "
            f"grouped_mm_13={tuple(grouped_mm_13.shape)}"
        )

    out = torch.empty_strided(
        (2 * d, m),
        (1, 2 * d),
        device=arg77_1.device,
        dtype=arg77_1.dtype,
    )
    total = m * d
    block = 512
    grid = (triton.cdiv(total, block),)
    _layout_fused_swiglu_kernel[grid](
        arg77_1,
        grouped_mm_13,
        out,
        total,
        d,
        arg77_1.stride(0),
        arg77_1.stride(1),
        grouped_mm_13.stride(0),
        grouped_mm_13.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK=block,
        num_warps=4,
    )
    return out


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
