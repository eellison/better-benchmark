"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full RMSNorm backward repro with the same fused row epilogue plus partial column reduction that Inductor emits, whereas Inductor already uses this mixed-order schedule and only leaves the required partial collapse as a second reduction; Inductor cannot materially reduce the remaining time because the full bf16 matrix read/write and f32 partial traffic dominate; the fix is BANDWIDTH_BOUND: keep this fused layout and avoid changing the exact dtype and operation ordering."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


ROWS_PER_BLOCK = 64
XBLOCK = 8
PIPELINE_STAGES = 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _rmsnorm_bwd_mixed_order_kernel(
        scale_ptr,
        weight_ptr,
        x_ptr,
        rstd_ptr,
        out_ptr,
        partial_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        ROWS_PER_BLOCK_: tl.constexpr,
        XBLOCK_: tl.constexpr,
        PIPELINE_STAGES_: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, N)
        row_index = pid * ROWS_PER_BLOCK_ + tl.arange(0, XBLOCK_)

        scale0 = tl.load(scale_ptr + 0).to(tl.float32)
        scale = scale0.to(tl.float32)
        weight = tl.load(weight_ptr + cols, eviction_policy="evict_last")
        scaled_weight = scale * weight
        accum = tl.full([N], 0.0, tl.float32)

        for _ in tl.range(0, ROWS_PER_BLOCK_, XBLOCK_, num_stages=PIPELINE_STAGES_):
            row_mask = row_index < M
            x_offsets = row_index[:, None] * N + cols[None, :]
            x_bf16 = tl.load(
                x_ptr + x_offsets,
                mask=row_mask[:, None],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            rstd = tl.load(
                rstd_ptr + row_index,
                mask=row_mask,
                other=0.0,
                eviction_policy="evict_last",
            )[:, None]

            x_f32 = x_bf16.to(tl.float32)
            prod_weight_x = scaled_weight[None, :] * x_f32
            row_terms = tl.where(row_mask[:, None], prod_weight_x, 0.0)
            row_sum = tl.sum(row_terms, axis=1)[:, None].to(tl.float32)

            first_term = scaled_weight[None, :] * rstd
            neg_half = tl.full([1, 1], -0.5, tl.float32)
            scaled_row_sum = row_sum * neg_half
            rstd_sq = rstd * rstd
            rstd_cu = rstd_sq * rstd
            row_scale = scaled_row_sum * rstd_cu
            inv_n = tl.full([1, 1], 0.001953125, tl.float32)
            row_scale = row_scale * inv_n
            two = tl.full([1, 1], 2.0, tl.float32)
            doubled_x = x_f32 * two
            second_term = row_scale * doubled_x
            out_f32 = first_term + second_term
            out_f32 = out_f32.to(tl.float32)

            col_terms = x_f32 * rstd
            col_terms = scale * col_terms
            col_sum = tl.sum(col_terms, axis=0)
            accum = accum + col_sum

            tl.store(out_ptr + x_offsets, out_f32, mask=row_mask[:, None])
            row_index += XBLOCK_

        tl.store(partial_ptr + pid * N + cols, accum)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is not available")
    if len(inputs) != 7:
        raise ValueError(f"expected 7 inputs, got {len(inputs)}")

    arg3_1, arg0_1, arg2_1, arg1_1, shape0, shape1, shape2 = inputs
    if not isinstance(arg3_1, torch.Tensor) or arg3_1.shape != ():
        raise ValueError("expected bf16 scalar arg3_1")
    if arg3_1.dtype != torch.bfloat16:
        raise ValueError("expected arg3_1 dtype torch.bfloat16")
    if arg0_1.dtype != torch.bfloat16 or arg2_1.dtype != torch.float32 or arg1_1.dtype != torch.float32:
        raise ValueError("unexpected input dtype")
    if arg0_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    m = int(shape0[0])
    n = int(shape0[1])
    if n != 512:
        raise ValueError(f"expected N=512, got {n}")
    if list(shape1) != [n] or list(shape2) != [m, n]:
        raise ValueError("unexpected shape parameters")
    if tuple(arg0_1.shape) != (m, n) or tuple(arg2_1.shape) != (m, 1) or tuple(arg1_1.shape) != (n,):
        raise ValueError("unexpected tensor shape")
    if arg0_1.stride() != (n, 1) or arg2_1.stride() != (1, 1) or arg1_1.stride() != (1,):
        raise ValueError("unexpected tensor stride")

    return arg3_1, arg0_1, arg2_1, arg1_1, shape1, m, n


def oracle_forward(inputs):
    """Run the complete Repro.forward computation for the captured shape."""
    scale, x, rstd, weight, shape1, m, n = _validate_inputs(inputs)
    num_row_blocks = triton.cdiv(m, ROWS_PER_BLOCK)

    out1 = torch.empty_strided((m, n), (n, 1), device=x.device, dtype=torch.bfloat16)
    partials = torch.empty((num_row_blocks, n), device=x.device, dtype=torch.float32)

    _rmsnorm_bwd_mixed_order_kernel[(num_row_blocks,)](
        scale,
        weight,
        x,
        rstd,
        out1,
        partials,
        M=m,
        N=n,
        ROWS_PER_BLOCK_=ROWS_PER_BLOCK,
        XBLOCK_=XBLOCK,
        PIPELINE_STAGES_=PIPELINE_STAGES,
        num_warps=2,
        num_stages=1,
    )
    out0 = partials.sum(dim=0).view(shape1)
    return (out0, out1)


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
