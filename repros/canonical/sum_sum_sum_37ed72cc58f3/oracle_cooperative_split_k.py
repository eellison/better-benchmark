"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete `sum_sum_sum_37ed72cc58f3` Swin window-reverse/indexed layer-norm-backward/drop-path return tuple by applying the dynamic height/width index gather, reducing each 256-wide row for the input-gradient epilogue, writing the returned `[256, 100352]` transposed side output, and accumulating all three returned `[256]` column reductions from the same row-tiled producer, whereas Inductor currently schedules the window layout clone/indexing, row reductions, residual/drop-path pointwise epilogue, transposed side-output store, and sibling reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps dynamic indexed layout reconstruction, row-local reductions, a dependent transposed side output, and sibling column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible row-tiled reductions across the indexed window producer, fuse the dependent epilogue/store, and finalize the sibling channel sums together."""
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


REPRO_ID = "sum_sum_sum_37ed72cc58f3"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
HEIGHT = 28
WIDTH = 28
WINDOW = 7
WINDOWS_H = HEIGHT // WINDOW
WINDOWS_W = WIDTH // WINDOW
TOKENS = HEIGHT * WIDTH
ROWS = BATCH * TOKENS
C = 256
KEEP_PROB = 0.9913043472915888

ROW_SPLIT = 24
XBLOCK = 1
KERNEL_WARPS = 4


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_three_vectors_kernel(
        out_sum_x_rhs_ptr,
        out_sum_x_ptr,
        out_sum_out_ptr,
        C_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = c < C_
        zeros = tl.zeros((BLOCK_C,), dtype=tl.float32)
        tl.store(out_sum_x_rhs_ptr + c, zeros, mask=mask)
        tl.store(out_sum_x_ptr + c, zeros, mask=mask)
        tl.store(out_sum_out_ptr + c, zeros, mask=mask)

    @triton.jit
    def _indexed_swin_atomic_kernel(
        mm_ptr,
        index_ptr,
        weight_ptr,
        rhs_ptr,
        scale_ptr,
        residual_ptr,
        keep_ptr,
        out_transposed_ptr,
        out_sum_x_rhs_ptr,
        out_sum_x_ptr,
        out_sum_out_ptr,
        ROWS_: tl.constexpr,
        C_: tl.constexpr,
        TOKENS_: tl.constexpr,
        WIDTH_: tl.constexpr,
        WINDOW_: tl.constexpr,
        WINDOWS_H_: tl.constexpr,
        WINDOWS_W_: tl.constexpr,
        KEEP_PROB_: tl.constexpr,
        ROW_SPLIT: tl.constexpr,
        XBLOCK: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        c = tl.arange(0, BLOCK_C)
        c_mask = c < C_
        weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

        acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_out = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for start in tl.range(0, ROW_SPLIT, XBLOCK):
            row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
            row_mask = row < ROWS_
            out_mask = row_mask[:, None] & c_mask[None, :]
            out_offsets = row[:, None] * C_ + c[None, :]

            batch = row // TOKENS_
            token = row - batch * TOKENS_
            h = token // WIDTH_
            w = token - h * WIDTH_
            src_h = tl.load(index_ptr + h, mask=row_mask, other=0)
            src_w = tl.load(index_ptr + w, mask=row_mask, other=0)
            hwin = src_h // WINDOW_
            wh = src_h - hwin * WINDOW_
            wwin = src_w // WINDOW_
            ww = src_w - wwin * WINDOW_
            src_row = (
                ((batch * WINDOWS_H_ + hwin) * WINDOWS_W_ + wwin)
                * WINDOW_
                * WINDOW_
                + wh * WINDOW_
                + ww
            )
            src_offsets = src_row[:, None] * C_ + c[None, :]

            x = tl.load(mm_ptr + src_offsets, mask=out_mask, other=0.0).to(tl.float32)
            rhs = tl.load(rhs_ptr + out_offsets, mask=out_mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + out_offsets, mask=out_mask, other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
            keep = tl.load(keep_ptr + batch, mask=row_mask, other=0).to(tl.float32) / KEEP_PROB_

            weighted = x * weight[None, :]
            row_sum = tl.sum(tl.where(out_mask, weighted, 0.0), axis=1)
            row_dot = tl.sum(tl.where(out_mask, weighted * rhs, 0.0), axis=1)
            grad = scale[:, None] * (
                weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None]
            )
            out = (residual + grad) * keep[:, None]

            tl.store(out_transposed_ptr + out_offsets, out, mask=out_mask)
            acc_x_rhs += tl.sum(tl.where(out_mask, x * rhs, 0.0), axis=0)
            acc_x += tl.sum(tl.where(out_mask, x, 0.0), axis=0)
            acc_out += tl.sum(tl.where(out_mask, out, 0.0), axis=0)

        tl.atomic_add(out_sum_x_rhs_ptr + c, acc_x_rhs, sem="relaxed", mask=c_mask)
        tl.atomic_add(out_sum_x_ptr + c, acc_x, sem="relaxed", mask=c_mask)
        tl.atomic_add(out_sum_out_ptr + c, acc_out, sem="relaxed", mask=c_mask)


@oracle_impl(hardware="H100", shapes="(T([100352, 256], f32), T([28], i64, gen=Index(28)), T([256], f32), T([128, 28, 28, 256], f32), T([128, 28, 28, 1], f32), T([128, 28, 28, 256], f32), T([128, 1, 1], b8), S([2048, 49, 256]), S([2048, 7, 7, 256]), S([128, 4, 4, 7, 7, 256]), S([128, 28, 28, 256]), S([128, 784, 256]), S([100352, 256]), S([256]))")
def oracle_forward(inputs):
    """Run the oracle computation."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    (
        mm_175,
        fmod_6,
        primals_52,
        mul_32,
        div_115,
        view_1292,
        lt_3,
        *_shape_params,
    ) = inputs

    if mm_175.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm_175.shape == (ROWS, C)
    assert fmod_6.shape == (HEIGHT,)
    assert primals_52.shape == (C,)
    assert mul_32.shape == (BATCH, HEIGHT, WIDTH, C)
    assert div_115.shape == (BATCH, HEIGHT, WIDTH, 1)
    assert view_1292.shape == (BATCH, HEIGHT, WIDTH, C)
    assert lt_3.shape == (BATCH, 1, 1)

    mm = mm_175.contiguous()
    index = fmod_6.contiguous()
    weight = primals_52.contiguous()
    rhs = mul_32.contiguous().reshape(ROWS, C)
    scale = div_115.contiguous().reshape(ROWS)
    residual = view_1292.contiguous().reshape(ROWS, C)
    keep = lt_3.contiguous().reshape(BATCH)

    device = mm.device
    out_sum_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    out_transposed = torch.empty_strided(
        (C, ROWS),
        (1, C),
        device=device,
        dtype=torch.float32,
    )

    _zero_three_vectors_kernel[(triton.cdiv(C, 256),)](
        out_sum_x_rhs,
        out_sum_x,
        out_sum_out,
        C_=C,
        BLOCK_C=256,
        num_warps=KERNEL_WARPS,
    )
    _indexed_swin_atomic_kernel[(triton.cdiv(ROWS, ROW_SPLIT),)](
        mm,
        index,
        weight,
        rhs,
        scale,
        residual,
        keep,
        out_transposed,
        out_sum_x_rhs,
        out_sum_x,
        out_sum_out,
        ROWS_=ROWS,
        C_=C,
        TOKENS_=TOKENS,
        WIDTH_=WIDTH,
        WINDOW_=WINDOW,
        WINDOWS_H_=WINDOWS_H,
        WINDOWS_W_=WINDOWS_W,
        KEEP_PROB_=KEEP_PROB,
        ROW_SPLIT=ROW_SPLIT,
        XBLOCK=XBLOCK,
        BLOCK_C=triton.next_power_of_2(C),
        num_warps=KERNEL_WARPS,
    )

    return out_sum_x_rhs, out_sum_x, out_transposed, out_sum_out


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
