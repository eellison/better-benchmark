"""
Oracle for sum_sum_sum_87fd4ec7f051

Gap diagnosis:
  Classification: COOPERATIVE_SPLIT_K
  What oracle does differently: The oracle streams the full Swin window-unpartitioned activation, both layernorm-backward row reductions, and all five returned channel reductions through one multi-accumulator Triton producer plus a small partial finalizer.
  What Inductor change would fix: Inductor needs a cooperative split-K multi-output reduction template that can keep row-local C reductions and sibling NHW channel accumulators in one coordinated schedule instead of materializing and rereading the dependent layernorm-backward intermediates.
"""
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
HEIGHT = 56
WIDTH = 56
CHANNELS = 128
ROWS = BATCH * HEIGHT * WIDTH
WINDOW = 7
WINDOW_BLOCKS = HEIGHT // WINDOW
WINDOW_AREA = WINDOW * WINDOW
INV_CHANNELS = 1.0 / CHANNELS
ORACLE_ROWS_PER_GROUP = 768
ORACLE_BLOCK_R = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _swin_ln_partials_kernel(
        mm_ptr,
        weight1_ptr,
        conv_ptr,
        mean0_ptr,
        rsqrt0_ptr,
        weight0_ptr,
        bias0_ptr,
        mean1_ptr,
        rsqrt1_ptr,
        residual_ptr,
        partials_ptr,
        ROWS_: tl.constexpr,
        HEIGHT_: tl.constexpr,
        WIDTH_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        WINDOW_: tl.constexpr,
        WINDOW_BLOCKS_: tl.constexpr,
        WINDOW_AREA_: tl.constexpr,
        INV_CHANNELS_: tl.constexpr,
        ROWS_PER_GROUP: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        group = tl.program_id(0)
        c = tl.arange(0, BLOCK_C)
        c_mask = c < CHANNELS_

        weight1 = tl.load(weight1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        weight0 = tl.load(weight0_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        bias0 = tl.load(bias0_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

        acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc4 = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
            row = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
            row_mask = row < ROWS_

            hw = row % (HEIGHT_ * WIDTH_)
            batch = row // (HEIGHT_ * WIDTH_)
            h = hw // WIDTH_
            w = hw - h * WIDTH_
            window_h = h // WINDOW_
            window_w = w // WINDOW_
            inner_h = h - window_h * WINDOW_
            inner_w = w - window_w * WINDOW_
            src_row = (
                ((batch * WINDOW_BLOCKS_ + window_h) * WINDOW_BLOCKS_ + window_w)
                * WINDOW_AREA_
                + inner_h * WINDOW_
                + inner_w
            )

            mask = row_mask[:, None] & c_mask[None, :]
            dst_offsets = row[:, None] * CHANNELS_ + c[None, :]
            src_offsets = src_row[:, None] * CHANNELS_ + c[None, :]

            x = tl.load(mm_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
            conv = tl.load(conv_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)

            mean0 = tl.load(mean0_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
            rsqrt0 = tl.load(rsqrt0_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
            mean1 = tl.load(mean1_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
            rsqrt1 = tl.load(rsqrt1_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

            norm0 = (conv - mean0[:, None]) * rsqrt0[:, None]
            norm1_source = norm0 * weight0[None, :] + bias0[None, :]
            norm1 = (norm1_source - mean1[:, None]) * rsqrt1[:, None]

            weighted1 = x * weight1[None, :]
            row_sum1 = tl.sum(tl.where(mask, weighted1, 0.0), axis=1)
            row_dot1 = tl.sum(tl.where(mask, weighted1 * norm1, 0.0), axis=1)
            grad1 = rsqrt1[:, None] * INV_CHANNELS_ * (
                weighted1 * CHANNELS_ - row_sum1[:, None] - norm1 * row_dot1[:, None]
            )
            add1 = residual + grad1

            weighted0 = add1 * weight0[None, :]
            row_sum0 = tl.sum(tl.where(mask, weighted0, 0.0), axis=1)
            row_dot0 = tl.sum(tl.where(mask, weighted0 * norm0, 0.0), axis=1)
            grad0 = rsqrt0[:, None] * INV_CHANNELS_ * (
                weighted0 * CHANNELS_ - row_sum0[:, None] - norm0 * row_dot0[:, None]
            )

            valid_x = tl.where(mask, x, 0.0)
            valid_norm1 = tl.where(mask, norm1, 0.0)
            valid_norm0 = tl.where(mask, norm0, 0.0)
            valid_add1 = tl.where(mask, add1, 0.0)
            valid_grad0 = tl.where(mask, grad0, 0.0)

            acc0 += tl.sum(valid_x * valid_norm1, axis=0)
            acc1 += tl.sum(valid_x, axis=0)
            acc2 += tl.sum(valid_add1 * valid_norm0, axis=0)
            acc3 += tl.sum(valid_add1, axis=0)
            acc4 += tl.sum(valid_grad0, axis=0)

        partial_base = group * 5 * CHANNELS_ + c
        tl.store(partials_ptr + partial_base, acc0, mask=c_mask)
        tl.store(partials_ptr + partial_base + CHANNELS_, acc1, mask=c_mask)
        tl.store(partials_ptr + partial_base + 2 * CHANNELS_, acc2, mask=c_mask)
        tl.store(partials_ptr + partial_base + 3 * CHANNELS_, acc3, mask=c_mask)
        tl.store(partials_ptr + partial_base + 4 * CHANNELS_, acc4, mask=c_mask)


    @triton.jit
    def _finalize_partials_kernel(
        partials_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out3_ptr,
        out4_ptr,
        NUM_GROUPS: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        c = tl.program_id(0)
        group = tl.arange(0, BLOCK_G)
        mask = group < NUM_GROUPS
        base = group * 5 * CHANNELS_ + c

        vals0 = tl.load(partials_ptr + base, mask=mask, other=0.0).to(tl.float32)
        vals1 = tl.load(partials_ptr + base + CHANNELS_, mask=mask, other=0.0).to(tl.float32)
        vals2 = tl.load(partials_ptr + base + 2 * CHANNELS_, mask=mask, other=0.0).to(tl.float32)
        vals3 = tl.load(partials_ptr + base + 3 * CHANNELS_, mask=mask, other=0.0).to(tl.float32)
        vals4 = tl.load(partials_ptr + base + 4 * CHANNELS_, mask=mask, other=0.0).to(tl.float32)

        tl.store(out0_ptr + c, tl.sum(vals0, axis=0))
        tl.store(out1_ptr + c, tl.sum(vals1, axis=0))
        tl.store(out2_ptr + c, tl.sum(vals2, axis=0))
        tl.store(out3_ptr + c, tl.sum(vals3, axis=0))
        tl.store(out4_ptr + c, tl.sum(vals4, axis=0))


def _require_cuda_triton():
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _oracle_impl(inputs, rows_per_group: int = ORACLE_ROWS_PER_GROUP, block_r: int = ORACLE_BLOCK_R):
    _require_cuda_triton()
    (
        mm_201,
        primals_6,
        convolution,
        getitem_1,
        rsqrt,
        primals_4,
        primals_5,
        getitem_3,
        rsqrt_1,
        view_1390,
        *_shape_params,
    ) = inputs

    assert mm_201.shape == (ROWS, CHANNELS)
    assert primals_6.shape == (CHANNELS,)
    assert convolution.shape == (BATCH, CHANNELS, HEIGHT, WIDTH)
    assert getitem_1.shape == (BATCH, HEIGHT, WIDTH, 1)
    assert rsqrt.shape == (BATCH, HEIGHT, WIDTH, 1)
    assert primals_4.shape == (CHANNELS,)
    assert primals_5.shape == (CHANNELS,)
    assert getitem_3.shape == (BATCH, HEIGHT, WIDTH, 1)
    assert rsqrt_1.shape == (BATCH, HEIGHT, WIDTH, 1)
    assert view_1390.shape == (BATCH, HEIGHT, WIDTH, CHANNELS)

    device = mm_201.device
    num_groups = triton.cdiv(ROWS, rows_per_group)
    partials = torch.empty((num_groups, 5, CHANNELS), device=device, dtype=torch.float32)
    out0 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out2 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out3 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out4 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    _swin_ln_partials_kernel[(num_groups,)](
        mm_201,
        primals_6,
        convolution,
        getitem_1,
        rsqrt,
        primals_4,
        primals_5,
        getitem_3,
        rsqrt_1,
        view_1390,
        partials,
        ROWS_=ROWS,
        HEIGHT_=HEIGHT,
        WIDTH_=WIDTH,
        CHANNELS_=CHANNELS,
        WINDOW_=WINDOW,
        WINDOW_BLOCKS_=WINDOW_BLOCKS,
        WINDOW_AREA_=WINDOW_AREA,
        INV_CHANNELS_=INV_CHANNELS,
        ROWS_PER_GROUP=rows_per_group,
        BLOCK_R=block_r,
        BLOCK_C=CHANNELS,
        num_warps=4,
    )
    _finalize_partials_kernel[(CHANNELS,)](
        partials,
        out0,
        out1,
        out2,
        out3,
        out4,
        NUM_GROUPS=num_groups,
        CHANNELS_=CHANNELS,
        BLOCK_G=triton.next_power_of_2(num_groups),
        num_warps=4,
    )
    return out0, out1, out2, out3, out4


def oracle_forward(inputs):
    """Run the full-scope oracle for Repro()(*make_inputs())."""
    return _oracle_impl(inputs)


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true", help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
