"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Whisper layer-norm backward return tuple by reducing each row's hidden-dimension summaries inside a row-tiled producer that writes the returned `[8, 1500, 384]` input-gradient add while cooperatively accumulating both returned `[384]` column reductions, whereas Inductor currently schedules the row sums, the input-gradient pointwise epilogue, and the two sibling `sum([0, 1])` reductions as separate generic reduction/pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reductions, a materialized side output, and sibling column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split the large row dimension for compatible layer-norm-backward column reductions, coordinate the partial accumulators, and fuse the dependent input-gradient side store."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_sum_d574fc7bdc59"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
SEQ = 1500
M = BATCH * SEQ
D = 384
INV_D = 1.0 / D


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_6,
        mm_9,
        mm_10,
        arg1_1,
        arg9_1,
        arg10_1,
        arg0_1,
        add_3,
        *_shape_params,
    ) = inputs

    return (
        mm_6.contiguous(),
        mm_9.contiguous(),
        mm_10.contiguous(),
        arg1_1.reshape(M, D).contiguous(),
        arg9_1.reshape(M).contiguous(),
        arg10_1.reshape(M).contiguous(),
        arg0_1.contiguous(),
        add_3.reshape(M, D).contiguous(),
    )


@triton.jit
def _layernorm_backward_tile_kernel(
    mm6_ptr,
    mm9_ptr,
    mm10_ptr,
    x_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    residual_ptr,
    partial_sum_dy_xhat_ptr,
    partial_sum_dy_ptr,
    out_md_ptr,
    M_: tl.constexpr,
    D_: tl.constexpr,
    INV_D_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, BLOCK_D)
    m_mask = m < M_
    d_mask = d < D_
    mask = m_mask[:, None] & d_mask[None, :]
    offsets = m[:, None] * D_ + d[None, :]

    dy = (
        tl.load(mm6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(mm9_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(mm10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    xhat = (x - mean[:, None]) * rstd[:, None]
    weighted = dy * weight[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * xhat, 0.0), axis=1)
    ln_grad = rstd[:, None] * INV_D_ * (
        weighted * D_ - row_sum[:, None] - xhat * row_dot[:, None]
    )
    out = residual + ln_grad
    tl.store(out_md_ptr + offsets, out, mask=mask)

    sum_dy_xhat = tl.sum(tl.where(mask, dy * xhat, 0.0), axis=0)
    sum_dy = tl.sum(tl.where(mask, dy, 0.0), axis=0)
    partial_offsets = tl.program_id(0) * D_ + d
    tl.store(partial_sum_dy_xhat_ptr + partial_offsets, sum_dy_xhat, mask=d_mask)
    tl.store(partial_sum_dy_ptr + partial_offsets, sum_dy, mask=d_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_sum_dy_xhat_ptr,
    partial_sum_dy_ptr,
    out_sum_dy_xhat_ptr,
    out_sum_dy_ptr,
    NUM_M_TILES: tl.constexpr,
    D_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile = tl.arange(0, BLOCK_TILES)
    d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (tile[:, None] < NUM_M_TILES) & (d[None, :] < D_)
    offsets = tile[:, None] * D_ + d[None, :]

    partial_xhat = tl.load(partial_sum_dy_xhat_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    partial_dy = tl.load(partial_sum_dy_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    d_mask = d < D_
    tl.store(out_sum_dy_xhat_ptr + d, tl.sum(partial_xhat, axis=0), mask=d_mask)
    tl.store(out_sum_dy_ptr + d, tl.sum(partial_dy, axis=0), mask=d_mask)


def oracle_full(
    mm_6: torch.Tensor,
    mm_9: torch.Tensor,
    mm_10: torch.Tensor,
    x_md: torch.Tensor,
    mean_m: torch.Tensor,
    rstd_m: torch.Tensor,
    weight_d: torch.Tensor,
    residual_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_6.shape == (M, D)
    assert mm_9.shape == (M, D)
    assert mm_10.shape == (M, D)
    assert x_md.shape == (M, D)
    assert mean_m.shape == (M,)
    assert rstd_m.shape == (M,)
    assert weight_d.shape == (D,)
    assert residual_md.shape == (M, D)
    assert mm_6.is_contiguous()
    assert mm_9.is_contiguous()
    assert mm_10.is_contiguous()
    assert x_md.is_contiguous()
    assert mean_m.is_contiguous()
    assert rstd_m.is_contiguous()
    assert weight_d.is_contiguous()
    assert residual_md.is_contiguous()

    device = mm_6.device
    block_m = 8
    block_d = 512
    num_m_tiles = triton.cdiv(M, block_m)
    partial_sum_dy_xhat = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_sum_dy = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    out_md = torch.empty((M, D), device=device, dtype=torch.float32)

    _layernorm_backward_tile_kernel[(num_m_tiles,)](
        mm_6,
        mm_9,
        mm_10,
        x_md,
        mean_m,
        rstd_m,
        weight_d,
        residual_md,
        partial_sum_dy_xhat,
        partial_sum_dy,
        out_md,
        M_=M,
        D_=D,
        INV_D_=INV_D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )

    sum_dy_xhat = torch.empty((D,), device=device, dtype=torch.float32)
    sum_dy = torch.empty((D,), device=device, dtype=torch.float32)
    block_final_d = 16
    block_tiles = 1 << (num_m_tiles - 1).bit_length()
    _finalize_column_sums_kernel[(triton.cdiv(D, block_final_d),)](
        partial_sum_dy_xhat,
        partial_sum_dy,
        sum_dy_xhat,
        sum_dy,
        NUM_M_TILES=num_m_tiles,
        D_=D,
        BLOCK_TILES=block_tiles,
        BLOCK_D=block_final_d,
        num_warps=8,
    )

    return sum_dy_xhat, sum_dy, out_md.reshape(BATCH, SEQ, D)


def oracle_forward(inputs):
    return oracle_full(*prepare_oracle_inputs(*inputs))


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
