"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle produces the complete 12-output `Repro.forward` tuple by handling each of the six BN/ReLU-backward branches as a repeated multi-output reduction template that shares the masked `mm / 64` slice producer across sibling channel reductions and the dependent input-gradient/vector epilogue, whereas Inductor currently schedules the expanded `mm`, channel slices, ReLU masks, sibling `sum([0, 2, 3])` reductions, and BN-backward epilogues as separate pointwise/reduction groups; Inductor cannot do this today because its scheduler/codegen does not build a reusable same-shape multi-output reduction template with shared producers and dependent epilogues for each sliced branch; the fix is SCHEDULER_FUSION: add a grouped reduction-plus-epilogue schedule for same-shape sibling reductions that reuses the masked producer and avoids materializing the expanded `mm / 64` intermediates."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_sum_fcd2cc2945b3"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
MM_C = 2048
H = 8
W = 8
HW = H * W
TOTAL_SPATIAL = N * HW
INV_EXPAND = 1.0 / 64.0
REDUCE_SCALE = 0.0001220703125



@triton.jit
def _bn_branch_partial_kernel(
    mm_ptr,
    activation_input_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    partial_sum1_ptr,
    partial_sum2_ptr,
    source_offset: tl.constexpr,
    num_tiles: tl.constexpr,
    C_: tl.constexpr,
    MM_C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_EXPAND_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < C_
    k_mask = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    branch_offsets = n[:, None] * (C_ * HW_) + c[None, :] * HW_ + hw[:, None]
    mm_offsets = n[:, None] * MM_C_ + (source_offset + c[None, :])
    active = k_mask[:, None] & c_mask[None, :]

    x = tl.load(activation_input_ptr + branch_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    source = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32) * INV_EXPAND_

    centered = x - mean[None, :]
    relu_input = centered * invstd[None, :] * gamma[None, :] + beta[None, :]
    where_self = tl.where(relu_input <= 0.0, 0.0, source)
    where_self = tl.where(active, where_self, 0.0)

    sum1 = tl.sum(where_self, axis=0)
    sum2 = tl.sum(where_self * centered, axis=0)
    partial_offsets = c * num_tiles + tl.program_id(1)
    tl.store(partial_sum1_ptr + partial_offsets, sum1, mask=c_mask)
    tl.store(partial_sum2_ptr + partial_offsets, sum2, mask=c_mask)


@triton.jit
def _bn_branch_finalize_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    invstd_ptr,
    sum1_ptr,
    sum2_ptr,
    vector_out_ptr,
    num_tiles: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    c_mask = c < C_
    tile_mask = tiles < num_tiles
    offsets = tiles[:, None] + c[None, :] * num_tiles

    values1 = tl.load(partial_sum1_ptr + offsets, mask=tile_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    values2 = tl.load(partial_sum2_ptr + offsets, mask=tile_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    sum1 = tl.sum(values1, axis=0)
    sum2 = tl.sum(values2, axis=0)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(sum1_ptr + c, sum1, mask=c_mask)
    tl.store(sum2_ptr + c, sum2, mask=c_mask)
    tl.store(vector_out_ptr + c, sum2 * invstd, mask=c_mask)


@triton.jit
def _bn_branch_epilogue_kernel(
    mm_ptr,
    activation_input_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    sum1_ptr,
    sum2_ptr,
    tensor_out_ptr,
    source_offset: tl.constexpr,
    numel: tl.constexpr,
    C_: tl.constexpr,
    MM_C_: tl.constexpr,
    HW_: tl.constexpr,
    INV_EXPAND_: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = offsets < numel

    hw = offsets % HW_
    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)
    branch_offsets = n * (C_ * HW_) + c * HW_ + hw
    mm_offsets = n * MM_C_ + source_offset + c

    x = tl.load(activation_input_ptr + branch_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=active, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)
    source = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32) * INV_EXPAND_

    centered = x - mean
    relu_input = centered * invstd * gamma + beta
    where_self = tl.where(relu_input <= 0.0, 0.0, source)
    variance_term = sum2 * REDUCE_SCALE_ * invstd * invstd
    mean_term = sum1 * REDUCE_SCALE_
    out = (where_self - centered * variance_term - mean_term) * (invstd * gamma)
    tl.store(tensor_out_ptr + offsets, out, mask=active)

def _run_bn_branch(
    mm: torch.Tensor,
    activation_input: torch.Tensor,
    mean: torch.Tensor,
    invstd: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    source_offset: int,
    channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mm.shape == (N, MM_C)
    assert activation_input.shape == (N, channels, H, W)
    assert mean.shape == (1, channels, 1, 1)
    assert invstd.shape == (1, channels, 1, 1)
    assert gamma.shape == (channels,)
    assert beta.shape == (channels,)
    assert mm.is_contiguous()
    assert activation_input.is_contiguous()

    device = mm.device
    block_c = 16
    block_k = 512
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum1 = torch.empty((channels, num_tiles), device=device, dtype=torch.float32)
    partial_sum2 = torch.empty((channels, num_tiles), device=device, dtype=torch.float32)
    sum1 = torch.empty((channels,), device=device, dtype=torch.float32)
    sum2 = torch.empty((channels,), device=device, dtype=torch.float32)
    vector_out = torch.empty((channels,), device=device, dtype=torch.float32)

    _bn_branch_partial_kernel[(triton.cdiv(channels, block_c), num_tiles)](
        mm,
        activation_input,
        mean,
        invstd,
        gamma,
        beta,
        partial_sum1,
        partial_sum2,
        source_offset=source_offset,
        num_tiles=num_tiles,
        C_=channels,
        MM_C_=MM_C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_EXPAND_=INV_EXPAND,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=8,
    )

    block_tiles = triton.next_power_of_2(num_tiles)
    _bn_branch_finalize_kernel[(triton.cdiv(channels, block_c),)](
        partial_sum1,
        partial_sum2,
        invstd,
        sum1,
        sum2,
        vector_out,
        num_tiles=num_tiles,
        C_=channels,
        BLOCK_C=block_c,
        BLOCK_TILES=block_tiles,
        num_warps=4,
    )

    tensor_out = torch.empty((N, channels, H, W), device=device, dtype=torch.float32)
    numel = N * channels * HW
    block_elems = 256
    _bn_branch_epilogue_kernel[(triton.cdiv(numel, block_elems),)](
        mm,
        activation_input,
        mean,
        invstd,
        gamma,
        beta,
        sum1,
        sum2,
        tensor_out,
        source_offset=source_offset,
        numel=numel,
        C_=channels,
        MM_C_=MM_C,
        HW_=HW,
        INV_EXPAND_=INV_EXPAND,
        REDUCE_SCALE_=REDUCE_SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )
    return tensor_out, vector_out


def oracle_fused(
    mm: torch.Tensor,
    arg540_1: torch.Tensor,
    arg541_1: torch.Tensor,
    arg542_1: torch.Tensor,
    arg233_1: torch.Tensor,
    arg234_1: torch.Tensor,
    arg536_1: torch.Tensor,
    arg537_1: torch.Tensor,
    arg538_1: torch.Tensor,
    arg230_1: torch.Tensor,
    arg231_1: torch.Tensor,
    arg533_1: torch.Tensor,
    arg534_1: torch.Tensor,
    arg535_1: torch.Tensor,
    arg227_1: torch.Tensor,
    arg228_1: torch.Tensor,
    arg524_1: torch.Tensor,
    arg525_1: torch.Tensor,
    arg526_1: torch.Tensor,
    arg220_1: torch.Tensor,
    arg221_1: torch.Tensor,
    arg521_1: torch.Tensor,
    arg522_1: torch.Tensor,
    arg523_1: torch.Tensor,
    arg217_1: torch.Tensor,
    arg218_1: torch.Tensor,
    arg515_1: torch.Tensor,
    arg516_1: torch.Tensor,
    arg517_1: torch.Tensor,
    arg212_1: torch.Tensor,
    arg213_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, ...]:
    del _shape_param_0, _shape_param_1
    mm = mm.contiguous()

    out_a = _run_bn_branch(mm, arg540_1.contiguous(), arg541_1.contiguous(), arg542_1.contiguous(), arg233_1.contiguous(), arg234_1.contiguous(), 1856, 192)
    out_b = _run_bn_branch(mm, arg536_1.contiguous(), arg537_1.contiguous(), arg538_1.contiguous(), arg230_1.contiguous(), arg231_1.contiguous(), 1472, 384)
    out_c = _run_bn_branch(mm, arg533_1.contiguous(), arg534_1.contiguous(), arg535_1.contiguous(), arg227_1.contiguous(), arg228_1.contiguous(), 1088, 384)
    out_d = _run_bn_branch(mm, arg524_1.contiguous(), arg525_1.contiguous(), arg526_1.contiguous(), arg220_1.contiguous(), arg221_1.contiguous(), 704, 384)
    out_e = _run_bn_branch(mm, arg521_1.contiguous(), arg522_1.contiguous(), arg523_1.contiguous(), arg217_1.contiguous(), arg218_1.contiguous(), 320, 384)
    out_f = _run_bn_branch(mm, arg515_1.contiguous(), arg516_1.contiguous(), arg517_1.contiguous(), arg212_1.contiguous(), arg213_1.contiguous(), 0, 320)

    return (
        out_a[0],
        out_a[1],
        out_b[0],
        out_b[1],
        out_c[0],
        out_c[1],
        out_d[0],
        out_d[1],
        out_e[0],
        out_e[1],
        out_f[0],
        out_f[1],
    )

@oracle_impl(hardware="H100", shapes="(T([128, 2048], f32), T([128, 192, 8, 8], f32), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([128, 384, 8, 8], f32), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32), T([128, 320, 8, 8], f32), T([1, 320, 1, 1], f32), T([1, 320, 1, 1], f32), T([320], f32), T([320], f32), S([128, 2048, 1, 1]), S([128, 2048, 8, 8]))")
def oracle_forward(inputs):
    return oracle_fused(*inputs)


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
