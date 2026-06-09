"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle keeps Inductor's split two-output f32 reduction order but fuses reduction finalization and writes the pointwise epilogue directly into the avg_pool2d_backward output, whereas Inductor materializes the 14x14 epilogue and expands it in a separate pool-backward kernel; Inductor cannot do this today because the scheduler cuts across the reduction consumers and the following layout-changing pool backward; the fix is SCHEDULER_FUSION: allow this fixed 2x2 avg-pool-backward expansion to fuse into its producer while preserving the reduction partial layout."""
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

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


N = 64
C = 256
H = 14
W = 14
HW = H * W
PARENT_NUMEL = N * C * HW


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _reduce_where_partials_kernel(
        mask_src,
        full_scalar,
        where_src,
        sub_src,
        mean_src,
        partial_mul,
        partial_where,
        BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)
        c = pid % 256
        half = pid // 256
        r = tl.arange(0, BLOCK_R)
        mask = r < 6272
        n = r // 196
        hw = r % 196
        idx = 1605632 * half + 50176 * n + 196 * c + hw

        mask_val = tl.load(mask_src + idx, mask=mask, other=0.0)
        where_val = tl.load(where_src + idx, mask=mask, other=0.0)
        sub_val = tl.load(sub_src + idx, mask=mask, other=0.0)
        mean_val = tl.load(mean_src + c)
        full_val = tl.load(full_scalar + 0)

        selected = tl.where(mask_val <= 0.0, full_val, where_val)
        centered = sub_val - mean_val
        mul_val = selected * centered

        sum_mul = tl.sum(tl.where(mask, mul_val, 0.0), axis=0)
        sum_where = tl.sum(tl.where(mask, selected, 0.0), axis=0)
        tl.store(partial_mul + pid, sum_mul)
        tl.store(partial_where + pid, sum_where)

    @triton.jit
    def _finalize_sums_kernel(
        partial_mul,
        partial_where,
        invstd,
        sum_mul,
        sum_where,
        output0,
        BLOCK_C: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = c < 256
        mul0 = tl.load(partial_mul + c, mask=mask, other=0.0)
        mul1 = tl.load(partial_mul + c + 256, mask=mask, other=0.0)
        where0 = tl.load(partial_where + c, mask=mask, other=0.0)
        where1 = tl.load(partial_where + c + 256, mask=mask, other=0.0)
        sum1 = mul0 + mul1
        sum0 = where0 + where1
        inv = tl.load(invstd + c, mask=mask, other=0.0)
        tl.store(sum_mul + c, sum1, mask=mask)
        tl.store(sum_where + c, sum0, mask=mask)
        tl.store(output0 + c, sum1 * inv, mask=mask)

    @triton.jit
    def _pooled_epilogue_kernel(
        out_pool,
        in0,
        in1,
        in2,
        in3,
        in4,
        in5,
        in6,
        in7,
        in8,
        in9,
        in10,
        in11,
        in12,
        in13,
        in14,
        in15,
        in16,
        mask_src,
        full_scalar,
        where_src,
        sub_src,
        mean_src,
        sum_mul,
        invstd,
        sum_where,
        affine_weight,
        in17,
        in18,
        in19,
        in20,
        in21,
        in22,
        in23,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < 3211264
        hw = offsets % 196
        c = (offsets // 196) % 256
        n = offsets // 50176

        x0 = hw + 196 * c
        tmp0 = tl.load(in0 + x0 + 200704 * n, mask=mask, other=0.0)
        tmp1 = tl.load(in1 + x0 + 194432 * n, mask=mask, other=0.0)
        tmp3 = tl.load(in2 + x0 + 188160 * n, mask=mask, other=0.0)
        tmp5 = tl.load(in3 + x0 + 181888 * n, mask=mask, other=0.0)
        tmp7 = tl.load(in4 + x0 + 175616 * n, mask=mask, other=0.0)
        tmp9 = tl.load(in5 + x0 + 169344 * n, mask=mask, other=0.0)
        tmp11 = tl.load(in6 + x0 + 163072 * n, mask=mask, other=0.0)
        tmp13 = tl.load(in7 + x0 + 156800 * n, mask=mask, other=0.0)
        tmp15 = tl.load(in8 + x0 + 150528 * n, mask=mask, other=0.0)
        tmp17 = tl.load(in9 + x0 + 144256 * n, mask=mask, other=0.0)
        tmp19 = tl.load(in10 + x0 + 137984 * n, mask=mask, other=0.0)
        tmp21 = tl.load(in11 + x0 + 131712 * n, mask=mask, other=0.0)
        tmp23 = tl.load(in12 + x0 + 125440 * n, mask=mask, other=0.0)
        tmp25 = tl.load(in13 + x0 + 119168 * n, mask=mask, other=0.0)
        tmp27 = tl.load(in14 + x0 + 112896 * n, mask=mask, other=0.0)
        tmp29 = tl.load(in15 + x0 + 106624 * n, mask=mask, other=0.0)
        tmp31 = tl.load(in16 + x0 + 100352 * n, mask=mask, other=0.0)

        tmp33 = tl.load(mask_src + offsets, mask=mask, other=0.0)
        tmp36 = tl.load(full_scalar + 0)
        tmp38 = tl.load(where_src + offsets, mask=mask, other=0.0)
        tmp40 = tl.load(sub_src + offsets, mask=mask, other=0.0)
        tmp41 = tl.load(mean_src + c, mask=mask, other=0.0, eviction_policy="evict_last")
        tmp43 = tl.load(sum_mul + c, mask=mask, other=0.0, eviction_policy="evict_last")
        tmp46 = tl.load(invstd + c, mask=mask, other=0.0, eviction_policy="evict_last")
        tmp51 = tl.load(sum_where + c, mask=mask, other=0.0, eviction_policy="evict_last")
        tmp54 = tl.load(affine_weight + c, mask=mask, other=0.0, eviction_policy="evict_last")

        tmp57 = tl.load(in17 + x0 + 94080 * n, mask=mask, other=0.0)
        tmp59 = tl.load(in18 + x0 + 87808 * n, mask=mask, other=0.0)
        tmp61 = tl.load(in19 + x0 + 81536 * n, mask=mask, other=0.0)
        tmp63 = tl.load(in20 + x0 + 75264 * n, mask=mask, other=0.0)
        tmp65 = tl.load(in21 + x0 + 68992 * n, mask=mask, other=0.0)
        tmp67 = tl.load(in22 + x0 + 62720 * n, mask=mask, other=0.0)
        tmp69 = tl.load(in23 + x0 + 56448 * n, mask=mask, other=0.0)

        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp10 = tmp8 + tmp9
        tmp12 = tmp10 + tmp11
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp20 = tmp18 + tmp19
        tmp22 = tmp20 + tmp21
        tmp24 = tmp22 + tmp23
        tmp26 = tmp24 + tmp25
        tmp28 = tmp26 + tmp27
        tmp30 = tmp28 + tmp29
        tmp32 = tmp30 + tmp31

        tmp35 = tmp33 <= 0.0
        tmp39 = tl.where(tmp35, tmp36, tmp38)
        tmp42 = tmp40 - tmp41
        tmp45 = tmp43 * 7.971938775510203e-05
        tmp47 = tmp46 * tmp46
        tmp48 = tmp45 * tmp47
        tmp49 = tmp42 * tmp48
        tmp50 = tmp39 - tmp49
        tmp52 = tmp51 * 7.971938775510203e-05
        tmp53 = tmp50 - tmp52
        tmp55 = tmp46 * tmp54
        tmp56 = tmp53 * tmp55

        tmp58 = tmp32 + tmp57
        tmp60 = tmp58 + tmp59
        tmp62 = tmp60 + tmp61
        tmp64 = tmp62 + tmp63
        tmp66 = tmp64 + tmp65
        tmp68 = tmp66 + tmp67
        tmp70 = tmp68 + tmp69
        tmp71 = tmp70 + tmp56
        pooled = tmp71 / 4.0

        h = hw // 14
        w = hw - h * 14
        out_base = n * 200704 + c * 784 + h * 56 + w * 2
        tl.store(out_pool + out_base, pooled, mask=mask)
        tl.store(out_pool + out_base + 1, pooled, mask=mask)
        tl.store(out_pool + out_base + 28, pooled, mask=mask)
        tl.store(out_pool + out_base + 29, pooled, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([64, 1024, 14, 14], f32), T([64, 992, 14, 14], f32), T([64, 960, 14, 14], f32), T([64, 928, 14, 14], f32), T([64, 896, 14, 14], f32), T([64, 864, 14, 14], f32), T([64, 832, 14, 14], f32), T([64, 800, 14, 14], f32), T([64, 768, 14, 14], f32), T([64, 736, 14, 14], f32), T([64, 704, 14, 14], f32), T([64, 672, 14, 14], f32), T([64, 640, 14, 14], f32), T([64, 608, 14, 14], f32), T([64, 576, 14, 14], f32), T([64, 544, 14, 14], f32), T([64, 512, 14, 14], f32), T([64, 480, 14, 14], f32), T([64, 448, 14, 14], f32), T([64, 416, 14, 14], f32), T([64, 384, 14, 14], f32), T([64, 352, 14, 14], f32), T([64, 320, 14, 14], f32), T([64, 288, 14, 14], f32), T([64, 256, 14, 14], f32), T([], f32), T([64, 256, 14, 14], f32), T([64, 256, 14, 14], f32), T([1, 256, 1, 1], f32), T([256], f32), T([256], f32), T([64, 256, 28, 28], f32))")
def oracle_forward(inputs):
    """Run the full repro-equivalent oracle computation."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        mul_306,
        mul_324,
        mul_342,
        mul_360,
        mul_378,
        mul_396,
        mul_414,
        mul_432,
        mul_450,
        mul_468,
        mul_486,
        mul_504,
        mul_522,
        mul_540,
        mul_558,
        mul_576,
        mul_594,
        mul_612,
        mul_630,
        mul_648,
        mul_666,
        mul_684,
        mul_702,
        mul_720,
        arg366_1,
        full,
        getitem_240,
        arg364_1,
        arg692_1,
        arg365_1,
        arg80_1,
        _arg363_1,
    ) = inputs

    device = mul_306.device
    partial_mul = torch.empty_strided((C, 2), (1, C), device=device, dtype=torch.float32)
    partial_where = torch.empty_strided((C, 2), (1, C), device=device, dtype=torch.float32)
    sum_mul = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    sum_where = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    output0 = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    output1 = torch.empty_strided(
        (N, C, 28, 28),
        (C * 28 * 28, 28 * 28, 28, 1),
        device=device,
        dtype=torch.float32,
    )

    _reduce_where_partials_kernel[(512,)](
        arg366_1,
        full,
        getitem_240,
        arg364_1,
        arg692_1,
        partial_mul,
        partial_where,
        BLOCK_R=8192,
        num_warps=8,
    )
    _finalize_sums_kernel[(1,)](
        partial_mul,
        partial_where,
        arg365_1,
        sum_mul,
        sum_where,
        output0,
        BLOCK_C=256,
        num_warps=8,
    )
    _pooled_epilogue_kernel[(triton.cdiv(PARENT_NUMEL, 256),)](
        output1,
        mul_306,
        mul_324,
        mul_342,
        mul_360,
        mul_378,
        mul_396,
        mul_414,
        mul_432,
        mul_450,
        mul_468,
        mul_486,
        mul_504,
        mul_522,
        mul_540,
        mul_558,
        mul_576,
        mul_594,
        arg366_1,
        full,
        getitem_240,
        arg364_1,
        arg692_1,
        sum_mul,
        arg365_1,
        sum_where,
        arg80_1,
        mul_612,
        mul_630,
        mul_648,
        mul_666,
        mul_684,
        mul_702,
        mul_720,
        BLOCK=256,
        num_warps=8,
    )
    return (output0, output1)


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
