"""
Oracle for sum_sum_sum_6107a2f54029

Gap diagnosis:
  Classification: COOPERATIVE_SPLIT_K
  What oracle does differently: The oracle streams the full Swin window-unpartition/roll producer once, computes the layernorm-backward row reductions, writes the returned transposed side output, and cooperatively accumulates all three returned channel reductions.
  What Inductor change would fix: Inductor needs a cooperative split-K multi-output reduction template that can keep row-local C reductions, a materialized transpose side output, and sibling NHW channel accumulators in one coordinated producer.
"""
from __future__ import annotations

import argparse
import json
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
HISTORICAL_BEST_COMPILE_US = 525.2799987792969
CLASSIFICATION = "COOPERATIVE_SPLIT_K"

ORACLE_ROWS_PER_GROUP = 64
ORACLE_BLOCK_R = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _swin_roll_ln_store_partials_kernel(
        mm_ptr,
        fmod_ptr,
        weight_ptr,
        rhs_ptr,
        scale_ptr,
        residual_ptr,
        out_base_ptr,
        partials_ptr,
        ROWS_: tl.constexpr,
        HEIGHT_: tl.constexpr,
        WIDTH_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        WINDOW_: tl.constexpr,
        WINDOW_BLOCKS_: tl.constexpr,
        WINDOW_AREA_: tl.constexpr,
        ROWS_PER_GROUP: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        group = tl.program_id(0)
        c = tl.arange(0, BLOCK_C)
        c_mask = c < CHANNELS_
        weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

        acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_add = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
            row = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
            row_mask = row < ROWS_

            hw = row % (HEIGHT_ * WIDTH_)
            batch = row // (HEIGHT_ * WIDTH_)
            h = hw // WIDTH_
            w = hw - h * WIDTH_

            src_h = tl.load(fmod_ptr + h, mask=row_mask, other=0)
            src_w = tl.load(fmod_ptr + w, mask=row_mask, other=0)
            window_h = src_h // WINDOW_
            window_w = src_w // WINDOW_
            inner_h = src_h - window_h * WINDOW_
            inner_w = src_w - window_w * WINDOW_
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
            rhs = tl.load(rhs_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + dst_offsets, mask=mask, other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

            weighted = x * weight[None, :]
            row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
            row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
            grad = scale[:, None] * (
                weighted * CHANNELS_ - row_sum[:, None] - rhs * row_dot[:, None]
            )
            add = residual + grad

            tl.store(out_base_ptr + dst_offsets, add, mask=mask)

            valid_x = tl.where(mask, x, 0.0)
            valid_rhs = tl.where(mask, rhs, 0.0)
            valid_add = tl.where(mask, add, 0.0)
            acc_x_rhs += tl.sum(valid_x * valid_rhs, axis=0)
            acc_x += tl.sum(valid_x, axis=0)
            acc_add += tl.sum(valid_add, axis=0)

        partial_base = group * 3 * CHANNELS_ + c
        tl.store(partials_ptr + partial_base, acc_x_rhs, mask=c_mask)
        tl.store(partials_ptr + partial_base + CHANNELS_, acc_x, mask=c_mask)
        tl.store(partials_ptr + partial_base + 2 * CHANNELS_, acc_add, mask=c_mask)


    @triton.jit
    def _finalize_partials_kernel(
        partials_ptr,
        out_x_rhs_ptr,
        out_x_ptr,
        out_add_ptr,
        NUM_GROUPS: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        c = tl.program_id(0)
        group = tl.arange(0, BLOCK_G)
        mask = group < NUM_GROUPS
        base = group * 3 * CHANNELS_ + c

        x_rhs = tl.load(partials_ptr + base, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(partials_ptr + base + CHANNELS_, mask=mask, other=0.0).to(tl.float32)
        add = tl.load(partials_ptr + base + 2 * CHANNELS_, mask=mask, other=0.0).to(tl.float32)

        tl.store(out_x_rhs_ptr + c, tl.sum(x_rhs, axis=0))
        tl.store(out_x_ptr + c, tl.sum(x, axis=0))
        tl.store(out_add_ptr + c, tl.sum(add, axis=0))


def _require_cuda_triton():
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _oracle_impl(
    inputs,
    rows_per_group: int = ORACLE_ROWS_PER_GROUP,
    block_r: int = ORACLE_BLOCK_R,
):
    _require_cuda_triton()
    (
        mm_193,
        fmod_2,
        primals_20,
        mul_10,
        div_120,
        view_1358,
        *_shape_params,
    ) = inputs

    assert mm_193.shape == (ROWS, CHANNELS)
    assert fmod_2.shape == (HEIGHT,)
    assert primals_20.shape == (CHANNELS,)
    assert mul_10.shape == (BATCH, HEIGHT, WIDTH, CHANNELS)
    assert div_120.shape == (BATCH, HEIGHT, WIDTH, 1)
    assert view_1358.shape == (BATCH, HEIGHT, WIDTH, CHANNELS)

    device = mm_193.device
    num_groups = triton.cdiv(ROWS, rows_per_group)
    partials = torch.empty((num_groups, 3, CHANNELS), device=device, dtype=torch.float32)
    out_x_rhs = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_base = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)
    out_add = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    _swin_roll_ln_store_partials_kernel[(num_groups,)](
        mm_193,
        fmod_2,
        primals_20,
        mul_10,
        div_120,
        view_1358,
        out_base,
        partials,
        ROWS_=ROWS,
        HEIGHT_=HEIGHT,
        WIDTH_=WIDTH,
        CHANNELS_=CHANNELS,
        WINDOW_=WINDOW,
        WINDOW_BLOCKS_=WINDOW_BLOCKS,
        WINDOW_AREA_=WINDOW_AREA,
        ROWS_PER_GROUP=rows_per_group,
        BLOCK_R=block_r,
        BLOCK_C=CHANNELS,
        num_warps=4,
    )
    _finalize_partials_kernel[(CHANNELS,)](
        partials,
        out_x_rhs,
        out_x,
        out_add,
        NUM_GROUPS=num_groups,
        CHANNELS_=CHANNELS,
        BLOCK_G=triton.next_power_of_2(num_groups),
        num_warps=8,
    )

    return out_x_rhs, out_x, out_base.permute(1, 0), out_add


def oracle_forward(inputs):
    """Run the full-scope oracle for Repro()(*make_inputs())."""
    return _oracle_impl(inputs)


def _do_bench(fn, warmup: int, rep: int) -> float:
    _require_cuda_triton()
    from triton.testing import do_bench

    return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _compile_with_config(instance, inputs, config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(instance)
        for _ in range(5):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def _run_required_bench(inputs, warmup: int, rep: int) -> dict[str, object]:
    _require_cuda_triton()

    with torch.no_grad():
        oracle_forward(inputs)
        torch.cuda.synchronize()
        oracle_us = _do_bench(lambda: oracle_forward(inputs), warmup=warmup, rep=rep)

    compile_configs = [
        ("coordinate_descent", {"coordinate_descent_tuning": True}),
        (
            "combo_looped_cd",
            {
                "combo_kernels": True,
                "combo_kernel_per_subkernel_blocks": True,
                "coordinate_descent_tuning": True,
                "benchmark_combo_kernel": True,
                "triton.multi_kernel": 3,
            },
        ),
    ]

    compile_times: dict[str, float] = {}
    for label, config in compile_configs:
        instance = get_repro_instance()
        with torch.no_grad():
            compiled = _compile_with_config(instance, inputs, config)
            compile_times[label] = _do_bench(
                lambda: compiled(*inputs),
                warmup=warmup,
                rep=rep,
            )

    cd_us = compile_times["coordinate_descent"]
    combo_us = compile_times["combo_looped_cd"]
    best_required_compile_us = min(cd_us, combo_us)
    true_floor = oracle_us < best_required_compile_us and oracle_us < HISTORICAL_BEST_COMPILE_US
    required_ratio = best_required_compile_us / oracle_us if oracle_us > 0 else 0.0
    cd_ratio = cd_us / oracle_us if oracle_us > 0 else 0.0

    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(cd_us, 3),
        "ratio": round(cd_ratio, 3),
        "status": "GOOD" if true_floor else "BAD_ORACLE",
        "combo_compile_us": round(combo_us, 3),
        "best_required_compile_us": round(best_required_compile_us, 3),
        "required_ratio": round(required_ratio, 3),
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "true_floor": true_floor,
        "classification": CLASSIFICATION,
    }
    print(json.dumps(result))
    if not true_floor:
        print(
            "WARNING: oracle is diagnosis-only/not a true floor "
            f"(required_ratio={required_ratio:.3f}x)"
        )
    return result


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
            _run_required_bench(inputs, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
