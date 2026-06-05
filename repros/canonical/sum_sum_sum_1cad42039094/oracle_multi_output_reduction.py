"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete `sum_sum_sum_1cad42039094` Swin MLP layer-norm-backward/drop-path/window-partition return tuple, including the two `[256]` reductions over the reshaped mm input, the row-local 256-wide layer-norm-backward reductions, residual add, per-batch drop-path mask, structured window-partition side store, returned `[256, 100352]` transpose view, and final `[256]` reduction over that side output, whereas Inductor currently schedules the sibling column reductions, row reductions, drop-path pointwise chain, clone/view/permute layout work, and final column reduction as separate generic templates over materialized intermediates; Inductor cannot do this today because its scheduler/codegen does not represent a cooperative split-K multi-output reduction that combines row-wise scalar reductions, column partials, a structured layout-producing side output, and a downstream reduction over that side output in one coordinated schedule; the fix is COOPERATIVE_SPLIT_K: add scheduler/codegen support for row-tiled dependent multi-output reductions that accumulate compatible column partials while writing structured window-layout epilogues and finalizing their column sums."""
from __future__ import annotations

import argparse
import importlib.util
import math
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



REPRO_ID = "sum_sum_sum_1cad42039094"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_001_8b656d5d"

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



def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


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
def _swin_ln_window_atomic_kernel(
    mm_ptr,
    weight_ptr,
    rhs_ptr,
    gate_ptr,
    residual_ptr,
    keep_ptr,
    out_base_ptr,
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

    acc_x_rhs = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_x = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_out = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * C_ + c[None, :]

        x = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weighted = x * weight[None, :]

        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
        gate = tl.load(gate_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

        b = row // TOKENS_
        token = row - b * TOKENS_
        h = token // WIDTH_
        w = token - h * WIDTH_
        hwin = h // WINDOW_
        wh = h - hwin * WINDOW_
        wwin = w // WINDOW_
        ww = w - wwin * WINDOW_
        final_row = ((((b * WINDOWS_H_ + hwin) * WINDOWS_W_ + wwin) * WINDOW_ + wh) * WINDOW_ + ww)
        keep = tl.load(keep_ptr + b, mask=row_mask, other=0).to(tl.float32) / KEEP_PROB_

        ln_grad = gate[:, None] * (weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None])
        out_value = (residual + ln_grad) * keep[:, None]

        tl.store(out_base_ptr + final_row[:, None] * C_ + c[None, :], out_value, mask=mask)
        acc_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_out += tl.sum(tl.where(mask, out_value, 0.0), axis=0)

    tl.atomic_add(out_sum_x_rhs_ptr + c, acc_x_rhs, sem="relaxed", mask=c_mask)
    tl.atomic_add(out_sum_x_ptr + c, acc_x, sem="relaxed", mask=c_mask)
    tl.atomic_add(out_sum_out_ptr + c, acc_out, sem="relaxed", mask=c_mask)


def oracle_full(
    mm_176: torch.Tensor,
    arg24_1: torch.Tensor,
    arg212_1: torch.Tensor,
    arg545_1: torch.Tensor,
    view_704: torch.Tensor,
    arg211_1: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mm_176.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm_176.shape == (ROWS, C)
    assert arg24_1.shape == (C,)
    assert arg212_1.shape == (BATCH, TOKENS, C)
    assert arg545_1.shape == (BATCH, TOKENS, 1)
    assert view_704.shape == (BATCH, TOKENS, C)
    assert arg211_1.shape == (BATCH, 1, 1, 1)
    assert mm_176.stride() == (C, 1)
    assert arg212_1.stride() == (TOKENS * C, C, 1)
    assert view_704.stride() == (TOKENS * C, C, 1)

    mm = mm_176.contiguous()
    weight = arg24_1.contiguous()
    rhs = arg212_1.contiguous().reshape(ROWS, C)
    gate = arg545_1.contiguous().reshape(ROWS)
    residual = view_704.contiguous().reshape(ROWS, C)
    keep = arg211_1.contiguous().reshape(BATCH)
    device = mm.device

    row_split = ROW_SPLIT
    xblock = XBLOCK
    block_c = triton.next_power_of_2(C)
    num_row_blocks = triton.cdiv(ROWS, row_split)

    out_base = torch.empty((ROWS, C), device=device, dtype=torch.float32)
    out_sum_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum_out = torch.empty((C,), device=device, dtype=torch.float32)

    _zero_three_vectors_kernel[(triton.cdiv(C, 256),)](
        out_sum_x_rhs,
        out_sum_x,
        out_sum_out,
        C_=C,
        BLOCK_C=256,
        num_warps=KERNEL_WARPS,
    )
    _swin_ln_window_atomic_kernel[(num_row_blocks,)](
        mm,
        weight,
        rhs,
        gate,
        residual,
        keep,
        out_base,
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
        ROW_SPLIT=row_split,
        XBLOCK=xblock,
        BLOCK_C=block_c,
        num_warps=KERNEL_WARPS,
    )

    return out_sum_x_rhs, out_sum_x, out_base.permute(1, 0), out_sum_out


def oracle_forward(inputs):
    return oracle_full(*inputs)


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
