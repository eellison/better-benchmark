"""Full-scope oracle for sum_5eec4697d8ff.

Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the
Longformer indexed scatter-add into the cropped attention window, the required
transposed `[768, 2048]` side output, and the sibling `[768]` reduction without
materializing the intermediate `[2359296] -> [24,1536,64] -> [24,1024,64]`
scatter/crop/permute chain. It initializes the returned transposed view from
the original `full_2` buffer, scatters the flattened `bmm_44` values directly
into that returned layout when the target index falls inside the live crop, and
then reduces the returned layout over the 2048 token/batch columns. Inductor
cannot do this today because `index_put(accumulate=True)` followed by
`as_strided`, negative padding crop, clone, view, transpose, and a sibling
reduction is scheduled as generic dense tensor materialization. The fix is
SCATTER_REDUCE: add an indexed scatter-reduce lowering that can target the live
cropped/transposed output layout directly while preserving the reduction
epilogue.
"""
from __future__ import annotations

import argparse
import importlib.util
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



REPRO_ID = "sum_5eec4697d8ff"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

A = 24
HEADS = 12
BATCH = 2
TOKENS = 1024
PADDED_TOKENS = 1536
C = 768
K = 64
SOURCE_BLOCKS = 96
SOURCE_NUMEL = SOURCE_BLOCKS * C * K
OUTPUT_ROWS = TOKENS * BATCH
OUTPUT_NUMEL = OUTPUT_ROWS * C
CROP_START = 256
STRIDE_A = PADDED_TOKENS * K

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
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



@triton.jit
def _init_transposed_output_kernel(
    full_ptr,
    out_base_ptr,
    OUTPUT_NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    K_: tl.constexpr,
    HEADS_: tl.constexpr,
    STRIDE_A_: tl.constexpr,
    CROP_START_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < OUTPUT_NUMEL_

    row = offsets // C_
    c = offsets - row * C_
    token = row // 2
    batch = row - token * 2
    head = c // K_
    k = c - head * K_
    a = batch * HEADS_ + head
    padded_token = token + CROP_START_
    source_offsets = a * STRIDE_A_ + padded_token * K_ + k
    values = tl.load(full_ptr + source_offsets, mask=active, other=0.0).to(tl.float32)
    tl.store(out_base_ptr + offsets, values, mask=active)


@triton.jit
def _scatter_live_crop_kernel(
    bmm_ptr,
    index_ptr,
    out_base_ptr,
    SOURCE_NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    K_: tl.constexpr,
    HEADS_: tl.constexpr,
    STRIDE_A_: tl.constexpr,
    CROP_START_: tl.constexpr,
    TOKENS_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < SOURCE_NUMEL_

    target = tl.load(index_ptr + offsets, mask=active, other=0).to(tl.int64)
    values = tl.load(bmm_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    a = target // STRIDE_A_
    rem = target - a * STRIDE_A_
    padded_token = rem // K_
    k = rem - padded_token * K_
    batch = a // HEADS_
    head = a - batch * HEADS_
    token = padded_token - CROP_START_
    c = head * K_ + k
    row = token * 2 + batch
    out_offsets = row * C_ + c

    in_crop = (
        active
        & (a >= 0)
        & (a < HEADS_ * 2)
        & (padded_token >= CROP_START_)
        & (padded_token < CROP_START_ + TOKENS_)
        & (k >= 0)
        & (k < K_)
    )
    tl.atomic_add(out_base_ptr + out_offsets, values, sem="relaxed", mask=in_crop)


@triton.jit
def _partial_reduce_columns_kernel(
    out_base_ptr,
    partial_ptr,
    num_tiles: tl.constexpr,
    C_: tl.constexpr,
    OUTPUT_ROWS_: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    active = cols < OUTPUT_ROWS_
    values = tl.load(out_base_ptr + cols * C_ + c, mask=active, other=0.0).to(tl.float32)
    tl.store(partial_ptr + c * num_tiles + tile, tl.sum(values, axis=0))


@triton.jit
def _finalize_reduce_kernel(
    partial_ptr,
    out_sum_ptr,
    num_tiles: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < num_tiles
    values = tl.load(partial_ptr + c * num_tiles + tiles, mask=active, other=0.0).to(tl.float32)
    tl.store(out_sum_ptr + c, tl.sum(values, axis=0))


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_longformer_scatter_reduce(
    bmm_44: torch.Tensor,
    full_2: torch.Tensor,
    view_19: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4
    assert bmm_44.shape == (SOURCE_BLOCKS, C, K)
    assert full_2.shape == (A * PADDED_TOKENS * K,)
    assert view_19.shape == (SOURCE_NUMEL,)
    assert bmm_44.is_contiguous()
    assert full_2.is_contiguous()
    assert view_19.is_contiguous()

    out_base = torch.empty((OUTPUT_ROWS, C), device=bmm_44.device, dtype=torch.float32)
    init_block = 256
    _init_transposed_output_kernel[(triton.cdiv(OUTPUT_NUMEL, init_block),)](
        full_2,
        out_base,
        OUTPUT_NUMEL_=OUTPUT_NUMEL,
        C_=C,
        K_=K,
        HEADS_=HEADS,
        STRIDE_A_=STRIDE_A,
        CROP_START_=CROP_START,
        BLOCK=init_block,
        num_warps=4,
    )

    scatter_block = 256
    _scatter_live_crop_kernel[(triton.cdiv(SOURCE_NUMEL, scatter_block),)](
        bmm_44,
        view_19,
        out_base,
        SOURCE_NUMEL_=SOURCE_NUMEL,
        C_=C,
        K_=K,
        HEADS_=HEADS,
        STRIDE_A_=STRIDE_A,
        CROP_START_=CROP_START,
        TOKENS_=TOKENS,
        BLOCK=scatter_block,
        num_warps=4,
    )

    reduce_block = 256
    num_tiles = triton.cdiv(OUTPUT_ROWS, reduce_block)
    partial = torch.empty((C, num_tiles), device=bmm_44.device, dtype=torch.float32)
    out_sum = torch.empty((C,), device=bmm_44.device, dtype=torch.float32)
    _partial_reduce_columns_kernel[(C, num_tiles)](
        out_base,
        partial,
        num_tiles=num_tiles,
        C_=C,
        OUTPUT_ROWS_=OUTPUT_ROWS,
        BLOCK_COLS=reduce_block,
        num_warps=4,
    )
    _finalize_reduce_kernel[(C,)](
        partial,
        out_sum,
        num_tiles=num_tiles,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=1,
    )
    return out_sum, out_base.t()


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def oracle_forward(inputs):
    return oracle_longformer_scatter_reduce(*inputs)


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
