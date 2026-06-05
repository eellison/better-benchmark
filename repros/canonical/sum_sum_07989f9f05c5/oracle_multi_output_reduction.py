"""
Full-scope Triton oracle for sum_sum_07989f9f05c5 (MT5 layer-norm backward tail).

Gap diagnosis (classification: SCHEDULER_FUSION): The oracle consumes the same
six tensor inputs and six shape parameters as repro.py, and returns the same
`[512]` column reduction plus `[512, 4096]` transposed epilogue with stride
`(1, 512)`. It differs from Inductor by fusing the per-row channel reduction
directly into the dropout-masked epilogue while separately computing the
orthogonal per-channel reduction with coalesced row tiles. Inductor cannot make
this a single ideal template today because the two sibling reductions reduce
different axes and one reduction feeds a full pointwise/layout epilogue, so the
scheduler keeps the reductions and transpose-shaped epilogue as separate
ordinary nodes instead of one cross-axis multi-output reduction plan. The fix
class is SCHEDULER_FUSION.
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



REPRO_ID = "sum_sum_07989f9f05c5"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 4096
C = 512
N = 32
T = 128
NUMEL = ROWS * C
DROPOUT_SCALE = 1.1111111111111112
INV_C = 1.0 / 512.0

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3"
        ),
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
def _row_epilogue_kernel(
    mm_279_ptr,
    mm_281_ptr,
    arg7_ptr,
    arg202_ptr,
    arg203_ptr,
    add_213_ptr,
    arg201_ptr,
    out_ptr,
    C_: tl.constexpr,
    INV_C_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    offsets = row * C_ + cols

    add_value = (
        tl.load(mm_279_ptr + offsets).to(tl.float32)
        + tl.load(mm_281_ptr + offsets).to(tl.float32)
    )
    arg7 = tl.load(arg7_ptr + cols).to(tl.float32)
    arg202 = tl.load(arg202_ptr + offsets).to(tl.float32)
    weighted = add_value * arg7
    row_sum = tl.sum(weighted * arg202, axis=0)

    arg203 = tl.load(arg203_ptr + row).to(tl.float32)
    arg203_cubed = arg203 * arg203 * arg203
    add_213 = tl.load(add_213_ptr + offsets).to(tl.float32)
    mask = tl.load(arg201_ptr + offsets).to(tl.int1)

    correction = row_sum * arg203_cubed * arg202 * INV_C_
    epilogue = (add_213 + weighted * arg203 - correction) * DROPOUT_SCALE_
    epilogue = tl.where(mask, epilogue, 0.0)
    tl.store(out_ptr + offsets, epilogue)


@triton.jit
def _col_reduce_partial_kernel(
    mm_279_ptr,
    mm_281_ptr,
    arg202_ptr,
    arg203_ptr,
    partial_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    num_row_tiles: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = col_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    active = (rows[:, None] < ROWS_) & (cols[None, :] < C_)
    offsets = rows[:, None] * C_ + cols[None, :]

    add_value = (
        tl.load(mm_279_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(mm_281_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    )
    arg202 = tl.load(arg202_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    arg203 = tl.load(arg203_ptr + rows, mask=rows < ROWS_, other=0.0).to(tl.float32)

    values = add_value * arg202 * arg203[:, None]
    partial = tl.sum(values, axis=0)
    partial_offsets = cols * num_row_tiles + row_tile
    tl.store(partial_ptr + partial_offsets, partial, mask=cols < C_)


@triton.jit
def _col_reduce_finalize_kernel(
    partial_ptr,
    out0_ptr,
    num_row_tiles: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    col = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < num_row_tiles
    values = tl.load(partial_ptr + col * num_row_tiles + tiles, mask=active, other=0.0).to(tl.float32)
    tl.store(out0_ptr + col, tl.sum(values, axis=0))


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    mm_279: torch.Tensor,
    mm_281: torch.Tensor,
    arg7_1: torch.Tensor,
    arg202_1: torch.Tensor,
    arg203_1: torch.Tensor,
    add_213: torch.Tensor,
    arg201_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mm_279.shape == (ROWS, C)
    assert mm_281.shape == (ROWS, C)
    assert arg7_1.shape == (C,)
    assert arg202_1.shape == (N, T, C)
    assert arg203_1.shape == (N, T, 1)
    assert add_213.shape == (N, T, C)
    assert arg201_1.shape == (N, T, C)
    assert tuple(_shape_param_0) == (N, T, C)
    assert tuple(_shape_param_1) == (N, T, C)
    assert tuple(_shape_param_2) == (C,)
    assert tuple(_shape_param_3) == (N, T, C)
    assert tuple(_shape_param_4) == (ROWS, C)
    assert mm_279.is_contiguous()
    assert mm_281.is_contiguous()
    assert arg7_1.is_contiguous()
    assert arg202_1.is_contiguous()
    assert arg203_1.is_contiguous()
    assert add_213.is_contiguous()
    assert arg201_1.is_contiguous()

    out0 = torch.empty((C,), device=mm_279.device, dtype=torch.float32)
    out1 = torch.empty_strided((C, ROWS), (1, C), device=mm_279.device, dtype=torch.float32)

    block_rows = 16
    block_cols = 64
    num_row_tiles = triton.cdiv(ROWS, block_rows)
    partial = torch.empty((C, num_row_tiles), device=mm_279.device, dtype=torch.float32)

    _row_epilogue_kernel[(ROWS,)](
        mm_279,
        mm_281,
        arg7_1,
        arg202_1,
        arg203_1,
        add_213,
        arg201_1,
        out1,
        C_=C,
        INV_C_=INV_C,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_C=C,
        num_warps=8,
    )

    _col_reduce_partial_kernel[(num_row_tiles, triton.cdiv(C, block_cols))](
        mm_279,
        mm_281,
        arg202_1,
        arg203_1,
        partial,
        ROWS_=ROWS,
        C_=C,
        num_row_tiles=num_row_tiles,
        BLOCK_ROWS=block_rows,
        BLOCK_C=block_cols,
        num_warps=4,
    )

    block_tiles = triton.next_power_of_2(num_row_tiles)
    _col_reduce_finalize_kernel[(C,)](
        partial,
        out0,
        num_row_tiles=num_row_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=4,
    )

    return out0, out1


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


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
