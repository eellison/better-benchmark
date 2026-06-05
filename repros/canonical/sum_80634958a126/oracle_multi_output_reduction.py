"""Full-scope Triton oracle for sum_80634958a126.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle consumes the same
four float32 inputs and five shape parameters as repro.py, computes the full
three-add producer, returns the same transposed f32[4096, 4096] view with
stride (1, 4096), and returns the same f32[4096] column sum. It differs from
Inductor by explicitly sharing the add producer between the materialized
transpose backing storage and the reduction: one Triton kernel writes the
contiguous add result and per-column partial sums, and a second Triton kernel
finalizes the vector. Inductor cannot express this exact schedule today because
its generic scheduler does not form a single producer-sharing template that
both materializes the backing storage for a returned non-contiguous view and
reduces that producer. The relevant Inductor change would be a scheduler/codegen
template for materialized-view plus reduction producer sharing, but this repro
is gated by the historical best: if the full-scope Triton path does not beat
both required local compile configs and historical best_compile_us, the result
is diagnosis-only/BANDWIDTH_BOUND rather than a true floor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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


REPRO_ID = "sum_80634958a126"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
SEQ = 512
HIDDEN = 4096
ROWS = BATCH * SEQ
COLS = HIDDEN
THREE_D_SHAPE = (BATCH, SEQ, HIDDEN)
TWO_D_SHAPE = (ROWS, COLS)
CONTIG_2D_STRIDE = (COLS, 1)
TRANSPOSE_STRIDE = (1, COLS)

HISTORICAL_BEST_COMPILE_US = 69.2799985408783
CLASSIFICATION = "BANDWIDTH_BOUND"
DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 32
DEFAULT_CANDIDATES = "32x32,32x64,64x16,64x32,128x16"

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
            "triton.multi_kernel=3"
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

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _add_and_partial_sum_kernel(
    mm_142_ptr,
    mul_341_ptr,
    mm_144_ptr,
    mm_146_ptr,
    out_base_ptr,
    partials_ptr,
    ROWS_: tl.constexpr,
    COLS_: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * COLS_ + cols[None, :]
    mask = (rows[:, None] < ROWS_) & (cols[None, :] < COLS_)

    mul_341 = tl.load(mul_341_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    value = mul_341 + tl.load(
        mm_142_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    value += tl.load(
        mm_144_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    value += tl.load(
        mm_146_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    tl.store(out_base_ptr + offsets, value, mask=mask)

    row_mask = rows < ROWS_
    col_mask = cols < COLS_
    partial = tl.sum(tl.where(row_mask[:, None], value, 0.0), axis=0)
    partial_offsets = row_block * COLS_ + cols
    tl.store(partials_ptr + partial_offsets, partial, mask=col_mask)


@triton.jit
def _finalize_sum_kernel(
    partials_ptr,
    sum_out_ptr,
    COLS_: tl.constexpr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_block = tl.program_id(0)
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = row_blocks[:, None] * COLS_ + cols[None, :]
    mask = (row_blocks[:, None] < NUM_ROW_BLOCKS) & (cols[None, :] < COLS_)
    values = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(values, axis=0)
    tl.store(sum_out_ptr + cols, sums, mask=cols < COLS_)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        x.cuda() if isinstance(x, torch.Tensor) and x.device.type != "cuda" else x
        for x in module.make_inputs()
    )


def _validate_power_of_two(name: str, value: int) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two")


def _validate_inputs(
    mm_142: torch.Tensor,
    mul_341: torch.Tensor,
    mm_144: torch.Tensor,
    mm_146: torch.Tensor,
    shape_param_0: object,
    shape_param_1: object,
    shape_param_2: object,
    shape_param_3: object,
    shape_param_4: object,
) -> None:
    assert tuple(mm_142.shape) == TWO_D_SHAPE
    assert tuple(mul_341.shape) == THREE_D_SHAPE
    assert tuple(mm_144.shape) == TWO_D_SHAPE
    assert tuple(mm_146.shape) == TWO_D_SHAPE
    assert tuple(shape_param_0) == THREE_D_SHAPE
    assert tuple(shape_param_1) == THREE_D_SHAPE
    assert tuple(shape_param_2) == THREE_D_SHAPE
    assert tuple(shape_param_3) == TWO_D_SHAPE
    assert tuple(shape_param_4) == (COLS,)
    assert mm_142.dtype == torch.float32
    assert mul_341.dtype == torch.float32
    assert mm_144.dtype == torch.float32
    assert mm_146.dtype == torch.float32
    assert mm_142.is_contiguous()
    assert mul_341.is_contiguous()
    assert mm_144.is_contiguous()
    assert mm_146.is_contiguous()


def _num_warps_for_tile(block_m: int, block_n: int) -> int:
    elems = block_m * block_n
    if elems <= 1024:
        return 4
    return 8


def _num_warps_for_finalize(block_row_blocks: int, block_n: int) -> int:
    elems = block_row_blocks * block_n
    if elems <= 1024:
        return 4
    return 8


def _allocate_workspace(
    device: torch.device,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_row_blocks = triton.cdiv(ROWS, block_m)
    out_base = torch.empty(TWO_D_SHAPE, device=device, dtype=torch.float32)
    sum_out = torch.empty((COLS,), device=device, dtype=torch.float32)
    partials = torch.empty((num_row_blocks, COLS), device=device, dtype=torch.float32)
    return out_base, sum_out, partials


def _oracle_into(
    mm_142: torch.Tensor,
    mul_341: torch.Tensor,
    mm_144: torch.Tensor,
    mm_146: torch.Tensor,
    out_base: torch.Tensor,
    sum_out: torch.Tensor,
    partials: torch.Tensor,
    block_m: int,
    block_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert out_base.shape == TWO_D_SHAPE
    assert out_base.stride() == CONTIG_2D_STRIDE
    assert sum_out.shape == (COLS,)
    assert partials.shape == (triton.cdiv(ROWS, block_m), COLS)

    num_row_blocks = triton.cdiv(ROWS, block_m)
    grid = (num_row_blocks, triton.cdiv(COLS, block_n))
    _add_and_partial_sum_kernel[grid](
        mm_142,
        mul_341,
        mm_144,
        mm_146,
        out_base,
        partials,
        ROWS_=ROWS,
        COLS_=COLS,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=_num_warps_for_tile(block_m, block_n),
    )

    block_row_blocks = triton.next_power_of_2(num_row_blocks)
    _finalize_sum_kernel[(triton.cdiv(COLS, block_n),)](
        partials,
        sum_out,
        COLS_=COLS,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_ROW_BLOCKS=block_row_blocks,
        BLOCK_N=block_n,
        num_warps=_num_warps_for_finalize(block_row_blocks, block_n),
    )

    return out_base.t(), sum_out


def oracle_fused(
    mm_142: torch.Tensor,
    mul_341: torch.Tensor,
    mm_144: torch.Tensor,
    mm_146: torch.Tensor,
    shape_param_0: object,
    shape_param_1: object,
    shape_param_2: object,
    shape_param_3: object,
    shape_param_4: object,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(
        mm_142,
        mul_341,
        mm_144,
        mm_146,
        shape_param_0,
        shape_param_1,
        shape_param_2,
        shape_param_3,
        shape_param_4,
    )
    _validate_power_of_two("block_m", block_m)
    _validate_power_of_two("block_n", block_n)
    out_base, sum_out, partials = _allocate_workspace(mm_142.device, block_m)
    return _oracle_into(
        mm_142,
        mul_341,
        mm_144,
        mm_146,
        out_base,
        sum_out,
        partials,
        block_m,
        block_n,
    )


def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Template-compatible full-scope oracle entry point."""
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        out = model(*inputs)
    return (out,) if isinstance(out, torch.Tensor) else tuple(out)


def main() -> None:
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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
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
