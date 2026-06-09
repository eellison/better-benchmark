"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete two-output Qwen/Llama/Mistral captured scope, fusing the shared add/cast/multiply producers with both the `[0,1]` column sum and the row-sum-driven bf16 transpose epilogue through a tiled multi-accumulator Triton pass plus a small partial-reduction finalizer, whereas Inductor currently schedules the decomposed sibling reductions and large pointwise transpose-producing epilogue as generic fused regions that replay or materialize more of the shared producer work; Inductor cannot do this today because its scheduler does not co-plan orthogonal row and column reductions with a full-size sibling epilogue as one multi-output reduction template; the fix is SCHEDULER_FUSION: add a guarded multi-output reduction schedule that keeps shared producers live across row and column accumulators and emits the required pointwise/transpose output in the same tiled pass."""
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


REPRO_ID = "sum_sum_aac6620b978d"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 2048
COLS = 1024
BLOCK_M = 8
BLOCK_N = 1024
N_ROW_BLOCKS = triton.cdiv(ROWS, BLOCK_M)


@triton.jit
def _full_scope_partial_kernel(
    mm_369,
    mm_371,
    arg21,
    arg347,
    arg348,
    add_392,
    partial_cols,
    out_transposed,
    BLOCK_M_: tl.constexpr,
    BLOCK_N_: tl.constexpr,
    COLS_: tl.constexpr,
):
    row_block = tl.program_id(0)
    rows = row_block * BLOCK_M_ + tl.arange(0, BLOCK_M_)
    cols = tl.arange(0, BLOCK_N_)
    offsets = rows[:, None] * COLS_ + cols[None, :]

    mm0 = tl.load(mm_369 + offsets).to(tl.float32)
    mm1 = tl.load(mm_371 + offsets).to(tl.float32)
    add_bf16 = (mm0 + mm1).to(tl.bfloat16)

    weight = tl.load(arg21 + cols).to(tl.float32)
    mul_bf16 = (add_bf16.to(tl.float32) * weight[None, :]).to(tl.bfloat16)

    arg347_f32 = tl.load(arg347 + offsets).to(tl.float32)
    scale = tl.load(arg348 + rows).to(tl.float32)
    scaled_arg_bf16 = (arg347_f32 * scale[:, None]).to(tl.bfloat16)

    col_terms = (add_bf16.to(tl.float32) * scaled_arg_bf16.to(tl.float32)).to(tl.bfloat16)
    col_partials = tl.sum(col_terms.to(tl.float32), axis=0)
    tl.store(partial_cols + row_block * COLS_ + cols, col_partials)

    row_terms = mul_bf16.to(tl.float32) * arg347_f32
    row_sums = tl.sum(row_terms, axis=1)

    scale3 = scale * scale * scale
    correction = (-0.0009765625) * row_sums * scale3
    epilogue_f32 = mul_bf16.to(tl.float32) * scale[:, None] + correction[:, None] * arg347_f32
    epilogue_bf16 = epilogue_f32.to(tl.bfloat16)
    residual = tl.load(add_392 + offsets).to(tl.float32)
    out_vals = (residual + epilogue_bf16.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_transposed + rows[:, None] * COLS_ + cols[None, :], out_vals)


@triton.jit
def _finalize_column_sum_kernel(
    partial_cols,
    out_sum,
    BLOCK_M_: tl.constexpr,
    COLS_: tl.constexpr,
):
    col = tl.program_id(0)
    blocks = tl.arange(0, BLOCK_M_)
    vals = tl.load(partial_cols + blocks * COLS_ + col).to(tl.float32)
    total = tl.sum(vals, axis=0)
    tl.store(out_sum + col, total.to(tl.bfloat16))


def get_inputs() -> list[object]:
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    return _harness_get_repro_instance(REPRO_DIR)


def _validate_tensor(name: str, tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if tensor.dtype is not dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


@oracle_impl(hardware="H100", shapes="(T([2048, 1024], bf16), T([2048, 1024], bf16), T([1024], bf16), T([4, 512, 1024], bf16), T([4, 512, 1], f32), T([4, 512, 1024], bf16), S([4, 512, 1024]), S([4, 512, 1024]), S([1024]), S([4, 512, 1024]), S([2048, 1024]))")
def oracle_forward(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, torch.Tensor]:
    (
        mm_369,
        mm_371,
        arg21_1,
        arg347_1,
        arg348_1,
        add_392,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
    ) = inputs

    _validate_tensor("mm_369", mm_369, (ROWS, COLS), torch.bfloat16)
    _validate_tensor("mm_371", mm_371, (ROWS, COLS), torch.bfloat16)
    _validate_tensor("arg21_1", arg21_1, (COLS,), torch.bfloat16)
    _validate_tensor("arg347_1", arg347_1, (4, 512, COLS), torch.bfloat16)
    _validate_tensor("arg348_1", arg348_1, (4, 512, 1), torch.float32)
    _validate_tensor("add_392", add_392, (4, 512, COLS), torch.bfloat16)

    out_sum = torch.empty((COLS,), device=mm_369.device, dtype=torch.bfloat16)
    out_transposed = torch.empty_strided(
        (COLS, ROWS),
        (1, COLS),
        device=mm_369.device,
        dtype=torch.bfloat16,
    )
    partial_cols = torch.empty((N_ROW_BLOCKS, COLS), device=mm_369.device, dtype=torch.float32)

    _full_scope_partial_kernel[(N_ROW_BLOCKS,)](
        mm_369,
        mm_371,
        arg21_1,
        arg347_1,
        arg348_1,
        add_392,
        partial_cols,
        out_transposed,
        BLOCK_M_=BLOCK_M,
        BLOCK_N_=BLOCK_N,
        COLS_=COLS,
        num_warps=8,
    )
    _finalize_column_sum_kernel[(COLS,)](
        partial_cols,
        out_sum,
        BLOCK_M_=N_ROW_BLOCKS,
        COLS_=COLS,
        num_warps=8,
    )
    return out_sum, out_transposed


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
