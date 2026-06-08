"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full ViT/DINO layernorm-backward slice by sharing the rowwise normalization reductions, then materializing the required permuted side output while accumulating the four sibling column reductions from the same per-element values, whereas Inductor schedules the row reductions, normalization epilogue, materialized multiply/permute view, and sibling reductions as separate generic kernels; Inductor cannot do this today because the scheduler does not build a single multi-output plan that keeps row reduction results live across a dependent pointwise epilogue and simultaneous column reductions with a large returned view; the fix is SCHEDULER_FUSION: teach reduction fusion/codegen to share same-shape dependent reductions and emit a fused materialize-plus-multi-accumulator column reduction kernel for this layernorm-backward pattern."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


N = 175360
C = 768
ROW_BLOCK = 16
ROW_C_BLOCK = 1024
COL_BLOCK_M = 64
COL_BLOCK_C = 64
NUM_ROW_TILES = 2740
FINAL_BLOCK_T = 4096
FINAL_BLOCK_C = 1


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> None:
    if len(inputs) != 12:
        raise ValueError(f"expected 12 repro inputs, got {len(inputs)}")
    tensors = inputs[:7]
    if any(not isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("first 7 repro inputs must be tensors")
    if any(x.device.type != "cuda" for x in tensors):
        raise RuntimeError("oracle expects CUDA tensors from make_inputs()")

    mm_90, arg7_1, arg112_1, arg306_1, add_43, arg111_1, arg6_1 = tensors
    expected = (
        (mm_90, (N, C)),
        (arg7_1, (C,)),
        (arg112_1, (128, 1370, C)),
        (arg306_1, (128, 1370, 1)),
        (add_43, (128, 1370, C)),
        (arg111_1, (N, C)),
        (arg6_1, (C,)),
    )
    for tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"unexpected tensor shape {tuple(tensor.shape)} != {shape}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"expected float32 tensor, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError("oracle expects contiguous captured inputs")

    expected_shapes = ([128, 1370, 768], [128, 1370, 768], [768], [175360, 768], [768])
    for got, want in zip(inputs[7:], expected_shapes, strict=True):
        if list(got) != want:
            raise ValueError(f"unexpected shape parameter {got}, expected {want}")


if triton is not None:

    @triton.jit
    def _f32_mul(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _f32_add(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _f32_sub(a, b):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _row_reductions_kernel(
        mm_ptr,
        gamma_ptr,
        norm_ptr,
        row_sum_ptr,
        row_norm_sum_ptr,
        N_TOTAL: tl.constexpr,
        C_TOTAL: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < N_TOTAL) & (cols[None, :] < C_TOTAL)
        offsets = rows[:, None] * C_TOTAL + cols[None, :]

        x = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=cols < C_TOTAL, other=0.0).to(tl.float32)
        normalized = tl.load(norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        weighted = _f32_mul(x, gamma[None, :])
        weighted_norm = _f32_mul(weighted, normalized)

        tl.store(row_sum_ptr + rows, tl.sum(weighted, axis=1), mask=rows < N_TOTAL)
        tl.store(row_norm_sum_ptr + rows, tl.sum(weighted_norm, axis=1), mask=rows < N_TOTAL)

    @triton.jit
    def _materialize_and_reduce_kernel(
        mm_ptr,
        gamma_ptr,
        norm_ptr,
        scale_ptr,
        residual_ptr,
        arg111_ptr,
        arg6_ptr,
        row_sum_ptr,
        row_norm_sum_ptr,
        materialized_ptr,
        partial_ptr,
        N_TOTAL: tl.constexpr,
        C_TOTAL: tl.constexpr,
        NUM_TILES: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        tile_m = tl.program_id(0)
        tile_c = tl.program_id(1)
        rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < N_TOTAL) & (cols[None, :] < C_TOTAL)
        offsets = rows[:, None] * C_TOTAL + cols[None, :]

        x = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=cols < C_TOTAL, other=0.0).to(tl.float32)
        normalized = tl.load(norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=rows < N_TOTAL, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        arg111 = tl.load(arg111_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        arg6 = tl.load(arg6_ptr + cols, mask=cols < C_TOTAL, other=0.0).to(tl.float32)
        row_sum = tl.load(row_sum_ptr + rows, mask=rows < N_TOTAL, other=0.0).to(tl.float32)
        row_norm_sum = tl.load(row_norm_sum_ptr + rows, mask=rows < N_TOTAL, other=0.0).to(tl.float32)

        weighted = _f32_mul(x, gamma[None, :])
        weighted_times_c = _f32_mul(weighted, 768.0)
        norm_row_sum = _f32_mul(normalized, row_norm_sum[:, None])
        centered = _f32_sub(weighted_times_c, row_sum[:, None])
        centered = _f32_sub(centered, norm_row_sum)
        scaled = _f32_mul(scale[:, None], centered)
        add_tensor = _f32_add(residual, scaled)

        out0_elem = _f32_mul(x, normalized)
        out1_elem = x
        out2_elem = _f32_mul(add_tensor, arg111)
        mat_elem = _f32_mul(add_tensor, arg6[None, :])

        tl.store(materialized_ptr + offsets, mat_elem, mask=mask)

        store_cols = cols < C_TOTAL
        partial_base = tile_m * C_TOTAL + cols
        stride = NUM_TILES * C_TOTAL
        tl.store(partial_ptr + partial_base, tl.sum(out0_elem, axis=0), mask=store_cols)
        tl.store(partial_ptr + stride + partial_base, tl.sum(out1_elem, axis=0), mask=store_cols)
        tl.store(partial_ptr + stride * 2 + partial_base, tl.sum(out2_elem, axis=0), mask=store_cols)
        tl.store(partial_ptr + stride * 3 + partial_base, tl.sum(mat_elem, axis=0), mask=store_cols)

    @triton.jit
    def _finalize_reductions_kernel(
        partial_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out4_ptr,
        C_TOTAL: tl.constexpr,
        NUM_TILES: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        tiles = tl.arange(0, BLOCK_T)
        mask = (tiles[:, None] < NUM_TILES) & (cols[None, :] < C_TOTAL)
        offsets = tiles[:, None] * C_TOTAL + cols[None, :]
        stride = NUM_TILES * C_TOTAL

        acc0 = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0), axis=0)
        acc1 = tl.sum(tl.load(partial_ptr + stride + offsets, mask=mask, other=0.0), axis=0)
        acc2 = tl.sum(tl.load(partial_ptr + stride * 2 + offsets, mask=mask, other=0.0), axis=0)
        acc4 = tl.sum(tl.load(partial_ptr + stride * 3 + offsets, mask=mask, other=0.0), axis=0)

        tl.store(out0_ptr + cols, acc0, mask=cols < C_TOTAL)
        tl.store(out1_ptr + cols, acc1, mask=cols < C_TOTAL)
        tl.store(out2_ptr + cols, acc2, mask=cols < C_TOTAL)
        tl.store(out4_ptr + cols, acc4, mask=cols < C_TOTAL)


def oracle_forward(inputs):
    """Run the full-scope Triton oracle for Repro.forward()."""
    _require_triton()
    _validate_inputs(inputs)

    mm_90, arg7_1, arg112_1, arg306_1, add_43, arg111_1, arg6_1, *_shape_params = inputs
    device = mm_90.device

    row_sum = torch.empty((N,), device=device, dtype=torch.float32)
    row_norm_sum = torch.empty((N,), device=device, dtype=torch.float32)
    materialized = torch.empty((N, C), device=device, dtype=torch.float32)
    partial = torch.empty((4, NUM_ROW_TILES, C), device=device, dtype=torch.float32)
    out0 = torch.empty((C,), device=device, dtype=torch.float32)
    out1 = torch.empty((C,), device=device, dtype=torch.float32)
    out2 = torch.empty((C,), device=device, dtype=torch.float32)
    out4 = torch.empty((C,), device=device, dtype=torch.float32)

    _row_reductions_kernel[(triton.cdiv(N, ROW_BLOCK),)](
        mm_90,
        arg7_1,
        arg112_1,
        row_sum,
        row_norm_sum,
        N_TOTAL=N,
        C_TOTAL=C,
        BLOCK_R=ROW_BLOCK,
        BLOCK_C=ROW_C_BLOCK,
        num_warps=8,
    )
    _materialize_and_reduce_kernel[(NUM_ROW_TILES, triton.cdiv(C, COL_BLOCK_C))](
        mm_90,
        arg7_1,
        arg112_1,
        arg306_1,
        add_43,
        arg111_1,
        arg6_1,
        row_sum,
        row_norm_sum,
        materialized,
        partial,
        N_TOTAL=N,
        C_TOTAL=C,
        NUM_TILES=NUM_ROW_TILES,
        BLOCK_M=COL_BLOCK_M,
        BLOCK_C=COL_BLOCK_C,
        num_warps=4,
    )
    _finalize_reductions_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial,
        out0,
        out1,
        out2,
        out4,
        C_TOTAL=C,
        NUM_TILES=NUM_ROW_TILES,
        BLOCK_T=FINAL_BLOCK_T,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
    )

    return (out0, out1, out2, materialized.permute(1, 0), out4)


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
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
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
