"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete T5 dropout/backward scope from Repro.forward, including the 24 matrix views plus residual add chain, the captured `mul_tensor_2` materialization boundary, both bool-mask scales, the channel reduction, the row reduction, and the dependent transposed dense epilogue; it co-plans the shared producer, sibling reductions, and final transpose-view store with Triton partial reductions plus a dense epilogue, while Inductor's compiled result is already within the CUDAGraph harness floor for the same required dense reads, f32 reductions, and full output store; Inductor cannot gain a meaningful local scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, or recompute improvement here because the full computation is dominated by required memory traffic and exact f32 reduction/epilogue work; the fix is BANDWIDTH_BOUND: record this as at-floor unless broader memory/reduction codegen improvements move both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 8192
COLS = 768
MASK_SCALE = 1.1111111111111112
ROW_BLOCK = 16
COL_BLOCK = 64
ROW_BLOCKS = ROWS // ROW_BLOCK
COL_BLOCKS = COLS // COL_BLOCK
OUT1_SHAPE = (COLS, ROWS)
OUT1_STRIDE = (1, COLS)
CLASSIFICATION = "BANDWIDTH_BOUND"


from oracle_harness import (  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _add_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _div_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _sum_add_chain(
        mm_9,
        arg844,
        mm_11,
        mm_29,
        mm_31,
        mm_49,
        mm_51,
        mm_69,
        mm_71,
        mm_89,
        mm_91,
        mm_109,
        mm_111,
        mm_129,
        mm_131,
        mm_149,
        mm_151,
        mm_169,
        mm_171,
        mm_189,
        mm_191,
        mm_209,
        mm_211,
        mm_229,
        mm_231,
        offsets,
        mask,
    ):
        acc = tl.load(arg844 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = _add_rn_f32(acc, tl.load(mm_9 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_11 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_29 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_31 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_49 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_51 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_69 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_71 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_89 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_91 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_109 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_111 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_129 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_131 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_149 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_151 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_169 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_171 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_189 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_191 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_209 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_211 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_229 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_231 + offsets, mask=mask, other=0.0).to(tl.float32))
        return acc

    @triton.jit
    def _partials_kernel(
        mm_9,
        arg844,
        mm_11,
        mm_29,
        mm_31,
        mm_49,
        mm_51,
        mm_69,
        mm_71,
        mm_89,
        mm_91,
        mm_109,
        mm_111,
        mm_129,
        mm_131,
        mm_149,
        mm_151,
        mm_169,
        mm_171,
        mm_189,
        mm_191,
        mm_209,
        mm_211,
        mm_229,
        mm_231,
        mask1,
        arg98,
        arg419,
        arg420,
        row_partial,
        out0,
        mul2_base,
        ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        ROW_BLOCK_: tl.constexpr,
        COL_BLOCK_: tl.constexpr,
        MASK_SCALE_: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        col_block = tl.program_id(1)
        rows = row_block * ROW_BLOCK_ + tl.arange(0, ROW_BLOCK_)
        cols = col_block * COL_BLOCK_ + tl.arange(0, COL_BLOCK_)
        offsets = rows[:, None] * COLS_ + cols[None, :]
        valid = (rows[:, None] < ROWS_) & (cols[None, :] < COLS_)

        total = _sum_add_chain(
            mm_9,
            arg844,
            mm_11,
            mm_29,
            mm_31,
            mm_49,
            mm_51,
            mm_69,
            mm_71,
            mm_89,
            mm_91,
            mm_109,
            mm_111,
            mm_129,
            mm_131,
            mm_149,
            mm_151,
            mm_169,
            mm_171,
            mm_189,
            mm_191,
            mm_209,
            mm_211,
            mm_229,
            mm_231,
            offsets,
            valid,
        )
        mask1_f = tl.load(mask1 + offsets, mask=valid, other=0).to(tl.float32)
        dropout = _mul_rn_f32(mask1_f, MASK_SCALE_)
        dropped = _mul_rn_f32(total, dropout)

        arg98_v = tl.load(arg98 + cols, mask=cols < COLS_, other=0.0).to(tl.float32)
        arg419_v = tl.load(arg419 + offsets, mask=valid, other=0.0).to(tl.float32)
        arg420_v = tl.load(arg420 + rows, mask=rows < ROWS_, other=0.0).to(tl.float32)

        scaled = _mul_rn_f32(dropped, arg98_v[None, :])
        row_terms = _mul_rn_f32(scaled, arg419_v)
        mul3 = _mul_rn_f32(arg419_v, arg420_v[:, None])
        col_terms = _mul_rn_f32(dropped, mul3)

        row_sums = tl.sum(tl.where(valid, row_terms, 0.0), axis=1)
        col_sums = tl.sum(tl.where(valid, col_terms, 0.0), axis=0)
        tl.store(mul2_base + offsets, scaled, mask=valid)
        tl.store(row_partial + col_block * ROWS_ + rows, row_sums, mask=rows < ROWS_)
        tl.atomic_add(out0 + cols, col_sums, sem="relaxed", mask=cols < COLS_)

    @triton.jit
    def _zero_vector_kernel(
        out0,
        COLS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(out0 + cols, tl.zeros((BLOCK_N,), tl.float32), mask=cols < COLS_)

    @triton.jit
    def _epilogue_kernel(
        mul2_base,
        arg419,
        arg420,
        mask2,
        row_partial,
        out_base,
        ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        ROW_BLOCK_: tl.constexpr,
        COL_BLOCK_: tl.constexpr,
        COL_BLOCKS_: tl.constexpr,
        BLOCK_CB: tl.constexpr,
        MASK_SCALE_: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        col_block = tl.program_id(1)
        rows = row_block * ROW_BLOCK_ + tl.arange(0, ROW_BLOCK_)
        cols = col_block * COL_BLOCK_ + tl.arange(0, COL_BLOCK_)
        offsets = rows[:, None] * COLS_ + cols[None, :]
        valid = (rows[:, None] < ROWS_) & (cols[None, :] < COLS_)

        mul2 = tl.load(mul2_base + offsets, mask=valid, other=0.0).to(tl.float32)
        arg419_v = tl.load(arg419 + offsets, mask=valid, other=0.0).to(tl.float32)
        arg420_v = tl.load(arg420 + rows, mask=rows < ROWS_, other=0.0).to(tl.float32)
        row_blocks = tl.arange(0, BLOCK_CB)
        row_vals = tl.load(
            row_partial + row_blocks[:, None] * ROWS_ + rows[None, :],
            mask=(row_blocks[:, None] < COL_BLOCKS_) & (rows[None, :] < ROWS_),
            other=0.0,
        ).to(tl.float32)
        row_total = tl.sum(row_vals, axis=0)

        mul6 = _mul_rn_f32(mul2, arg420_v[:, None])
        arg420_sq = _mul_rn_f32(arg420_v, arg420_v)
        arg420_cu = _mul_rn_f32(arg420_sq, arg420_v)
        neg_half_sum = _mul_rn_f32(row_total, -0.5)
        mul7 = _mul_rn_f32(neg_half_sum, arg420_cu)
        div = _div_rn_f32(mul7, 768.0)
        mul_scalar1 = _mul_rn_f32(arg419_v, 2.0)
        mul8 = _mul_rn_f32(div[:, None], mul_scalar1)
        combined = _add_rn_f32(mul6, mul8)

        mask2_f = tl.load(mask2 + offsets, mask=valid, other=0).to(tl.float32)
        dropout2 = _mul_rn_f32(mask2_f, MASK_SCALE_)
        out = _mul_rn_f32(combined, dropout2)
        tl.store(out_base + offsets, out, mask=valid)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 57:
        raise ValueError(f"{REPRO_ID} expects 57 inputs, got {len(inputs)}")

    tensors = inputs[:30]
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("the first 30 inputs must be tensors")
    device = tensors[0].device
    if device.type != "cuda":
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if any(value.device != device for value in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    for index in [0, *range(2, 25)]:
        tensor = tensors[index]
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != (ROWS, COLS):
            raise ValueError(
                f"input {index} expected f32[{ROWS}, {COLS}], "
                f"got dtype={tensor.dtype} shape={tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={tensor.stride()}")

    arg844 = tensors[1]
    if arg844.dtype != torch.float32 or tuple(arg844.shape) != (8, 1024, COLS):
        raise ValueError(f"arg844_1 has unexpected dtype/shape: {arg844.dtype} {tuple(arg844.shape)}")
    if not arg844.is_contiguous():
        raise ValueError(f"arg844_1 must be contiguous, got stride={arg844.stride()}")

    mask1 = tensors[25]
    arg98 = tensors[26]
    arg419 = tensors[27]
    arg420 = tensors[28]
    mask2 = tensors[29]
    if mask1.dtype != torch.bool or tuple(mask1.shape) != (8, 1024, COLS):
        raise ValueError(f"arg421_1 has unexpected dtype/shape: {mask1.dtype} {tuple(mask1.shape)}")
    if mask2.dtype != torch.bool or tuple(mask2.shape) != (8, 1024, COLS):
        raise ValueError(f"arg418_1 has unexpected dtype/shape: {mask2.dtype} {tuple(mask2.shape)}")
    if arg98.dtype != torch.float32 or tuple(arg98.shape) != (COLS,):
        raise ValueError(f"arg98_1 has unexpected dtype/shape: {arg98.dtype} {tuple(arg98.shape)}")
    if arg419.dtype != torch.float32 or tuple(arg419.shape) != (8, 1024, COLS):
        raise ValueError(f"arg419_1 has unexpected dtype/shape: {arg419.dtype} {tuple(arg419.shape)}")
    if arg420.dtype != torch.float32 or tuple(arg420.shape) != (8, 1024, 1):
        raise ValueError(f"arg420_1 has unexpected dtype/shape: {arg420.dtype} {tuple(arg420.shape)}")
    for index in (25, 26, 27, 28, 29):
        if not tensors[index].is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={tensors[index].stride()}")

    expected_shapes = [(8, 1024, COLS)] * 24 + [(COLS,), (8, 1024, COLS), (ROWS, COLS)]
    actual_shapes = [_shape_tuple(value) for value in inputs[30:]]
    if actual_shapes != expected_shapes:
        raise ValueError(f"unexpected shape params: {actual_shapes}")

    return tuple(tensors)  # type: ignore[return-value]


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact full Repro.forward scope."""
    (
        mm_9,
        arg844,
        mm_11,
        mm_29,
        mm_31,
        mm_49,
        mm_51,
        mm_69,
        mm_71,
        mm_89,
        mm_91,
        mm_109,
        mm_111,
        mm_129,
        mm_131,
        mm_149,
        mm_151,
        mm_169,
        mm_171,
        mm_189,
        mm_191,
        mm_209,
        mm_211,
        mm_229,
        mm_231,
        mask1,
        arg98,
        arg419,
        arg420,
        mask2,
    ) = _validate_inputs(inputs)

    row_partial = torch.empty((COL_BLOCKS, ROWS), device=mm_9.device, dtype=torch.float32)
    mul2_base = torch.empty_strided((ROWS, COLS), (COLS, 1), device=mm_9.device, dtype=torch.float32)
    out0 = torch.empty_strided((COLS,), (1,), device=mm_9.device, dtype=torch.float32)
    out_base = torch.empty_strided((ROWS, COLS), (COLS, 1), device=mm_9.device, dtype=torch.float32)

    _zero_vector_kernel[(triton.cdiv(COLS, 256),)](
        out0,
        COLS_=COLS,
        BLOCK_N=256,
        num_warps=4,
    )
    _partials_kernel[(ROW_BLOCKS, COL_BLOCKS)](
        mm_9,
        arg844,
        mm_11,
        mm_29,
        mm_31,
        mm_49,
        mm_51,
        mm_69,
        mm_71,
        mm_89,
        mm_91,
        mm_109,
        mm_111,
        mm_129,
        mm_131,
        mm_149,
        mm_151,
        mm_169,
        mm_171,
        mm_189,
        mm_191,
        mm_209,
        mm_211,
        mm_229,
        mm_231,
        mask1,
        arg98,
        arg419,
        arg420,
        row_partial,
        out0,
        mul2_base,
        ROWS_=ROWS,
        COLS_=COLS,
        ROW_BLOCK_=ROW_BLOCK,
        COL_BLOCK_=COL_BLOCK,
        MASK_SCALE_=MASK_SCALE,
        num_warps=8,
    )
    _epilogue_kernel[(ROW_BLOCKS, COL_BLOCKS)](
        mul2_base,
        arg419,
        arg420,
        mask2,
        row_partial,
        out_base,
        ROWS_=ROWS,
        COLS_=COLS,
        ROW_BLOCK_=ROW_BLOCK,
        COL_BLOCK_=COL_BLOCK,
        COL_BLOCKS_=COL_BLOCKS,
        BLOCK_CB=16,
        MASK_SCALE_=MASK_SCALE,
        num_warps=8,
    )
    return out0, torch.ops.aten.permute.default(out_base, [1, 0])


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
