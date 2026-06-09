"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MT5 dropout/backward scope from Repro.forward, sharing the 17-term f32 add/dropout producer across the sibling `[512]` channel reduction, the per-row reduction, and the dependent transposed dense epilogue; Inductor currently schedules the same full graph with extra separation around these sibling reductions and the layout-changing epilogue, so it cannot keep the shared producer and reduction partials co-planned across both outputs; the fix is SCHEDULER_FUSION: teach the scheduler/codegen to emit a multi-output reduction template that fuses a shared same-numel producer into sibling reductions plus the row-finalized permuted epilogue while preserving the captured f32 operation order."""
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

ROWS = 4096
COLS = 512
MASK_SCALE = 1.1111111111111112
ROW_BLOCK = 16
COL_BLOCK = 64
ROW_BLOCKS = ROWS // ROW_BLOCK
COL_BLOCKS = COLS // COL_BLOCK
OUT1_SHAPE = (COLS, ROWS)
OUT1_STRIDE = (1, COLS)
CLASSIFICATION = "SCHEDULER_FUSION"


from oracle_harness import (
    oracle_impl,  # noqa: E402
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
        mm_11,
        arg602,
        mm_13,
        mm_33,
        mm_35,
        mm_55,
        mm_57,
        mm_77,
        mm_79,
        mm_99,
        mm_101,
        mm_121,
        mm_123,
        mm_143,
        mm_145,
        mm_165,
        mm_167,
        offsets,
        mask,
    ):
        acc = tl.load(arg602 + offsets, mask=mask, other=0.0).to(tl.float32)
        acc = _add_rn_f32(acc, tl.load(mm_11 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_13 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_33 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_35 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_55 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_57 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_77 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_79 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_99 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_101 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_121 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_123 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_143 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_145 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_165 + offsets, mask=mask, other=0.0).to(tl.float32))
        acc = _add_rn_f32(acc, tl.load(mm_167 + offsets, mask=mask, other=0.0).to(tl.float32))
        return acc

    @triton.jit
    def _zero_vector_kernel(
        out0,
        COLS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(out0 + cols, tl.zeros((BLOCK_N,), tl.float32), mask=cols < COLS_)

    @triton.jit
    def _partials_kernel(
        mm_11,
        arg602,
        mm_13,
        mm_33,
        mm_35,
        mm_55,
        mm_57,
        mm_77,
        mm_79,
        mm_99,
        mm_101,
        mm_121,
        mm_123,
        mm_143,
        mm_145,
        mm_165,
        mm_167,
        mask1,
        arg74,
        arg315,
        arg316,
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
            mm_11,
            arg602,
            mm_13,
            mm_33,
            mm_35,
            mm_55,
            mm_57,
            mm_77,
            mm_79,
            mm_99,
            mm_101,
            mm_121,
            mm_123,
            mm_143,
            mm_145,
            mm_165,
            mm_167,
            offsets,
            valid,
        )
        mask1_f = tl.load(mask1 + offsets, mask=valid, other=0).to(tl.float32)
        dropout = _mul_rn_f32(mask1_f, MASK_SCALE_)
        dropped = _mul_rn_f32(total, dropout)

        arg74_v = tl.load(arg74 + cols, mask=cols < COLS_, other=0.0).to(tl.float32)
        arg315_v = tl.load(arg315 + offsets, mask=valid, other=0.0).to(tl.float32)
        arg316_v = tl.load(arg316 + rows, mask=rows < ROWS_, other=0.0).to(tl.float32)

        scaled = _mul_rn_f32(dropped, arg74_v[None, :])
        row_terms = _mul_rn_f32(scaled, arg315_v)
        mul3 = _mul_rn_f32(arg315_v, arg316_v[:, None])
        col_terms = _mul_rn_f32(dropped, mul3)

        row_sums = tl.sum(tl.where(valid, row_terms, 0.0), axis=1)
        col_sums = tl.sum(tl.where(valid, col_terms, 0.0), axis=0)
        tl.store(mul2_base + offsets, scaled, mask=valid)
        tl.store(row_partial + col_block * ROWS_ + rows, row_sums, mask=rows < ROWS_)
        tl.atomic_add(out0 + cols, col_sums, sem="relaxed", mask=cols < COLS_)

    @triton.jit
    def _epilogue_kernel(
        mul2_base,
        arg315,
        arg316,
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
        arg315_v = tl.load(arg315 + offsets, mask=valid, other=0.0).to(tl.float32)
        arg316_v = tl.load(arg316 + rows, mask=rows < ROWS_, other=0.0).to(tl.float32)
        row_blocks = tl.arange(0, BLOCK_CB)
        row_vals = tl.load(
            row_partial + row_blocks[:, None] * ROWS_ + rows[None, :],
            mask=(row_blocks[:, None] < COL_BLOCKS_) & (rows[None, :] < ROWS_),
            other=0.0,
        ).to(tl.float32)
        row_total = tl.sum(row_vals, axis=0)

        mul6 = _mul_rn_f32(mul2, arg316_v[:, None])
        arg316_sq = _mul_rn_f32(arg316_v, arg316_v)
        arg316_cu = _mul_rn_f32(arg316_sq, arg316_v)
        neg_half_sum = _mul_rn_f32(row_total, -0.5)
        mul7 = _mul_rn_f32(neg_half_sum, arg316_cu)
        div = _div_rn_f32(mul7, 512.0)
        mul_scalar1 = _mul_rn_f32(arg315_v, 2.0)
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
    if len(inputs) != 41:
        raise ValueError(f"{REPRO_ID} expects 41 inputs, got {len(inputs)}")

    tensors = inputs[:22]
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("the first 22 inputs must be tensors")
    device = tensors[0].device
    if device.type != "cuda":
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if any(value.device != device for value in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    for index in [0, *range(2, 17)]:
        tensor = tensors[index]
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != (ROWS, COLS):
            raise ValueError(
                f"input {index} expected f32[{ROWS}, {COLS}], "
                f"got dtype={tensor.dtype} shape={tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={tensor.stride()}")

    arg602 = tensors[1]
    if arg602.dtype != torch.float32 or tuple(arg602.shape) != (32, 128, COLS):
        raise ValueError(f"arg602_1 has unexpected dtype/shape: {arg602.dtype} {tuple(arg602.shape)}")
    if not arg602.is_contiguous():
        raise ValueError(f"arg602_1 must be contiguous, got stride={arg602.stride()}")

    mask1 = tensors[17]
    arg74 = tensors[18]
    arg315 = tensors[19]
    arg316 = tensors[20]
    mask2 = tensors[21]
    if mask1.dtype != torch.bool or tuple(mask1.shape) != (32, 128, COLS):
        raise ValueError(f"arg317_1 has unexpected dtype/shape: {mask1.dtype} {tuple(mask1.shape)}")
    if mask2.dtype != torch.bool or tuple(mask2.shape) != (32, 128, COLS):
        raise ValueError(f"arg314_1 has unexpected dtype/shape: {mask2.dtype} {tuple(mask2.shape)}")
    if arg74.dtype != torch.float32 or tuple(arg74.shape) != (COLS,):
        raise ValueError(f"arg74_1 has unexpected dtype/shape: {arg74.dtype} {tuple(arg74.shape)}")
    if arg315.dtype != torch.float32 or tuple(arg315.shape) != (32, 128, COLS):
        raise ValueError(f"arg315_1 has unexpected dtype/shape: {arg315.dtype} {tuple(arg315.shape)}")
    if arg316.dtype != torch.float32 or tuple(arg316.shape) != (32, 128, 1):
        raise ValueError(f"arg316_1 has unexpected dtype/shape: {arg316.dtype} {tuple(arg316.shape)}")
    for index in (17, 18, 19, 20, 21):
        if not tensors[index].is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={tensors[index].stride()}")

    expected_shapes = [(32, 128, COLS)] * 16 + [(COLS,), (32, 128, COLS), (ROWS, COLS)]
    actual_shapes = [_shape_tuple(value) for value in inputs[22:]]
    if actual_shapes != expected_shapes:
        raise ValueError(f"unexpected shape params: {actual_shapes}")

    return tuple(tensors)  # type: ignore[return-value]


@oracle_impl(hardware="H100", shapes="(T([4096, 512], f32), T([32, 128, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([32, 128, 512], b8), T([512], f32), T([32, 128, 512], f32), T([32, 128, 1], f32), T([32, 128, 512], b8), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([512]), S([32, 128, 512]), S([4096, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact full Repro.forward scope."""
    (
        mm_11,
        arg602,
        mm_13,
        mm_33,
        mm_35,
        mm_55,
        mm_57,
        mm_77,
        mm_79,
        mm_99,
        mm_101,
        mm_121,
        mm_123,
        mm_143,
        mm_145,
        mm_165,
        mm_167,
        mask1,
        arg74,
        arg315,
        arg316,
        mask2,
    ) = _validate_inputs(inputs)

    row_partial = torch.empty((COL_BLOCKS, ROWS), device=mm_11.device, dtype=torch.float32)
    mul2_base = torch.empty_strided((ROWS, COLS), (COLS, 1), device=mm_11.device, dtype=torch.float32)
    out0 = torch.empty_strided((COLS,), (1,), device=mm_11.device, dtype=torch.float32)
    out_base = torch.empty_strided((ROWS, COLS), (COLS, 1), device=mm_11.device, dtype=torch.float32)

    _zero_vector_kernel[(triton.cdiv(COLS, 256),)](
        out0,
        COLS_=COLS,
        BLOCK_N=256,
        num_warps=4,
    )
    _partials_kernel[(ROW_BLOCKS, COL_BLOCKS)](
        mm_11,
        arg602,
        mm_13,
        mm_33,
        mm_35,
        mm_55,
        mm_57,
        mm_77,
        mm_79,
        mm_99,
        mm_101,
        mm_121,
        mm_123,
        mm_143,
        mm_145,
        mm_165,
        mm_167,
        mask1,
        arg74,
        arg315,
        arg316,
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
        arg315,
        arg316,
        mask2,
        row_partial,
        out_base,
        ROWS_=ROWS,
        COLS_=COLS,
        ROW_BLOCK_=ROW_BLOCK,
        COL_BLOCK_=COL_BLOCK,
        COL_BLOCKS_=COL_BLOCKS,
        BLOCK_CB=8,
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
