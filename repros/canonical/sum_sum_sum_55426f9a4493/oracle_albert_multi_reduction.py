"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ALBERT backward fragment from `Repro.forward` by materializing the required layernorm-backward tensor with the captured f32 operation order, then fusing the twelve product reductions, twelve plain column reductions, and twelve matrix/tensor column reductions into one 128-row partial-reduction Triton pass plus one finalizer that returns the same three vectors and non-contiguous transpose view; Inductor currently emits separate partial-reduction kernels for each sibling source and separate finalizers even though the reductions share the same `[4096, 4096] -> [4096]` axis, because its scheduler cannot group multi-source same-axis reductions across a required materialized side-output producer; the fix is SCHEDULER_FUSION: teach reduction scheduling to form a multi-output partial-reduction group for compatible sibling sources and include the materialized side-output reduction in the shared finalizer."""
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


ROWS = 4096
COLS = 4096
VIEW_SHAPE = (8, 512, COLS)
FLAT_SHAPE = (ROWS, COLS)
SUM_SHAPE = (COLS,)
ARG180_SHAPE = (8, 512, 1)
TRANSPOSE_STRIDE = (1, COLS)

ROW_CHUNK = 128
NUM_CHUNKS = ROWS // ROW_CHUNK
PARTIAL_SOURCES_PER_OUTPUT = 12
PARTIAL_SOURCE_COUNT = 3 * PARTIAL_SOURCES_PER_OUTPUT
PARTIAL_STRIDE = NUM_CHUNKS * COLS
ROWWISE_BLOCK_N = 4096
PARTIAL_BLOCK_C = 64
FINAL_BLOCK_C = 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} stride {tuple(value.stride())} does not match {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} dtype {value.dtype} does not match torch.float32")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have storage_offset=0, got {value.storage_offset()}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 52:
        raise ValueError(f"{REPRO_ID} expects 52 inputs, got {len(inputs)}")

    tensor_inputs: list[torch.Tensor] = []
    flat_stride = (2097152, 4096, 1)
    matrix_stride = (4096, 1)
    for base in range(0, 33, 3):
        tensor_inputs.append(_expect_tensor(f"input {base}", inputs[base], VIEW_SHAPE, flat_stride))
        tensor_inputs.append(_expect_tensor(f"input {base + 1}", inputs[base + 1], VIEW_SHAPE, flat_stride))
        tensor_inputs.append(_expect_tensor(f"input {base + 2}", inputs[base + 2], FLAT_SHAPE, matrix_stride))

    tensor_inputs.append(_expect_tensor("input 33", inputs[33], FLAT_SHAPE, matrix_stride))
    tensor_inputs.append(_expect_tensor("input 34", inputs[34], VIEW_SHAPE, flat_stride))
    tensor_inputs.append(_expect_tensor("input 35", inputs[35], SUM_SHAPE, (1,)))
    tensor_inputs.append(_expect_tensor("input 36", inputs[36], VIEW_SHAPE, flat_stride))
    tensor_inputs.append(_expect_tensor("input 37", inputs[37], ARG180_SHAPE, (512, 1, 1)))

    for i in range(38, 49):
        if _shape_tuple(inputs[i]) != SUM_SHAPE:
            raise ValueError(f"shape input {i} is {inputs[i]!r}, expected {SUM_SHAPE}")
    if _shape_tuple(inputs[49]) != VIEW_SHAPE:
        raise ValueError(f"shape input 49 is {inputs[49]!r}, expected {VIEW_SHAPE}")
    if _shape_tuple(inputs[50]) != FLAT_SHAPE:
        raise ValueError(f"shape input 50 is {inputs[50]!r}, expected {FLAT_SHAPE}")
    if _shape_tuple(inputs[51]) != SUM_SHAPE:
        raise ValueError(f"shape input 51 is {inputs[51]!r}, expected {SUM_SHAPE}")
    return tuple(tensor_inputs)


if triton is not None:

    @triton.jit
    def _rowwise_ln_backward_kernel(
        mul_323_ptr,
        mm_138_ptr,
        arg9_ptr,
        arg22_ptr,
        arg180_ptr,
        out_base_ptr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols_base = tl.arange(0, BLOCK_N)

        row_sum = 0.0
        row_dot = 0.0
        for col_start in tl.range(0, 4096, BLOCK_N):
            cols = col_start + cols_base
            mask = cols < 4096
            offsets = row * 4096 + cols
            add30 = (
                tl.load(mul_323_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                + tl.load(mm_138_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            )
            weighted = add30 * tl.load(arg9_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            arg22 = tl.load(arg22_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            row_sum += tl.sum(tl.where(mask, weighted, 0.0), axis=0)
            row_dot += tl.sum(tl.where(mask, weighted * arg22, 0.0), axis=0)

        row_scale = tl.load(arg180_ptr + row).to(tl.float32)
        for col_start in tl.range(0, 4096, BLOCK_N):
            cols = col_start + cols_base
            mask = cols < 4096
            offsets = row * 4096 + cols
            add30 = (
                tl.load(mul_323_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                + tl.load(mm_138_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            )
            weighted = add30 * tl.load(arg9_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            arg22 = tl.load(arg22_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            out = row_scale * ((weighted * 4096.0 - row_sum) - arg22 * row_dot)
            tl.store(out_base_ptr + offsets, out, mask=mask)

    @triton.jit
    def _store_pair_partials(
        a_ptr,
        b_ptr,
        partials_ptr,
        cols,
        rows,
        col_mask,
        chunk,
        src: tl.constexpr,
    ):
        offsets = cols + 4096 * rows
        a = tl.load(a_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        prod_sum = tl.sum(a * b, axis=1)[:, None]
        a_sum = tl.sum(a, axis=1)[:, None]
        store_offsets = chunk * 4096 + cols
        tl.store(partials_ptr + src * 131072 + store_offsets, prod_sum, mask=col_mask)
        tl.store(partials_ptr + (12 + src) * 131072 + store_offsets, a_sum, mask=col_mask)

    @triton.jit
    def _store_matrix_partials(
        matrix_ptr,
        partials_ptr,
        cols,
        rows,
        col_mask,
        chunk,
        src: tl.constexpr,
    ):
        offsets = cols + 4096 * rows
        values = tl.load(matrix_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        value_sum = tl.sum(values, axis=1)[:, None]
        store_offsets = chunk * 4096 + cols
        tl.store(partials_ptr + (24 + src) * 131072 + store_offsets, value_sum, mask=col_mask)

    @triton.jit
    def _store_last_partials(
        mul_323_ptr,
        mm_138_ptr,
        arg22_ptr,
        out_base_ptr,
        partials_ptr,
        cols,
        rows,
        col_mask,
        chunk,
    ):
        offsets = cols + 4096 * rows
        add30 = (
            tl.load(mul_323_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
            + tl.load(mm_138_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        )
        arg22 = tl.load(arg22_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        out_values = tl.load(out_base_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
        store_offsets = chunk * 4096 + cols
        tl.store(partials_ptr + 11 * 131072 + store_offsets, tl.sum(add30 * arg22, axis=1)[:, None], mask=col_mask)
        tl.store(partials_ptr + 23 * 131072 + store_offsets, tl.sum(add30, axis=1)[:, None], mask=col_mask)
        tl.store(partials_ptr + 35 * 131072 + store_offsets, tl.sum(out_values, axis=1)[:, None], mask=col_mask)

    @triton.jit
    def _fused_partials_kernel(
        add0_ptr,
        mul0_ptr,
        mat0_ptr,
        add1_ptr,
        mul1_ptr,
        mat1_ptr,
        add2_ptr,
        mul2_ptr,
        mat2_ptr,
        add3_ptr,
        mul3_ptr,
        mat3_ptr,
        add4_ptr,
        mul4_ptr,
        mat4_ptr,
        add5_ptr,
        mul5_ptr,
        mat5_ptr,
        add6_ptr,
        mul6_ptr,
        mat6_ptr,
        add7_ptr,
        mul7_ptr,
        mat7_ptr,
        add8_ptr,
        mul8_ptr,
        mat8_ptr,
        add9_ptr,
        mul9_ptr,
        mat9_ptr,
        add10_ptr,
        mul10_ptr,
        mat10_ptr,
        mm_138_ptr,
        mul_323_ptr,
        arg22_ptr,
        out_base_ptr,
        partials_ptr,
        BLOCK_C: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        chunk = tl.program_id(1)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
        rows = chunk * BLOCK_R + tl.arange(0, BLOCK_R)[None, :]
        col_mask = cols < 4096

        _store_pair_partials(add0_ptr, mul0_ptr, partials_ptr, cols, rows, col_mask, chunk, 0)
        _store_pair_partials(add1_ptr, mul1_ptr, partials_ptr, cols, rows, col_mask, chunk, 1)
        _store_pair_partials(add2_ptr, mul2_ptr, partials_ptr, cols, rows, col_mask, chunk, 2)
        _store_pair_partials(add3_ptr, mul3_ptr, partials_ptr, cols, rows, col_mask, chunk, 3)
        _store_pair_partials(add4_ptr, mul4_ptr, partials_ptr, cols, rows, col_mask, chunk, 4)
        _store_pair_partials(add5_ptr, mul5_ptr, partials_ptr, cols, rows, col_mask, chunk, 5)
        _store_pair_partials(add6_ptr, mul6_ptr, partials_ptr, cols, rows, col_mask, chunk, 6)
        _store_pair_partials(add7_ptr, mul7_ptr, partials_ptr, cols, rows, col_mask, chunk, 7)
        _store_pair_partials(add8_ptr, mul8_ptr, partials_ptr, cols, rows, col_mask, chunk, 8)
        _store_pair_partials(add9_ptr, mul9_ptr, partials_ptr, cols, rows, col_mask, chunk, 9)
        _store_pair_partials(add10_ptr, mul10_ptr, partials_ptr, cols, rows, col_mask, chunk, 10)
        _store_last_partials(mul_323_ptr, mm_138_ptr, arg22_ptr, out_base_ptr, partials_ptr, cols, rows, col_mask, chunk)

        _store_matrix_partials(mat0_ptr, partials_ptr, cols, rows, col_mask, chunk, 0)
        _store_matrix_partials(mat1_ptr, partials_ptr, cols, rows, col_mask, chunk, 1)
        _store_matrix_partials(mat2_ptr, partials_ptr, cols, rows, col_mask, chunk, 2)
        _store_matrix_partials(mat3_ptr, partials_ptr, cols, rows, col_mask, chunk, 3)
        _store_matrix_partials(mat4_ptr, partials_ptr, cols, rows, col_mask, chunk, 4)
        _store_matrix_partials(mat5_ptr, partials_ptr, cols, rows, col_mask, chunk, 5)
        _store_matrix_partials(mat6_ptr, partials_ptr, cols, rows, col_mask, chunk, 6)
        _store_matrix_partials(mat7_ptr, partials_ptr, cols, rows, col_mask, chunk, 7)
        _store_matrix_partials(mat8_ptr, partials_ptr, cols, rows, col_mask, chunk, 8)
        _store_matrix_partials(mat9_ptr, partials_ptr, cols, rows, col_mask, chunk, 9)
        _store_matrix_partials(mat10_ptr, partials_ptr, cols, rows, col_mask, chunk, 10)

    @triton.jit
    def _sum_partial_source(
        partials_ptr,
        cols,
        chunks,
        mask,
        src: tl.constexpr,
    ):
        values = tl.load(
            partials_ptr + src * 131072 + chunks * 4096 + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        return tl.sum(values, axis=1)[:, None]

    @triton.jit
    def _finalize_partials_kernel(
        partials_ptr,
        out0_ptr,
        out1_ptr,
        out3_ptr,
        BLOCK_C: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
        chunks = tl.arange(0, 32)[None, :]
        mask = cols < 4096

        p0 = _sum_partial_source(partials_ptr, cols, chunks, mask, 0)
        p1 = _sum_partial_source(partials_ptr, cols, chunks, mask, 1)
        p2 = _sum_partial_source(partials_ptr, cols, chunks, mask, 2)
        p3 = _sum_partial_source(partials_ptr, cols, chunks, mask, 3)
        p4 = _sum_partial_source(partials_ptr, cols, chunks, mask, 4)
        p5 = _sum_partial_source(partials_ptr, cols, chunks, mask, 5)
        p6 = _sum_partial_source(partials_ptr, cols, chunks, mask, 6)
        p7 = _sum_partial_source(partials_ptr, cols, chunks, mask, 7)
        p8 = _sum_partial_source(partials_ptr, cols, chunks, mask, 8)
        p9 = _sum_partial_source(partials_ptr, cols, chunks, mask, 9)
        p10 = _sum_partial_source(partials_ptr, cols, chunks, mask, 10)
        p11 = _sum_partial_source(partials_ptr, cols, chunks, mask, 11)
        out0 = (((((((((((p0 + p1) + p2) + p3) + p4) + p5) + p6) + p7) + p8) + p9) + p10) + p11)

        a0 = _sum_partial_source(partials_ptr, cols, chunks, mask, 12)
        a1 = _sum_partial_source(partials_ptr, cols, chunks, mask, 13)
        a2 = _sum_partial_source(partials_ptr, cols, chunks, mask, 14)
        a3 = _sum_partial_source(partials_ptr, cols, chunks, mask, 15)
        a4 = _sum_partial_source(partials_ptr, cols, chunks, mask, 16)
        a5 = _sum_partial_source(partials_ptr, cols, chunks, mask, 17)
        a6 = _sum_partial_source(partials_ptr, cols, chunks, mask, 18)
        a7 = _sum_partial_source(partials_ptr, cols, chunks, mask, 19)
        a8 = _sum_partial_source(partials_ptr, cols, chunks, mask, 20)
        a9 = _sum_partial_source(partials_ptr, cols, chunks, mask, 21)
        a10 = _sum_partial_source(partials_ptr, cols, chunks, mask, 22)
        a11 = _sum_partial_source(partials_ptr, cols, chunks, mask, 23)
        out1 = (((((((((((a0 + a1) + a2) + a3) + a4) + a5) + a6) + a7) + a8) + a9) + a10) + a11)

        m0 = _sum_partial_source(partials_ptr, cols, chunks, mask, 24)
        m1 = _sum_partial_source(partials_ptr, cols, chunks, mask, 25)
        m2 = _sum_partial_source(partials_ptr, cols, chunks, mask, 26)
        m3 = _sum_partial_source(partials_ptr, cols, chunks, mask, 27)
        m4 = _sum_partial_source(partials_ptr, cols, chunks, mask, 28)
        m5 = _sum_partial_source(partials_ptr, cols, chunks, mask, 29)
        m6 = _sum_partial_source(partials_ptr, cols, chunks, mask, 30)
        m7 = _sum_partial_source(partials_ptr, cols, chunks, mask, 31)
        m8 = _sum_partial_source(partials_ptr, cols, chunks, mask, 32)
        m9 = _sum_partial_source(partials_ptr, cols, chunks, mask, 33)
        m10 = _sum_partial_source(partials_ptr, cols, chunks, mask, 34)
        m11 = _sum_partial_source(partials_ptr, cols, chunks, mask, 35)
        out3 = (((((((((((m0 + m1) + m2) + m3) + m4) + m5) + m6) + m7) + m8) + m9) + m10) + m11)

        tl.store(out0_ptr + cols, out0, mask=mask)
        tl.store(out1_ptr + cols, out1, mask=mask)
        tl.store(out3_ptr + cols, out3, mask=mask)


def oracle_forward(inputs):
    """Run the full Repro.forward computation for the canonical ALBERT shape."""
    (
        add_9,
        arg110_1,
        view_18,
        add_23,
        arg102_1,
        view_48,
        add_47,
        arg94_1,
        view_78,
        add_71,
        arg86_1,
        view_108,
        add_95,
        arg78_1,
        view_138,
        add_119,
        arg70_1,
        view_168,
        add_143,
        arg62_1,
        view_198,
        add_167,
        arg54_1,
        view_228,
        add_191,
        arg46_1,
        view_258,
        add_215,
        arg38_1,
        view_288,
        add_239,
        arg30_1,
        view_318,
        mm_138,
        mul_323,
        arg9_1,
        arg22_1,
        arg180_1,
    ) = _validate_inputs(inputs)

    out_base = torch.empty(FLAT_SHAPE, device=mul_323.device, dtype=torch.float32)
    partials = torch.empty(
        (PARTIAL_SOURCE_COUNT, NUM_CHUNKS, COLS),
        device=mul_323.device,
        dtype=torch.float32,
    )
    out0 = torch.empty(SUM_SHAPE, device=mul_323.device, dtype=torch.float32)
    out1 = torch.empty(SUM_SHAPE, device=mul_323.device, dtype=torch.float32)
    out3 = torch.empty(SUM_SHAPE, device=mul_323.device, dtype=torch.float32)

    _rowwise_ln_backward_kernel[(ROWS,)](
        mul_323,
        mm_138,
        arg9_1,
        arg22_1,
        arg180_1,
        out_base,
        BLOCK_N=ROWWISE_BLOCK_N,
        num_warps=4,
    )
    _fused_partials_kernel[(triton.cdiv(COLS, PARTIAL_BLOCK_C), NUM_CHUNKS)](
        add_9,
        arg110_1,
        view_18,
        add_23,
        arg102_1,
        view_48,
        add_47,
        arg94_1,
        view_78,
        add_71,
        arg86_1,
        view_108,
        add_95,
        arg78_1,
        view_138,
        add_119,
        arg70_1,
        view_168,
        add_143,
        arg62_1,
        view_198,
        add_167,
        arg54_1,
        view_228,
        add_191,
        arg46_1,
        view_258,
        add_215,
        arg38_1,
        view_288,
        add_239,
        arg30_1,
        view_318,
        mm_138,
        mul_323,
        arg22_1,
        out_base,
        partials,
        BLOCK_C=PARTIAL_BLOCK_C,
        BLOCK_R=ROW_CHUNK,
        num_warps=4,
    )
    _finalize_partials_kernel[(triton.cdiv(COLS, FINAL_BLOCK_C),)](
        partials,
        out0,
        out1,
        out3,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=4,
    )

    return (out0, out1, out_base.permute(1, 0), out3)


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [x for x in out if isinstance(x, torch.Tensor)]
    return []


def _strict_scope_check(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    actual_list = _normalize_outputs(actual)
    if len(eager_list) != len(actual_list):
        print(
            f"  strict scope: FAIL output_count oracle={len(actual_list)} "
            f"eager={len(eager_list)}"
        )
        return False

    ok = True
    for idx, (got, expected) in enumerate(zip(actual_list, eager_list)):
        shape_ok = got.shape == expected.shape
        dtype_ok = got.dtype == expected.dtype
        stride_ok = got.stride() == expected.stride()
        ok = ok and shape_ok and dtype_ok and stride_ok
        print(
            f"  strict scope output {idx}: "
            f"shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok} "
            f"oracle_stride={got.stride()} eager_stride={expected.stride()}"
        )
    print(f"  strict scope: {'PASS' if ok else 'FAIL'}")
    return ok


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
        strict_ok = _strict_scope_check(instance, inputs)
        ok = ok and strict_ok
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
