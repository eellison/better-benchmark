"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DINOv2 layernorm-backward/projection fragment returned by `Repro.forward`, sharing the rowwise layernorm reduction scalars across the sibling channel reductions and writing the required transposed side output in one tiled epilogue pass, whereas Inductor's compiled version is already within the same bandwidth floor for the full-scope computation; Inductor cannot materially improve this today because the required dense side output plus row-dependent f32 reductions force streaming the same large tensors and writing the same 512 MiB strided-view backing storage; the fix is BANDWIDTH_BOUND: no scheduler change is justified unless a future template beats this full-scope floor while preserving exact f32 operation ordering."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
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
    get_shape_key,
    has_stochastic_ops,
)


BATCH = 128
TOKENS = 1370
CHANNELS = 768
ROWS = BATCH * TOKENS
ROW_REDUCE_ROWS = 32
TILE_ROWS = 64
TILE_COLS = 64
REDUCE_BLOCKS = (ROWS + TILE_ROWS - 1) // TILE_ROWS
FINAL_BLOCKS = 4096

EXPECTED_SHAPE_PARAMS = (
    [BATCH, TOKENS, CHANNELS],
    [BATCH, TOKENS, CHANNELS],
    [CHANNELS],
    [ROWS, CHANNELS],
    [CHANNELS],
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

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
    def _sub_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;",
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
    def _row_reductions_kernel(
        mm_ptr,
        gamma_ptr,
        xhat_ptr,
        row_sum_ptr,
        row_dot_ptr,
        mm_stride_r: tl.constexpr,
        mm_stride_c: tl.constexpr,
        gamma_stride_c: tl.constexpr,
        xhat_stride_r: tl.constexpr,
        xhat_stride_c: tl.constexpr,
        rows_total: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < rows_total) & (cols[None, :] < channels)

        mm = tl.load(
            mm_ptr + rows[:, None] * mm_stride_r + cols[None, :] * mm_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(
            gamma_ptr + cols * gamma_stride_c,
            mask=cols < channels,
            other=0.0,
        ).to(tl.float32)
        xhat = tl.load(
            xhat_ptr + rows[:, None] * xhat_stride_r + cols[None, :] * xhat_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        weighted = _mul_rn_f32(mm, gamma[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, _mul_rn_f32(weighted, xhat), 0.0), axis=1)
        tl.store(row_sum_ptr + rows, row_sum, mask=rows < rows_total)
        tl.store(row_dot_ptr + rows, row_dot, mask=rows < rows_total)

    @triton.jit
    def _epilogue_tile_kernel(
        mm_ptr,
        gamma_ptr,
        xhat_ptr,
        div_ptr,
        residual_ptr,
        proj_ptr,
        side_scale_ptr,
        row_sum_ptr,
        row_dot_ptr,
        side_base_ptr,
        partials_ptr,
        mm_stride_r: tl.constexpr,
        mm_stride_c: tl.constexpr,
        gamma_stride_c: tl.constexpr,
        xhat_stride_r: tl.constexpr,
        xhat_stride_c: tl.constexpr,
        div_stride_b: tl.constexpr,
        div_stride_t: tl.constexpr,
        residual_stride_r: tl.constexpr,
        residual_stride_c: tl.constexpr,
        proj_stride_r: tl.constexpr,
        proj_stride_c: tl.constexpr,
        side_scale_stride_c: tl.constexpr,
        side_stride_r: tl.constexpr,
        side_stride_c: tl.constexpr,
        rows_total: tl.constexpr,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        reduce_blocks: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        rb = tl.program_id(0)
        cb = tl.program_id(1)
        rows = rb * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < rows_total) & (cols[None, :] < channels)

        mm = tl.load(
            mm_ptr + rows[:, None] * mm_stride_r + cols[None, :] * mm_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(
            gamma_ptr + cols * gamma_stride_c,
            mask=cols < channels,
            other=0.0,
        ).to(tl.float32)
        xhat = tl.load(
            xhat_ptr + rows[:, None] * xhat_stride_r + cols[None, :] * xhat_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        batch = rows // tokens
        token = rows - batch * tokens
        div = tl.load(
            div_ptr + batch * div_stride_b + token * div_stride_t,
            mask=rows < rows_total,
            other=0.0,
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr
            + rows[:, None] * residual_stride_r
            + cols[None, :] * residual_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        proj = tl.load(
            proj_ptr + rows[:, None] * proj_stride_r + cols[None, :] * proj_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        side_scale = tl.load(
            side_scale_ptr + cols * side_scale_stride_c,
            mask=cols < channels,
            other=0.0,
        ).to(tl.float32)
        sum0 = tl.load(row_sum_ptr + rows, mask=rows < rows_total, other=0.0).to(
            tl.float32
        )
        dot0 = tl.load(row_dot_ptr + rows, mask=rows < rows_total, other=0.0).to(
            tl.float32
        )

        weighted = _mul_rn_f32(mm, gamma[None, :])
        scaled = _mul_rn_f32(weighted, 768.0)
        dot_term = _mul_rn_f32(xhat, dot0[:, None])
        centered = _sub_rn_f32(scaled, sum0[:, None])
        centered = _sub_rn_f32(centered, dot_term)
        ln_grad = _mul_rn_f32(div[:, None], centered)
        add_tensor = _add_rn_f32(residual, ln_grad)
        side_value = _mul_rn_f32(add_tensor, side_scale[None, :])

        tl.store(
            side_base_ptr + rows[:, None] * side_stride_r + cols[None, :] * side_stride_c,
            side_value,
            mask=mask,
        )

        valid_values = mask
        out0_vals = _mul_rn_f32(mm, xhat)
        out2_vals = _mul_rn_f32(add_tensor, proj)
        out0 = tl.sum(tl.where(valid_values, out0_vals, 0.0), axis=0)
        out1 = tl.sum(tl.where(valid_values, mm, 0.0), axis=0)
        out2 = tl.sum(tl.where(valid_values, out2_vals, 0.0), axis=0)
        out4 = tl.sum(tl.where(valid_values, side_value, 0.0), axis=0)

        partial_offsets = rb * channels + cols
        c_mask = cols < channels
        tl.store(partials_ptr + partial_offsets, out0, mask=c_mask)
        tl.store(
            partials_ptr + (reduce_blocks * channels) + partial_offsets,
            out1,
            mask=c_mask,
        )
        tl.store(
            partials_ptr + (2 * reduce_blocks * channels) + partial_offsets,
            out2,
            mask=c_mask,
        )
        tl.store(
            partials_ptr + (3 * reduce_blocks * channels) + partial_offsets,
            out4,
            mask=c_mask,
        )

    @triton.jit
    def _finalize_kernel(
        partials_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out4_ptr,
        channels: tl.constexpr,
        reduce_blocks: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        c = tl.program_id(0)
        blocks = tl.arange(0, BLOCK_B)
        mask = blocks < reduce_blocks
        offsets = blocks * channels + c
        plane = reduce_blocks * channels

        out0 = tl.sum(tl.load(partials_ptr + offsets, mask=mask, other=0.0), axis=0)
        out1 = tl.sum(
            tl.load(partials_ptr + plane + offsets, mask=mask, other=0.0),
            axis=0,
        )
        out2 = tl.sum(
            tl.load(partials_ptr + 2 * plane + offsets, mask=mask, other=0.0),
            axis=0,
        )
        out4 = tl.sum(
            tl.load(partials_ptr + 3 * plane + offsets, mask=mask, other=0.0),
            axis=0,
        )

        tl.store(out0_ptr + c, out0)
        tl.store(out1_ptr + c, out1)
        tl.store(out2_ptr + c, out2)
        tl.store(out4_ptr + c, out4)


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
        raise ValueError(f"{name} storage_offset {value.storage_offset()} is unsupported")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")
    (
        mm_90,
        primals_13,
        mul_3,
        div_23,
        add_130,
        addmm_1,
        primals_12,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs

    mm_90 = _expect_tensor("mm_90", mm_90, (ROWS, CHANNELS), (CHANNELS, 1))
    primals_13 = _expect_tensor("primals_13", primals_13, (CHANNELS,), (1,))
    mul_3 = _expect_tensor(
        "mul_3",
        mul_3,
        (BATCH, TOKENS, CHANNELS),
        (TOKENS * CHANNELS, CHANNELS, 1),
    )
    div_23 = _expect_tensor("div_23", div_23, (BATCH, TOKENS, 1), (1376, 1, 1))
    add_130 = _expect_tensor(
        "add_130",
        add_130,
        (BATCH, TOKENS, CHANNELS),
        (TOKENS * CHANNELS, CHANNELS, 1),
    )
    addmm_1 = _expect_tensor("addmm_1", addmm_1, (ROWS, CHANNELS), (CHANNELS, 1))
    primals_12 = _expect_tensor("primals_12", primals_12, (CHANNELS,), (1,))

    for idx, (actual, expected) in enumerate(
        zip((shape0, shape1, shape2, shape3, shape4), EXPECTED_SHAPE_PARAMS),
        start=7,
    ):
        if list(actual) != expected:
            raise ValueError(f"shape parameter {idx} is {actual}, expected {expected}")

    return mm_90, primals_13, mul_3, div_23, add_130, addmm_1, primals_12


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    (
        mm_90,
        primals_13,
        mul_3,
        div_23,
        add_130,
        addmm_1,
        primals_12,
    ) = _validate_inputs(inputs)

    device = mm_90.device
    row_sum = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=device, dtype=torch.float32)
    partials = torch.empty((4, REDUCE_BLOCKS, CHANNELS), device=device, dtype=torch.float32)
    side_base = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)
    out0 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out2 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out4 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    _row_reductions_kernel[(triton.cdiv(ROWS, ROW_REDUCE_ROWS),)](
        mm_90,
        primals_13,
        mul_3,
        row_sum,
        row_dot,
        mm_stride_r=mm_90.stride(0),
        mm_stride_c=mm_90.stride(1),
        gamma_stride_c=primals_13.stride(0),
        xhat_stride_r=mul_3.stride(1),
        xhat_stride_c=mul_3.stride(2),
        rows_total=ROWS,
        channels=CHANNELS,
        BLOCK_R=ROW_REDUCE_ROWS,
        BLOCK_C=1024,
        num_warps=8,
    )
    _epilogue_tile_kernel[(REDUCE_BLOCKS, triton.cdiv(CHANNELS, TILE_COLS))](
        mm_90,
        primals_13,
        mul_3,
        div_23,
        add_130,
        addmm_1,
        primals_12,
        row_sum,
        row_dot,
        side_base,
        partials,
        mm_stride_r=mm_90.stride(0),
        mm_stride_c=mm_90.stride(1),
        gamma_stride_c=primals_13.stride(0),
        xhat_stride_r=mul_3.stride(1),
        xhat_stride_c=mul_3.stride(2),
        div_stride_b=div_23.stride(0),
        div_stride_t=div_23.stride(1),
        residual_stride_r=add_130.stride(1),
        residual_stride_c=add_130.stride(2),
        proj_stride_r=addmm_1.stride(0),
        proj_stride_c=addmm_1.stride(1),
        side_scale_stride_c=primals_12.stride(0),
        side_stride_r=side_base.stride(0),
        side_stride_c=side_base.stride(1),
        rows_total=ROWS,
        tokens=TOKENS,
        channels=CHANNELS,
        reduce_blocks=REDUCE_BLOCKS,
        BLOCK_R=TILE_ROWS,
        BLOCK_C=TILE_COLS,
        num_warps=8,
    )
    _finalize_kernel[(CHANNELS,)](
        partials,
        out0,
        out1,
        out2,
        out4,
        channels=CHANNELS,
        reduce_blocks=REDUCE_BLOCKS,
        BLOCK_B=FINAL_BLOCKS,
        num_warps=8,
    )

    return (out0, out1, out2, side_base.t(), out4)


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
