"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin window-reverse, two gather indices, channel-scale algebra, three returned channel reductions, masked residual update, final `[256,100352]` transpose view, and flattened channel sum with one fused row-block producer plus small final reducers, whereas Inductor lowers the layout/gather producer, sibling reductions, residual epilogue, mask multiply, transpose view, and final reduction through generic scheduled regions with exposed intermediate traffic; Inductor cannot do this today because its scheduler does not fuse a nontrivial layout/gather producer with multiple downstream reductions and a large transposed pointwise consumer into one multi-output row/channel reduction plan; the fix is SCHEDULER_FUSION: teach the scheduler to keep this window-reverse gather virtual while emitting a multi-accumulator reduction epilogue and direct final-layout writer."""
from __future__ import annotations

import argparse
import math
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
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


BATCH = 128
HEIGHT = 28
WIDTH = 28
WINDOW = 7
WINDOWS_PER_DIM = 4
WINDOW_AREA = WINDOW * WINDOW
CHANNELS = 256
ROWS_PER_BATCH = HEIGHT * WIDTH
ROWS = BATCH * ROWS_PER_BATCH
MASK_DENOM = 0.9913043472915888
MASK_SCALE = 1.0 / MASK_DENOM
BLOCK_ROWS = 8
REDUCE_BLOCK = 1024


if triton is not None:

    @triton.jit
    def _swin_row_block_kernel(
        mm_ptr,
        index_ptr,
        channel_scale_ptr,
        arg217_ptr,
        arg544_ptr,
        residual_ptr,
        mask_ptr,
        flat_out_ptr,
        partial_weighted_ptr,
        partial_plain_ptr,
        partial_final_ptr,
        ROWS_: tl.constexpr,
        ROWS_PER_BATCH_: tl.constexpr,
        WIDTH_: tl.constexpr,
        WINDOW_: tl.constexpr,
        WINDOWS_PER_DIM_: tl.constexpr,
        WINDOW_AREA_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        MASK_SCALE_: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        rows = block_id * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        channels = tl.arange(0, BLOCK_CHANNELS_)
        valid_rows = rows < ROWS_

        batch = rows // ROWS_PER_BATCH_
        rem = rows - batch * ROWS_PER_BATCH_
        gather_h_pos = rem // WIDTH_
        gather_w_pos = rem - gather_h_pos * WIDTH_

        gather_h = tl.load(index_ptr + gather_h_pos, mask=valid_rows, other=0).to(tl.int64)
        gather_w = tl.load(index_ptr + gather_w_pos, mask=valid_rows, other=0).to(tl.int64)

        window_h = gather_h // WINDOW_
        window_w = gather_w // WINDOW_
        inner_h = gather_h - window_h * WINDOW_
        inner_w = gather_w - window_w * WINDOW_
        window_row = (
            ((batch * WINDOWS_PER_DIM_ + window_h) * WINDOWS_PER_DIM_ + window_w)
            * WINDOW_AREA_
            + inner_h * WINDOW_
            + inner_w
        )

        matrix_offsets = window_row[:, None] * CHANNELS_ + channels[None, :]
        row_channel_offsets = rows[:, None] * CHANNELS_ + channels[None, :]
        matrix_mask = valid_rows[:, None]

        gathered = tl.load(mm_ptr + matrix_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
        arg217 = tl.load(arg217_ptr + row_channel_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
        channel_scale = tl.load(channel_scale_ptr + channels).to(tl.float32)
        scaled = gathered * channel_scale[None, :]

        row_scaled_sum = tl.sum(scaled, axis=1)
        row_scaled_arg217_sum = tl.sum(scaled * arg217, axis=1)

        partial_weighted = tl.sum(tl.where(matrix_mask, gathered * arg217, 0.0), axis=0)
        partial_plain = tl.sum(tl.where(matrix_mask, gathered, 0.0), axis=0)

        gate = tl.load(mask_ptr + batch, mask=valid_rows, other=0).to(tl.float32) * MASK_SCALE_
        arg544 = tl.load(arg544_ptr + rows, mask=valid_rows, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + row_channel_offsets, mask=matrix_mask, other=0.0).to(tl.float32)

        updated = residual + arg544[:, None] * (
            scaled * 256.0
            - row_scaled_sum[:, None]
            - arg217 * row_scaled_arg217_sum[:, None]
        )
        final = updated * gate[:, None]

        tl.store(flat_out_ptr + row_channel_offsets, final, mask=matrix_mask)
        partial_final = tl.sum(tl.where(matrix_mask, final, 0.0), axis=0)

        partial_offsets = block_id * CHANNELS_ + channels
        tl.store(partial_weighted_ptr + partial_offsets, partial_weighted)
        tl.store(partial_plain_ptr + partial_offsets, partial_plain)
        tl.store(partial_final_ptr + partial_offsets, partial_final)

    @triton.jit
    def _reduce_stage1_kernel(
        partial_weighted_ptr,
        partial_plain_ptr,
        partial_final_ptr,
        tmp_weighted_ptr,
        tmp_plain_ptr,
        tmp_final_ptr,
        num_blocks: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        channel = tl.program_id(1)
        offsets = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < num_blocks
        partial_offsets = offsets * CHANNELS_ + channel

        weighted = tl.load(partial_weighted_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        plain = tl.load(partial_plain_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        final = tl.load(partial_final_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)

        tmp_offset = chunk * CHANNELS_ + channel
        tl.store(tmp_weighted_ptr + tmp_offset, tl.sum(weighted, axis=0))
        tl.store(tmp_plain_ptr + tmp_offset, tl.sum(plain, axis=0))
        tl.store(tmp_final_ptr + tmp_offset, tl.sum(final, axis=0))

    @triton.jit
    def _reduce_stage2_kernel(
        tmp_weighted_ptr,
        tmp_plain_ptr,
        tmp_final_ptr,
        out_weighted_ptr,
        out_plain_ptr,
        out_final_ptr,
        num_chunks: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < num_chunks
        tmp_offsets = offsets * CHANNELS_ + channel

        weighted = tl.load(tmp_weighted_ptr + tmp_offsets, mask=mask, other=0.0).to(tl.float32)
        plain = tl.load(tmp_plain_ptr + tmp_offsets, mask=mask, other=0.0).to(tl.float32)
        final = tl.load(tmp_final_ptr + tmp_offsets, mask=mask, other=0.0).to(tl.float32)

        tl.store(out_weighted_ptr + channel, tl.sum(weighted, axis=0))
        tl.store(out_plain_ptr + channel, tl.sum(plain, axis=0))
        tl.store(out_final_ptr + channel, tl.sum(final, axis=0))


def _require_cuda_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 14:
        raise ValueError(f"{REPRO_ID} expects 14 inputs, got {len(inputs)}")

    (
        mm_172,
        arg221_1,
        arg27_1,
        arg217_1,
        arg544_1,
        view_679,
        arg216_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
    ) = inputs

    expected_shapes = (
        [2048, 49, 256],
        [2048, 7, 7, 256],
        [128, 4, 4, 7, 7, 256],
        [128, 28, 28, 256],
        [128, 784, 256],
        [100352, 256],
        [256],
    )
    got_shapes = (shape0, shape1, shape2, shape3, shape4, shape5, shape6)
    if tuple(list(s) for s in got_shapes) != expected_shapes:
        raise ValueError(f"unexpected shape parameters: {got_shapes!r}")

    return (
        _require_cuda_tensor("mm_172", mm_172, (ROWS, CHANNELS), torch.float32),
        _require_cuda_tensor("arg221_1", arg221_1, (HEIGHT,), torch.int64),
        _require_cuda_tensor("arg27_1", arg27_1, (CHANNELS,), torch.float32),
        _require_cuda_tensor("arg217_1", arg217_1, (BATCH, HEIGHT, WIDTH, CHANNELS), torch.float32),
        _require_cuda_tensor("arg544_1", arg544_1, (BATCH, HEIGHT, WIDTH, 1), torch.float32),
        _require_cuda_tensor("view_679", view_679, (BATCH, HEIGHT, WIDTH, CHANNELS), torch.float32),
        _require_cuda_tensor("arg216_1", arg216_1, (BATCH, 1, 1), torch.bool),
    )


@oracle_impl(hardware="H100", shapes="(T([100352, 256], f32), T([28], i64, gen=Index(28)), T([256], f32), T([128, 28, 28, 256], f32), T([128, 28, 28, 1], f32), T([128, 28, 28, 256], f32), T([128, 1, 1], b8), S([2048, 49, 256]), S([2048, 7, 7, 256]), S([128, 4, 4, 7, 7, 256]), S([128, 28, 28, 256]), S([128, 784, 256]), S([100352, 256]), S([256]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns
    the same four outputs: two channel reductions, the transposed final view,
    and the flattened channel sum.
    """
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")

    (
        mm_172,
        arg221_1,
        arg27_1,
        arg217_1,
        arg544_1,
        view_679,
        arg216_1,
    ) = _validate_inputs(inputs)

    num_row_blocks = triton.cdiv(ROWS, BLOCK_ROWS)
    num_reduce_chunks = triton.cdiv(num_row_blocks, REDUCE_BLOCK)
    final_reduce_block = 1 << math.ceil(math.log2(num_reduce_chunks))

    flat_out = torch.empty((ROWS, CHANNELS), device=mm_172.device, dtype=torch.float32)
    partial_weighted = torch.empty((num_row_blocks, CHANNELS), device=mm_172.device, dtype=torch.float32)
    partial_plain = torch.empty_like(partial_weighted)
    partial_final = torch.empty_like(partial_weighted)
    tmp_weighted = torch.empty((num_reduce_chunks, CHANNELS), device=mm_172.device, dtype=torch.float32)
    tmp_plain = torch.empty_like(tmp_weighted)
    tmp_final = torch.empty_like(tmp_weighted)
    out_weighted = torch.empty((CHANNELS,), device=mm_172.device, dtype=torch.float32)
    out_plain = torch.empty_like(out_weighted)
    out_final = torch.empty_like(out_weighted)

    _swin_row_block_kernel[(num_row_blocks,)](
        mm_172,
        arg221_1,
        arg27_1,
        arg217_1,
        arg544_1,
        view_679,
        arg216_1,
        flat_out,
        partial_weighted,
        partial_plain,
        partial_final,
        ROWS_=ROWS,
        ROWS_PER_BATCH_=ROWS_PER_BATCH,
        WIDTH_=WIDTH,
        WINDOW_=WINDOW,
        WINDOWS_PER_DIM_=WINDOWS_PER_DIM,
        WINDOW_AREA_=WINDOW_AREA,
        CHANNELS_=CHANNELS,
        MASK_SCALE_=MASK_SCALE,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_CHANNELS_=CHANNELS,
        num_warps=8,
    )
    _reduce_stage1_kernel[(num_reduce_chunks, CHANNELS)](
        partial_weighted,
        partial_plain,
        partial_final,
        tmp_weighted,
        tmp_plain,
        tmp_final,
        num_blocks=num_row_blocks,
        CHANNELS_=CHANNELS,
        BLOCK_N=REDUCE_BLOCK,
        num_warps=4,
    )
    _reduce_stage2_kernel[(CHANNELS,)](
        tmp_weighted,
        tmp_plain,
        tmp_final,
        out_weighted,
        out_plain,
        out_final,
        num_chunks=num_reduce_chunks,
        CHANNELS_=CHANNELS,
        BLOCK_N=final_reduce_block,
        num_warps=1,
    )

    return (
        out_weighted,
        out_plain,
        flat_out.as_strided((CHANNELS, ROWS), (1, CHANNELS)),
        out_final,
    )


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
