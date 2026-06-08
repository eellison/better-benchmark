"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete BERT dropout-weighted softmax-backward row update with the broadcast fill mask loaded once per batch-row across a block of heads, whereas Inductor lowers the reduction as independent flattened rows that reload the [batch,1,row,col] mask for each head; Inductor cannot form this head-blocked row program for a broadcast non-reduction operand today, so the fix is NEW_PATTERN: add a small-row-reduction template that tiles broadcast head dimensions while keeping the dependent fma and final scale fused."""
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
HEADS = 12
ROWS_PER_HEAD = 128
COLS = 128
HEAD_BLOCK = 16
DROPOUT_SCALE_F32 = 1.1111111640930176
FINAL_SCALE_F32 = 0.125
VIEW_SHAPE = (BATCH, HEADS, ROWS_PER_HEAD, COLS)
OUT_SHAPE = (BATCH * HEADS, ROWS_PER_HEAD, COLS)
OUT_STRIDE = (ROWS_PER_HEAD * COLS, COLS, 1)
TENSOR4_STRIDE = (HEADS * ROWS_PER_HEAD * COLS, ROWS_PER_HEAD * COLS, COLS, 1)
MASK_SHAPE = (BATCH, 1, ROWS_PER_HEAD, COLS)
MASK_STRIDE = (ROWS_PER_HEAD * COLS, ROWS_PER_HEAD * COLS, COLS, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _fma_rn_f32(a, b, c):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=[a, b, c],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _head_blocked_mask_kernel(
        bmm_ptr,
        dropout_ptr,
        grad_ptr,
        fill_mask_ptr,
        fill_ptr,
        out_ptr,
        dropout_scale: tl.constexpr,
        final_scale: tl.constexpr,
        head_block: tl.constexpr,
        heads: tl.constexpr,
        rows_per_head: tl.constexpr,
        cols_per_row: tl.constexpr,
    ):
        batch_row = tl.program_id(0)
        head_tile = tl.program_id(1)

        batch = batch_row // rows_per_head
        row = batch_row - batch * rows_per_head
        heads_offsets = head_tile * head_block + tl.arange(0, head_block)
        cols = tl.arange(0, cols_per_row)
        valid_heads = heads_offsets < heads

        offsets = (
            batch * (heads * rows_per_head * cols_per_row)
            + heads_offsets[:, None] * (rows_per_head * cols_per_row)
            + row * cols_per_row
            + cols[None, :]
        )
        head_mask = valid_heads[:, None]

        bmm = tl.load(bmm_ptr + offsets, mask=head_mask, other=0.0).to(tl.float32)
        keep = tl.load(dropout_ptr + offsets, mask=head_mask, other=0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=head_mask, other=0.0).to(tl.float32)

        scale = tl.full((), dropout_scale, tl.float32)
        scaled_keep = keep * scale
        scaled_bmm = bmm * scaled_keep
        product = scaled_bmm * grad
        row_sum = tl.sum(product, axis=1)[:, None].to(tl.float32)

        mask_offsets = batch * (rows_per_head * cols_per_row) + row * cols_per_row + cols
        fill_mask = tl.load(fill_mask_ptr + mask_offsets).to(tl.int1)
        fill = tl.load(fill_ptr + 0).to(tl.float32)

        epilogue = _fma_rn_f32(-grad, row_sum, product)
        selected = tl.where(fill_mask[None, :], fill, epilogue)
        out = selected * tl.full((), final_scale, tl.float32)

        tl.store(out_ptr + offsets, out, mask=head_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    bmm_41, arg123_1, arg122_1, eq_1, full_1, shape0, shape1 = inputs
    tensor_inputs = (bmm_41, arg123_1, arg122_1, eq_1, full_1)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first five inputs must be tensors")

    expected = (
        (OUT_SHAPE, OUT_STRIDE, torch.float32),
        (VIEW_SHAPE, TENSOR4_STRIDE, torch.bool),
        (VIEW_SHAPE, TENSOR4_STRIDE, torch.float32),
        (MASK_SHAPE, MASK_STRIDE, torch.bool),
        ((), (), torch.float32),
    )
    for index, (tensor, (shape, stride, dtype)) in enumerate(zip(tensor_inputs, expected)):
        if tuple(tensor.shape) != shape or tuple(tensor.stride()) != stride or tensor.dtype != dtype:
            raise ValueError(
                f"input {index} expected shape={shape} stride={stride} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
                f"dtype={tensor.dtype}"
            )

    if _shape_tuple(shape0) != VIEW_SHAPE:
        raise ValueError(f"unexpected view shape: {shape0}")
    if _shape_tuple(shape1) != OUT_SHAPE:
        raise ValueError(f"unexpected output shape: {shape1}")
    if not (bmm_41.device == arg123_1.device == arg122_1.device == eq_1.device == full_1.device):
        raise ValueError("all tensor inputs must be on the same device")
    if bmm_41.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for this oracle")

    return bmm_41, arg123_1, arg122_1, eq_1, full_1


def oracle_forward(inputs):
    """Run the full Repro.forward computation."""
    bmm_41, arg123_1, arg122_1, eq_1, full_1 = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_41.device,
        dtype=torch.float32,
    )
    grid = (BATCH * ROWS_PER_HEAD, triton.cdiv(HEADS, HEAD_BLOCK))
    _head_blocked_mask_kernel[grid](
        bmm_41,
        arg123_1,
        arg122_1,
        eq_1,
        full_1,
        out,
        dropout_scale=DROPOUT_SCALE_F32,
        final_scale=FINAL_SCALE_F32,
        head_block=HEAD_BLOCK,
        heads=HEADS,
        rows_per_head=ROWS_PER_HEAD,
        cols_per_row=COLS,
        num_warps=4,
    )
    return out


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
