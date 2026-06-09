"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin qkv layout clone and sibling column sum returned by Repro.forward by materializing the required transpose backing buffer while also emitting per-column partial sums from the same loaded tiles, whereas Inductor currently writes the clone buffer and then rereads it through two reduction kernels for the sum; Inductor cannot do this today because its scheduler does not generate a multi-output layout-copy plus sibling-reduction partial kernel when the layout-copy result is also returned; the fix is SCHEDULER_FUSION: allow returned pointwise/layout producers to feed reduction partials in the same schedule while preserving the output storage contract."""
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
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


BATCH = 8192
HEADS = 4
WINDOW = 49
HEAD_DIM = 32
QKV = 3
COLUMNS = QKV * HEADS * HEAD_DIM
ROWS = BATCH * WINDOW
REDUCTION_CHUNKS = 196
ROWS_PER_CHUNK = 2048
INPUT_ROWS = BATCH * HEADS
INPUT_STRIDE0 = WINDOW * HEAD_DIM

INPUT0_SHAPE = (INPUT_ROWS, WINDOW, HEAD_DIM)
INPUT1_SHAPE = (INPUT_ROWS, HEAD_DIM, WINDOW)
INPUT2_SHAPE = (INPUT_ROWS, WINDOW, HEAD_DIM)
INPUT0_STRIDE = (WINDOW * HEAD_DIM, HEAD_DIM, 1)
INPUT1_STRIDE = (HEAD_DIM * WINDOW, WINDOW, 1)
INPUT2_STRIDE = (WINDOW * HEAD_DIM, HEAD_DIM, 1)

OUTPUT0_SHAPE = (COLUMNS, ROWS)
OUTPUT0_STRIDE = (1, COLUMNS)
OUTPUT1_SHAPE = (COLUMNS,)
OUTPUT1_STRIDE = (1,)
BASE_SHAPE = (ROWS, COLUMNS)
BASE_STRIDE = (COLUMNS, 1)
PARTIAL_SHAPE = (REDUCTION_CHUNKS, COLUMNS)
PARTIAL_STRIDE = (COLUMNS, 1)

SHAPE_PARAM_EXPECTED = (
    (BATCH, HEADS, WINDOW, HEAD_DIM),
    (BATCH, HEADS, HEAD_DIM, WINDOW),
    (BATCH, HEADS, WINDOW, HEAD_DIM),
    (QKV, BATCH, HEADS, WINDOW, HEAD_DIM),
    (BATCH, WINDOW, COLUMNS),
    (ROWS, COLUMNS),
    (COLUMNS,),
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _expect_shape_param(name: str, value: Any, expected: tuple[int, ...]) -> None:
    actual = tuple(int(dim) for dim in value)
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual}")


def _expect_f32_cuda(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    bmm_92 = _expect_f32_cuda("bmm_92", inputs[0], INPUT0_SHAPE, INPUT0_STRIDE)
    bmm_94 = _expect_f32_cuda("bmm_94", inputs[1], INPUT1_SHAPE, INPUT1_STRIDE)
    bmm_95 = _expect_f32_cuda("bmm_95", inputs[2], INPUT2_SHAPE, INPUT2_STRIDE)
    if bmm_92.device != bmm_94.device or bmm_92.device != bmm_95.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    for index, expected in enumerate(SHAPE_PARAM_EXPECTED, start=3):
        _expect_shape_param(f"_shape_param_{index - 3}", inputs[index], expected)

    return bmm_92, bmm_94, bmm_95


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _materialize_chunk_partial_kernel(
        bmm_92,
        bmm_94,
        bmm_95,
        base,
        partials,
        P: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        head = tl.program_id(1)
        row_offsets = tl.arange(0, BLOCK_ROWS)[:, None]
        d = tl.arange(0, 32)[None, :]
        d_flat = tl.arange(0, 32)
        column = P * 128 + head * 32 + d
        column_flat = P * 128 + head * 32 + d_flat
        partial = tl.zeros((32,), tl.float32)

        for block_start in tl.range(0, 2048, BLOCK_ROWS):
            row = chunk * 2048 + block_start + row_offsets
            batch = row // 49
            window_index = row - batch * 49
            input_batch_head = batch * 4 + head
            qv_offsets = input_batch_head * 1568 + window_index * 32 + d
            k_offsets = input_batch_head * 1568 + d * 49 + window_index

            if P == 0:
                values = tl.load(bmm_95 + qv_offsets) * 0.1767766952966369
            elif P == 1:
                values = tl.load(bmm_94 + k_offsets)
            else:
                values = tl.load(bmm_92 + qv_offsets)

            tl.store(base + row * 384 + column, values)
            partial += tl.sum(values, axis=0)

        tl.store(partials + chunk * 384 + column_flat, partial)

    @triton.jit
    def _final_sum_kernel(
        partials,
        out,
        BLOCK_BATCH: tl.constexpr,
    ):
        column = tl.program_id(0)
        batches = tl.arange(0, BLOCK_BATCH)
        mask = batches < 196
        values = tl.load(
            partials + batches * 384 + column,
            mask=mask,
            other=0.0,
        )
        total = tl.sum(values, axis=0)
        tl.store(out + column, total)


@oracle_impl(hardware="H100", shapes="(T([32768, 49, 32], f32), T([32768, 32, 49], f32), T([32768, 49, 32], f32), S([8192, 4, 49, 32]), S([8192, 4, 32, 49]), S([8192, 4, 49, 32]), S([3, 8192, 4, 49, 32]), S([8192, 49, 384]), S([401408, 384]), S([384]))")
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
    bmm_92, bmm_94, bmm_95 = _validate_inputs(inputs)
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=bmm_92.device,
        dtype=torch.float32,
    )
    partials = torch.empty_strided(
        PARTIAL_SHAPE,
        PARTIAL_STRIDE,
        device=bmm_92.device,
        dtype=torch.float32,
    )
    summed = torch.empty_strided(
        OUTPUT1_SHAPE,
        OUTPUT1_STRIDE,
        device=bmm_92.device,
        dtype=torch.float32,
    )

    grid = (REDUCTION_CHUNKS, HEADS)
    _materialize_chunk_partial_kernel[grid](
        bmm_92,
        bmm_94,
        bmm_95,
        base,
        partials,
        P=0,
        BLOCK_ROWS=64,
        num_warps=8,
    )
    _materialize_chunk_partial_kernel[grid](
        bmm_92,
        bmm_94,
        bmm_95,
        base,
        partials,
        P=1,
        BLOCK_ROWS=64,
        num_warps=8,
    )
    _materialize_chunk_partial_kernel[grid](
        bmm_92,
        bmm_94,
        bmm_95,
        base,
        partials,
        P=2,
        BLOCK_ROWS=64,
        num_warps=8,
    )
    _final_sum_kernel[(COLUMNS,)](
        partials,
        summed,
        BLOCK_BATCH=triton.next_power_of_2(REDUCTION_CHUNKS),
        num_warps=8,
    )

    output0 = base.as_strided(OUTPUT0_SHAPE, OUTPUT0_STRIDE)
    return output0, summed


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
