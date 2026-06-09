"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete f32 bias add, Longformer constant-pad, overlapping as_strided clone materialization, and final non-contiguous `[96, 64, 768]` returned layout into one Triton kernel, whereas Inductor currently schedules the add, pad, clone, and layout materialization through separate generic regions; Inductor cannot do this today because padded affine as_strided clone materialization is a scheduler fusion barrier even when the final consumer is only a stride-changing view; the fix is SCHEDULER_FUSION: teach the scheduler to fuse pointwise producers and constant padding into affine overlapping layout materialization kernels while preserving the exact final output stride."""
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


SEQ = 1024
BATCH = 2
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
HEAD_BATCH = BATCH * HEADS
WINDOWS = 4
WINDOW_SIZE = 768
WINDOW_STEP = 256
PAD_BEFORE = 256
BASE_SHAPE = (HEAD_BATCH * WINDOWS, WINDOW_SIZE, HEAD_DIM)
BASE_STRIDE = (WINDOW_SIZE * HEAD_DIM, HEAD_DIM, 1)
RETURN_SHAPE = (HEAD_BATCH * WINDOWS, HEAD_DIM, WINDOW_SIZE)
RETURN_STRIDE = (WINDOW_SIZE * HEAD_DIM, 1, HEAD_DIM)
NUMEL = HEAD_BATCH * WINDOWS * WINDOW_SIZE * HEAD_DIM
BLOCK_P = 16


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _padded_bias_layout_kernel(
        mm_ptr,
        bias_ptr,
        out_ptr,
        head_dim: tl.constexpr,
        window_size: tl.constexpr,
        window_step: tl.constexpr,
        pad_before: tl.constexpr,
        seq_len: tl.constexpr,
        heads: tl.constexpr,
        batch_size: tl.constexpr,
        hidden: tl.constexpr,
        windows: tl.constexpr,
        block_p: tl.constexpr,
    ):
        out_n = tl.program_id(0)
        pos = tl.program_id(1) * block_p + tl.arange(0, block_p)
        dim = tl.arange(0, head_dim)
        window = out_n % windows
        head_batch = out_n // windows
        batch = head_batch // heads
        head = head_batch - batch * heads

        source_seq = pos + window * window_step - pad_before
        source_feature = head * head_dim + dim
        source_row = source_seq * batch_size + batch
        source_offsets = source_row[:, None] * hidden + source_feature[None, :]
        valid = (pos < window_size) & (source_seq >= 0) & (source_seq < seq_len)
        safe_offsets = tl.where(valid[:, None], source_offsets, 0)

        mm = tl.load(mm_ptr + safe_offsets, mask=valid[:, None], other=0.0)
        bias = tl.load(bias_ptr + source_feature)
        value = tl.where(valid[:, None], mm + bias[None, :], -1.0)

        output_offsets = out_n * window_size * head_dim + pos[:, None] * head_dim + dim[None, :]
        tl.store(out_ptr + output_offsets, value, mask=pos[:, None] < window_size)


def _shape_tuple(value, name):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _expect_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm_46, arg184_1, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(mm_46, torch.Tensor) or not isinstance(arg184_1, torch.Tensor):
        raise TypeError("expected first two inputs to be tensors")
    if not mm_46.is_cuda or not arg184_1.is_cuda:
        raise RuntimeError("This oracle is a CUDA/Triton oracle and requires CUDA inputs")
    if mm_46.dtype != torch.float32 or arg184_1.dtype != torch.float32:
        raise ValueError(f"expected f32 inputs, got {mm_46.dtype=} {arg184_1.dtype=}")
    if tuple(mm_46.shape) != (SEQ * BATCH, HIDDEN):
        raise ValueError(f"unexpected mm_46 shape: {tuple(mm_46.shape)}")
    if tuple(arg184_1.shape) != (HIDDEN,):
        raise ValueError(f"unexpected arg184_1 shape: {tuple(arg184_1.shape)}")
    if tuple(mm_46.stride()) != (HIDDEN, 1):
        raise ValueError(f"mm_46 must be contiguous, got stride={tuple(mm_46.stride())}")
    if tuple(arg184_1.stride()) != (1,):
        raise ValueError(f"arg184_1 must be contiguous, got stride={tuple(arg184_1.stride())}")

    expected_shapes = (
        (SEQ, BATCH, HIDDEN),
        (SEQ, BATCH, HEADS, HEAD_DIM),
        (HEAD_BATCH, SEQ, HEAD_DIM),
        BASE_SHAPE,
    )
    actual_shapes = (
        _shape_tuple(shape0, "_shape_param_0"),
        _shape_tuple(shape1, "_shape_param_1"),
        _shape_tuple(shape2, "_shape_param_2"),
        _shape_tuple(shape3, "_shape_param_3"),
    )
    if actual_shapes != expected_shapes:
        raise ValueError(f"unexpected shape parameters: {actual_shapes}")

    return mm_46, arg184_1


def _check_output_layout(output: torch.Tensor) -> bool:
    return (
        tuple(output.shape) == RETURN_SHAPE
        and tuple(output.stride()) == RETURN_STRIDE
        and output.dtype == torch.float32
        and output.storage_offset() == 0
    )


@oracle_impl(hardware="H100", shapes="(T([2048, 768], f32), T([768], f32), S([1024, 2, 768]), S([1024, 2, 12, 64]), S([24, 1024, 64]), S([96, 768, 64]))")
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
    mm_46, arg184_1 = _expect_inputs(inputs)

    output = torch.empty_strided(
        RETURN_SHAPE,
        RETURN_STRIDE,
        device=mm_46.device,
        dtype=torch.float32,
    )
    grid = (HEAD_BATCH * WINDOWS, triton.cdiv(WINDOW_SIZE, BLOCK_P))
    _padded_bias_layout_kernel[grid](
        mm_46,
        arg184_1,
        output,
        head_dim=HEAD_DIM,
        window_size=WINDOW_SIZE,
        window_step=WINDOW_STEP,
        pad_before=PAD_BEFORE,
        seq_len=SEQ,
        heads=HEADS,
        batch_size=BATCH,
        hidden=HIDDEN,
        windows=WINDOWS,
        block_p=BLOCK_P,
        num_warps=4,
        num_stages=1,
    )
    return output


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
        with torch.no_grad():
            layout_output = oracle_forward(inputs)
            if layout_output.is_cuda:
                torch.cuda.synchronize()
        layout_ok = _check_output_layout(layout_output)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_output.shape)} stride={list(layout_output.stride())} "
            f"dtype={layout_output.dtype})"
        )
        ok = ok and layout_ok
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
