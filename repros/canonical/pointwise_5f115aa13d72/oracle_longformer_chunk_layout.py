"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete Longformer bias add, head/batch layout rewrite, overlapping three-window as_strided stencil clone, and final returned `[288,512,64]` strided view into one Triton materialization kernel, whereas Inductor currently schedules the bias add and overlapping clone/layout materialization through generic pointwise/layout work with avoidable intermediate traffic; Inductor cannot do this today because clone/as_strided materialization with overlapping affine sequence windows is a scheduler fusion barrier even when the consumer is only a stride-changing view; the fix is SCHEDULER_FUSION: teach the scheduler to sink pointwise producers into fixed-overlap layout materialization kernels while preserving the final view stride."""
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
BATCH = 8
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
HEAD_BATCH = BATCH * HEADS
WINDOWS = 3
WINDOW_SIZE = 512
WINDOW_STEP = 256
BASE_SHAPE = (HEAD_BATCH * WINDOWS, HEAD_DIM, WINDOW_SIZE)
BASE_STRIDE = (HEAD_DIM * WINDOW_SIZE, WINDOW_SIZE, 1)
RETURN_SHAPE = (HEAD_BATCH * WINDOWS, WINDOW_SIZE, HEAD_DIM)
RETURN_STRIDE = (HEAD_DIM * WINDOW_SIZE, 1, WINDOW_SIZE)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_D": 16, "BLOCK_P": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_D": 32, "BLOCK_P": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_D": 64, "BLOCK_P": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_D": 64, "BLOCK_P": 32}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL_CHUNKS"],
    )
    @triton.jit
    def _chunk_layout_kernel(
        mm_ptr,
        bias_ptr,
        out_ptr,
        TOTAL_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        pos = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
        dim = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)

        head_batch = chunk // 3
        window = chunk - head_batch * 3
        batch = head_batch // 12
        head = head_batch - batch * 12

        source_seq = window * 256 + pos[None, :]
        source_feature = head * 64 + dim[:, None]

        load_offsets = (source_seq * 8 + batch) * 768 + source_feature
        store_offsets = chunk * 64 * 512 + dim[:, None] * 512 + pos[None, :]
        mask = (chunk < TOTAL_CHUNKS) & (dim[:, None] < 64) & (pos[None, :] < 512)

        values = tl.load(mm_ptr + load_offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + source_feature, mask=dim[:, None] < 64, other=0.0)
        tl.store(out_ptr + store_offsets, values + bias, mask=mask)


def _shape_tuple(value, name):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _expect_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    (
        mm_45,
        arg182_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs
    if not isinstance(mm_45, torch.Tensor) or not isinstance(arg182_1, torch.Tensor):
        raise TypeError("expected first two inputs to be tensors")
    if not mm_45.is_cuda or not arg182_1.is_cuda:
        raise RuntimeError("This oracle is a CUDA/Triton oracle and requires CUDA inputs")
    if mm_45.device != arg182_1.device:
        raise ValueError("input tensors must be on the same device")
    if mm_45.dtype != torch.float32 or arg182_1.dtype != torch.float32:
        raise ValueError(f"expected f32 inputs, got {mm_45.dtype=} {arg182_1.dtype=}")
    if tuple(mm_45.shape) != (SEQ * BATCH, HIDDEN):
        raise ValueError(f"unexpected mm_45 shape: {tuple(mm_45.shape)}")
    if tuple(arg182_1.shape) != (HIDDEN,):
        raise ValueError(f"unexpected arg182_1 shape: {tuple(arg182_1.shape)}")
    if tuple(mm_45.stride()) != (HIDDEN, 1):
        raise ValueError(f"mm_45 must be contiguous, got stride={tuple(mm_45.stride())}")
    if tuple(arg182_1.stride()) != (1,):
        raise ValueError(f"arg182_1 must be contiguous, got stride={tuple(arg182_1.stride())}")

    expected_shapes = (
        (SEQ, BATCH, HIDDEN),
        (SEQ, BATCH, HEADS, HEAD_DIM),
        (HEAD_BATCH, SEQ, HEAD_DIM),
        (HEAD_BATCH, 2, WINDOW_SIZE, HEAD_DIM),
        BASE_SHAPE,
    )
    actual_shapes = (
        _shape_tuple(shape0, "_shape_param_0"),
        _shape_tuple(shape1, "_shape_param_1"),
        _shape_tuple(shape2, "_shape_param_2"),
        _shape_tuple(shape3, "_shape_param_3"),
        _shape_tuple(shape4, "_shape_param_4"),
    )
    if actual_shapes != expected_shapes:
        raise ValueError(f"unexpected shape parameters: {actual_shapes}")

    return mm_45, arg182_1


def _check_output_layout(output: torch.Tensor) -> bool:
    return (
        tuple(output.shape) == RETURN_SHAPE
        and output.dtype == torch.float32
        and tuple(output.stride()) == RETURN_STRIDE
        and output.storage_offset() == 0
    )


@oracle_impl(hardware="H100", shapes="(T([8192, 768], f32), T([768], f32), S([1024, 8, 768]), S([1024, 8, 12, 64]), S([96, 1024, 64]), S([96, 2, 512, 64]), S([288, 64, 512]))")
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
    mm_45, arg182_1 = _expect_inputs(inputs)

    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mm_45.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (
        BASE_SHAPE[0],
        triton.cdiv(WINDOW_SIZE, meta["BLOCK_P"]),
        triton.cdiv(HEAD_DIM, meta["BLOCK_D"]),
    )
    _chunk_layout_kernel[grid](
        mm_45,
        arg182_1,
        base,
        TOTAL_CHUNKS=BASE_SHAPE[0],
    )
    return base.permute(0, 2, 1)


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
            output = oracle_forward(inputs)
            if output.is_cuda:
                torch.cuda.synchronize()
        layout_ok = _check_output_layout(output)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(output.shape)} dtype={output.dtype} stride={list(output.stride())})"
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
