"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete Longformer bias add, head/batch layout split, overlapping 512-token as_strided window materialization, transpose, and final contiguous clone into one tiled Triton materialization kernel, whereas Inductor currently schedules the pointwise bias add and overlapping clone/layout materialization through more generic layout-indexing code; Inductor cannot do this today because the scheduler treats the as_strided/permute/clone layout consumer as a materialization barrier instead of sinking the bias-add producer into the affine overlapping window store; the fix is SCHEDULER_FUSION: teach layout clone codegen to inline affine pointwise producers through overlapping as_strided window indexing and write the final contiguous chunk layout directly."""
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
OUTPUT_SHAPE = (HEAD_BATCH * WINDOWS, HEAD_DIM, WINDOW_SIZE)
OUTPUT_STRIDE = (HEAD_DIM * WINDOW_SIZE, WINDOW_SIZE, 1)


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
            triton.Config({"BLOCK_T": 16, "BLOCK_D": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_T": 32, "BLOCK_D": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_T": 32, "BLOCK_D": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_T": 64, "BLOCK_D": 16}, num_warps=4, num_stages=3),
        ],
        key=["TOTAL"],
    )
    @triton.jit
    def _longformer_layout_stencil_kernel(
        mm_ptr,
        bias_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        WINDOWS_C: tl.constexpr,
        WINDOW_STEP_C: tl.constexpr,
        WINDOW_SIZE_C: tl.constexpr,
        HEADS_C: tl.constexpr,
        HEAD_DIM_C: tl.constexpr,
        BATCH_HIDDEN_C: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        dim_offsets = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        out_chunk = tl.program_id(2)

        head_batch = out_chunk // WINDOWS_C
        window = out_chunk - head_batch * WINDOWS_C
        head = head_batch % HEADS_C

        source_token = window * WINDOW_STEP_C + token_offsets
        source_feature = head * HEAD_DIM_C + dim_offsets
        source_offsets = (
            source_token[:, None] * BATCH_HIDDEN_C
            + head_batch * HEAD_DIM_C
            + dim_offsets[None, :]
        )
        load_mask = (token_offsets[:, None] < WINDOW_SIZE_C) & (dim_offsets[None, :] < HEAD_DIM_C)

        values = tl.load(mm_ptr + source_offsets, mask=load_mask, other=0.0)
        bias = tl.load(
            bias_ptr + source_feature,
            mask=dim_offsets < HEAD_DIM_C,
            other=0.0,
        )
        values = values + bias[None, :]

        store_offsets = (
            out_chunk * HEAD_DIM_C * WINDOW_SIZE_C
            + dim_offsets[:, None] * WINDOW_SIZE_C
            + token_offsets[None, :]
        )
        store_mask = (dim_offsets[:, None] < HEAD_DIM_C) & (token_offsets[None, :] < WINDOW_SIZE_C)
        tl.store(out_ptr + store_offsets, tl.trans(values), mask=store_mask)


def _shape_tuple(value, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _expect_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    mm_45, arg182_1, shape0, shape1, shape2, shape3, shape4 = inputs
    if not isinstance(mm_45, torch.Tensor) or not isinstance(arg182_1, torch.Tensor):
        raise TypeError("expected first two inputs to be tensors")
    if not mm_45.is_cuda or not arg182_1.is_cuda:
        raise RuntimeError("This oracle is a CUDA/Triton oracle and requires CUDA inputs")
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
        OUTPUT_SHAPE,
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


def oracle_forward(inputs):
    """Run the full Repro.forward scope and return the exact eager output layout.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single f32 tensor with shape [288, 64, 512] and contiguous stride
    [32768, 512, 1].
    """
    mm_45, arg182_1 = _expect_inputs(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=mm_45.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (
        triton.cdiv(WINDOW_SIZE, meta["BLOCK_T"]),
        triton.cdiv(HEAD_DIM, meta["BLOCK_D"]),
        HEAD_BATCH * WINDOWS,
    )
    _longformer_layout_stencil_kernel[grid](
        mm_45,
        arg182_1,
        output,
        TOTAL=output.numel(),
        WINDOWS_C=WINDOWS,
        WINDOW_STEP_C=WINDOW_STEP,
        WINDOW_SIZE_C=WINDOW_SIZE,
        HEADS_C=HEADS,
        HEAD_DIM_C=HEAD_DIM,
        BATCH_HIDDEN_C=BATCH * HIDDEN,
    )
    return output


def _check_layout(output: torch.Tensor) -> bool:
    return (
        tuple(output.shape) == OUTPUT_SHAPE
        and tuple(output.stride()) == OUTPUT_STRIDE
        and output.dtype is torch.float32
        and output.storage_offset() == 0
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
        with torch.no_grad():
            layout_output = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout(layout_output)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_output.shape)} stride={list(layout_output.stride())})"
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
