"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete Longformer bias add, scale by 1/8, head/batch layout rewrite, overlapping three-window as_strided stencil clone, and final contiguous `[288,512,64]` return into one Triton materialization kernel, whereas Inductor currently handles the cheap pointwise producer and the overlapping clone/layout materialization as separate generic scheduling work with extra memory traffic; Inductor cannot do this today because fixed-overlap as_strided clone materialization is treated as a scheduler fusion barrier even when the producer is affine pointwise arithmetic and the consumer is only a view; the fix is SCHEDULER_FUSION: teach layout clone codegen to sink affine pointwise producers through fixed-overlap as_strided window indexing and write the final contiguous backing storage directly."""
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
OUTPUT_SHAPE = (HEAD_BATCH * WINDOWS, WINDOW_SIZE, HEAD_DIM)
OUTPUT_STRIDE = (WINDOW_SIZE * HEAD_DIM, HEAD_DIM, 1)


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
            triton.Config({"BLOCK_P": 16, "BLOCK_D": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_P": 32, "BLOCK_D": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_P": 32, "BLOCK_D": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_P": 64, "BLOCK_D": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_P": 64, "BLOCK_D": 64}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL_CHUNKS"],
    )
    @triton.jit
    def _scaled_layout_stencil_kernel(
        mm_ptr,
        bias_ptr,
        out_ptr,
        TOTAL_CHUNKS: tl.constexpr,
        WINDOWS_C: tl.constexpr,
        WINDOW_STEP_C: tl.constexpr,
        WINDOW_SIZE_C: tl.constexpr,
        BATCH_C: tl.constexpr,
        HIDDEN_C: tl.constexpr,
        HEADS_C: tl.constexpr,
        HEAD_DIM_C: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        pos = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
        dim = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)

        head_batch = chunk // WINDOWS_C
        window = chunk - head_batch * WINDOWS_C
        batch = head_batch // HEADS_C
        head = head_batch - batch * HEADS_C

        source_seq = window * WINDOW_STEP_C + pos[:, None]
        source_feature = head * HEAD_DIM_C + dim[None, :]

        load_offsets = (source_seq * BATCH_C + batch) * HIDDEN_C + source_feature
        store_offsets = chunk * WINDOW_SIZE_C * HEAD_DIM_C + pos[:, None] * HEAD_DIM_C + dim[None, :]
        mask = (
            (chunk < TOTAL_CHUNKS)
            & (pos[:, None] < WINDOW_SIZE_C)
            & (dim[None, :] < HEAD_DIM_C)
        )

        values = tl.load(mm_ptr + load_offsets, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + source_feature, mask=dim[None, :] < HEAD_DIM_C, other=0.0)
        tl.store(out_ptr + store_offsets, (values + bias) * 0.125, mask=mask)


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

    mm_44, arg180_1, shape0, shape1, shape2, shape3, shape4 = inputs
    if not isinstance(mm_44, torch.Tensor) or not isinstance(arg180_1, torch.Tensor):
        raise TypeError("expected first two inputs to be tensors")
    if not mm_44.is_cuda or not arg180_1.is_cuda:
        raise RuntimeError("This oracle is a CUDA/Triton oracle and requires CUDA inputs")
    if mm_44.device != arg180_1.device:
        raise ValueError("input tensors must be on the same device")
    if mm_44.dtype != torch.float32 or arg180_1.dtype != torch.float32:
        raise ValueError(f"expected f32 inputs, got {mm_44.dtype=} {arg180_1.dtype=}")
    if tuple(mm_44.shape) != (SEQ * BATCH, HIDDEN):
        raise ValueError(f"unexpected mm_44 shape: {tuple(mm_44.shape)}")
    if tuple(arg180_1.shape) != (HIDDEN,):
        raise ValueError(f"unexpected arg180_1 shape: {tuple(arg180_1.shape)}")
    if tuple(mm_44.stride()) != (HIDDEN, 1):
        raise ValueError(f"mm_44 must be contiguous, got stride={tuple(mm_44.stride())}")
    if tuple(arg180_1.stride()) != (1,):
        raise ValueError(f"arg180_1 must be contiguous, got stride={tuple(arg180_1.stride())}")

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

    return mm_44, arg180_1


def oracle_forward(inputs):
    """Run the full Repro.forward scope and return the exact contiguous layout."""
    mm_44, arg180_1 = _expect_inputs(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=mm_44.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (
        OUTPUT_SHAPE[0],
        triton.cdiv(WINDOW_SIZE, meta["BLOCK_P"]),
        triton.cdiv(HEAD_DIM, meta["BLOCK_D"]),
    )
    _scaled_layout_stencil_kernel[grid](
        mm_44,
        arg180_1,
        output,
        TOTAL_CHUNKS=OUTPUT_SHAPE[0],
        WINDOWS_C=WINDOWS,
        WINDOW_STEP_C=WINDOW_STEP,
        WINDOW_SIZE_C=WINDOW_SIZE,
        BATCH_C=BATCH,
        HIDDEN_C=HIDDEN,
        HEADS_C=HEADS,
        HEAD_DIM_C=HEAD_DIM,
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
