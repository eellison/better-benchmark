"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Whisper view -> scale -> head-split view -> permute -> contiguous clone scope by directly materializing the returned contiguous `[B, H, T, 64]` tensor from the captured `[B*T, H*64]` input in one head-vectorized Triton kernel, whereas Inductor already lowers the same scale plus attention-head layout materialization into an equivalent single compiled layout-copy/pointwise kernel within timing noise; Inductor cannot materially improve this isolated repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute because the returned clone requires the input read, scaled reordered write, allocation, and one graph-captured launch; the fix is BANDWIDTH_BOUND: record this as an at-floor scaled head-layout materialization unless broader launch overhead or generic layout-copy codegen improvements move the baseline."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


HEAD_DIM = 64
SCALE = 0.125
BLOCK_ROWS = 64
NUM_WARPS = 8
NUM_STAGES = 3


def _shape_tuple(value, name):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _scaled_head_layout_kernel(
        input_ptr,
        output_ptr,
        TOTAL_ROWS: tl.constexpr,
        TOKENS: tl.constexpr,
        HEADS: tl.constexpr,
        HIDDEN: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        dim = tl.arange(0, HEAD_DIM_)
        row_mask = rows < TOTAL_ROWS

        rows_per_batch = HEADS * TOKENS
        batch = rows // rows_per_batch
        batch_rem = rows - batch * rows_per_batch
        head = batch_rem // TOKENS
        token = batch_rem - head * TOKENS

        input_offsets = (
            batch[:, None] * (TOKENS * HIDDEN)
            + token[:, None] * HIDDEN
            + head[:, None] * HEAD_DIM_
            + dim[None, :]
        )
        output_offsets = rows[:, None] * HEAD_DIM_ + dim[None, :]

        values = tl.load(input_ptr + input_offsets, mask=row_mask[:, None], other=0.0)
        tl.store(output_ptr + output_offsets, values * SCALE_, mask=row_mask[:, None])


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    addmm_15, shape0, shape1 = inputs
    if not isinstance(addmm_15, torch.Tensor):
        raise TypeError(f"expected addmm_15 tensor, got {type(addmm_15)!r}")
    if addmm_15.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if addmm_15.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"expected f16 or f32 input, got {addmm_15.dtype}")
    if addmm_15.dim() != 2:
        raise ValueError(f"expected rank-2 input, got shape={tuple(addmm_15.shape)}")
    if not addmm_15.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(addmm_15.stride())}")

    view_shape = _shape_tuple(shape0, "_shape_param_0")
    split_shape = _shape_tuple(shape1, "_shape_param_1")
    if len(view_shape) != 3:
        raise ValueError(f"_shape_param_0 must be rank 3, got {view_shape}")
    if len(split_shape) != 4:
        raise ValueError(f"_shape_param_1 must be rank 4, got {split_shape}")

    batch, tokens, hidden = view_shape
    if hidden % HEAD_DIM != 0:
        raise ValueError(f"hidden size {hidden} is not divisible by {HEAD_DIM}")
    heads = hidden // HEAD_DIM

    expected_input_shape = (batch * tokens, hidden)
    if tuple(addmm_15.shape) != expected_input_shape:
        raise ValueError(f"input shape {tuple(addmm_15.shape)} != expected {expected_input_shape}")
    if split_shape[0] != batch or split_shape[1] != tokens or split_shape[3] != HEAD_DIM:
        raise ValueError(f"_shape_param_1 {split_shape} is inconsistent with {view_shape}")
    if split_shape[2] not in (-1, heads):
        raise ValueError(f"_shape_param_1 head dim {split_shape[2]} != inferred heads {heads}")

    output_shape = (batch, heads, tokens, HEAD_DIM)
    output_stride = (heads * tokens * HEAD_DIM, tokens * HEAD_DIM, HEAD_DIM, 1)
    total_rows = batch * heads * tokens
    return addmm_15, output_shape, output_stride, total_rows, tokens, heads, hidden


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
    addmm_15, output_shape, output_stride, total_rows, tokens, heads, hidden = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        output_stride,
        device=addmm_15.device,
        dtype=addmm_15.dtype,
    )
    grid = (triton.cdiv(total_rows, BLOCK_ROWS),)
    _scaled_head_layout_kernel[grid](
        addmm_15,
        output,
        TOTAL_ROWS=total_rows,
        TOKENS=tokens,
        HEADS=heads,
        HIDDEN=hidden,
        HEAD_DIM_=HEAD_DIM,
        SCALE_=SCALE,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
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
