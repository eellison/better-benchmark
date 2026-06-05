"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Reformer layout transform returned by Repro.forward, including the [B*12*64,64,64] to [B,12,64,64,64] and [B,12,4096,64] views, the [B,4096,12,64] permute, the contiguous clone, and the final [B*4096,768] view, by emitting one shape-specialized Triton layout-copy kernel that preserves contiguous 64-wide head chunks; whereas Inductor currently lowers the decomposed view/permute/clone/view graph through a generic layout materialization path; Inductor cannot do this today because its scheduler/codegen does not specialize this fixed Reformer head/sequence transpose into a blocked copy over the innermost head dimension; the fix is SCHEDULER_FUSION: add a guarded layout-copy schedule for reshape/permute/clone patterns that writes the final flattened output layout directly."""
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


HEAD_DIM = 64


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
            triton.Config({"BLOCK_Q1": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_Q1": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_Q1": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_Q1": 64}, num_warps=8, num_stages=3),
        ],
        key=["BATCH", "HEADS", "Q0", "Q1"],
    )
    @triton.jit
    def _reformer_layout_kernel(
        input_ptr,
        output_ptr,
        BATCH: tl.constexpr,
        HEADS: tl.constexpr,
        Q0: tl.constexpr,
        Q1: tl.constexpr,
        HEAD_DIM_SIZE: tl.constexpr,
        BLOCK_Q1: tl.constexpr,
    ):
        tile_q1 = tl.program_id(0)
        q0 = tl.program_id(1)
        head = tl.program_id(2) % HEADS
        batch = tl.program_id(2) // HEADS

        q1_offsets = tile_q1 * BLOCK_Q1 + tl.arange(0, BLOCK_Q1)
        d_offsets = tl.arange(0, HEAD_DIM_SIZE)
        q1_mask = q1_offsets < Q1
        mask = q1_mask[:, None] & (d_offsets[None, :] < HEAD_DIM_SIZE)

        input_base = (((batch * HEADS + head) * Q0 + q0) * Q1) * HEAD_DIM_SIZE
        input_offsets = input_base + q1_offsets[:, None] * HEAD_DIM_SIZE + d_offsets[None, :]

        output_row = batch * (Q0 * Q1) + q0 * Q1 + q1_offsets
        output_col = head * HEAD_DIM_SIZE + d_offsets
        output_offsets = output_row[:, None] * (HEADS * HEAD_DIM_SIZE) + output_col[None, :]

        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


def _shape_tuple(value):
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    bmm_11, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(bmm_11, torch.Tensor):
        raise TypeError("expected first input to be a tensor")
    if bmm_11.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if not bmm_11.is_contiguous():
        raise ValueError("oracle expects the captured contiguous input layout")

    shape0 = _shape_tuple(shape0)
    shape1 = _shape_tuple(shape1)
    shape2 = _shape_tuple(shape2)
    shape3 = _shape_tuple(shape3)
    if len(shape0) != 5 or len(shape1) != 4 or len(shape2) != 3 or len(shape3) != 2:
        raise ValueError("unexpected captured view rank")

    batch, heads, q0, q1, head_dim = shape0
    sequence = q0 * q1
    hidden = heads * head_dim
    expected_input = (batch * heads * q0, q1, head_dim)

    if head_dim != HEAD_DIM:
        raise ValueError(f"expected head dimension {HEAD_DIM}, got {head_dim}")
    if shape1 != (batch, heads, sequence, head_dim):
        raise ValueError(f"unexpected second view shape {shape1}")
    if shape2 != (batch, sequence, hidden):
        raise ValueError(f"unexpected third view shape {shape2}")
    if shape3 != (batch * sequence, hidden):
        raise ValueError(f"unexpected output view shape {shape3}")
    if tuple(bmm_11.shape) != expected_input:
        raise ValueError(f"expected input shape {expected_input}, got {tuple(bmm_11.shape)}")

    return bmm_11, batch, heads, q0, q1, shape3


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
    bmm_11, batch, heads, q0, q1, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        (output_shape[1], 1),
        device=bmm_11.device,
        dtype=bmm_11.dtype,
    )
    grid = lambda meta: (triton.cdiv(q1, meta["BLOCK_Q1"]), q0, batch * heads)
    _reformer_layout_kernel[grid](
        bmm_11,
        output,
        BATCH=batch,
        HEADS=heads,
        Q0=q0,
        Q1=q1,
        HEAD_DIM_SIZE=HEAD_DIM,
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
