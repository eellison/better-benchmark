"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full timm ViT permute -> contiguous clone -> unsafe_view -> final view scope by directly materializing the required contiguous float32[6304,384] output from the captured float32[32,6,197,64] input in one head-vectorized Triton layout-copy kernel, whereas Inductor already lowers this layout-only chain to a single materialization kernel with comparable CUDAGraph timing; Inductor cannot remove the dominant work today because the returned tensor is the newly materialized contiguous clone storage, so the full-scope contract requires the same read and write traffic; the fix is BANDWIDTH_BOUND: no algorithmic scheduler change is indicated for this isolated repro beyond minor layout-copy index specialization."""
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


BATCH = 32
HEADS = 6
SEQ = 197
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM
TOTAL_ROWS = BATCH * SEQ
OUT_SHAPE = (TOTAL_ROWS, HIDDEN)
OUT_STRIDE = (HIDDEN, 1)
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _permute_clone_view_kernel(
        input_ptr,
        output_ptr,
        TOTAL_ROWS_: tl.constexpr,
        SEQ_: tl.constexpr,
        HEADS_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        head = tl.program_id(1)
        dim = tl.arange(0, HEAD_DIM_)
        mask = rows[:, None] < TOTAL_ROWS_

        batch = rows // SEQ_
        seq_idx = rows - batch * SEQ_
        input_base = (
            batch[:, None] * (HEADS_ * SEQ_ * HEAD_DIM_)
            + head * SEQ_ * HEAD_DIM_
            + seq_idx[:, None] * HEAD_DIM_
            + dim[None, :]
        )
        output_base = rows[:, None] * HIDDEN_ + head * HEAD_DIM_ + dim[None, :]

        values = tl.load(input_ptr + input_base, mask=mask, other=0.0)
        tl.store(output_ptr + output_base, values, mask=mask)


def _shape_tuple(value, name):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected arg84_1 to be a tensor, got {type(x)!r}")
    if tuple(x.shape) != (BATCH, HEADS, SEQ, HEAD_DIM):
        raise ValueError(f"expected input shape {(BATCH, HEADS, SEQ, HEAD_DIM)}, got {tuple(x.shape)}")
    if tuple(x.stride()) != (HEADS * SEQ * HEAD_DIM, SEQ * HEAD_DIM, HEAD_DIM, 1):
        raise ValueError(f"unexpected input stride {tuple(x.stride())}")
    if x.storage_offset() != 0:
        raise ValueError(f"unexpected input storage_offset {x.storage_offset()}")
    if x.dtype != torch.float32:
        raise TypeError(f"expected torch.float32 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError("Triton oracle requires CUDA inputs")

    out_shape = _shape_tuple(shape_param, "_shape_param_0")
    if out_shape != OUT_SHAPE:
        raise ValueError(f"unexpected _shape_param_0 {out_shape}, expected {OUT_SHAPE}")

    return x, out_shape


@oracle_impl(hardware="H100", shapes="(T([32, 6, 197, 64], f32), S([6304, 384]))")
def oracle_forward(inputs):
    """Run the full permute -> contiguous clone -> unsafe_view -> view scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, out_shape = _validate_inputs(tuple(inputs))
    output = torch.empty_strided(
        out_shape,
        OUT_STRIDE,
        device=x.device,
        dtype=x.dtype,
    )
    grid = lambda META: (triton.cdiv(TOTAL_ROWS, META["BLOCK_ROWS"]), HEADS)
    _permute_clone_view_kernel[grid](
        x,
        output,
        TOTAL_ROWS_=TOTAL_ROWS,
        SEQ_=SEQ,
        HEADS_=HEADS,
        HEAD_DIM_=HEAD_DIM,
        HIDDEN_=HIDDEN,
        BLOCK_ROWS=64,
        num_warps=4,
        num_stages=3,
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
