"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full `view f16[12000,256] -> [8,1500,256]` plus `mean(dim=1)` scope in one fixed-shape Triton reduction that co-reduces adjacent feature columns and stores only the final f16 `[8,256]` output, whereas Inductor lowers the view into a generic middle-dimension mean reduction template with more general indexing and scheduling overhead for this tiny fixed batch; Inductor cannot do this today because its reduction scheduler does not specialize fixed-shape sequence means into a feature-blocked, coalesced middle-dimension reduction tile; the fix is SCHEDULER_FUSION: teach the scheduler to fuse reshape-only producers into a shape-specialized non-contiguous-dim reduction schedule that groups adjacent output columns per program."""
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


BATCH = 8
SEQ = 1500
HIDDEN = 256
BLOCK_SEQ = 2048
BLOCK_HIDDEN = 8

if triton is not None:

    @triton.jit
    def _sequence_mean_kernel(
        x_ptr,
        out_ptr,
        SEQ_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_SEQ_: tl.constexpr,
        BLOCK_HIDDEN_: tl.constexpr,
    ):
        batch = tl.program_id(0)
        hidden_block = tl.program_id(1)
        seq_offsets = tl.arange(0, BLOCK_SEQ_)
        hidden_offsets = hidden_block * BLOCK_HIDDEN_ + tl.arange(0, BLOCK_HIDDEN_)

        offsets = (
            batch * SEQ_ * HIDDEN_
            + seq_offsets[:, None] * HIDDEN_
            + hidden_offsets[None, :]
        )
        mask = (seq_offsets[:, None] < SEQ_) & (hidden_offsets[None, :] < HIDDEN_)
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        means = tl.sum(values, axis=0) * (1.0 / SEQ_)
        tl.store(
            out_ptr + batch * HIDDEN_ + hidden_offsets,
            means,
            mask=hidden_offsets < HIDDEN_,
        )


def _require_input_tensor(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"addmm_20 must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != (BATCH * SEQ, HIDDEN):
        raise ValueError(
            f"addmm_20 has shape {tuple(value.shape)}, expected {(BATCH * SEQ, HIDDEN)}"
        )
    if value.dtype != torch.float16:
        raise TypeError(f"addmm_20 has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise RuntimeError("addmm_20 must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != (HIDDEN, 1):
        raise ValueError(f"addmm_20 has stride {tuple(value.stride())}, expected {(HIDDEN, 1)}")
    return value


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, shape_param = inputs
    x = _require_input_tensor(x)
    if list(shape_param) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")
    return x


def oracle_forward(inputs):
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_sequence_mean.py")

    x = _validate_inputs(inputs)
    output = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=x.device,
        dtype=torch.float16,
    )
    grid = (BATCH, triton.cdiv(HIDDEN, BLOCK_HIDDEN))
    _sequence_mean_kernel[grid](
        x,
        output,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BLOCK_SEQ_=BLOCK_SEQ,
        BLOCK_HIDDEN_=BLOCK_HIDDEN,
        num_warps=8,
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
