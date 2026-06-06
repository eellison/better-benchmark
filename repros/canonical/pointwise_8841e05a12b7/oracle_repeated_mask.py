"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete repeated Qwen causal segment-mask scope by evaluating the shared `j <= i and cumsum[b, i] == cumsum[b, j]` predicate once per element and writing all 28 independent bf16 outputs from one Triton program, whereas the measured Inductor lowering is already at the same CUDAGraph-captured floor for this full scope; Inductor cannot materially improve this case today because the required work is dominated by coalesced stores of 28 separate 2 MiB outputs rather than removable arithmetic or launch overhead; the fix is BANDWIDTH_BOUND: keep this repro closed as at-floor unless a future representation can legally avoid returning independent materialized mask tensors."""
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


OUT_SHAPE = (4, 1, 512, 512)
OUT_STRIDE = (262144, 262144, 512, 1)
N_OUTPUTS = 28
N_ELEMENTS = 4 * 512 * 512
BLOCK_SIZE = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _repeated_mask_kernel(
        cumsum_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out3_ptr,
        out4_ptr,
        out5_ptr,
        out6_ptr,
        out7_ptr,
        out8_ptr,
        out9_ptr,
        out10_ptr,
        out11_ptr,
        out12_ptr,
        out13_ptr,
        out14_ptr,
        out15_ptr,
        out16_ptr,
        out17_ptr,
        out18_ptr,
        out19_ptr,
        out20_ptr,
        out21_ptr,
        out22_ptr,
        out23_ptr,
        out24_ptr,
        out25_ptr,
        out26_ptr,
        out27_ptr,
        n_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements

        col = offsets % 512
        row = (offsets // 512) % 512
        batch = offsets // 262144
        cumsum_base = batch * 512

        row_value = tl.load(cumsum_ptr + cumsum_base + row, mask=mask, other=0)
        col_value = tl.load(cumsum_ptr + cumsum_base + col, mask=mask, other=1)
        keep = (col <= row) & (row_value == col_value)
        value = tl.where(keep, 0.0, -float("inf"))

        tl.store(out0_ptr + offsets, value, mask=mask)
        tl.store(out1_ptr + offsets, value, mask=mask)
        tl.store(out2_ptr + offsets, value, mask=mask)
        tl.store(out3_ptr + offsets, value, mask=mask)
        tl.store(out4_ptr + offsets, value, mask=mask)
        tl.store(out5_ptr + offsets, value, mask=mask)
        tl.store(out6_ptr + offsets, value, mask=mask)
        tl.store(out7_ptr + offsets, value, mask=mask)
        tl.store(out8_ptr + offsets, value, mask=mask)
        tl.store(out9_ptr + offsets, value, mask=mask)
        tl.store(out10_ptr + offsets, value, mask=mask)
        tl.store(out11_ptr + offsets, value, mask=mask)
        tl.store(out12_ptr + offsets, value, mask=mask)
        tl.store(out13_ptr + offsets, value, mask=mask)
        tl.store(out14_ptr + offsets, value, mask=mask)
        tl.store(out15_ptr + offsets, value, mask=mask)
        tl.store(out16_ptr + offsets, value, mask=mask)
        tl.store(out17_ptr + offsets, value, mask=mask)
        tl.store(out18_ptr + offsets, value, mask=mask)
        tl.store(out19_ptr + offsets, value, mask=mask)
        tl.store(out20_ptr + offsets, value, mask=mask)
        tl.store(out21_ptr + offsets, value, mask=mask)
        tl.store(out22_ptr + offsets, value, mask=mask)
        tl.store(out23_ptr + offsets, value, mask=mask)
        tl.store(out24_ptr + offsets, value, mask=mask)
        tl.store(out25_ptr + offsets, value, mask=mask)
        tl.store(out26_ptr + offsets, value, mask=mask)
        tl.store(out27_ptr + offsets, value, mask=mask)


def _new_output(device):
    return torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=device,
        dtype=torch.bfloat16,
    )


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
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    cumsum, shape_param = inputs
    if tuple(cumsum.shape) != (4, 512) or cumsum.dtype != torch.int64:
        raise ValueError("oracle_repeated_mask only covers make_inputs() shape [4, 512] int64")
    if list(shape_param) != [4, -1, 512, 512]:
        raise ValueError("oracle_repeated_mask only covers make_inputs() expand shape")

    outputs = tuple(_new_output(cumsum.device) for _ in range(N_OUTPUTS))
    grid = (triton.cdiv(N_ELEMENTS, BLOCK_SIZE),)
    _repeated_mask_kernel[grid](
        cumsum,
        *outputs,
        n_elements=N_ELEMENTS,
        block_size=BLOCK_SIZE,
    )
    return outputs


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
