"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full sliced complex RoPE factor broadcast and both complex64 multiplies in one Triton kernel, whereas Inductor currently handles the sliced/viewed complex factor and two sibling complex mul outputs through its generic pointwise scheduling path; Inductor cannot do this today because the scheduler/codegen does not reliably recognize this shared broadcasted complex factor as a single reusable full-scope multi-output pointwise group with explicit real/imag arithmetic; the fix is SCHEDULER_FUSION: fuse sibling complex pointwise consumers of the same sliced broadcast factor into one generated kernel and CSE the factor load/indexing across outputs."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


INPUT_FACTOR_SHAPE = (2048, 32)
INPUT_DATA_SHAPE = (32, 32, 8, 32)
SHAPE_PARAM = [1, 32, 1, 32]
COMPLEX_NUMEL = 32 * 32 * 8 * 32


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _complex_rope_slice_mul_kernel(
        factor_ptr,
        x0_ptr,
        x1_ptr,
        out0_ptr,
        out1_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        d3 = offsets & 31
        d1 = (offsets >> 8) & 31
        factor_complex_offsets = (d1 + 1) * 32 + d3
        factor_real_offsets = factor_complex_offsets * 2

        data_real_offsets = offsets * 2
        factor_r = tl.load(factor_ptr + factor_real_offsets, mask=mask, other=0.0).to(tl.float32)
        factor_i = tl.load(factor_ptr + factor_real_offsets + 1, mask=mask, other=0.0).to(tl.float32)

        x0_r = tl.load(x0_ptr + data_real_offsets, mask=mask, other=0.0).to(tl.float32)
        x0_i = tl.load(x0_ptr + data_real_offsets + 1, mask=mask, other=0.0).to(tl.float32)
        y0_r = x0_r * factor_r - x0_i * factor_i
        y0_i = x0_r * factor_i + x0_i * factor_r
        tl.store(out0_ptr + data_real_offsets, y0_r, mask=mask)
        tl.store(out0_ptr + data_real_offsets + 1, y0_i, mask=mask)

        x1_r = tl.load(x1_ptr + data_real_offsets, mask=mask, other=0.0).to(tl.float32)
        x1_i = tl.load(x1_ptr + data_real_offsets + 1, mask=mask, other=0.0).to(tl.float32)
        y1_r = x1_r * factor_r - x1_i * factor_i
        y1_i = x1_r * factor_i + x1_i * factor_r
        tl.store(out1_ptr + data_real_offsets, y1_r, mask=mask)
        tl.store(out1_ptr + data_real_offsets + 1, y1_i, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([2048, 32], torch.complex64), T([32, 32, 8, 32], torch.complex64), T([32, 32, 8, 32], torch.complex64), S([1, 32, 1, 32]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope for the sliced complex RoPE multiply."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_complex_rope_slice_mul.py")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    arg2_1, view_as_complex, view_as_complex_1, shape_param = inputs
    for name, tensor, expected_shape in (
        ("arg2_1", arg2_1, INPUT_FACTOR_SHAPE),
        ("view_as_complex", view_as_complex, INPUT_DATA_SHAPE),
        ("view_as_complex_1", view_as_complex_1, INPUT_DATA_SHAPE),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{REPRO_ID} {name} must be a tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{REPRO_ID} {name} expects shape {expected_shape}, got {tuple(tensor.shape)}")
        if tensor.dtype is not torch.complex64:
            raise ValueError(f"{REPRO_ID} {name} expects torch.complex64, got {tensor.dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"{REPRO_ID} {name} expects a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{REPRO_ID} {name} expects contiguous input, got stride={tensor.stride()}")
    if list(shape_param) != SHAPE_PARAM:
        raise ValueError(f"{REPRO_ID} expects shape parameter {SHAPE_PARAM}, got {shape_param}")

    out0 = torch.empty_like(view_as_complex)
    out1 = torch.empty_like(view_as_complex_1)

    block_n = 512
    grid = (triton.cdiv(COMPLEX_NUMEL, block_n),)
    _complex_rope_slice_mul_kernel[grid](
        torch.view_as_real(arg2_1),
        torch.view_as_real(view_as_complex),
        torch.view_as_real(view_as_complex_1),
        torch.view_as_real(out0),
        torch.view_as_real(out1),
        N=COMPLEX_NUMEL,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=3,
    )
    return (out0, out1)


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
