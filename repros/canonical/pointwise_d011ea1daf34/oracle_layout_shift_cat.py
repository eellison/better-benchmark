"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the divide, rotated slice/cat window, second cat, and final non-contiguous layout into one Triton storage-order write, whereas Inductor schedules the layout/cat graphlet as generic indexed pointwise work; Inductor cannot do this today because the scheduler does not virtualize this rotated cat/permute indexing pattern into one destination-layout kernel; the fix is SCHEDULER_FUSION: add a layout-indexing fusion path for rotated cat windows that writes the final physical layout directly."""
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


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
        ],
        key=["N_ELEMENTS"],
    )
    @triton.jit
    def _layout_shift_cat_kernel(
        mm_ptr,
        out_ptr,
        N_ELEMENTS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Write the final [B*768, 64, 128] view in physical storage order."""
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N_ELEMENTS

        d = offsets % 64
        k = (offsets // 64) % 128
        n = offsets // 8192

        q = n % 64
        h = (n // 64) % 12
        b = n // 768

        p = k % 64
        shifted_q = tl.where(k < 64, (q + 63) % 64, q)
        seq = shifted_q * 64 + p
        input_offsets = (b * 4096 + seq) * 768 + h * 64 + d

        value = tl.load(mm_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, value * 0.125, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([4096, 768], f16), S([1, 4096, 768]), S([1, 4096, 12, 64]), S([1, 12, 64, 64, 64]), S([1, 12, 64, 64, 128]), S([768, 64, 128]))")
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
        raise RuntimeError("Triton is required for oracle_layout_shift_cat.py")

    (
        mm_15,
        shape_0,
        shape_1,
        shape_2,
        shape_3,
        shape_4,
    ) = inputs

    if not mm_15.is_cuda:
        raise ValueError("oracle_layout_shift_cat.py expects CUDA inputs")
    if mm_15.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"unexpected input dtype: {mm_15.dtype}")
    if mm_15.ndim != 2 or mm_15.shape[1] != 768 or mm_15.shape[0] % 4096 != 0:
        raise ValueError(f"unexpected input shape: {tuple(mm_15.shape)}")
    if tuple(mm_15.stride()) != (768, 1):
        raise ValueError(f"unexpected input stride: {tuple(mm_15.stride())}")

    batch = mm_15.shape[0] // 4096
    expected_shapes = (
        (batch, 4096, 768),
        (batch, 4096, 12, 64),
        (batch, 12, 64, 64, 64),
        (batch, 12, 64, 64, 128),
        (batch * 768, 64, 128),
    )
    actual_shapes = (
        tuple(shape_0),
        tuple(shape_1),
        tuple(shape_2),
        tuple(shape_3),
        tuple(shape_4),
    )
    if actual_shapes != expected_shapes:
        raise ValueError(f"unexpected shape parameters: {actual_shapes}")

    output = torch.empty_strided(
        actual_shapes[-1],
        (8192, 1, 64),
        device=mm_15.device,
        dtype=mm_15.dtype,
    )
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_N"]),)
    _layout_shift_cat_kernel[grid](
        mm_15,
        output,
        N_ELEMENTS=n_elements,
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
