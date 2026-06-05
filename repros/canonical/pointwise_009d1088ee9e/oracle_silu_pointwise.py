"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileViT MLP activation scope, including the `[8192,480] -> [512,16,480]` metadata view, the f32 SiLU value `x / (exp(-x) + 1)`, and the final contiguous `[8192,480]` output layout, in one storage-linear Triton pointwise kernel using the canonical `x * sigmoid(x)` form, but the required CUDAGraph benchmark is AT_FLOOR at 1.005x versus tuned Inductor; Inductor already lowers this isolated activation and metadata-only reshape to the same one-launch pointwise math and memory-traffic envelope, so the oracle does not expose a material scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, recompute-fusion, or new-pattern gap; the fix is BANDWIDTH_BOUND: record this row as at floor unless broader pointwise math-codegen or launch-overhead work moves both implementations."""
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


# --- Oracle kernel(s) ---
ROWS = 8192
COLS = 480
INPUT_SHAPE = (ROWS, COLS)
MID_VIEW_SHAPE = (512, 16, COLS)
OUTPUT_SHAPE = INPUT_SHAPE
OUTPUT_STRIDE = (COLS, 1)
N_ELEMENTS = ROWS * COLS


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=4),
        ],
        key=["N"],
    )
    @triton.jit
    def _silu_pointwise_kernel(
        input_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = x * tl.sigmoid(x)
        tl.store(output_ptr + offsets, out, mask=mask)


def _require_shape(name: str, value, expected: tuple[int, ...]) -> None:
    shape = tuple(int(dim) for dim in value)
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_silu_pointwise.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    addmm_34, shape0, shape1 = inputs
    if not isinstance(addmm_34, torch.Tensor):
        raise TypeError(f"addmm_34 must be a tensor, got {type(addmm_34)!r}")
    if addmm_34.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if addmm_34.dtype != torch.float32:
        raise TypeError(f"addmm_34 must be f32, got {addmm_34.dtype}")
    if tuple(addmm_34.shape) != INPUT_SHAPE:
        raise ValueError(f"addmm_34 has shape {tuple(addmm_34.shape)}, expected {INPUT_SHAPE}")
    if tuple(addmm_34.stride()) != OUTPUT_STRIDE:
        raise ValueError(f"addmm_34 has stride {tuple(addmm_34.stride())}, expected {OUTPUT_STRIDE}")

    _require_shape("_shape_param_0", shape0, MID_VIEW_SHAPE)
    _require_shape("_shape_param_1", shape1, OUTPUT_SHAPE)
    return addmm_34


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
    addmm_34 = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm_34.device,
        dtype=addmm_34.dtype,
    )
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK_N"]),)
    _silu_pointwise_kernel[grid](addmm_34, output, N=N_ELEMENTS)
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
