"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileBERT pointwise affine chain in one storage-linear Triton kernel, including both metadata-only `[32768,128] -> [256,128,128]` input views, the two hidden-dimension broadcasts, the add between branches, and the final contiguous `[32768,128]` view of a fresh `[256,128,128]` buffer, whereas tuned Inductor already lowers this isolated pointwise graph to the same single-kernel memory-traffic envelope; Inductor cannot materially improve it today with scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the remaining work is dominated by the mandatory two activation reads, four cached vector-parameter reads, and output store rather than avoidable intermediate materialization; the fix is BANDWIDTH_BOUND: record this as at floor unless broader pointwise bandwidth or launch-overhead work moves both implementations."""
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
ROWS = 32768
HIDDEN = 128
VIEW_SHAPE = (256, 128, 128)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
N_ELEMENTS = ROWS * HIDDEN

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
    def _affine_chain_kernel(
        addmm_346_ptr,
        scale0_ptr,
        bias0_ptr,
        addmm_351_ptr,
        scale1_ptr,
        bias1_ptr,
        out_ptr,
        N: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        hidden = offsets % HIDDEN_SIZE

        x0 = tl.load(addmm_346_ptr + offsets, mask=mask, other=0.0)
        s0 = tl.load(scale0_ptr + hidden, mask=mask, other=0.0)
        b0 = tl.load(bias0_ptr + hidden, mask=mask, other=0.0)
        x1 = tl.load(addmm_351_ptr + offsets, mask=mask, other=0.0)
        s1 = tl.load(scale1_ptr + hidden, mask=mask, other=0.0)
        b1 = tl.load(bias1_ptr + hidden, mask=mask, other=0.0)

        first = x0 * s0 + b0
        second = x1 + first
        out = second * s1 + b1
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_shape(name: str, value, expected: tuple[int, ...]) -> None:
    shape = tuple(int(dim) for dim in value)
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")


def _require_tensor(name: str, value, expected: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride {tuple(value.stride())}")
    return value


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_affine_chain.py")
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects 9 inputs, got {len(inputs)}")

    addmm_346 = _require_tensor("addmm_346", inputs[0], OUTPUT_SHAPE)
    scale0 = _require_tensor("arg1069_1", inputs[1], (HIDDEN,))
    bias0 = _require_tensor("arg1070_1", inputs[2], (HIDDEN,))
    addmm_351 = _require_tensor("addmm_351", inputs[3], OUTPUT_SHAPE)
    scale1 = _require_tensor("arg1083_1", inputs[4], (HIDDEN,))
    bias1 = _require_tensor("arg1084_1", inputs[5], (HIDDEN,))
    _require_shape("_shape_param_0", inputs[6], VIEW_SHAPE)
    _require_shape("_shape_param_1", inputs[7], VIEW_SHAPE)
    _require_shape("_shape_param_2", inputs[8], OUTPUT_SHAPE)
    return addmm_346, scale0, bias0, addmm_351, scale1, bias1


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
    addmm_346, scale0, bias0, addmm_351, scale1, bias1 = _validate_inputs(inputs)
    output_base = torch.empty_strided(
        VIEW_SHAPE,
        (HIDDEN * HIDDEN, HIDDEN, 1),
        device=addmm_346.device,
        dtype=addmm_346.dtype,
    )
    output = output_base.view(OUTPUT_SHAPE)
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK_N"]),)
    _affine_chain_kernel[grid](
        addmm_346,
        scale0,
        bias0,
        addmm_351,
        scale1,
        bias1,
        output,
        N=N_ELEMENTS,
        HIDDEN_SIZE=HIDDEN,
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
