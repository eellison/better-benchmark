"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete view -> ReLU -> view plus ReLU<=0 mask scope in one contiguous Triton load/max/two-store pass over all 33,554,432 elements, whereas tuned Inductor already lowers the isolated metadata-view and pointwise fanout graph into the same required input-read, f32-output-write, bool-mask-write envelope; Inductor cannot materially improve this repro through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, recompute fusion, or a new pattern because there is no avoidable producer, reduction, scatter, or intermediate materialization beyond the mandated output tensors; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise bandwidth case unless broader pointwise launch or memory-codegen work moves both implementations."""
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


MASK_VIEW_SHAPE = (64, 128, 4096)
OUTPUT_SHAPE = (8192, 4096)

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _relu_mask_kernel(
        input_ptr,
        relu_out_ptr,
        mask_out_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < N
        x = tl.load(input_ptr + offsets, mask=valid, other=0.0)
        relu = tl.where(x != x, x, tl.maximum(x, 0.0))
        tl.store(relu_out_ptr + offsets, relu, mask=valid)
        tl.store(mask_out_ptr + offsets, relu <= 0.0, mask=valid)


def _shape_tuple(value) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_relu_mask.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    addmm_4, view_shape, output_shape = inputs
    if not isinstance(addmm_4, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor")
    if not addmm_4.is_cuda:
        raise RuntimeError(f"{REPRO_ID} expects CUDA tensor inputs")
    if addmm_4.dtype is not torch.float32:
        raise TypeError(f"{REPRO_ID} expects a float32 input, got {addmm_4.dtype}")
    if not addmm_4.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects the captured contiguous input layout")

    view_shape = _shape_tuple(view_shape)
    output_shape = _shape_tuple(output_shape)
    n_elements = addmm_4.numel()
    if view_shape != MASK_VIEW_SHAPE:
        raise ValueError(f"expected _shape_param_0 {MASK_VIEW_SHAPE}, got {view_shape}")
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"expected _shape_param_1 {OUTPUT_SHAPE}, got {output_shape}")
    if _numel(view_shape) != n_elements or _numel(output_shape) != n_elements:
        raise ValueError("shape parameters must have the same element count as input 0")
    return addmm_4, view_shape, output_shape, n_elements


def oracle_forward(inputs):
    """Run the full-scope Repro.forward computation."""
    addmm_4, mask_shape, output_shape, n_elements = _validate_inputs(inputs)
    relu_out = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=addmm_4.device,
        dtype=addmm_4.dtype,
    )
    mask_out = torch.empty_strided(
        mask_shape,
        _contiguous_stride(mask_shape),
        device=addmm_4.device,
        dtype=torch.bool,
    )

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _relu_mask_kernel[grid](addmm_4, relu_out, mask_out, N=n_elements)
    return relu_out, mask_out


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
