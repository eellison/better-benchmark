"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete add-plus-affine LayerNorm scope in one shape-specialized Triton row kernel, including the `[4096,4096]` to `[8,512,4096]` view, activation add, correction=0 var_mean over hidden size 4096, `rsqrt(var + 1e-12)`, affine epilogue, final `[4096,4096]` view, and sibling `rsqrt / 4096` output, whereas Inductor currently emits one fused generic Welford reduction kernel for the same scope; Inductor cannot do this today because its correction=0 var_mean lowering keeps the general Welford state machine instead of selecting a fixed-width LayerNorm algebra that reuses the resident row tile for the affine epilogue and saved inverse-scale store; the fix is ALGEBRAIC_ELIMINATION: add a guarded correction=0 LayerNorm lowering that replaces generic Welford bookkeeping with direct mean and centered-variance reductions for static hidden dimensions while preserving the saved side output."""
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


ROWS = 4096
HIDDEN = 4096
INPUT_VIEW_SHAPE = (8, 512, 4096)
OUTPUT_SHAPE = (4096, 4096)
INVSTD_SHAPE = (8, 512, 1)
EPS = 1.0e-12
BLOCK_H = 4096


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs):
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_70, add_103, arg18_1, arg19_1, shape0, shape1 = inputs
    tensor_inputs = (addmm_70, add_103, arg18_1, arg19_1)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (OUTPUT_SHAPE, INPUT_VIEW_SHAPE, (HIDDEN,), (HIDDEN,))
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    if shape0_tuple != INPUT_VIEW_SHAPE:
        raise ValueError(f"unexpected input view shape parameter: {shape0!r}")
    if shape1_tuple != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {shape1!r}")

    return addmm_70, add_103, arg18_1, arg19_1, shape1_tuple


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _add_layernorm_saved_scale_kernel(
        addmm_ptr,
        add_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets).to(tl.float32)
        addend = tl.load(add_ptr + offsets).to(tl.float32)
        x = addend + addmm

        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(centered * centered, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        bias = tl.load(bias_ptr + cols).to(tl.float32)
        out = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, out)
        tl.store(invstd_div_ptr + row, invstd / hidden)


@oracle_impl(hardware="H100", shapes="(T([4096, 4096], f32), T([8, 512, 4096], f32), T([4096], f32), T([4096], f32), S([8, 512, 4096]), S([4096, 4096]))")
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

    addmm_70, add_103, arg18_1, arg19_1, output_shape = _validate_inputs(inputs)
    output_view = torch.empty_strided(
        INPUT_VIEW_SHAPE,
        (512 * HIDDEN, HIDDEN, 1),
        device=addmm_70.device,
        dtype=torch.float32,
    )
    invstd_div = torch.empty_strided(
        INVSTD_SHAPE,
        (512, 1, 1),
        device=addmm_70.device,
        dtype=torch.float32,
    )
    _add_layernorm_saved_scale_kernel[(ROWS,)](
        addmm_70,
        add_103,
        arg18_1,
        arg19_1,
        output_view,
        invstd_div,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK=BLOCK_H,
        num_warps=8,
        num_stages=3,
    )
    return output_view.view(output_shape), invstd_div


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
