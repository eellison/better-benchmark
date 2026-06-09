"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Reformer population LayerNorm scope in one shape-specialized Triton row kernel, including `var_mean(correction=0, keepdim=True)`, `rsqrt(var + 1e-12)`, affine scale/bias, and both returned `[32768,256]` view aliases from one output buffer, whereas Inductor emits a generic fused Welford var_mean reduction kernel for the same decomposed graph; Inductor cannot do this today because its correction=0 var_mean lowering keeps general Welford bookkeeping instead of selecting fixed-width population LayerNorm algebra and preserving duplicate view aliases directly from the epilogue; the fix is ALGEBRAIC_ELIMINATION: add a guarded correction=0 LayerNorm lowering that replaces generic Welford state with direct mean/centered-variance reductions and emits alias-only view returns from the normalized output."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


ROWS = 32768
BATCH = 8
SEQ = 4096
HIDDEN = 256
INPUT_SHAPE = (BATCH, SEQ, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
INPUT_STRIDE = (SEQ * HIDDEN, HIDDEN, 1)
OUTPUT_STRIDE = (HIDDEN, 1)
EPS = 1.0e-12
BLOCK_M = 8
BLOCK_N = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int]]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    x, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (x, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")

    expected_shapes = (INPUT_SHAPE, (HIDDEN,), (HIDDEN,))
    expected_strides = (INPUT_STRIDE, (1,), (1,))
    for index, (value, shape, stride) in enumerate(
        zip(tensor_inputs, expected_shapes, expected_strides)
    ):
        if tuple(value.shape) != shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {shape}")
        if tuple(value.stride()) != stride:
            raise ValueError(f"input {index} stride {tuple(value.stride())} != {stride}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if value.device != x.device:
            raise ValueError(f"input {index} device {value.device} != {x.device}")
        if value.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for the Triton oracle")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    if shape0_tuple != OUTPUT_SHAPE:
        raise ValueError(f"unexpected first output view shape parameter: {shape0!r}")
    if shape1_tuple != OUTPUT_SHAPE:
        raise ValueError(f"unexpected second output view shape parameter: {shape1!r}")

    return x, weight, bias, shape0_tuple, shape1_tuple


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _layernorm_aliases_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        rows_total: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        cols = tl.arange(0, BLOCK_N)[None, :]
        mask = (rows < rows_total) & (cols < hidden)
        offsets = rows * hidden + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=1) / hidden
        centered = x - mean[:, None]
        centered_for_reduce = tl.where(mask, centered, 0.0)
        variance = tl.sum(centered_for_reduce * centered_for_reduce, axis=1) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
        out = (centered * invstd[:, None]) * weight + bias

        tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([8, 4096, 256], f32), T([256], f32), T([256], f32), S([32768, 256]), S([32768, 256]))")
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
        raise RuntimeError("Triton is required for oracle_reformer_layernorm_aliases.py")

    x, weight, bias, shape0, shape1 = _validate_inputs(inputs)
    base = torch.empty_strided(
        INPUT_SHAPE,
        INPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    _layernorm_aliases_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        x,
        weight,
        bias,
        base,
        rows_total=ROWS,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return base.view(shape0), base.view(shape1)


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
