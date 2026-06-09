"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Longformer inference residual-add LayerNorm scope in one shape-specialized Triton row kernel, including the `[8192,768] -> [8,1024,768]` view, f32 residual add, correction=0 mean and centered variance over hidden size 768, `variance + 1e-5`, generated `libdevice.rsqrt`, affine multiply-then-add order, and contiguous `[8,1024,768]` output layout, whereas Inductor already emits a fused persistent reduction kernel for the same full scope; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new narrow pattern because the required activation/residual/affine reads, one row reduction, reciprocal square root, and output store dominate; the fix is BANDWIDTH_BOUND: record this as an at-floor LayerNorm case unless broader normalization-template or memory-traffic work moves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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


ROWS = 8192
BATCH = 8
SEQ_LEN = 1024
HIDDEN = 768
INPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
EPS = 1.0e-5
BLOCK_M = 2
BLOCK_N = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_23, add_139, weight, bias, shape_param = inputs
    tensor_inputs = (addmm_23, add_139, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (INPUT_SHAPE, OUTPUT_SHAPE, (HIDDEN,), (HIDDEN,))
    expected_strides = ((HIDDEN, 1), OUTPUT_STRIDE, (1,), (1,))
    for index, (value, shape, stride) in enumerate(
        zip(tensor_inputs, expected_shapes, expected_strides)
    ):
        if tuple(value.shape) != shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {shape}")
        if tuple(value.stride()) != stride:
            raise ValueError(f"input {index} stride {tuple(value.stride())} != {stride}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if value.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for the Triton oracle")

    if _shape_tuple(shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")

    return addmm_23, add_139, weight, bias


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)[:, None]
        cols = tl.arange(0, block_n)[None, :]
        mask = cols < hidden
        offsets = rows * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        x = addmm + residual
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=1)[:, None].to(tl.float32) / hidden
        centered = x - mean
        centered_for_reduce = tl.where(mask, centered * centered, 0.0)
        variance = tl.sum(centered_for_reduce, axis=1)[:, None].to(tl.float32) / hidden
        invstd = libdevice.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        out = (centered * invstd) * weight + bias
        tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([8192, 768], f32), T([8, 1024, 768], f32), T([768], f32), T([768], f32), S([8, 1024, 768]))")
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
        raise RuntimeError("Triton is required for oracle_residual_layernorm.py")

    addmm_23, add_139, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm_23.device,
        dtype=torch.float32,
    )
    _residual_layernorm_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        addmm_23,
        add_139,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        eps=EPS,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return out


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
