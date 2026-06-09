"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DeiT training residual LayerNorm scope in one hidden-size-768 Triton row kernel, including the `[25344,768] -> [128,198,768]` view, residual add, fp32 correction=0 mean and centered variance over the hidden dimension, eps=1e-6 before libdevice rsqrt, affine scale/bias, final `[25344,768]` view, and sibling `invstd / 768` output, whereas Inductor already emits the same full fused normalization region through its persistent reduction template; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new local pattern because the mandatory activation/residual/affine reads, one row reduction, output store, and side-output store dominate the full scope; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader normalization-template or memory-traffic work moves both implementations."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


BATCH = 128
TOKENS = 198
HIDDEN = 768
ROWS = BATCH * TOKENS
INPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, TOKENS, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
INVSTD_SHAPE = (BATCH, TOKENS, 1)
EPS = 1.0e-6
INV_HIDDEN = 0.0013020833333333333
BLOCK_H = 1024


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_45, add_77, primals_145, primals_146, shape0, shape1 = inputs
    tensor_inputs = (addmm_45, add_77, primals_145, primals_146)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (INPUT_SHAPE, RESIDUAL_SHAPE, (HIDDEN,), (HIDDEN,))
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if value.device != addmm_45.device:
            raise ValueError(f"input {index} device {value.device} != {addmm_45.device}")

    if _shape_tuple(shape0) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected input view shape parameter: {shape0!r}")
    output_shape = _shape_tuple(shape1)
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {shape1!r}")

    return addmm_45, add_77, primals_145, primals_146, output_shape


if triton is not None:

    @triton.jit
    def _residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        inv_hidden: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
        x = residual + addmm

        x_sum = tl.sum(tl.where(mask, x, 0.0), axis=0).to(tl.float32)
        mean = x_sum / hidden
        centered = x - mean
        variance_sum = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0).to(tl.float32)
        variance = variance_sum / hidden
        invstd = libdevice.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
        normalized = centered * invstd
        output = normalized * weight + bias

        tl.store(output_ptr + offsets, output, mask=mask)
        tl.store(invstd_div_ptr + row, invstd * inv_hidden)


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm_45, add_77, primals_145, primals_146, output_shape = _validate_inputs(inputs)
    addmm_view = torch.ops.aten.reshape.default(addmm_45, RESIDUAL_SHAPE)
    add_tensor = torch.ops.aten.add.Tensor(add_77, addmm_view)
    var, mean = torch.ops.aten.var_mean.correction(add_tensor, [2], correction=0, keepdim=True)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var, EPS))
    centered = torch.ops.aten.sub.Tensor(add_tensor, mean)
    normalized = torch.ops.aten.mul.Tensor(centered, invstd)
    scaled = torch.ops.aten.mul.Tensor(normalized, primals_145)
    affine = torch.ops.aten.add.Tensor(scaled, primals_146)
    output = torch.ops.aten.reshape.default(affine, output_shape)
    return output, torch.ops.aten.mul.Tensor(invstd, INV_HIDDEN)


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
    addmm_45, add_77, primals_145, primals_146, output_shape = _validate_inputs(inputs)
    if (
        triton is None
        or libdevice is None
        or not addmm_45.is_cuda
        or not addmm_45.is_contiguous()
        or not add_77.is_contiguous()
        or not primals_145.is_contiguous()
        or not primals_146.is_contiguous()
    ):
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        output_shape,
        (HIDDEN, 1),
        device=addmm_45.device,
        dtype=torch.float32,
    )
    invstd_div = torch.empty_strided(
        INVSTD_SHAPE,
        (TOKENS, 1, 1),
        device=addmm_45.device,
        dtype=torch.float32,
    )
    _residual_layernorm_kernel[(ROWS,)](
        addmm_45,
        add_77,
        primals_145,
        primals_146,
        output,
        invstd_div,
        hidden=HIDDEN,
        eps=EPS,
        inv_hidden=INV_HIDDEN,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=4,
    )
    return output, invstd_div


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
