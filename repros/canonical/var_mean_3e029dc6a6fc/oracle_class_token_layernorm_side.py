"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DINOv2 residual LayerNorm scope, including `[175360,768] -> [128,1370,768]`, gamma multiply before residual add, fp32 population var_mean, eps-before-rsqrt, affine class-token clone, and full `[128,1370,1]` `invstd / 768` side output, but only evaluates affine scale/bias for the token-0 rows that Repro.forward returns; Inductor currently schedules the affine LayerNorm epilogue across all `[128,1370,768]` rows before applying the trailing token select and clone, and it cannot do this today because the normalization scheduler does not push a constant token select through a row-local LayerNorm while preserving all-row saved inverse scale; the fix is ALGEBRAIC_ELIMINATION: sink fixed-token selects into the affine epilogue and keep the side-output store over the original row domain."""
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
    has_stochastic_ops,
)


BATCH = 128
TOKENS = 1370
HIDDEN = 768
ROWS = BATCH * TOKENS
INPUT_SHAPE = (ROWS, HIDDEN)
RESHAPE = (BATCH, TOKENS, HIDDEN)
VECTOR_SHAPE = (HIDDEN,)
OUTPUT_SHAPE = (BATCH, HIDDEN)
SIDE_SHAPE = (BATCH, TOKENS, 1)
EPS = 1.0e-6
BLOCK_H = 1024


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
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_47, gamma, residual, weight, bias, shape_param = inputs
    tensor_inputs = (addmm_47, gamma, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (INPUT_SHAPE, VECTOR_SHAPE, RESHAPE, VECTOR_SHAPE, VECTOR_SHAPE)
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if _shape_tuple(shape_param) != RESHAPE:
        raise ValueError(f"unexpected reshape parameter: {shape_param!r}")

    device = addmm_47.device
    if not all(value.device == device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm_47, gamma, residual, weight, bias


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _class_token_layernorm_side_kernel(
        addmm_ptr,
        gamma_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        side_ptr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scaled = addmm * gamma
        x = residual + scaled
        reduce_x = tl.where(mask, x, 0.0)

        mean = tl.sum(reduce_x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = libdevice.rsqrt(variance + eps)
        tl.store(side_ptr + row, invstd / hidden)

        token = row - (row // tokens) * tokens
        is_class_token = token == 0
        weight = tl.load(weight_ptr + cols, mask=mask & is_class_token, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask & is_class_token, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias
        batch = row // tokens
        tl.store(out_ptr + batch * hidden + cols, y, mask=mask & is_class_token)


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
    if triton is None or tl is None or libdevice is None:
        raise RuntimeError("Triton with Inductor libdevice helpers is required")

    addmm_47, gamma, residual, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        (HIDDEN, 1),
        device=addmm_47.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        (TOKENS, 1, 1),
        device=addmm_47.device,
        dtype=torch.float32,
    )

    _class_token_layernorm_side_kernel[(ROWS,)](
        addmm_47,
        gamma,
        residual,
        weight,
        bias,
        output,
        side,
        hidden=HIDDEN,
        tokens=TOKENS,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=4,
    )
    return output, side


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
