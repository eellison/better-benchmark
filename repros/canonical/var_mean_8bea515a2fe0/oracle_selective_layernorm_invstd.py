"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete view, broadcast multiply, residual add, fp32 population var_mean, all-token rsqrt/768 side output, and token-0 affine clone with scale/bias work only for returned rows, whereas Inductor currently schedules the decomposed LayerNorm affine producer across all [128, 1370, 768] rows before applying select/clone while also materializing the rsqrt side output; Inductor cannot do this today because its normalization scheduler does not sink a constant token select through row-independent view/multiply/add/var_mean/affine when another consumer still needs per-row invstd; the fix is ALGEBRAIC_ELIMINATION: split the LayerNorm lowering into an all-row statistics side-output loop and a narrowed affine epilogue loop for selected token rows."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
BATCH = 128
TOKENS = 1370
HIDDEN = 768
ROWS = BATCH * TOKENS
ADDMM_SHAPE = (ROWS, HIDDEN)
BASE_SHAPE = (BATCH, TOKENS, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
SELECTED_SHAPE = (BATCH, HIDDEN)
SELECTED_STRIDE = (HIDDEN, 1)
INVSTD_SHAPE = (BATCH, TOKENS, 1)
INVSTD_STRIDE = (TOKENS, 1, 1)
EPS = 1.0e-6
BLOCK_H = 1024

if triton is not None:

    @triton.jit
    def _invstd_side_kernel(
        addmm_ptr,
        scale_ptr,
        residual_ptr,
        invstd_side_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        values = residual + addmm * scale
        values_for_reduce = tl.where(mask, values, 0.0)

        mean = tl.sum(values_for_reduce, axis=0) / hidden
        centered = values - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        tl.store(invstd_side_ptr + row, invstd / hidden)

    @triton.jit
    def _selected_token_layernorm_kernel(
        addmm_ptr,
        scale_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        selected_out_ptr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        batch = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        row = batch * tokens
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        values = residual + addmm * scale
        values_for_reduce = tl.where(mask, values, 0.0)

        mean = tl.sum(values_for_reduce, axis=0) / hidden
        centered = values - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        output = centered * invstd * weight + bias
        tl.store(selected_out_ptr + batch * hidden + cols, output, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_47 = _require_tensor("addmm_47", inputs[0], ADDMM_SHAPE, torch.float32, (HIDDEN, 1))
    scale = _require_tensor("arg172_1", inputs[1], AFFINE_SHAPE, torch.float32, (1,))
    residual = _require_tensor(
        "add_80",
        inputs[2],
        BASE_SHAPE,
        torch.float32,
        (TOKENS * HIDDEN, HIDDEN, 1),
    )
    weight = _require_tensor("arg173_1", inputs[3], AFFINE_SHAPE, torch.float32, (1,))
    bias = _require_tensor("arg174_1", inputs[4], AFFINE_SHAPE, torch.float32, (1,))

    shape_param = _shape_tuple(inputs[5])
    if shape_param != BASE_SHAPE:
        raise ValueError(f"_shape_param_0 is {shape_param}, expected {BASE_SHAPE}")

    device = addmm_47.device
    for tensor in (scale, residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm_47, scale, residual, weight, bias


@oracle_impl(hardware="H100", shapes="(T([175360, 768], f32), T([768], f32), T([128, 1370, 768], f32), T([768], f32), T([768], f32), S([128, 1370, 768]))")
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
        raise RuntimeError("Triton is required for oracle_selective_layernorm_invstd.py")

    addmm_47, scale, residual, weight, bias = _validate_inputs(inputs)
    selected_out = torch.empty_strided(
        SELECTED_SHAPE,
        SELECTED_STRIDE,
        device=addmm_47.device,
        dtype=torch.float32,
    )
    invstd_side = torch.empty_strided(
        INVSTD_SHAPE,
        INVSTD_STRIDE,
        device=addmm_47.device,
        dtype=torch.float32,
    )

    _invstd_side_kernel[(ROWS,)](
        addmm_47,
        scale,
        residual,
        invstd_side,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=3,
    )
    _selected_token_layernorm_kernel[(BATCH,)](
        addmm_47,
        scale,
        residual,
        weight,
        bias,
        selected_out,
        tokens=TOKENS,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=3,
    )
    return selected_out, invstd_side


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
