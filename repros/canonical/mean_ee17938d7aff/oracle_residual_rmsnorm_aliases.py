"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete residual-add RMSNorm scope, including the low-precision residual add, fp32 mean-square reduction, rsqrt normalization, low-precision round-trip, affine weight multiply, and three returned alias views from one Triton row kernel, whereas Inductor already emits one full-scope fused reduction kernel for the same graph but remains slightly slower than this shape-specialized row-normalization kernel; Inductor cannot materially improve this today through missing-op fusion because the work is already fused and dominated by fixed row reads, reduction math, and the final write; the fix is BANDWIDTH_BOUND: treat this as near-floor normalization-template tuning unless a more specialized residual-RMSNorm schedule consistently beats the generic fused reduction."""
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

EPS = 1.0e-6

if triton is not None:

    @triton.jit
    def _residual_rmsnorm_f16_kernel(
        mm_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < hidden
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        added = (mm + residual).to(tl.float16).to(tl.float32)

        square_sum = tl.sum(tl.where(mask, added * added, 0.0), axis=0)
        inv_rms = tl.rsqrt(square_sum * (1.0 / hidden) + eps)
        normalized = (added * inv_rms).to(tl.float16)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        out = (weight * normalized).to(tl.float16)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _residual_rmsnorm_bf16_kernel(
        mm_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < hidden
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        added = (mm + residual).to(tl.bfloat16).to(tl.float32)

        square_sum = tl.sum(tl.where(mask, added * added, 0.0), axis=0)
        inv_rms = tl.rsqrt(square_sum * (1.0 / hidden) + eps)
        normalized = (added * inv_rms).to(tl.float16).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = weight * normalized
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_tensor(
    name: str,
    value: Any,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _require_shape(name: str, value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a shape sequence, got {value!r}") from exc


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    mm = _require_tensor("mm_216", inputs[0])
    residual = _require_tensor("add_216", inputs[1], mm.dtype)
    weight = _require_tensor("arg282_1", inputs[2], mm.dtype)

    if mm.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"mm_216 dtype {mm.dtype} is not supported")
    if mm.ndim != 2:
        raise ValueError(f"mm_216 must be rank-2, got shape={tuple(mm.shape)}")
    if residual.ndim != 3:
        raise ValueError(f"add_216 must be rank-3, got shape={tuple(residual.shape)}")

    rows, hidden = int(mm.shape[0]), int(mm.shape[1])
    if tuple(weight.shape) != (hidden,):
        raise ValueError(f"arg282_1 shape {tuple(weight.shape)} != {(hidden,)}")
    if residual.numel() != mm.numel() or int(residual.shape[-1]) != hidden:
        raise ValueError(
            f"add_216 shape {tuple(residual.shape)} is not view-compatible with "
            f"mm_216 shape {tuple(mm.shape)}"
        )
    if any(value.device != mm.device for value in (residual, weight)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    base_shape = _require_shape("_shape_param_0", inputs[3])
    output_shapes = (
        _require_shape("_shape_param_1", inputs[4]),
        _require_shape("_shape_param_2", inputs[5]),
        _require_shape("_shape_param_3", inputs[6]),
    )
    if base_shape != tuple(residual.shape):
        raise ValueError(f"_shape_param_0 is {base_shape}, expected {tuple(residual.shape)}")
    for index, shape in enumerate(output_shapes, start=1):
        if _numel(shape) != rows * hidden:
            raise ValueError(
                f"_shape_param_{index} has numel {_numel(shape)}, expected {rows * hidden}"
            )

    return mm, residual, weight, base_shape, output_shapes


@oracle_impl(hardware="H100", shapes="(T([512, 4096], f16), T([1, 512, 4096], f16), T([4096], f16), S([1, 512, 4096]), S([512, 4096]), S([512, 4096]), S([512, 4096]))")
def oracle_forward(inputs):
    """Run the complete residual-add RMSNorm repro computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same three low-precision view outputs. The returned tensors are views
    of one contiguous base tensor, matching the repro's repeated view outputs.

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_rmsnorm_aliases.py")

    mm, residual, weight, base_shape, output_shapes = _validate_inputs(inputs)
    rows, hidden = int(mm.shape[0]), int(mm.shape[1])
    out_base = torch.empty_strided(
        base_shape,
        (base_shape[1] * base_shape[2], base_shape[2], 1),
        device=mm.device,
        dtype=torch.float32 if mm.dtype is torch.bfloat16 else mm.dtype,
    )
    block_h = triton.next_power_of_2(hidden)
    kernel = (
        _residual_rmsnorm_bf16_kernel
        if mm.dtype is torch.bfloat16
        else _residual_rmsnorm_f16_kernel
    )
    kernel[(rows,)](
        mm,
        residual,
        weight,
        out_base,
        hidden=hidden,
        eps=EPS,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return tuple(out_base.view(shape) for shape in output_shapes)


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
