"""Gap diagnosis (classification: RECOMPUTE_FUSION): this oracle computes the complete residual-add RMSNorm affine scope in one shape-specialized Triton row kernel, including the metadata-only input view, low-precision residual-add rounding, fp32 mean-square reduction over the hidden dimension, eps=1e-6 rsqrt, fp16-rounded normalized activation, promoted affine multiply, and final output view, whereas Inductor lowers the same graph through a generic reduction schedule that must reload or recompute the residual-add producer for the normalization epilogue; Inductor cannot do this today because its row-reduction scheduler does not keep the full hidden tile's producer values live across the reduction and directly feed the affine store for this residual-RMSNorm pattern; the fix is RECOMPUTE_FUSION: add a guarded RMSNorm row schedule/template that retains the residual-add tile through the reduction and emits the fp16-normalized affine epilogue without the second input pass."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

EPS = 1.0e-6

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
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _residual_rmsnorm_kernel(
        mm_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        input_is_bf16: tl.constexpr,
        output_is_f32: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < hidden
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        added = mm + residual
        if input_is_bf16:
            norm_input = added.to(tl.bfloat16).to(tl.float32)
        else:
            norm_input = added.to(tl.float16).to(tl.float32)

        square_sum = tl.sum(tl.where(mask, norm_input * norm_input, 0.0), axis=0)
        inv_rms = tl.rsqrt(square_sum * (1.0 / hidden) + eps)
        normalized = (norm_input * inv_rms).to(tl.float16)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        if output_is_f32:
            out = weight.to(tl.float32) * normalized.to(tl.float32)
        else:
            out = (weight * normalized).to(tl.float16)
        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"shape argument must be a sequence, got {value!r}") from exc


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _resolve_shape(shape: tuple[int, ...], total_elements: int) -> tuple[int, ...]:
    infer_index = None
    known_product = 1
    for index, dim in enumerate(shape):
        if dim == -1:
            if infer_index is not None:
                raise ValueError(f"only one inferred dimension is allowed, got {shape!r}")
            infer_index = index
        elif dim <= 0:
            raise ValueError(f"invalid non-positive shape dimension in {shape!r}")
        else:
            known_product *= dim

    if infer_index is None:
        if known_product != total_elements:
            raise ValueError(f"shape {shape!r} has numel {known_product}, expected {total_elements}")
        return shape

    if total_elements % known_product != 0:
        raise ValueError(f"cannot infer shape {shape!r} for {total_elements} elements")
    inferred = total_elements // known_product
    resolved = list(shape)
    resolved[infer_index] = inferred
    return tuple(resolved)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mm, residual, weight, view_shape_arg, out_shape_arg = inputs
    if not isinstance(mm, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(mm)!r}")
    if not isinstance(residual, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("residual and weight inputs must be tensors")
    if mm.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"input dtype {mm.dtype} is not supported")
    if residual.dtype != mm.dtype or weight.dtype != mm.dtype:
        raise TypeError(
            f"expected all tensor inputs to have dtype {mm.dtype}, "
            f"got residual={residual.dtype} weight={weight.dtype}"
        )
    if not mm.is_cuda or not residual.is_cuda or not weight.is_cuda:
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if any(value.device != mm.device for value in (residual, weight)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if not mm.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        raise ValueError(
            "expected contiguous inputs, got "
            f"mm_stride={mm.stride()} residual_stride={residual.stride()} "
            f"weight_stride={weight.stride()}"
        )
    if mm.ndim != 2:
        raise ValueError(f"mm input must be rank-2, got shape={tuple(mm.shape)}")

    rows, hidden = int(mm.shape[0]), int(mm.shape[1])
    view_shape = _shape_tuple(view_shape_arg)
    if tuple(residual.shape) != view_shape:
        raise ValueError(f"residual shape {tuple(residual.shape)} != view shape {view_shape}")
    if residual.numel() != mm.numel() or int(residual.shape[-1]) != hidden:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} is not view-compatible with "
            f"mm shape {tuple(mm.shape)}"
        )
    if tuple(weight.shape) != (hidden,):
        raise ValueError(f"weight shape {tuple(weight.shape)} != {(hidden,)}")

    out_shape = _resolve_shape(_shape_tuple(out_shape_arg), rows * hidden)
    if _numel(out_shape) != rows * hidden:
        raise ValueError(f"output shape {out_shape!r} is not compatible with {rows * hidden} elements")

    return mm, residual, weight, out_shape


@oracle_impl(hardware="H100", shapes="(T([512, 4096], f16), T([1, 512, 4096], f16), T([4096], f16), S([1, 512, 4096]), S([512, 4096]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward residual RMSNorm affine computation.

    SCOPE INVARIANT: accepts the same five inputs as Repro.forward() and
    returns the same single output tensor with the same shape, dtype, and
    contiguous view stride.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_rmsnorm.py")

    mm, residual, weight, out_shape = _validate_inputs(inputs)
    rows, hidden = int(mm.shape[0]), int(mm.shape[1])
    output_dtype = torch.float32 if mm.dtype == torch.bfloat16 else torch.float16
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=mm.device,
        dtype=output_dtype,
    )
    block_h = triton.next_power_of_2(hidden)
    _residual_rmsnorm_kernel[(rows,)](
        mm,
        residual,
        weight,
        out,
        hidden=hidden,
        eps=EPS,
        input_is_bf16=mm.dtype == torch.bfloat16,
        output_is_f32=output_dtype == torch.float32,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 2048 else 4,
        num_stages=2,
    )
    return out


def _normalize_outputs(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = True
    for index, (expected_item, actual_item) in enumerate(
        zip(_normalize_outputs(expected), _normalize_outputs(actual))
    ):
        if not isinstance(expected_item, torch.Tensor) or not isinstance(actual_item, torch.Tensor):
            continue
        item_ok = (
            expected_item.dtype == actual_item.dtype
            and expected_item.stride() == actual_item.stride()
        )
        print(
            f"  output {index} layout: {'PASS' if item_ok else 'FAIL'} "
            f"(expected_dtype={expected_item.dtype} oracle_dtype={actual_item.dtype} "
            f"expected_stride={expected_item.stride()} oracle_stride={actual_item.stride()})"
        )
        ok = item_ok and ok
    return ok


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
        ok = _check_layout(instance, inputs) and ok
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
