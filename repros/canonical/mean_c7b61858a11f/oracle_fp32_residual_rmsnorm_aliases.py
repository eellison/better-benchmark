"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete fp32 MT5 residual RMSNorm alias scope in one Triton row kernel, including the metadata-only matmul view, residual add, fp32 mean(square) over hidden size 512, eps=1e-6 rsqrt, affine weight multiply, and both returned aliasing [4096,512] views over one contiguous [32,128,512] base buffer, whereas Inductor already reaches the same measured performance floor for this full scope with its compiled reduction schedule; Inductor cannot materially improve this repro through a local fusion or algebraic rewrite because the remaining work is dominated by the required two activation reads, one weight read, one fp32 row reduction, rsqrt, and final stores rather than an avoidable extra kernel; the fix is BANDWIDTH_BOUND: record this as an at-floor fp32 residual-RMSNorm alias case unless broader normalization-template or launch-overhead changes move both paths."""
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

BATCH = 32
SEQ_LEN = 128
ROWS = BATCH * SEQ_LEN
HIDDEN = 512
BASE_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
BASE_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
OUTPUT_COUNT = 2
EPS = 1.0e-6

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
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
    def _fp32_residual_rmsnorm_aliases_kernel(
        mm_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_H)
        mask = cols[None, :] < hidden
        offsets = rows[:, None] * hidden + cols[None, :]

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        x = mm + residual
        sum_square = tl.sum(tl.where(mask, x * x, 0.0), axis=1)
        inv_rms = tl.rsqrt(sum_square / hidden + eps)
        weight = tl.load(weight_ptr + cols, mask=cols < hidden, other=0.0)
        out = x * inv_rms[:, None] * weight[None, :]
        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(name: str, value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a shape sequence, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm = _require_tensor("mm_140", inputs[0], OUTPUT_SHAPE, torch.float32)
    residual = _require_tensor("add_136", inputs[1], BASE_SHAPE, torch.float32)
    weight = _require_tensor("arg186_1", inputs[2], (HIDDEN,), torch.float32)

    view_shape = _shape_tuple("_shape_param_0", inputs[3])
    output_shapes = (
        _shape_tuple("_shape_param_1", inputs[4]),
        _shape_tuple("_shape_param_2", inputs[5]),
    )
    if view_shape != BASE_SHAPE:
        raise ValueError(f"_shape_param_0 is {view_shape}, expected {BASE_SHAPE}")
    for index, shape in enumerate(output_shapes, start=1):
        if shape != OUTPUT_SHAPE:
            raise ValueError(f"_shape_param_{index} is {shape}, expected {OUTPUT_SHAPE}")
    if residual.device != mm.device or weight.device != mm.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mm, residual, weight, output_shapes


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the complete fp32 residual RMSNorm alias repro computation.

    SCOPE INVARIANT: accepts the same six inputs as Repro.forward() and returns
    the same two fp32 [4096,512] contiguous views, both aliasing one base result.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_fp32_residual_rmsnorm_aliases.py")

    mm, residual, weight, output_shapes = _validate_inputs(inputs)
    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mm.device,
        dtype=torch.float32,
    )
    block_m = 2
    _fp32_residual_rmsnorm_aliases_kernel[(ROWS // block_m,)](
        mm,
        residual,
        weight,
        out_base,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_M=block_m,
        BLOCK_H=HIDDEN,
        num_warps=4,
        num_stages=2,
    )
    return tuple(out_base.view(shape) for shape in output_shapes)


def _normalize_outputs(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _layout_signature(value: torch.Tensor) -> tuple[torch.dtype, tuple[int, ...], tuple[int, ...]]:
    return value.dtype, tuple(value.shape), tuple(value.stride())


def _check_layout_and_alias(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = _normalize_outputs(instance(*inputs))
        actual = _normalize_outputs(oracle_forward(inputs))
        torch.cuda.synchronize()

    ok = True
    for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
        if not isinstance(expected_item, torch.Tensor) or not isinstance(actual_item, torch.Tensor):
            continue
        item_ok = _layout_signature(expected_item) == _layout_signature(actual_item)
        print(
            f"  output {index} layout: {'PASS' if item_ok else 'FAIL'} "
            f"(expected={_layout_signature(expected_item)} oracle={_layout_signature(actual_item)})"
        )
        ok = item_ok and ok

    expected_alias = (
        isinstance(expected[0], torch.Tensor)
        and isinstance(expected[1], torch.Tensor)
        and expected[0].untyped_storage().data_ptr() == expected[1].untyped_storage().data_ptr()
        and expected[0].storage_offset() == expected[1].storage_offset()
    )
    actual_alias = (
        isinstance(actual[0], torch.Tensor)
        and isinstance(actual[1], torch.Tensor)
        and actual[0].untyped_storage().data_ptr() == actual[1].untyped_storage().data_ptr()
        and actual[0].storage_offset() == actual[1].storage_offset()
    )
    alias_ok = expected_alias == actual_alias
    print(
        f"  output alias: {'PASS' if alias_ok else 'FAIL'} "
        f"(expected={expected_alias} oracle={actual_alias})"
    )
    return ok and alias_ok


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
        ok = _check_layout_and_alias(instance, inputs) and ok
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
