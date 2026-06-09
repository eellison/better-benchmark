"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete fp32 MT5 residual RMSNorm scope in one Triton row kernel, including the metadata-only `[4096,512] -> [32,128,512]` matmul view, residual add, fp32 mean(square) over hidden size 512, eps=1e-6 rsqrt, affine weight multiply, and final contiguous `[4096,512]` output view; measured full-scope benchmarking should classify this repro as AT_FLOOR because Inductor already emits a single compiled reduction kernel in the same required memory-traffic envelope; the fix is BANDWIDTH_BOUND: record this as an at-floor fp32 residual-RMSNorm case unless broader normalization-template or memory-traffic changes move both paths."""
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
OUTPUT_STRIDE = (HIDDEN, 1)
EPS = 1.0e-6

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
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
    def _fp32_residual_rmsnorm_kernel(
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
        col_mask = cols < hidden
        mask = col_mask[None, :]
        offsets = rows[:, None] * hidden + cols[None, :]

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        x = mm + residual
        sum_square = tl.sum(tl.where(mask, x * x, 0.0), axis=1)
        inv_rms = tl.rsqrt(sum_square / hidden + eps)
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mm = _require_tensor("mm_143", inputs[0], OUTPUT_SHAPE, torch.float32)
    residual = _require_tensor("add_138", inputs[1], BASE_SHAPE, torch.float32)
    weight = _require_tensor("arg190_1", inputs[2], (HIDDEN,), torch.float32)

    view_shape = _shape_tuple("_shape_param_0", inputs[3])
    output_shape = _shape_tuple("_shape_param_1", inputs[4])
    if view_shape != BASE_SHAPE:
        raise ValueError(f"_shape_param_0 is {view_shape}, expected {BASE_SHAPE}")
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"_shape_param_1 is {output_shape}, expected {OUTPUT_SHAPE}")
    if residual.device != mm.device or weight.device != mm.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mm, residual, weight, output_shape


@oracle_impl(hardware="H100", shapes="(T([4096, 512], f32), T([32, 128, 512], f32), T([512], f32), S([32, 128, 512]), S([4096, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete fp32 residual RMSNorm repro computation.

    SCOPE INVARIANT: accepts the same five inputs as Repro.forward() and returns
    the same single fp32 `[4096,512]` contiguous view output.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_fp32_residual_rmsnorm.py")

    mm, residual, weight, output_shape = _validate_inputs(inputs)
    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mm.device,
        dtype=torch.float32,
    )
    block_m = 2
    _fp32_residual_rmsnorm_kernel[(ROWS // block_m,)](
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
    return out_base.view(output_shape)


def _layout_signature(value: torch.Tensor) -> tuple[torch.dtype, tuple[int, ...], tuple[int, ...]]:
    return value.dtype, tuple(value.shape), tuple(value.stride())


def _check_layout(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    ok = _layout_signature(expected) == _layout_signature(actual)
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected={_layout_signature(expected)} oracle={_layout_signature(actual)})"
    )
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
