"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT residual affine pointwise scope with a row/hidden Triton tile that reuses the broadcast scale and bias across several contiguous rows while returning the same final view, whereas Inductor currently lowers the decomposed view/add/mul/add/view graph as a generic flattened pointwise loop over every element; Inductor cannot do this today because pointwise codegen does not select a broadcast-aware two-dimensional row template for fixed hidden-size affine chains, so it relies on generic linear indexing and per-element broadcast addressing; the fix is NEW_PATTERN: add a guarded row/hidden pointwise template for contiguous last-dimension affine broadcasts and benchmark-gate it against the generic pointwise schedule."""
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


BATCH = 256
SEQ = 128
ROWS = BATCH * SEQ
INPUT_512_SHAPE = (ROWS, 512)
INPUT_128_SHAPE = (ROWS, 128)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 2, "BLOCK_N": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 2, "BLOCK_N": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=8, num_stages=3),
        ],
        key=["N_ROWS", "N_COLS"],
    )
    @triton.jit
    def _row_affine_kernel(
        addmm_ptr,
        residual_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < N_ROWS) & (cols[None, :] < N_COLS)
        col_mask = cols < N_COLS
        offsets = rows[:, None] * N_COLS + cols[None, :]

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        scale = tl.load(scale_ptr + cols, mask=col_mask, other=0.0)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0)

        values = (addmm + residual) * scale[None, :] + bias[None, :]
        tl.store(out_ptr + offsets, values, mask=mask)


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    shape = tuple(int(dim) for dim in value)
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")


def _require_tensor(
    name: str,
    value: Any,
    expected_shape: tuple[int, ...],
    expected_stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected_shape}")
    if tuple(value.stride()) != expected_stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {expected_stride}")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have storage_offset 0, got {value.storage_offset()}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
    tuple[int, int],
]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_row_affine.py")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm, residual_3d, scale, bias, shape0_raw, shape1_raw = inputs
    shape0 = tuple(int(dim) for dim in shape0_raw)
    shape1 = tuple(int(dim) for dim in shape1_raw)
    if len(shape0) != 3 or len(shape1) != 2:
        raise ValueError(f"unexpected shape params: {shape0_raw!r}, {shape1_raw!r}")
    if shape0[0] != BATCH or shape0[1] != SEQ or shape1[0] != ROWS or shape0[2] != shape1[1]:
        raise ValueError(f"unexpected shape params: {shape0_raw!r}, {shape1_raw!r}")
    if shape1 not in (INPUT_128_SHAPE, INPUT_512_SHAPE):
        raise ValueError(f"unsupported captured shape: {shape1}")

    cols = shape1[1]
    _require_shape("_shape_param_0", shape0_raw, (BATCH, SEQ, cols))
    _require_shape("_shape_param_1", shape1_raw, (ROWS, cols))
    addmm = _require_tensor("addmm_360", addmm, (ROWS, cols), (cols, 1))
    residual_3d = _require_tensor("add_348", residual_3d, (BATCH, SEQ, cols), (SEQ * cols, cols, 1))
    scale = _require_tensor("arg1111_1", scale, (cols,), (1,))
    bias = _require_tensor("arg1112_1", bias, (cols,), (1,))
    if not (addmm.device == residual_3d.device == scale.device == bias.device):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm, residual_3d, scale, bias, shape0, shape1


@oracle_impl(hardware="H100", shapes="(T([32768, 512], f32), T([256, 128, 512], f32), T([512], f32), T([512], f32), S([256, 128, 512]), S([32768, 512]))")
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
    addmm, residual_3d, scale, bias, shape0, shape1 = _validate_inputs(inputs)
    cols = shape1[1]
    base = torch.empty_strided(
        shape0,
        (SEQ * cols, cols, 1),
        device=addmm.device,
        dtype=addmm.dtype,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_M"]),)
    _row_affine_kernel[grid](
        addmm,
        residual_3d,
        scale,
        bias,
        base,
        N_ROWS=ROWS,
        N_COLS=cols,
    )
    return base.view(shape1)


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
