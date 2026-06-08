"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet gated product-reduction scope, multiplying the two `[512, 120, 28, 28]` inputs, reducing each `(N,C)` row over spatial dimensions in f32, applying the captured `arg280_1 > -3.0 & arg280_1 < 3.0` gate with the exact `0.16666666666666666` scale/full fallback, then finalizing the `[120]` channel sum; Inductor already lowers this as the same bandwidth-heavy fused product plus dependent reductions, and the oracle mainly exposes a row-boundary partial schedule without changing the math; Inductor cannot remove much more traffic for this graph because both paths must stream the two dense inputs and produce one small reduction result, so the fix is BANDWIDTH_BOUND: mark this repro at floor unless a broader reduction-template or memory-bandwidth improvement moves both implementations together."""
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


N = 512
C = 120
H = 28
W = 28
HW = H * W
N_BLOCK = 4
N_TILES = (N + N_BLOCK - 1) // N_BLOCK
BLOCK_HW = 1024
FINAL_BLOCK_N = 128

NCHW_SHAPE = (N, C, H, W)
NCHW_STRIDE = (C * HW, HW, W, 1)
NC11_SHAPE = (N, C, 1, 1)
NC11_STRIDE = (C, 1, 1, 1)
OUT_SHAPE = (C,)
OUT_STRIDE = (1,)
SCALAR_SHAPE = ()
SCALAR_STRIDE = ()


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _row_partial_kernel(
        lhs_ptr,
        rhs_ptr,
        gate_ptr,
        full_ptr,
        partial_ptr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        N_BLOCK_: tl.constexpr,
        N_TILES_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n_tile = tl.program_id(1)

        n_offsets = n_tile * N_BLOCK_ + tl.arange(0, N_BLOCK_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        n_mask = n_offsets < N_
        hw_mask = hw_offsets < HW_

        offsets = n_offsets[:, None] * (C_ * HW_) + c * HW_ + hw_offsets[None, :]
        mask = n_mask[:, None] & hw_mask[None, :]
        lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        product = lhs * rhs
        row_sums = tl.sum(tl.where(mask, product, 0.0), axis=1)

        gate = tl.load(gate_ptr + n_offsets * C_ + c, mask=n_mask, other=0.0).to(tl.float32)
        gt_scalar = gate > -3.0
        lt_scalar = gate < 3.0
        bitwise_and_tensor = gt_scalar & lt_scalar
        scaled = row_sums * 0.16666666666666666
        full_value = tl.load(full_ptr).to(tl.float32)
        row_values = tl.where(bitwise_and_tensor, scaled, full_value)
        row_values = tl.where(n_mask, row_values, 0.0)
        partial = tl.sum(row_values, axis=0)
        tl.store(partial_ptr + c * N_TILES_ + n_tile, partial)

    @triton.jit
    def _finalize_kernel(
        partial_ptr,
        out_ptr,
        N_TILES_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        c = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_N_)
        mask = offsets < N_TILES_
        values = tl.load(partial_ptr + c * N_TILES_ + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        total = tl.sum(values, axis=0)
        tl.store(out_ptr + c, total)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    getitem_195, arg277_1, arg280_1, full = inputs
    getitem_195 = _require_f32_tensor("getitem_195", getitem_195, NCHW_SHAPE, NCHW_STRIDE)
    arg277_1 = _require_f32_tensor("arg277_1", arg277_1, NCHW_SHAPE, NCHW_STRIDE)
    arg280_1 = _require_f32_tensor("arg280_1", arg280_1, NC11_SHAPE, NC11_STRIDE)
    full = _require_f32_tensor("full", full, SCALAR_SHAPE, SCALAR_STRIDE)

    device = getitem_195.device
    if arg277_1.device != device or arg280_1.device != device or full.device != device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return getitem_195, arg277_1, arg280_1, full


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
        raise RuntimeError("Triton is required for oracle_gated_product_channel_sum.py")

    getitem_195, arg277_1, arg280_1, full = _validate_inputs(inputs)

    partial = torch.empty_strided(
        (C, N_TILES),
        (N_TILES, 1),
        device=getitem_195.device,
        dtype=torch.float32,
    )
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=getitem_195.device,
        dtype=torch.float32,
    )

    _row_partial_kernel[(C, N_TILES)](
        getitem_195,
        arg277_1,
        arg280_1,
        full,
        partial,
        N,
        C,
        HW,
        N_BLOCK,
        N_TILES,
        BLOCK_HW,
    )
    _finalize_kernel[(C,)](
        partial,
        output,
        N_TILES,
        FINAL_BLOCK_N,
    )
    return output


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
