"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileNetV3 f32 BatchNorm-affine, ReLU, upstream-gradient product, spatial sum, gate-mask/scale/full fallback, and final channel sum for Repro.forward by atomically accumulating one contribution per `(n-block, c)` spatial tile into the final channel vector, whereas Inductor materializes the dependent `[512, 72, 1, 1]` spatial-reduction result before a separate channel reduction but measures at the same CUDAGraph floor for this full scope; Inductor cannot materially improve this repro through a local scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, recompute-fusion, or new-pattern change because the required reads of the two full NCHW tensors plus f32 pointwise work dominate; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader reduction memory-traffic or launch-overhead work moves both implementations."""
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

N = 512
C = 72
H = 28
W = 28
HW = H * W
BLOCK_HW = 1024
BLOCK_N = 1
HW_CHUNKS = (HW + BLOCK_HW - 1) // BLOCK_HW

NCHW_SHAPE = (N, C, H, W)
NCHW_STRIDE = (C * HW, HW, W, 1)
CHANNEL_11_SHAPE = (1, C, 1, 1)
CHANNEL_11_STRIDE = (C, 1, 1, 1)
CHANNEL_SHAPE = (C,)
CHANNEL_STRIDE = (1,)
NC11_SHAPE = (N, C, 1, 1)
NC11_STRIDE = (C, 1, 1, 1)

if triton is not None:

    @triton.jit
    def _zero_output_kernel(
        out_ptr,
        C_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_C_)
        tl.store(out_ptr + offsets, tl.zeros((BLOCK_C_,), dtype=tl.float32), mask=offsets < C_)

    @triton.jit
    def _spatial_atomic_channel_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        grad_ptr,
        gate_ptr,
        full_ptr,
        out_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        N_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        HW_CHUNKS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n_start = tl.program_id(1) * BLOCK_N_

        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        weight = tl.load(weight_ptr + c).to(tl.float32)
        bias = tl.load(bias_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        block_sum = tl.zeros((), dtype=tl.float32)

        for i in tl.static_range(0, BLOCK_N_):
            n = n_start + i
            row_mask = n < N_
            spatial_sum = tl.zeros((), dtype=tl.float32)

            for chunk in tl.static_range(0, HW_CHUNKS_):
                hw = chunk * BLOCK_HW_ + tl.arange(0, BLOCK_HW_)
                hw_mask = hw < HW_
                offsets = n * C_ * HW_ + c * HW_ + hw

                x = tl.load(x_ptr + offsets, mask=hw_mask & row_mask, other=0.0).to(tl.float32)
                grad = tl.load(grad_ptr + offsets, mask=hw_mask & row_mask, other=0.0).to(tl.float32)

                centered = x - mean
                normalized = centered * invstd
                affine_mul = normalized * weight
                affine = affine_mul + bias
                relu = tl.where(affine < 0.0, 0.0, affine)
                product = grad * relu
                spatial_sum += tl.sum(tl.where(hw_mask, product, 0.0), axis=0)

            gate = tl.load(gate_ptr + n * C_ + c, mask=row_mask, other=0.0).to(tl.float32)
            gt = gate > -3.0
            lt = gate < 3.0
            in_range = gt & lt
            scaled = spatial_sum * 0.16666666666666666
            contribution = tl.where(in_range, scaled, full_value)
            block_sum += tl.where(row_mask, contribution, 0.0)

        tl.atomic_add(out_ptr + c, block_sum, sem="relaxed")


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
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        arg165_1,
        arg166_1,
        arg167_1,
        arg23_1,
        arg24_1,
        getitem_147,
        arg170_1,
        full,
    ) = inputs

    arg165_1 = _require_f32_tensor("arg165_1", arg165_1, NCHW_SHAPE, NCHW_STRIDE)
    arg166_1 = _require_f32_tensor("arg166_1", arg166_1, CHANNEL_11_SHAPE, CHANNEL_11_STRIDE)
    arg167_1 = _require_f32_tensor("arg167_1", arg167_1, CHANNEL_11_SHAPE, CHANNEL_11_STRIDE)
    arg23_1 = _require_f32_tensor("arg23_1", arg23_1, CHANNEL_SHAPE, CHANNEL_STRIDE)
    arg24_1 = _require_f32_tensor("arg24_1", arg24_1, CHANNEL_SHAPE, CHANNEL_STRIDE)
    getitem_147 = _require_f32_tensor("getitem_147", getitem_147, NCHW_SHAPE, NCHW_STRIDE)
    arg170_1 = _require_f32_tensor("arg170_1", arg170_1, NC11_SHAPE, NC11_STRIDE)
    full = _require_f32_tensor("full", full, (), ())

    device = arg165_1.device
    if any(
        tensor.device != device
        for tensor in (
            arg166_1,
            arg167_1,
            arg23_1,
            arg24_1,
            getitem_147,
            arg170_1,
            full,
        )
    ):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return (
        arg165_1,
        arg166_1,
        arg167_1,
        arg23_1,
        arg24_1,
        getitem_147,
        arg170_1,
        full,
    )


@oracle_impl(hardware="H100", shapes="(T([512, 72, 28, 28], f32), T([1, 72, 1, 1], f32), T([1, 72, 1, 1], f32), T([72], f32), T([72], f32), T([512, 72, 28, 28], f32), T([512, 72, 1, 1], f32), T([], f32))")
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
        raise RuntimeError("Triton is required for oracle_algebraic_channel_reduction.py")

    (
        arg165_1,
        arg166_1,
        arg167_1,
        arg23_1,
        arg24_1,
        getitem_147,
        arg170_1,
        full,
    ) = _validate_inputs(inputs)

    output = torch.empty_strided(
        CHANNEL_SHAPE,
        CHANNEL_STRIDE,
        device=arg165_1.device,
        dtype=torch.float32,
    )

    _zero_output_kernel[(1,)](
        output,
        C_=C,
        BLOCK_C_=triton.next_power_of_2(C),
        num_warps=1,
        num_stages=1,
    )
    _spatial_atomic_channel_kernel[(C, triton.cdiv(N, BLOCK_N))](
        arg165_1,
        arg166_1,
        arg167_1,
        arg23_1,
        arg24_1,
        getitem_147,
        arg170_1,
        full,
        output,
        C_=C,
        HW_=HW,
        N_=N,
        BLOCK_N_=BLOCK_N,
        HW_CHUNKS_=HW_CHUNKS,
        BLOCK_HW_=BLOCK_HW,
        num_warps=4,
        num_stages=1,
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
