"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileNetV3 BatchNorm-affine, hard-swish, upstream-gradient product, spatial sum, gate-mask/scale/full fallback, and final channel sum returned by Repro.forward, using a split batch/channel reduction where the first Triton kernel emits the exact post-spatial partial for each `(N,C)` row and a second kernel finalizes the `[480]` channel vector; Inductor currently lowers the decomposed broadcast pointwise hard-swish producer and the dependent spatial-plus-batch reductions through generic reduction scheduling, but bench_oracle measures the full-scope Triton floor only 1.019x faster than compiled code; the fix is BANDWIDTH_BOUND: record this repro as at floor unless a broader bandwidth, launch-overhead, or reduction-template improvement moves both paths together."""
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


# --- Oracle kernel(s) ---

N = 512
C = 480
H = 14
W = 14
HW = H * W
N_BLOCKS = N

NCHW_SHAPE = (N, C, H, W)
NCHW_STRIDE = (C * HW, HW, W, 1)
CHANNEL_11_SHAPE = (1, C, 1, 1)
CHANNEL_11_STRIDE = (C, 1, 1, 1)
CHANNEL_SHAPE = (C,)
CHANNEL_STRIDE = (1,)
NC11_SHAPE = (N, C, 1, 1)
NC11_STRIDE = (C, 1, 1, 1)
SCALAR_SHAPE = ()
SCALAR_STRIDE = ()

REDUCE_BLOCK_HW = 256
REDUCE_BLOCK_C = 4
FINAL_BLOCK_N = 512

CLASSIFICATION = "BANDWIDTH_BOUND"

if triton is not None:

    @triton.jit
    def _spatial_partial_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        grad_ptr,
        gate_ptr,
        full_ptr,
        partial_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        n = tl.program_id(1)
        c = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        hw = tl.arange(0, BLOCK_HW_)
        c_mask = c < C_
        hw_mask = hw < HW_
        offsets = n * C_ * HW_ + c[:, None] * HW_ + hw[None, :]
        mask = c_mask[:, None] & hw_mask[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

        sub_tensor = x - mean[:, None]
        mul_tensor = sub_tensor * invstd[:, None]
        mul_tensor_1 = mul_tensor * weight[:, None]
        add_tensor = mul_tensor_1 + bias[:, None]
        add_tensor_1 = add_tensor + 3.0
        clamp_min_default = tl.where(add_tensor_1 < 0.0, 0.0, add_tensor_1)
        clamp_max_default = tl.where(clamp_min_default > 6.0, 6.0, clamp_min_default)
        mul_tensor_2 = add_tensor * clamp_max_default
        div_tensor = mul_tensor_2 / 6.0
        mul_tensor_3 = grad * div_tensor
        sum_dim_int_list = tl.sum(tl.where(hw_mask[None, :], mul_tensor_3, 0.0), axis=1)

        gate = tl.load(gate_ptr + n * C_ + c, mask=c_mask, other=0.0).to(tl.float32)
        gt_scalar = gate > -3.0
        lt_scalar = gate < 3.0
        bitwise_and_tensor = gt_scalar & lt_scalar
        mul_tensor_4 = sum_dim_int_list * 0.16666666666666666
        full_value = tl.load(full_ptr).to(tl.float32)
        where_self = tl.where(bitwise_and_tensor, mul_tensor_4, full_value)
        tl.store(partial_ptr + n * C_ + c, where_self, mask=c_mask)

    @triton.jit
    def _finalize_channel_sum_kernel(
        partial_ptr,
        out_ptr,
        C_: tl.constexpr,
        N_BLOCKS_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        n_offsets = tl.arange(0, BLOCK_N_)
        c = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        n_mask = n_offsets < N_BLOCKS_
        c_mask = c < C_
        offsets = n_offsets[:, None] * C_ + c[None, :]
        vals = tl.load(
            partial_ptr + offsets,
            mask=n_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        total = tl.sum(vals, axis=0)
        tl.store(out_ptr + c, total, mask=c_mask)


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
        arg249_1,
        arg250_1,
        arg251_1,
        arg83_1,
        arg84_1,
        getitem_66,
        arg254_1,
        full,
    ) = inputs

    arg249_1 = _require_f32_tensor("arg249_1", arg249_1, NCHW_SHAPE, NCHW_STRIDE)
    arg250_1 = _require_f32_tensor("arg250_1", arg250_1, CHANNEL_11_SHAPE, CHANNEL_11_STRIDE)
    arg251_1 = _require_f32_tensor("arg251_1", arg251_1, CHANNEL_11_SHAPE, CHANNEL_11_STRIDE)
    arg83_1 = _require_f32_tensor("arg83_1", arg83_1, CHANNEL_SHAPE, CHANNEL_STRIDE)
    arg84_1 = _require_f32_tensor("arg84_1", arg84_1, CHANNEL_SHAPE, CHANNEL_STRIDE)
    getitem_66 = _require_f32_tensor("getitem_66", getitem_66, NCHW_SHAPE, NCHW_STRIDE)
    arg254_1 = _require_f32_tensor("arg254_1", arg254_1, NC11_SHAPE, NC11_STRIDE)
    full = _require_f32_tensor("full", full, SCALAR_SHAPE, SCALAR_STRIDE)

    device = arg249_1.device
    if any(
        tensor.device != device
        for tensor in (
            arg250_1,
            arg251_1,
            arg83_1,
            arg84_1,
            getitem_66,
            arg254_1,
            full,
        )
    ):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return (
        arg249_1,
        arg250_1,
        arg251_1,
        arg83_1,
        arg84_1,
        getitem_66,
        arg254_1,
        full,
    )


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
        raise RuntimeError("Triton is required for oracle_split_batch_hardswish_sum.py")

    (
        arg249_1,
        arg250_1,
        arg251_1,
        arg83_1,
        arg84_1,
        getitem_66,
        arg254_1,
        full,
    ) = _validate_inputs(inputs)

    partial = torch.empty_strided(
        (N_BLOCKS, C),
        (C, 1),
        device=arg249_1.device,
        dtype=torch.float32,
    )
    output = torch.empty_strided(
        CHANNEL_SHAPE,
        CHANNEL_STRIDE,
        device=arg249_1.device,
        dtype=torch.float32,
    )

    _spatial_partial_kernel[(triton.cdiv(C, REDUCE_BLOCK_C), N)](
        arg249_1,
        arg250_1,
        arg251_1,
        arg83_1,
        arg84_1,
        getitem_66,
        arg254_1,
        full,
        partial,
        C_=C,
        HW_=HW,
        BLOCK_HW_=REDUCE_BLOCK_HW,
        BLOCK_C_=REDUCE_BLOCK_C,
        num_warps=4,
        num_stages=4,
    )
    _finalize_channel_sum_kernel[(triton.cdiv(C, REDUCE_BLOCK_C),)](
        partial,
        output,
        C_=C,
        N_BLOCKS_=N_BLOCKS,
        BLOCK_N_=FINAL_BLOCK_N,
        BLOCK_C_=REDUCE_BLOCK_C,
        num_warps=4,
        num_stages=4,
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
