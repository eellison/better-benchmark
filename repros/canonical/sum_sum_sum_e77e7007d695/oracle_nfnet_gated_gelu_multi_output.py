"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet gated exact-GELU-backward reduction scope from Repro.forward by streaming the shared sigmoid/gate/GELU-gradient pointwise producer once into per-(N,C) spatial summaries and finalizing the scalar loss-gradient sum plus the two [512] channel reductions, whereas Inductor currently schedules the fused producer, global sum, spatial sigmoid-gradient reduction, and add_tensor_3 channel reduction as separate generic regions over materialized intermediates; Inductor cannot do this today because the scheduler does not form one multi-output reduction plan for sibling reductions with different output ranks when one output has a per-(N,C) reduced epilogue before the final channel sum; the fix is SCHEDULER_FUSION: teach Inductor to keep the shared pointwise producer inside a multi-accumulator spatial reduction and lower the scalar and channel finalizers together."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 512
H = 24
W = 24
HW = H * W

SCALE_GETITEM = 0.9805806756909201
GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327
POINTWISE_SCALE = 0.2

SPATIAL_BLOCK_C = 8
SPATIAL_BLOCK_HW = 256
FINAL_BLOCK_C = 16
FINAL_BLOCK_N = 128


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _spatial_summaries_kernel(
        getitem_ptr,
        sigmoid_arg_ptr,
        arg208_ptr,
        scalar_ptr,
        arg193_ptr,
        mul785_ptr,
        scalar_nc_ptr,
        channel_nc_ptr,
        add3_nc_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        SCALE_GETITEM_: tl.constexpr,
        GAMMA_: tl.constexpr,
        RSQRT2_: tl.constexpr,
        NORMAL_PDF_SCALE_: tl.constexpr,
        POINTWISE_SCALE_: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n = tl.program_id(0)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c < C_

        sigmoid_arg = tl.load(sigmoid_arg_ptr + n * C_ + c, mask=c_mask, other=0.0).to(tl.float32)
        sigmoid = tl.sigmoid(sigmoid_arg)
        scalar = tl.load(scalar_ptr).to(tl.float32)

        acc_scalar = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_channel_spatial = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_add3 = tl.zeros((BLOCK_C,), dtype=tl.float32)

        hw_offsets_base = tl.arange(0, BLOCK_HW)
        for hw_base in tl.static_range(0, HW_, BLOCK_HW):
            hw = hw_base + hw_offsets_base
            hw_mask = hw < HW_
            mask = c_mask[:, None] & hw_mask[None, :]
            offsets = n * C_ * HW_ + c[:, None] * HW_ + hw[None, :]

            getitem = tl.load(getitem_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            arg208 = tl.load(arg208_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            arg193 = tl.load(arg193_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            mul785 = tl.load(mul785_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            mul_tensor = getitem * SCALE_GETITEM_
            mul_tensor_1 = mul_tensor * GAMMA_
            mul_tensor_2 = arg208 * sigmoid[:, None]
            mul_tensor_3 = mul_tensor_2 * 2.0
            mul_tensor_4 = mul_tensor_3 * scalar
            mul_tensor_5 = mul_tensor_4 * POINTWISE_SCALE_
            add_tensor = mul_tensor_5 + arg193

            mul_tensor_6 = add_tensor * RSQRT2_
            erf_default = tl.math.erf(mul_tensor_6)
            add_tensor_1 = erf_default + 1.0
            mul_tensor_7 = add_tensor_1 * 0.5
            mul_tensor_8 = add_tensor * add_tensor
            mul_tensor_9 = mul_tensor_8 * -0.5
            exp_default = tl.exp(mul_tensor_9)
            mul_tensor_10 = exp_default * NORMAL_PDF_SCALE_
            mul_tensor_11 = add_tensor * mul_tensor_10
            add_tensor_2 = mul_tensor_7 + mul_tensor_11
            mul_tensor_12 = mul_tensor_1 * add_tensor_2
            add_tensor_3 = mul785 + mul_tensor_12

            mul_tensor_13 = add_tensor_3 * POINTWISE_SCALE_
            mul_tensor_14 = mul_tensor_13 * mul_tensor_3
            mul_tensor_15 = mul_tensor_13 * scalar
            mul_tensor_16 = mul_tensor_15 * 2.0
            mul_tensor_17 = mul_tensor_16 * arg208

            acc_scalar += tl.sum(tl.where(mask, mul_tensor_14, 0.0), axis=1)
            acc_channel_spatial += tl.sum(tl.where(mask, mul_tensor_17, 0.0), axis=1)
            acc_add3 += tl.sum(tl.where(mask, add_tensor_3, 0.0), axis=1)

        sigmoid_grad = sigmoid * (1.0 - sigmoid)
        out_offsets = n * C_ + c
        tl.store(scalar_nc_ptr + out_offsets, acc_scalar, mask=c_mask)
        tl.store(channel_nc_ptr + out_offsets, acc_channel_spatial * sigmoid_grad, mask=c_mask)
        tl.store(add3_nc_ptr + out_offsets, acc_add3, mask=c_mask)

    @triton.jit
    def _finalize_channels_kernel(
        scalar_nc_ptr,
        channel_nc_ptr,
        add3_nc_ptr,
        scalar_blocks_ptr,
        out_channel_ptr,
        out_add3_ptr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        block = tl.program_id(0)
        n = tl.arange(0, BLOCK_N)
        c = block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (n[:, None] < N_) & (c[None, :] < C_)
        offsets = n[:, None] * C_ + c[None, :]

        scalar_vals = tl.load(scalar_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        channel_vals = tl.load(channel_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add3_vals = tl.load(add3_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        scalar_per_channel = tl.sum(scalar_vals, axis=0)
        channel_sum = tl.sum(channel_vals, axis=0)
        add3_sum = tl.sum(add3_vals, axis=0)

        c_mask = c < C_
        tl.store(out_channel_ptr + c, channel_sum, mask=c_mask)
        tl.store(out_add3_ptr + c, add3_sum, mask=c_mask)
        tl.store(scalar_blocks_ptr + block, tl.sum(tl.where(c_mask, scalar_per_channel, 0.0), axis=0))

    @triton.jit
    def _finalize_scalar_kernel(
        scalar_blocks_ptr,
        out_scalar_ptr,
        NUM_BLOCKS: tl.constexpr,
        BLOCKS: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCKS)
        mask = offsets < NUM_BLOCKS
        vals = tl.load(scalar_blocks_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_scalar_ptr, tl.sum(vals, axis=0))


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    getitem_186, arg211_1, arg208_1, arg33_1, arg193_1, mul_785 = inputs
    expected_4d = (N, C, H, W)
    expected_gate = (N, C, 1, 1)
    tensors = (
        ("getitem_186", getitem_186, expected_4d),
        ("arg211_1", arg211_1, expected_gate),
        ("arg208_1", arg208_1, expected_4d),
        ("arg193_1", arg193_1, expected_4d),
        ("mul_785", mul_785, expected_4d),
    )

    for name, tensor, shape in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} expected shape={shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} expected torch.float32, got {tensor.dtype}")
        if not tensor.is_cuda:
            raise RuntimeError(f"{REPRO_ID} expects CUDA inputs")

    if not isinstance(arg33_1, torch.Tensor):
        raise TypeError(f"arg33_1 must be a tensor, got {type(arg33_1)!r}")
    if tuple(arg33_1.shape) != () or arg33_1.dtype != torch.float32 or not arg33_1.is_cuda:
        raise ValueError("arg33_1 must be a CUDA f32 scalar tensor")

    return (
        getitem_186.contiguous(),
        arg211_1.contiguous(),
        arg208_1.contiguous(),
        arg33_1.contiguous(),
        arg193_1.contiguous(),
        mul_785.contiguous(),
    )


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward computation with fused multi-output reductions."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    getitem_186, arg211_1, arg208_1, arg33_1, arg193_1, mul_785 = _validate_inputs(inputs)
    device = getitem_186.device

    scalar_nc = torch.empty((N, C), device=device, dtype=torch.float32)
    channel_nc = torch.empty((N, C), device=device, dtype=torch.float32)
    add3_nc = torch.empty((N, C), device=device, dtype=torch.float32)

    _spatial_summaries_kernel[(N, triton.cdiv(C, SPATIAL_BLOCK_C))](
        getitem_186,
        arg211_1,
        arg208_1,
        arg33_1,
        arg193_1,
        mul_785,
        scalar_nc,
        channel_nc,
        add3_nc,
        C_=C,
        HW_=HW,
        SCALE_GETITEM_=SCALE_GETITEM,
        GAMMA_=GAMMA,
        RSQRT2_=RSQRT2,
        NORMAL_PDF_SCALE_=NORMAL_PDF_SCALE,
        POINTWISE_SCALE_=POINTWISE_SCALE,
        BLOCK_C=SPATIAL_BLOCK_C,
        BLOCK_HW=SPATIAL_BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )

    out_scalar = torch.empty((), device=device, dtype=torch.float32)
    out_channel = torch.empty((C,), device=device, dtype=torch.float32)
    out_add3 = torch.empty((C,), device=device, dtype=torch.float32)
    num_final_blocks = triton.cdiv(C, FINAL_BLOCK_C)
    scalar_blocks = torch.empty((num_final_blocks,), device=device, dtype=torch.float32)

    _finalize_channels_kernel[(num_final_blocks,)](
        scalar_nc,
        channel_nc,
        add3_nc,
        scalar_blocks,
        out_channel,
        out_add3,
        N_=N,
        C_=C,
        BLOCK_N=FINAL_BLOCK_N,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _finalize_scalar_kernel[(1,)](
        scalar_blocks,
        out_scalar,
        NUM_BLOCKS=num_final_blocks,
        BLOCKS=triton.next_power_of_2(num_final_blocks),
        num_warps=4,
        num_stages=3,
    )

    return out_scalar, out_channel, out_add3


def main() -> None:
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

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
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
