"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete `sum_sum_sum_6f9b333ed892` NFNet backward fragment by streaming the avg_pool2d_backward expansion, broadcast sigmoid gate, exact GELU-derivative pointwise chain, scalar sum, sigmoid-gradient channel reduction, and sibling `mul_tensor_12` channel reduction through one Triton producer with three per-`(N,C)` accumulators, whereas Inductor currently materializes `mul_tensor_12`, separately materializes the spatial reduction feeding `sigmoid(arg188_1) * (1 - sigmoid(arg188_1))`, and launches separate reductions for the scalar and sibling channel sum; Inductor cannot do this today because its algebraic simplifier/reduction codegen does not flatten the linear `sum([2,3]) -> sigmoid-derivative multiply -> sum([0,2,3])` chain into the same multi-output reduction as the scalar and sibling channel reductions; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to reassociate this dependent reduction pattern and emit one multi-accumulator channel-reduction template over the shared fused producer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None
    libdevice = None

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
C = 256
H = 48
W = 48
HW = H * W
POOL_H = 24
POOL_W = 24
POOL_HW = POOL_H * POOL_W

AVG_POOL_BACKWARD_SCALE = 4.0
SCALE_GETITEM = 0.9805806756909201
GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327
POINTWISE_SCALE = 0.2

SPATIAL_BLOCK_C = 4
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
        getitem_207_ptr,
        getitem_204_ptr,
        gate_arg_ptr,
        arg185_ptr,
        scalar_ptr,
        arg170_ptr,
        scalar_nc_ptr,
        gate_nc_ptr,
        main_nc_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        W_: tl.constexpr,
        POOL_HW_: tl.constexpr,
        POOL_W_: tl.constexpr,
        AVG_POOL_BACKWARD_SCALE_: tl.constexpr,
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

        gate_arg = tl.load(gate_arg_ptr + n * C_ + c, mask=c_mask, other=0.0).to(tl.float32)
        # Inductor lowers captured aten.sigmoid to tl.sigmoid for this repro.
        sigmoid_default = tl.sigmoid(gate_arg)
        scalar = tl.load(scalar_ptr).to(tl.float32)

        acc_scalar = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_gate = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_main = tl.zeros((BLOCK_C,), dtype=tl.float32)

        hw_offsets_base = tl.arange(0, BLOCK_HW)
        for hw_base in tl.static_range(0, HW_, BLOCK_HW):
            hw = hw_base + hw_offsets_base
            hw_mask = hw < HW_
            h = hw // W_
            w = hw - h * W_
            pool_hw = (h // 2) * POOL_W_ + (w // 2)

            mask = c_mask[:, None] & hw_mask[None, :]
            dense_offsets = n * C_ * HW_ + c[:, None] * HW_ + hw[None, :]
            pool_offsets = n * C_ * POOL_HW_ + c[:, None] * POOL_HW_ + pool_hw[None, :]

            getitem_204 = tl.load(getitem_204_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
            pool_grad = tl.load(getitem_207_ptr + pool_offsets, mask=mask, other=0.0).to(tl.float32)
            arg185 = tl.load(arg185_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
            arg170 = tl.load(arg170_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)

            avg_pool2d_backward_default = pool_grad / AVG_POOL_BACKWARD_SCALE_
            add_tensor = getitem_204 + avg_pool2d_backward_default
            mul_tensor = add_tensor * SCALE_GETITEM_
            mul_tensor_1 = mul_tensor * GAMMA_

            mul_tensor_2 = arg185 * sigmoid_default[:, None]
            mul_tensor_3 = mul_tensor_2 * 2.0
            mul_tensor_4 = mul_tensor_3 * scalar
            mul_tensor_5 = mul_tensor_4 * POINTWISE_SCALE_
            add_tensor_1 = mul_tensor_5 + arg170

            mul_tensor_6 = add_tensor_1 * RSQRT2_
            erf_default = libdevice.erf(mul_tensor_6)
            add_tensor_2 = erf_default + 1.0
            mul_tensor_7 = add_tensor_2 * 0.5
            mul_tensor_8 = add_tensor_1 * add_tensor_1
            mul_tensor_9 = mul_tensor_8 * -0.5
            exp_default = libdevice.exp(mul_tensor_9)
            mul_tensor_10 = exp_default * NORMAL_PDF_SCALE_
            mul_tensor_11 = add_tensor_1 * mul_tensor_10
            add_tensor_3 = mul_tensor_7 + mul_tensor_11

            mul_tensor_12 = mul_tensor_1 * add_tensor_3
            mul_tensor_13 = mul_tensor_12 * POINTWISE_SCALE_
            mul_tensor_14 = mul_tensor_13 * mul_tensor_3
            mul_tensor_15 = mul_tensor_13 * scalar
            mul_tensor_16 = mul_tensor_15 * 2.0
            mul_tensor_17 = mul_tensor_16 * arg185

            acc_scalar += tl.sum(tl.where(mask, mul_tensor_14, 0.0), axis=1)
            acc_gate += tl.sum(tl.where(mask, mul_tensor_17, 0.0), axis=1)
            acc_main += tl.sum(tl.where(mask, mul_tensor_12, 0.0), axis=1)

        sub_tensor = 1.0 - sigmoid_default
        mul_tensor_18 = sigmoid_default * sub_tensor
        out_offsets = n * C_ + c
        tl.store(scalar_nc_ptr + out_offsets, acc_scalar, mask=c_mask)
        tl.store(gate_nc_ptr + out_offsets, acc_gate * mul_tensor_18, mask=c_mask)
        tl.store(main_nc_ptr + out_offsets, acc_main, mask=c_mask)

    @triton.jit
    def _finalize_channels_kernel(
        scalar_nc_ptr,
        gate_nc_ptr,
        main_nc_ptr,
        scalar_blocks_ptr,
        out_gate_ptr,
        out_main_ptr,
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
        gate_vals = tl.load(gate_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        main_vals = tl.load(main_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        scalar_per_channel = tl.sum(scalar_vals, axis=0)
        gate_sum = tl.sum(gate_vals, axis=0)
        main_sum = tl.sum(main_vals, axis=0)

        c_mask = c < C_
        tl.store(out_gate_ptr + c, gate_sum, mask=c_mask)
        tl.store(out_main_ptr + c, main_sum, mask=c_mask)
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


def _expect_tensor(name: str, value: Any, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} expected shape={shape}, got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} expected torch.float32, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{REPRO_ID} expects CUDA inputs")
    if not value.is_contiguous():
        raise ValueError(f"{name} expected captured contiguous layout, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    getitem_207, arg189_1, getitem_204, arg188_1, arg185_1, arg20_1, arg170_1 = inputs
    _expect_tensor("getitem_207", getitem_207, (N, C, POOL_H, POOL_W))
    _expect_tensor("arg189_1", arg189_1, (N, C, H, W))
    _expect_tensor("getitem_204", getitem_204, (N, C, H, W))
    _expect_tensor("arg188_1", arg188_1, (N, C, 1, 1))
    _expect_tensor("arg185_1", arg185_1, (N, C, H, W))
    _expect_tensor("arg170_1", arg170_1, (N, C, H, W))
    if not isinstance(arg20_1, torch.Tensor):
        raise TypeError(f"arg20_1 must be a tensor, got {type(arg20_1)!r}")
    if tuple(arg20_1.shape) != () or arg20_1.dtype != torch.float32 or not arg20_1.is_cuda:
        raise ValueError("arg20_1 must be a CUDA f32 scalar tensor")

    # avg_pool2d_backward.default reads arg189_1 only for the captured input shape.
    return getitem_207, getitem_204, arg188_1, arg185_1, arg20_1, arg170_1


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward computation with fused multi-output reductions."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    getitem_207, getitem_204, arg188_1, arg185_1, arg20_1, arg170_1 = _validate_inputs(inputs)
    device = getitem_207.device

    scalar_nc = torch.empty((N, C), device=device, dtype=torch.float32)
    gate_nc = torch.empty((N, C), device=device, dtype=torch.float32)
    main_nc = torch.empty((N, C), device=device, dtype=torch.float32)

    _spatial_summaries_kernel[(N, triton.cdiv(C, SPATIAL_BLOCK_C))](
        getitem_207,
        getitem_204,
        arg188_1,
        arg185_1,
        arg20_1,
        arg170_1,
        scalar_nc,
        gate_nc,
        main_nc,
        C_=C,
        HW_=HW,
        W_=W,
        POOL_HW_=POOL_HW,
        POOL_W_=POOL_W,
        AVG_POOL_BACKWARD_SCALE_=AVG_POOL_BACKWARD_SCALE,
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
    out_gate = torch.empty((C,), device=device, dtype=torch.float32)
    out_main = torch.empty((C,), device=device, dtype=torch.float32)
    num_final_blocks = triton.cdiv(C, FINAL_BLOCK_C)
    scalar_blocks = torch.empty((num_final_blocks,), device=device, dtype=torch.float32)

    _finalize_channels_kernel[(num_final_blocks,)](
        scalar_nc,
        gate_nc,
        main_nc,
        scalar_blocks,
        out_gate,
        out_main,
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

    return out_scalar, out_gate, out_main


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
