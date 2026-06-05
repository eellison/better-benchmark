"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full MobileNetV3 average-pool-backward, hard-sigmoid derivative mask, and batch-norm-backward return tuple directly from the pooled `[512, 960, 1, 1]` gradient and `[512, 960, 7, 7]` activation, whereas Inductor materializes the zero-fill `as_strided_scatter -> as_strided -> expand -> div` pool-gradient tensor before scheduling the mask, sibling channel reductions, and dependent pointwise epilogue as generic consumers; Inductor cannot do this today because scheduler/codegen does not represent this structured scatter/expand producer as a source-space gather-mask-reduce that can feed both reductions and the full tensor epilogue; the fix is SCATTER_REDUCE: add a structured average-pool-backward scatter-reduce lowering that tiles the pooled-gradient source with the activation layout, accumulates the channel sums, and emits the full BN-backward tuple without the scatter intermediate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - allows CPU-only syntax checks.
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

N = 512
C = 960
H = 7
W = 7
HW = H * W
N_HW = N * HW
INV_HW = 1.0 / HW
REDUCTION_SCALE = 1.0 / (N * HW)
BLOCK_NHW = 32768
BLOCK_HW = 64
BLOCK_N = 512
BLOCK_C = 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def oracle_torch(inputs):
    """Reference implementation with the scatter/expand algebraically removed."""
    (
        pooled_grad,
        x,
        mean_4d,
        invstd_4d,
        weight,
        bias,
        full,
        _shape_param_0,
    ) = inputs

    mean = mean_4d.reshape(C)
    invstd = invstd_4d.reshape(C)
    centered = x - mean[None, :, None, None]
    affine = centered * invstd[None, :, None, None] * weight[None, :, None, None]
    affine = affine + bias[None, :, None, None]
    pool_grad = pooled_grad.reshape(N, C)[:, :, None, None] * INV_HW
    middle = pool_grad * (affine / 3.0 + 0.5)
    grad_bn_out = torch.where(affine < 3.0, middle, pool_grad)
    grad_bn_out = torch.where(affine <= -3.0, full, grad_bn_out)

    grad_sum = grad_bn_out.sum(dim=(0, 2, 3))
    centered_grad_sum = (grad_bn_out * centered).sum(dim=(0, 2, 3))
    mean_term = grad_sum * REDUCTION_SCALE
    var_term = centered_grad_sum * REDUCTION_SCALE * invstd * invstd
    input_scale = invstd * weight

    out0_value = (
        grad_bn_out
        - centered * var_term[None, :, None, None]
        - mean_term[None, :, None, None]
    ) * input_scale[None, :, None, None]
    out0 = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    out0.copy_(out0_value)
    out1 = centered_grad_sum * invstd
    return out0, out1


if triton is not None:

    @triton.jit
    def _avgpool_hsigmoid_bn_onepass_channel_kernel(
        pooled_grad_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        out0_ptr,
        out1_ptr,
        pooled_stride_n: tl.constexpr,
        pooled_stride_c: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_w: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_NHW_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_NHW_)
        mask = offsets < N_HW_

        n_idx = offsets // HW_
        hw_idx = offsets - n_idx * HW_
        h_idx = hw_idx // W_
        w_idx = hw_idx - h_idx * W_
        x_offsets = (
            n_idx * x_stride_n
            + channel * x_stride_c
            + h_idx * x_stride_h
            + w_idx * x_stride_w
        )
        out_offsets = (
            n_idx * out_stride_n
            + channel * out_stride_c
            + h_idx * out_stride_h
            + w_idx * out_stride_w
        )

        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        pool_vals = tl.load(
            pooled_grad_ptr + n_idx * pooled_stride_n + channel * pooled_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32) * INV_HW_
        mean = tl.load(mean_ptr + channel).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        channel_weight = tl.load(weight_ptr + channel).to(tl.float32)
        channel_bias = tl.load(bias_ptr + channel).to(tl.float32)
        full_val = tl.load(full_ptr).to(tl.float32)

        centered = x_vals - mean
        affine = centered * invstd * channel_weight + channel_bias
        middle = pool_vals * (affine / 3.0 + 0.5)
        grad_bn_out = tl.where(affine < 3.0, middle, pool_vals)
        grad_bn_out = tl.where(affine <= -3.0, full_val, grad_bn_out)
        grad_bn_out = tl.where(mask, grad_bn_out, 0.0)

        sum0 = tl.sum(grad_bn_out, axis=0)
        sum1 = tl.sum(grad_bn_out * centered, axis=0)
        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * invstd * invstd
        input_scale = invstd * channel_weight
        out_vals = (grad_bn_out - centered * var_term - mean_term) * input_scale

        tl.store(out0_ptr + out_offsets, out_vals, mask=mask)
        tl.store(out1_ptr + channel, sum1 * invstd)

    @triton.jit
    def _avgpool_hsigmoid_bn_reduce_per_sample_kernel(
        pooled_grad_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partial0_ptr,
        partial1_ptr,
        pooled_stride_n: tl.constexpr,
        pooled_stride_c: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        n_idx = tl.program_id(1)
        offs_c = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        offs_hw = tl.arange(0, BLOCK_HW_)
        c_mask = offs_c < C_
        hw_mask = offs_hw < HW_

        h_idx = offs_hw // W_
        w_idx = offs_hw - h_idx * W_
        x_offsets = (
            n_idx * x_stride_n
            + offs_c[None, :] * x_stride_c
            + h_idx[:, None] * x_stride_h
            + w_idx[:, None] * x_stride_w
        )
        mask = hw_mask[:, None] & c_mask[None, :]

        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        pool_vals = tl.load(
            pooled_grad_ptr + n_idx * pooled_stride_n + offs_c * pooled_stride_c,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32) * INV_HW_
        mean = tl.load(mean_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        channel_weight = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        channel_bias = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        full_val = tl.load(full_ptr).to(tl.float32)

        centered = x_vals - mean[None, :]
        affine = centered * invstd[None, :] * channel_weight[None, :] + channel_bias[None, :]
        middle = pool_vals[None, :] * (affine / 3.0 + 0.5)
        grad_bn_out = tl.where(affine < 3.0, middle, pool_vals[None, :])
        grad_bn_out = tl.where(affine <= -3.0, full_val, grad_bn_out)
        grad_bn_out = tl.where(mask, grad_bn_out, 0.0)

        sum0 = tl.sum(grad_bn_out, axis=0)
        sum1 = tl.sum(grad_bn_out * centered, axis=0)

        tl.store(partial0_ptr + n_idx * C_ + offs_c, sum0, mask=c_mask)
        tl.store(partial1_ptr + n_idx * C_ + offs_c, sum1, mask=c_mask)

    @triton.jit
    def _avgpool_hsigmoid_bn_finalize_stats_kernel(
        partial0_ptr,
        partial1_ptr,
        invstd_ptr,
        weight_ptr,
        stats_ptr,
        out1_ptr,
        C_: tl.constexpr,
        N_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        offs_c = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        offs_n = tl.arange(0, BLOCK_N_)
        c_mask = offs_c < C_
        n_mask = offs_n < N_
        offsets = offs_n[:, None] * C_ + offs_c[None, :]
        mask = n_mask[:, None] & c_mask[None, :]

        partial0 = tl.load(partial0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        partial1 = tl.load(partial1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum0 = tl.sum(partial0, axis=0)
        sum1 = tl.sum(partial1, axis=0)

        invstd = tl.load(invstd_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        channel_weight = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * invstd * invstd
        input_scale = invstd * channel_weight

        tl.store(stats_ptr + offs_c, mean_term, mask=c_mask)
        tl.store(stats_ptr + C_ + offs_c, var_term, mask=c_mask)
        tl.store(stats_ptr + 2 * C_ + offs_c, input_scale, mask=c_mask)
        tl.store(out1_ptr + offs_c, sum1 * invstd, mask=c_mask)

    @triton.jit
    def _avgpool_hsigmoid_bn_epilogue_kernel(
        pooled_grad_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        stats_ptr,
        out0_ptr,
        pooled_stride_n: tl.constexpr,
        pooled_stride_c: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_w: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        n_idx = tl.program_id(1)
        offs_c = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        offs_hw = tl.arange(0, BLOCK_HW_)
        c_mask = offs_c < C_
        hw_mask = offs_hw < HW_

        h_idx = offs_hw // W_
        w_idx = offs_hw - h_idx * W_
        x_offsets = (
            n_idx * x_stride_n
            + offs_c[None, :] * x_stride_c
            + h_idx[:, None] * x_stride_h
            + w_idx[:, None] * x_stride_w
        )
        out_offsets = (
            n_idx * out_stride_n
            + offs_c[None, :] * out_stride_c
            + h_idx[:, None] * out_stride_h
            + w_idx[:, None] * out_stride_w
        )
        mask = hw_mask[:, None] & c_mask[None, :]

        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        pool_vals = tl.load(
            pooled_grad_ptr + n_idx * pooled_stride_n + offs_c * pooled_stride_c,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32) * INV_HW_
        mean = tl.load(mean_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        channel_weight = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        channel_bias = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        full_val = tl.load(full_ptr).to(tl.float32)
        mean_term = tl.load(stats_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        var_term = tl.load(stats_ptr + C_ + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        input_scale = tl.load(stats_ptr + 2 * C_ + offs_c, mask=c_mask, other=0.0).to(tl.float32)

        centered = x_vals - mean[None, :]
        affine = centered * invstd[None, :] * channel_weight[None, :] + channel_bias[None, :]
        middle = pool_vals[None, :] * (affine / 3.0 + 0.5)
        grad_bn_out = tl.where(affine < 3.0, middle, pool_vals[None, :])
        grad_bn_out = tl.where(affine <= -3.0, full_val, grad_bn_out)
        out_vals = (
            grad_bn_out
            - centered * var_term[None, :]
            - mean_term[None, :]
        ) * input_scale[None, :]

        tl.store(out0_ptr + out_offsets, out_vals, mask=mask)


def _validate_inputs(inputs):
    if len(inputs) != 8:
        raise ValueError(f"expected 8 inputs, got {len(inputs)}")
    tensors = inputs[:7]
    if not all(isinstance(inp, torch.Tensor) for inp in tensors):
        raise ValueError("first seven inputs must be tensors")

    expected_shapes = (
        (N, C, 1, 1),
        (N, C, H, W),
        (1, C, 1, 1),
        (1, C, 1, 1),
        (C,),
        (C,),
        (),
    )
    actual_shapes = tuple(tuple(inp.shape) for inp in tensors)
    if actual_shapes != expected_shapes:
        raise ValueError(f"unsupported input shapes: expected {expected_shapes}, got {actual_shapes}")

    if not all(inp.dtype is torch.float32 for inp in tensors):
        actual_dtypes = tuple(inp.dtype for inp in tensors)
        raise ValueError(f"unsupported input dtypes: expected all torch.float32, got {actual_dtypes}")


def oracle_triton(inputs):
    """Run the full-scope Triton oracle."""
    if triton is None:
        raise RuntimeError("triton is not available")
    _validate_inputs(inputs)
    (
        pooled_grad,
        x,
        mean_4d,
        invstd_4d,
        weight,
        bias,
        full,
        _shape_param_0,
    ) = inputs
    if pooled_grad.device.type != "cuda":
        return oracle_torch(inputs)

    out0 = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    out1 = torch.empty_like(weight)
    _avgpool_hsigmoid_bn_onepass_channel_kernel[(C,)](
        pooled_grad,
        x,
        mean_4d,
        invstd_4d,
        weight,
        bias,
        full,
        out0,
        out1,
        pooled_stride_n=pooled_grad.stride(0),
        pooled_stride_c=pooled_grad.stride(1),
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        out_stride_n=out0.stride(0),
        out_stride_c=out0.stride(1),
        out_stride_h=out0.stride(2),
        out_stride_w=out0.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        N_HW_=N_HW,
        INV_HW_=INV_HW,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_NHW_=BLOCK_NHW,
        num_warps=16,
    )
    return out0, out1


def oracle_triton_split(inputs):
    """Run the three-kernel fallback oracle."""
    if triton is None:
        raise RuntimeError("triton is not available")
    _validate_inputs(inputs)
    (
        pooled_grad,
        x,
        mean_4d,
        invstd_4d,
        weight,
        bias,
        full,
        _shape_param_0,
    ) = inputs
    if pooled_grad.device.type != "cuda":
        return oracle_torch(inputs)

    out0 = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    out1 = torch.empty_like(weight)
    partial0 = torch.empty((N, C), device=x.device, dtype=torch.float32)
    partial1 = torch.empty((N, C), device=x.device, dtype=torch.float32)
    stats = torch.empty((3, C), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(C, BLOCK_C), N)
    _avgpool_hsigmoid_bn_reduce_per_sample_kernel[grid](
        pooled_grad,
        x,
        mean_4d,
        invstd_4d,
        weight,
        bias,
        full,
        partial0,
        partial1,
        pooled_stride_n=pooled_grad.stride(0),
        pooled_stride_c=pooled_grad.stride(1),
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        INV_HW_=INV_HW,
        BLOCK_HW_=BLOCK_HW,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )
    _avgpool_hsigmoid_bn_finalize_stats_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial0,
        partial1,
        invstd_4d,
        weight,
        stats,
        out1,
        C_=C,
        N_=N,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_N_=BLOCK_N,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )
    _avgpool_hsigmoid_bn_epilogue_kernel[grid](
        pooled_grad,
        x,
        mean_4d,
        invstd_4d,
        weight,
        bias,
        full,
        stats,
        out0,
        pooled_stride_n=pooled_grad.stride(0),
        pooled_stride_c=pooled_grad.stride(1),
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        out_stride_n=out0.stride(0),
        out_stride_c=out0.stride(1),
        out_stride_h=out0.stride(2),
        out_stride_w=out0.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        INV_HW_=INV_HW,
        BLOCK_HW_=BLOCK_HW,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )
    return out0, out1


def oracle_forward(inputs):
    """Run the oracle computation for the exact Repro() input tuple."""
    return oracle_triton(inputs)


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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
