"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full MobileNetV2 adaptive-average-pool backward, hard-relu6 derivative, and batch-norm-backward return tuple directly from the original `[128, 1280]` pooled gradient and strided channels-last `[128, 1280, 7, 7]` activation without materializing the zero-fill `as_strided_scatter -> as_strided -> expand -> div` pool-gradient tensor, whereas Inductor lowers that structured scatter/expand producer as ordinary tensor work and schedules the hard-relu6 mask, two sibling channel reductions, and dependent full pointwise BN-backward epilogue as separate generic consumers; Inductor cannot do this today because scheduler/codegen does not model zero-fill view/as_strided scatter followed by broadcasted average-pool backward as a structured scatter-reduce feeding both reductions and a layout-preserving epilogue; the fix is SCATTER_REDUCE: add a structured average-pool-backward scatter/expand lowering that tiles the source-space pooled gradient with the activation layout, accumulates the sibling channel reductions, and emits the full BN-backward tuple from one fused template."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - allows CPU-only syntax checks.
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

REPRO_ID = "sum_sum_e75c5e897993"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_mobilenetv2_100_train_6cf4583b_strided"

N = 128
C = 1280
H = 7
W = 7
HW = H * W
N_HW = N * HW
INV_HW = 1.0 / HW
REDUCTION_SCALE = 1.0 / (N * HW)
BLOCK_HW = 64
BLOCK_N = 128
BLOCK_C = 64

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]



def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        config = next(iter(configs.values()))
        config = {
            "inputs": [
                {**spec, "device": str(device)}
                if isinstance(spec, dict) and spec.get("kind") == "tensor"
                else spec
                for spec in config["inputs"]
            ]
        }
        inputs = make_inputs_from_config(config)
    else:
        module = _load_repro_module()
        inputs = module.make_inputs()

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def oracle_torch(
    mm: torch.Tensor,
    convolution_51: torch.Tensor,
    getitem_103: torch.Tensor,
    rsqrt_51: torch.Tensor,
    primals_312: torch.Tensor,
    primals_313: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-scope source-space average-pool-backward + hard-relu6 + BN backward."""
    mean = getitem_103.reshape(C)
    invstd = rsqrt_51.reshape(C)
    centered = convolution_51 - mean[None, :, None, None]
    affine = (
        centered
        * invstd[None, :, None, None]
        * primals_312[None, :, None, None]
        + primals_313[None, :, None, None]
    )
    pool_grad = mm[:, :, None, None] * INV_HW
    grad_bn_out = torch.where((affine <= 0.0) | (affine >= 6.0), 0.0, pool_grad)

    grad_sum = grad_bn_out.sum(dim=(0, 2, 3))
    centered_grad_sum = (grad_bn_out * centered).sum(dim=(0, 2, 3))
    mean_term = grad_sum * REDUCTION_SCALE
    var_term = centered_grad_sum * REDUCTION_SCALE * invstd * invstd
    input_scale = invstd * primals_312

    out0_value = (
        grad_bn_out
        - centered * var_term[None, :, None, None]
        - mean_term[None, :, None, None]
    ) * input_scale[None, :, None, None]
    out0 = torch.empty_strided(
        tuple(convolution_51.shape),
        tuple(convolution_51.stride()),
        device=convolution_51.device,
        dtype=convolution_51.dtype,
    )
    out0.copy_(out0_value)
    out1 = centered_grad_sum * invstd
    return out0, out1


if triton is not None:

    @triton.jit
    def _avgpool_hrelu_bn_reduce_per_sample_kernel(
        mm_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        partial0_ptr,
        partial1_ptr,
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
        mm_vals = tl.load(mm_ptr + n_idx * C_ + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        channel_weight = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        channel_bias = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)

        centered = x_vals - mean[None, :]
        affine = centered * invstd[None, :] * channel_weight[None, :] + channel_bias[None, :]
        active = (affine > 0.0) & (affine < 6.0) & mask
        grad_bn_out = tl.where(active, mm_vals[None, :] * INV_HW_, 0.0)
        sum0 = tl.sum(grad_bn_out, axis=0)
        sum1 = tl.sum(grad_bn_out * centered, axis=0)

        tl.store(partial0_ptr + n_idx * C_ + offs_c, sum0, mask=c_mask)
        tl.store(partial1_ptr + n_idx * C_ + offs_c, sum1, mask=c_mask)

    @triton.jit
    def _avgpool_hrelu_bn_finalize_stats_kernel(
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
    def _avgpool_hrelu_bn_epilogue_kernel(
        mm_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        stats_ptr,
        out0_ptr,
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
        mm_vals = tl.load(mm_ptr + n_idx * C_ + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        channel_weight = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        channel_bias = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        mean_term = tl.load(stats_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        var_term = tl.load(stats_ptr + C_ + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        input_scale = tl.load(stats_ptr + 2 * C_ + offs_c, mask=c_mask, other=0.0).to(tl.float32)

        centered = x_vals - mean[None, :]
        affine = centered * invstd[None, :] * channel_weight[None, :] + channel_bias[None, :]
        active = (affine > 0.0) & (affine < 6.0) & mask
        grad_bn_out = tl.where(active, mm_vals[None, :] * INV_HW_, 0.0)
        out_vals = (
            grad_bn_out
            - centered * var_term[None, :]
            - mean_term[None, :]
        ) * input_scale[None, :]

        tl.store(out0_ptr + out_offsets, out_vals, mask=mask)


def oracle_triton(
    mm: torch.Tensor,
    convolution_51: torch.Tensor,
    getitem_103: torch.Tensor,
    rsqrt_51: torch.Tensor,
    primals_312: torch.Tensor,
    primals_313: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    out0 = torch.empty_strided(
        tuple(convolution_51.shape),
        tuple(convolution_51.stride()),
        device=convolution_51.device,
        dtype=convolution_51.dtype,
    )
    out1 = torch.empty_like(primals_312)
    partial0 = torch.empty((N, C), device=convolution_51.device, dtype=torch.float32)
    partial1 = torch.empty((N, C), device=convolution_51.device, dtype=torch.float32)
    stats = torch.empty((3, C), device=convolution_51.device, dtype=torch.float32)

    grid = (triton.cdiv(C, BLOCK_C), N)
    _avgpool_hrelu_bn_reduce_per_sample_kernel[grid](
        mm,
        convolution_51,
        getitem_103,
        rsqrt_51,
        primals_312,
        primals_313,
        partial0,
        partial1,
        x_stride_n=convolution_51.stride(0),
        x_stride_c=convolution_51.stride(1),
        x_stride_h=convolution_51.stride(2),
        x_stride_w=convolution_51.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        INV_HW_=INV_HW,
        BLOCK_HW_=BLOCK_HW,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )
    _avgpool_hrelu_bn_finalize_stats_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial0,
        partial1,
        rsqrt_51,
        primals_312,
        stats,
        out1,
        C_=C,
        N_=N,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_N_=BLOCK_N,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )
    _avgpool_hrelu_bn_epilogue_kernel[grid](
        mm,
        convolution_51,
        getitem_103,
        rsqrt_51,
        primals_312,
        primals_313,
        stats,
        out0,
        x_stride_n=convolution_51.stride(0),
        x_stride_c=convolution_51.stride(1),
        x_stride_h=convolution_51.stride(2),
        x_stride_w=convolution_51.stride(3),
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


def oracle_structured_pool_upsample_reduce(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    mm = inputs[0]
    if not isinstance(mm, torch.Tensor):
        raise TypeError("first input must be a tensor")
    if impl == "auto":
        impl = "triton" if mm.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


class OracleModule(torch.nn.Module):
    def __init__(self, impl: str = "torch") -> None:
        super().__init__()
        self.impl = impl

    def forward(self, *inputs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return oracle_structured_pool_upsample_reduce(*inputs, impl=self.impl)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    return model(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(max(1, warmup)):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def oracle_forward(inputs):
    return oracle_triton(*inputs)


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
