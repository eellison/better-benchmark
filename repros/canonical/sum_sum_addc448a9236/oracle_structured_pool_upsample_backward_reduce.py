"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete two-output EfficientNet adaptive-average-pool backward feeding SiLU-gated batch-norm backward by mapping the zero-fill/as_strided_scatter plus expand/div pool-gradient reconstruction directly back to `mm`, accumulating both per-channel reductions, and emitting the dependent contiguous input-gradient tensor plus vector scale-gradient output, whereas Inductor currently lowers the pool-gradient reconstruction, sigmoid/SiLU derivative, sibling channel reductions, and full-tensor BN-backward epilogue as separate generic kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not model the zero-fill structured scatter/expand as an average-pool-backward scatter-reduce producer that can feed both channel reductions and a required full side-output store; the fix is SCATTER_REDUCE: add a structured average-pool-backward scatter-reduce lowering that maps each pooled-gradient source directly into the fused SiLU/BN reduction producer and emits the dependent BN-backward output for the complete return tuple."""
from __future__ import annotations

import argparse
import sys
import importlib.util
import math
import time
from collections.abc import Callable
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - allows py_compile without Triton.
    triton = None
    tl = None



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

REPRO_ID = "sum_sum_addc448a9236"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_tf_efficientnet_b0_train_6cf4583b"

N = 128
C = 1280
H = 7
W = 7
HW = H * W
N_HW = N * HW
INV_HW = 1.0 / HW
REDUCTION_SCALE = 1.0 / N_HW
BLOCK_M = 128
BLOCK_C = 32
BLOCK_TILES = 64
NUM_M_TILES = triton.cdiv(N_HW, BLOCK_M) if triton is not None else math.ceil(N_HW / BLOCK_M)



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


if triton is not None:

    @triton.jit
    def _partial_silu_bn_sums_kernel(
        mm_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        partial0_ptr,
        partial1_ptr,
        mm_stride_n: tl.constexpr,
        mm_stride_c: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // HW_
        spatial = m_offsets - n_idx * HW_
        h_idx = spatial // W_
        w_idx = spatial - h_idx * W_

        mm_offsets = n_idx[:, None] * mm_stride_n + c_offsets[None, :] * mm_stride_c
        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + h_idx[:, None] * x_stride_h
            + w_idx[:, None] * x_stride_w
        )

        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets * weight_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets * bias_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        pooled_grad = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32)

        centered = x - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        sigmoid = 1.0 / (tl.exp(-affine) + 1.0)
        grad = (pooled_grad * INV_HW_) * sigmoid * (affine * (1.0 - sigmoid) + 1.0)
        grad = tl.where(active, grad, 0.0)
        centered = tl.where(active, centered, 0.0)

        sum_grad = tl.sum(grad, axis=0)
        sum_centered = tl.sum(grad * centered, axis=0)
        partial_offsets = pid_m * C_ + c_offsets
        partial_mask = c_offsets < C_
        tl.store(partial0_ptr + partial_offsets, sum_grad, mask=partial_mask)
        tl.store(partial1_ptr + partial_offsets, sum_centered, mask=partial_mask)

    @triton.jit
    def _finalize_silu_bn_sums_kernel(
        partial0_ptr,
        partial1_ptr,
        invstd_ptr,
        sum0_ptr,
        sum1_ptr,
        out1_ptr,
        invstd_stride_c: tl.constexpr,
        C_: tl.constexpr,
        NUM_M_TILES_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        tile_offsets = tl.arange(0, BLOCK_TILES_)
        active = (tile_offsets[:, None] < NUM_M_TILES_) & (c_offsets[None, :] < C_)
        partial_offsets = tile_offsets[:, None] * C_ + c_offsets[None, :]

        sum0_vals = tl.load(partial0_ptr + partial_offsets, mask=active, other=0.0).to(tl.float32)
        sum1_vals = tl.load(partial1_ptr + partial_offsets, mask=active, other=0.0).to(tl.float32)
        sum0 = tl.sum(sum0_vals, axis=0)
        sum1 = tl.sum(sum1_vals, axis=0)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)

        mask = c_offsets < C_
        tl.store(sum0_ptr + c_offsets, sum0, mask=mask)
        tl.store(sum1_ptr + c_offsets, sum1, mask=mask)
        tl.store(out1_ptr + c_offsets, sum1 * invstd, mask=mask)

    @triton.jit
    def _silu_bn_input_grad_kernel(
        mm_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        sum0_ptr,
        sum1_ptr,
        out0_ptr,
        mm_stride_n: tl.constexpr,
        mm_stride_c: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
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
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // HW_
        spatial = m_offsets - n_idx * HW_
        h_idx = spatial // W_
        w_idx = spatial - h_idx * W_

        mm_offsets = n_idx[:, None] * mm_stride_n + c_offsets[None, :] * mm_stride_c
        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + h_idx[:, None] * x_stride_h
            + w_idx[:, None] * x_stride_w
        )
        out_offsets = (
            n_idx[:, None] * out_stride_n
            + c_offsets[None, :] * out_stride_c
            + h_idx[:, None] * out_stride_h
            + w_idx[:, None] * out_stride_w
        )

        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets * weight_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets * bias_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        sum0 = tl.load(sum0_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        sum1 = tl.load(sum1_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        pooled_grad = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32)

        centered = x - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        sigmoid = 1.0 / (tl.exp(-affine) + 1.0)
        grad = (pooled_grad * INV_HW_) * sigmoid * (affine * (1.0 - sigmoid) + 1.0)

        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * invstd * invstd
        scale = invstd * weight
        out = (grad - centered * var_term[None, :] - mean_term[None, :]) * scale[None, :]
        tl.store(out0_ptr + out_offsets, out, mask=active)


def oracle_structured_pool_upsample_backward_reduce(
    mm: torch.Tensor,
    convolution_80: torch.Tensor,
    getitem_97: torch.Tensor,
    rsqrt_48: torch.Tensor,
    primals_358: torch.Tensor,
    primals_359: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")
    if convolution_80.shape != (N, C, H, W) or mm.shape != (N, C):
        raise ValueError(
            f"unexpected input shapes: mm={tuple(mm.shape)} convolution_80={tuple(convolution_80.shape)}"
        )

    out0 = torch.empty_strided(
        tuple(convolution_80.shape),
        (C * H * W, H * W, W, 1),
        device=convolution_80.device,
        dtype=convolution_80.dtype,
    )
    out1 = torch.empty((C,), device=convolution_80.device, dtype=convolution_80.dtype)
    partial0 = torch.empty((NUM_M_TILES, C), device=mm.device, dtype=torch.float32)
    partial1 = torch.empty_like(partial0)
    sum0 = torch.empty((C,), device=mm.device, dtype=torch.float32)
    sum1 = torch.empty_like(sum0)

    grid = (triton.cdiv(C, BLOCK_C), NUM_M_TILES)
    _partial_silu_bn_sums_kernel[grid](
        mm,
        convolution_80,
        getitem_97,
        rsqrt_48,
        primals_358,
        primals_359,
        partial0,
        partial1,
        mm_stride_n=mm.stride(0),
        mm_stride_c=mm.stride(1),
        x_stride_n=convolution_80.stride(0),
        x_stride_c=convolution_80.stride(1),
        x_stride_h=convolution_80.stride(2),
        x_stride_w=convolution_80.stride(3),
        mean_stride_c=getitem_97.stride(1),
        invstd_stride_c=rsqrt_48.stride(1),
        weight_stride_c=primals_358.stride(0),
        bias_stride_c=primals_359.stride(0),
        C_=C,
        W_=W,
        HW_=HW,
        N_HW_=N_HW,
        INV_HW_=INV_HW,
        BLOCK_M_=BLOCK_M,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
    )

    _finalize_silu_bn_sums_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial0,
        partial1,
        rsqrt_48,
        sum0,
        sum1,
        out1,
        invstd_stride_c=rsqrt_48.stride(1),
        C_=C,
        NUM_M_TILES_=NUM_M_TILES,
        BLOCK_C_=BLOCK_C,
        BLOCK_TILES_=BLOCK_TILES,
        num_warps=1,
    )

    _silu_bn_input_grad_kernel[grid](
        mm,
        convolution_80,
        getitem_97,
        rsqrt_48,
        primals_358,
        primals_359,
        sum0,
        sum1,
        out0,
        mm_stride_n=mm.stride(0),
        mm_stride_c=mm.stride(1),
        x_stride_n=convolution_80.stride(0),
        x_stride_c=convolution_80.stride(1),
        x_stride_h=convolution_80.stride(2),
        x_stride_w=convolution_80.stride(3),
        mean_stride_c=getitem_97.stride(1),
        invstd_stride_c=rsqrt_48.stride(1),
        weight_stride_c=primals_358.stride(0),
        bias_stride_c=primals_359.stride(0),
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
        BLOCK_M_=BLOCK_M,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
    )
    return out0, out1


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    if device.type != "cuda":
        raise RuntimeError("reference repro uses a captured CUDA device literal; run checks on CUDA")
    module = _load_repro_module()
    return module.Repro().to(device)(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    if device.type == "cuda" and triton is not None:
        return triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1_000.0

    for _ in range(max(0, warmup)):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


@oracle_impl(hardware="H100", shapes="(T([128, 1280], f32), T([128, 1280, 7, 7], f32, stride=(62720, 1, 8960, 1280)), T([1, 1280, 1, 1], f32), T([1, 1280, 1, 1], f32), T([1280], f32), T([1280], f32), S([128, 1280, 1, 1]), S([128, 1280, 7, 7]))")
def oracle_forward(inputs):
    return oracle_structured_pool_upsample_backward_reduce(*inputs)


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
