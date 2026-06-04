"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete two-output EfficientNet adaptive-average-pool backward feeding SiLU-gated batch-norm backward by mapping the zero-fill/as_strided_scatter plus expand/div pool-gradient reconstruction directly back to `mm`, accumulating both per-channel reductions, and emitting the dependent contiguous input-gradient tensor plus vector scale-gradient output, whereas Inductor currently lowers the pool-gradient reconstruction, sigmoid/SiLU derivative, sibling channel reductions, and full-tensor BN-backward epilogue as separate generic kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not model the zero-fill structured scatter/expand as an average-pool-backward scatter-reduce producer that can feed both channel reductions and a required full side-output store; the fix is SCATTER_REDUCE: add a structured average-pool-backward scatter-reduce lowering that maps each pooled-gradient source directly into the fused SiLU/BN reduction producer and emits the dependent BN-backward output for the complete return tuple."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
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


REPRO_ID = "sum_sum_addc448a9236"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
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
BLOCK_C = 16
BLOCK_TILES = 64
NUM_M_TILES = triton.cdiv(N_HW, BLOCK_M) if triton is not None else math.ceil(N_HW / BLOCK_M)



def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_structured_pool_upsample_backward_reduce(*inputs)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        ok = ok and shape_ok and dtype_ok and stride_ok and value_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
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


def run_bench(device: torch.device, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    logical_read_bytes = (2 * N * C * HW + N * C) * 4
    logical_write_bytes = (N * C * HW + C) * 4
    print(
        f"oracle shape: mm=f32[{N}, {C}], activation=f32[{N}, {C}, {H}, {W}] "
        f"shape={SHAPE_LABEL} device={device}"
    )
    print(f"direct logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_structured_pool_upsample_backward_reduce(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_structured_pool_upsample_backward_reduce(*inputs),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_structured_pool_upsample_backward_reduce: {oracle_us:.3f} us "
        f"impl=triton shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs and strides to repro.py")
    parser.add_argument("--bench", action="store_true", help="time the Triton oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
