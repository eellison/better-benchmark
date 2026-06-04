"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GoogleFnet layer-norm backward, dgamma/dbeta reductions, and select_scatter output with Triton row reductions plus atomic cooperative split-K column reductions, whereas Inductor currently emits separate generic reduction and select_scatter kernels around materialized intermediates; Inductor cannot do this today because its scheduler/codegen cannot fuse LN-backward row reductions with sibling column reductions and full select_scatter side-output stores while cooperatively splitting the reduction across the batch-sequence dimension; the fix is COOPERATIVE_SPLIT_K: teach Inductor to generate a specialized layer-norm-backward schedule that keeps row reductions and select_scatter stores in one output kernel while accumulating dgamma and dbeta with cooperative split-K reductions."""
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
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_b30da3bff8d4"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_GoogleFnet_train_001_a1091d37"

B = 32
S = 512
M = B * S
C = 768
LANES = 2
BLOCK_ROW_C = 1024
ROWS_PER_SPLIT = 4
NUM_ROW_SPLITS = math.ceil(M / ROWS_PER_SPLIT)


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
    def _fused_ln_select_atomic_split_k_kernel(
        mm_ptr,
        residual_ptr,
        gamma_ptr,
        xhat_ptr,
        scale_ptr,
        full_ptr,
        out_ptr,
        dgamma_ptr,
        dbeta_ptr,
        mm_stride_m: tl.constexpr,
        mm_stride_c: tl.constexpr,
        residual_stride_b: tl.constexpr,
        residual_stride_s: tl.constexpr,
        residual_stride_c: tl.constexpr,
        gamma_stride_c: tl.constexpr,
        xhat_stride_b: tl.constexpr,
        xhat_stride_s: tl.constexpr,
        xhat_stride_c: tl.constexpr,
        scale_stride_b: tl.constexpr,
        scale_stride_s: tl.constexpr,
        scale_stride_c: tl.constexpr,
        full_stride_b: tl.constexpr,
        full_stride_s: tl.constexpr,
        full_stride_c: tl.constexpr,
        full_stride_l: tl.constexpr,
        out_stride_b: tl.constexpr,
        out_stride_s: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_l: tl.constexpr,
        ROWS_PER_SPLIT_: tl.constexpr,
        S_: tl.constexpr,
        M_: tl.constexpr,
        C_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid = tl.program_id(0)
        c_offsets = tl.arange(0, BLOCK_C_)
        c_mask = c_offsets < C_
        gamma = tl.load(gamma_ptr + c_offsets * gamma_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        accum_dgamma = tl.full([BLOCK_C_], 0.0, tl.float32)
        accum_dbeta = tl.full([BLOCK_C_], 0.0, tl.float32)

        for row_offset in tl.static_range(0, ROWS_PER_SPLIT_):
            row = pid * ROWS_PER_SPLIT_ + row_offset
            row_active = row < M_
            b_idx = row // S_
            s_idx = row - b_idx * S_
            mask = c_mask & row_active

            mm_offsets = row * mm_stride_m + c_offsets * mm_stride_c
            residual_offsets = (
                b_idx * residual_stride_b
                + s_idx * residual_stride_s
                + c_offsets * residual_stride_c
            )
            xhat_offsets = b_idx * xhat_stride_b + s_idx * xhat_stride_s + c_offsets * xhat_stride_c
            scale_offset = b_idx * scale_stride_b + s_idx * scale_stride_s
            full_lane1_offsets = (
                b_idx * full_stride_b
                + s_idx * full_stride_s
                + c_offsets * full_stride_c
                + full_stride_l
            )
            out_lane0_offsets = (
                b_idx * out_stride_b
                + s_idx * out_stride_s
                + c_offsets * out_stride_c
            )

            mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
            xhat = tl.load(xhat_ptr + xhat_offsets, mask=mask, other=0.0).to(tl.float32)
            row_scale = tl.load(scale_ptr + scale_offset + 0 * scale_stride_c, mask=row_active, other=0.0).to(tl.float32)

            upstream = mm + residual
            weighted = upstream * gamma
            sum_weighted = tl.sum(weighted, axis=0)
            sum_weighted_xhat = tl.sum(weighted * xhat, axis=0)
            dx = row_scale * (weighted * C_ - sum_weighted - xhat * sum_weighted_xhat)
            lane1 = tl.load(full_ptr + full_lane1_offsets, mask=mask, other=0.0).to(tl.float32)

            lane_offsets = tl.arange(0, 2)
            out_pair_offsets = out_lane0_offsets[:, None] + lane_offsets[None, :] * out_stride_l
            out_pair = tl.where(lane_offsets[None, :] == 0, dx[:, None], lane1[:, None])
            tl.store(out_ptr + out_pair_offsets, out_pair, mask=mask[:, None])
            accum_dbeta += upstream
            accum_dgamma += upstream * xhat

        tl.atomic_add(dbeta_ptr + c_offsets, accum_dbeta, sem="relaxed", mask=c_mask)
        tl.atomic_add(dgamma_ptr + c_offsets, accum_dgamma, sem="relaxed", mask=c_mask)


def oracle_cooperative_split_k(
    mm_50: torch.Tensor,
    mul_312: torch.Tensor,
    arg6_1: torch.Tensor,
    arg60_1: torch.Tensor,
    arg164_1: torch.Tensor,
    full_2: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm_50.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    dgamma = torch.empty((C,), device=mm_50.device, dtype=mm_50.dtype)
    dbeta = torch.empty((C,), device=mm_50.device, dtype=mm_50.dtype)
    select_scatter = torch.empty_strided(
        tuple(full_2.shape),
        tuple(full_2.stride()),
        device=full_2.device,
        dtype=full_2.dtype,
    )
    dgamma.zero_()
    dbeta.zero_()

    _fused_ln_select_atomic_split_k_kernel[(NUM_ROW_SPLITS,)](
        mm_50,
        mul_312,
        arg6_1,
        arg60_1,
        arg164_1,
        full_2,
        select_scatter,
        dgamma,
        dbeta,
        mm_stride_m=mm_50.stride(0),
        mm_stride_c=mm_50.stride(1),
        residual_stride_b=mul_312.stride(0),
        residual_stride_s=mul_312.stride(1),
        residual_stride_c=mul_312.stride(2),
        gamma_stride_c=arg6_1.stride(0),
        xhat_stride_b=arg60_1.stride(0),
        xhat_stride_s=arg60_1.stride(1),
        xhat_stride_c=arg60_1.stride(2),
        scale_stride_b=arg164_1.stride(0),
        scale_stride_s=arg164_1.stride(1),
        scale_stride_c=arg164_1.stride(2),
        full_stride_b=full_2.stride(0),
        full_stride_s=full_2.stride(1),
        full_stride_c=full_2.stride(2),
        full_stride_l=full_2.stride(3),
        out_stride_b=select_scatter.stride(0),
        out_stride_s=select_scatter.stride(1),
        out_stride_c=select_scatter.stride(2),
        out_stride_l=select_scatter.stride(3),
        ROWS_PER_SPLIT_=ROWS_PER_SPLIT,
        S_=S,
        M_=M,
        C_=C,
        BLOCK_C_=BLOCK_ROW_C,
        num_warps=4,
    )
    return dgamma, dbeta, select_scatter


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
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
        actual = oracle_cooperative_split_k(*inputs)
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
    logical_read_bytes = (M * C * 3 + M + M * C) * 4
    logical_write_bytes = (M * C * LANES + C * 2) * 4
    print(
        f"oracle shape: upstream=f32[{B}, {S}, {C}] full=f32[{B}, {S}, {C}, {LANES}] "
        f"shape={SHAPE_LABEL} device={device}"
    )
    print(f"direct logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_cooperative_split_k(*inputs)
        synchronize(device)
        oracle_us = benchmark(lambda: oracle_cooperative_split_k(*inputs), device, warmup, rep)
    print(
        f"oracle_cooperative_split_k: {oracle_us:.3f} us "
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
