"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full three-output Swin relative-position-bias backward with one vectorized Triton producer per `(channel, row, batch tile)` that emits the `[8192, 49, 49]` softmax-backward side output and atomically accumulates both tile batch-reduction partials into duplicate `[169, 16]` relative-position buckets, whereas Inductor currently lowers the two `sum(dim=0) -> permute/view -> index_put(accumulate=True)` branches and the `mul/sum/neg/fma/view` branch as separate generic reduction, layout, scatter, and pointwise kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not recognize duplicate-index relative-position `index_put(accumulate=True)` as a structured scatter-reduce that can share the rowwise softmax-backward producer with required side-output stores; the fix is SCATTER_REDUCE: add a structured relative-position scatter-reduce lowering that fuses batch reductions with indexed accumulation and emits any full-tensor side output from the same producer."""
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


REPRO_ID = "sum_sum_sum_3eab263249bb"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_001_37c9ce01"

N = 512
C = 16
H = 49
W = 49
NC = N * C
BUCKETS = 169
BLOCK_N = 16
BLOCK_W = 64
N_BLOCKS = (N + BLOCK_N - 1) // BLOCK_N
ZERO_BLOCK = 1024



if triton is not None:

    @triton.jit
    def _zero_pair_kernel(
        out0_ptr,
        out1_ptr,
        total: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        zeros = tl.zeros((BLOCK,), tl.float32)
        tl.store(out0_ptr + offsets, zeros, mask=mask)
        tl.store(out1_ptr + offsets, zeros, mask=mask)

    @triton.jit
    def _relative_position_softmax_bwd_kernel(
        fma_2_ptr,
        index0_ptr,
        bmm_ptr,
        arg395_ptr,
        index1_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        fma_2_stride_n: tl.constexpr,
        fma_2_stride_c: tl.constexpr,
        fma_2_stride_h: tl.constexpr,
        fma_2_stride_w: tl.constexpr,
        index0_stride_h: tl.constexpr,
        index0_stride_w: tl.constexpr,
        bmm_stride_m: tl.constexpr,
        bmm_stride_h: tl.constexpr,
        bmm_stride_w: tl.constexpr,
        arg395_stride_n: tl.constexpr,
        arg395_stride_c: tl.constexpr,
        arg395_stride_h: tl.constexpr,
        arg395_stride_w: tl.constexpr,
        index1_stride_h: tl.constexpr,
        index1_stride_w: tl.constexpr,
        out2_stride_m: tl.constexpr,
        out2_stride_h: tl.constexpr,
        out2_stride_w: tl.constexpr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        H_: tl.constexpr,
        W_: tl.constexpr,
        BUCKETS_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_W_: tl.constexpr,
    ):
        c = tl.program_id(0)
        h = tl.program_id(1)
        n_block = tl.program_id(2)

        n_offsets = n_block * BLOCK_N_ + tl.arange(0, BLOCK_N_)[:, None]
        w_vec = tl.arange(0, BLOCK_W_)
        w_offsets = w_vec[None, :]
        mask = (n_offsets < N_) & (w_offsets < W_)
        m_offsets = n_offsets * C_ + c

        fma_2_offsets = (
            n_offsets * fma_2_stride_n
            + c * fma_2_stride_c
            + h * fma_2_stride_h
            + w_offsets * fma_2_stride_w
        )
        fma_2_vals = tl.load(fma_2_ptr + fma_2_offsets, mask=mask, other=0.0).to(tl.float32)
        partial0 = tl.sum(tl.where(mask, fma_2_vals, 0.0), axis=0)

        bmm_offsets = m_offsets * bmm_stride_m + h * bmm_stride_h + w_offsets * bmm_stride_w
        arg395_offsets = (
            n_offsets * arg395_stride_n
            + c * arg395_stride_c
            + h * arg395_stride_h
            + w_offsets * arg395_stride_w
        )
        bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0).to(tl.float32)
        arg395_vals = tl.load(arg395_ptr + arg395_offsets, mask=mask, other=0.0).to(tl.float32)
        mul_vals = bmm_vals * arg395_vals
        row_sum = tl.sum(tl.where(mask, mul_vals, 0.0), axis=1)
        fma_vals = mul_vals - arg395_vals * row_sum[:, None]

        out2_offsets = m_offsets * out2_stride_m + h * out2_stride_h + w_offsets * out2_stride_w
        tl.store(out2_ptr + out2_offsets, fma_vals, mask=mask)
        partial1 = tl.sum(tl.where(mask, fma_vals, 0.0), axis=0)

        bucket0 = tl.load(
            index0_ptr + h * index0_stride_h + w_vec * index0_stride_w,
            mask=w_vec < W_,
            other=0,
        ).to(tl.int64)
        bucket1 = tl.load(
            index1_ptr + h * index1_stride_h + w_vec * index1_stride_w,
            mask=w_vec < W_,
            other=0,
        ).to(tl.int64)
        scatter0_mask = (w_vec < W_) & (bucket0 >= 0) & (bucket0 < BUCKETS_)
        scatter1_mask = (w_vec < W_) & (bucket1 >= 0) & (bucket1 < BUCKETS_)
        tl.atomic_add(out0_ptr + bucket0 * C_ + c, partial0, sem="relaxed", mask=scatter0_mask)
        tl.atomic_add(out1_ptr + bucket1 * C_ + c, partial1, sem="relaxed", mask=scatter1_mask)


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


def _validate_inputs(
    fma_2: torch.Tensor,
    arg157_1: torch.Tensor,
    bmm_13: torch.Tensor,
    arg395_1: torch.Tensor,
    arg150_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    if fma_2.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    expected = [
        (tuple(fma_2.shape), (N, C, H, W), "fma_2"),
        (tuple(arg157_1.shape), (H, W), "arg157_1"),
        (tuple(bmm_13.shape), (NC, H, W), "bmm_13"),
        (tuple(arg395_1.shape), (N, C, H, W), "arg395_1"),
        (tuple(arg150_1.shape), (H, W), "arg150_1"),
    ]
    for got, want, name in expected:
        if got != want:
            raise ValueError(f"unexpected {name} shape: got={got} expected={want}")
    if fma_2.dtype != torch.float32 or bmm_13.dtype != torch.float32 or arg395_1.dtype != torch.float32:
        raise ValueError("expected float32 source tensors")
    if arg157_1.dtype != torch.int64 or arg150_1.dtype != torch.int64:
        raise ValueError("expected int64 relative-position indices")
    if list(_shape_param_0) != [H * W, C] or list(_shape_param_2) != [H * W, C]:
        raise ValueError("unexpected scatter view shape parameter")
    if list(_shape_param_1) != [N, C, H, W] or list(_shape_param_3) != [NC, H, W]:
        raise ValueError("unexpected softmax-backward view shape parameter")


def oracle_relative_position_scatter_reduce(
    fma_2: torch.Tensor,
    arg157_1: torch.Tensor,
    bmm_13: torch.Tensor,
    arg395_1: torch.Tensor,
    arg150_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the complete Repro.forward tuple with fused source-space scatter reductions."""
    _validate_inputs(
        fma_2,
        arg157_1,
        bmm_13,
        arg395_1,
        arg150_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )

    out0 = torch.empty_strided((BUCKETS, C), (C, 1), device=fma_2.device, dtype=fma_2.dtype)
    out1 = torch.empty_strided((BUCKETS, C), (C, 1), device=fma_2.device, dtype=fma_2.dtype)
    out2 = torch.empty_strided((NC, H, W), (H * W, W, 1), device=fma_2.device, dtype=fma_2.dtype)

    _zero_pair_kernel[(triton.cdiv(BUCKETS * C, ZERO_BLOCK),)](
        out0,
        out1,
        total=BUCKETS * C,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )

    grid = (C, H, N_BLOCKS)
    _relative_position_softmax_bwd_kernel[grid](
        fma_2,
        arg157_1,
        bmm_13,
        arg395_1,
        arg150_1,
        out0,
        out1,
        out2,
        fma_2_stride_n=fma_2.stride(0),
        fma_2_stride_c=fma_2.stride(1),
        fma_2_stride_h=fma_2.stride(2),
        fma_2_stride_w=fma_2.stride(3),
        index0_stride_h=arg157_1.stride(0),
        index0_stride_w=arg157_1.stride(1),
        bmm_stride_m=bmm_13.stride(0),
        bmm_stride_h=bmm_13.stride(1),
        bmm_stride_w=bmm_13.stride(2),
        arg395_stride_n=arg395_1.stride(0),
        arg395_stride_c=arg395_1.stride(1),
        arg395_stride_h=arg395_1.stride(2),
        arg395_stride_w=arg395_1.stride(3),
        index1_stride_h=arg150_1.stride(0),
        index1_stride_w=arg150_1.stride(1),
        out2_stride_m=out2.stride(0),
        out2_stride_h=out2.stride(1),
        out2_stride_w=out2.stride(2),
        N_=N,
        C_=C,
        H_=H,
        W_=W,
        BUCKETS_=BUCKETS,
        BLOCK_N_=BLOCK_N,
        BLOCK_W_=BLOCK_W,
        num_warps=4,
    )
    return out0, out1, out2


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    out = model(*inputs)
    if not isinstance(out, tuple):
        raise TypeError(f"expected tuple output from repro, got {type(out)!r}")
    return out


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
        actual = oracle_relative_position_scatter_reduce(*inputs)
        synchronize(device)

    ok = True
    if len(actual) != len(expected):
        print(f"output_count: actual={len(actual)} expected={len(expected)}")
        ok = False

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
    if triton is not None and device.type == "cuda":
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

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
    logical_read_bytes = (N * C * H * W + 2 * NC * H * W) * 4 + 2 * H * W * 8
    logical_write_bytes = (2 * BUCKETS * C + NC * H * W) * 4
    print(
        f"oracle shape: fma_2=f32[{N}, {C}, {H}, {W}], bmm=f32[{NC}, {H}, {W}] "
        f"shape={SHAPE_LABEL} device={device}"
    )
    print(f"direct logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_relative_position_scatter_reduce(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_relative_position_scatter_reduce(*inputs),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_relative_position_scatter_reduce: {oracle_us:.3f} us "
        f"impl=triton block_n={BLOCK_N} block_w={BLOCK_W} "
        f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs with Repro.forward")
    parser.add_argument("--bench", action="store_true", help="benchmark the Triton oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-2)
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
