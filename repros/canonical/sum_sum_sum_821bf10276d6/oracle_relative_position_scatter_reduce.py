"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full three-output Swin relative-position-bias backward by reducing both `[512, 16, 49, 49]` sources over batch directly into duplicate `[169, 16]` relative-position buckets while emitting the dependent softmax-backward tensor as `[8192, 49, 49]`, whereas Inductor currently lowers the two `sum(dim=0) -> permute/reshape -> index_put(accumulate=True)` branches and the `mul/sum/neg/fma/reshape` branch as separate generic reduction, layout, scatter, and pointwise kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not recognize duplicate-index relative-position `index_put(accumulate=True)` as a structured scatter-reduce that can share the rowwise softmax-backward producer with required side-output stores; the fix is SCATTER_REDUCE: add a structured relative-position scatter-reduce lowering that fuses batch reductions with indexed accumulation and emits any full-tensor side output from the same producer."""
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
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_821bf10276d6"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_37c9ce01"

N = 512
C = 16
H = 49
W = 49
NC = N * C
BUCKETS = 169
BLOCK_N = 16
BLOCK_W = 64
ZERO_BLOCK = 1024

sys.path.insert(0, str(REPO_ROOT))


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
        primals_322_ptr,
        bmm_61_ptr,
        div_58_ptr,
        primals_307_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        fma_2_stride_n: tl.constexpr,
        fma_2_stride_c: tl.constexpr,
        fma_2_stride_h: tl.constexpr,
        fma_2_stride_w: tl.constexpr,
        primals_322_stride_h: tl.constexpr,
        primals_322_stride_w: tl.constexpr,
        bmm_61_stride_m: tl.constexpr,
        bmm_61_stride_h: tl.constexpr,
        bmm_61_stride_w: tl.constexpr,
        div_58_stride_n: tl.constexpr,
        div_58_stride_c: tl.constexpr,
        div_58_stride_h: tl.constexpr,
        div_58_stride_w: tl.constexpr,
        primals_307_stride_h: tl.constexpr,
        primals_307_stride_w: tl.constexpr,
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
        n_mask = n_offsets < N_
        w_mask = w_offsets < W_
        mask = n_mask & w_mask
        m_offsets = n_offsets * C_ + c

        fma_2_offsets = (
            n_offsets * fma_2_stride_n
            + c * fma_2_stride_c
            + h * fma_2_stride_h
            + w_offsets * fma_2_stride_w
        )
        fma_2_vals = tl.load(fma_2_ptr + fma_2_offsets, mask=mask, other=0.0).to(tl.float32)
        partial0 = tl.sum(tl.where(mask, fma_2_vals, 0.0), axis=0)

        bmm_61_offsets = m_offsets * bmm_61_stride_m + h * bmm_61_stride_h + w_offsets * bmm_61_stride_w
        div_58_offsets = (
            n_offsets * div_58_stride_n
            + c * div_58_stride_c
            + h * div_58_stride_h
            + w_offsets * div_58_stride_w
        )
        bmm_61_vals = tl.load(bmm_61_ptr + bmm_61_offsets, mask=mask, other=0.0).to(tl.float32)
        div_58_vals = tl.load(div_58_ptr + div_58_offsets, mask=mask, other=0.0).to(tl.float32)
        mul_vals = bmm_61_vals * div_58_vals
        row_sum = tl.sum(tl.where(mask, mul_vals, 0.0), axis=1)
        fma_bwd_vals = mul_vals - div_58_vals * row_sum[:, None]

        out2_offsets = m_offsets * out2_stride_m + h * out2_stride_h + w_offsets * out2_stride_w
        tl.store(out2_ptr + out2_offsets, fma_bwd_vals, mask=mask)
        partial1 = tl.sum(tl.where(mask, fma_bwd_vals, 0.0), axis=0)

        bucket0 = tl.load(
            primals_322_ptr + h * primals_322_stride_h + w_vec * primals_322_stride_w,
            mask=w_vec < W_,
            other=0,
        ).to(tl.int64)
        bucket1 = tl.load(
            primals_307_ptr + h * primals_307_stride_h + w_vec * primals_307_stride_w,
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


def _config_on_device(config: dict, device: torch.device) -> dict:
    return {
        "inputs": [
            {**spec, "device": str(device)}
            if isinstance(spec, dict) and spec.get("kind") == "tensor"
            else spec
            for spec in config["inputs"]
        ]
    }


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        inputs = make_inputs_from_config(
            _config_on_device(next(iter(configs.values())), device)
        )
    else:
        inputs = _load_repro_module().make_inputs()

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def _validate_inputs(
    fma_2: torch.Tensor,
    primals_322: torch.Tensor,
    bmm_61: torch.Tensor,
    div_58: torch.Tensor,
    primals_307: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> None:
    expected = [
        (tuple(fma_2.shape), (N, C, H, W), "fma_2"),
        (tuple(primals_322.shape), (H, W), "primals_322"),
        (tuple(bmm_61.shape), (NC, H, W), "bmm_61"),
        (tuple(div_58.shape), (N, C, H, W), "div_58"),
        (tuple(primals_307.shape), (H, W), "primals_307"),
    ]
    for got, want, name in expected:
        if got != want:
            raise ValueError(f"unexpected {name} shape: got={got} expected={want}")

    if fma_2.dtype != torch.float32 or bmm_61.dtype != torch.float32 or div_58.dtype != torch.float32:
        raise ValueError("expected float32 source tensors")
    if primals_322.dtype != torch.int64 or primals_307.dtype != torch.int64:
        raise ValueError("expected int64 relative-position indices")
    if list(_shape_param_0) != [H * W, C] or list(_shape_param_2) != [H * W, C]:
        raise ValueError("unexpected scatter view shape parameter")
    if list(_shape_param_1) != [N, C, H, W] or list(_shape_param_3) != [NC, H, W]:
        raise ValueError("unexpected softmax-backward view shape parameter")


def _oracle_relative_position_scatter_reduce_torch(
    fma_2: torch.Tensor,
    primals_322: torch.Tensor,
    bmm_61: torch.Tensor,
    div_58: torch.Tensor,
    primals_307: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_inputs(
        fma_2,
        primals_322,
        bmm_61,
        div_58,
        primals_307,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )

    values0 = fma_2.sum(dim=0).permute(1, 2, 0).reshape(_shape_param_0)
    out0 = torch.empty_strided((BUCKETS, C), (C, 1), device=fma_2.device, dtype=fma_2.dtype)
    out0.zero_()
    out0.index_put_((primals_322.reshape(-1),), values0, accumulate=True)

    bmm_view = bmm_61.reshape(_shape_param_1)
    mul = bmm_view * div_58
    fma_default = mul - div_58 * mul.sum(dim=-1, keepdim=True)

    values1 = fma_default.sum(dim=0).permute(1, 2, 0).reshape(_shape_param_2)
    out1 = torch.empty_strided((BUCKETS, C), (C, 1), device=fma_2.device, dtype=fma_2.dtype)
    out1.zero_()
    out1.index_put_((primals_307.reshape(-1),), values1, accumulate=True)
    out2 = fma_default.reshape(_shape_param_3)
    return out0, out1, out2


def _oracle_relative_position_scatter_reduce_triton(
    fma_2: torch.Tensor,
    primals_322: torch.Tensor,
    bmm_61: torch.Tensor,
    div_58: torch.Tensor,
    primals_307: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if fma_2.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    _validate_inputs(
        fma_2,
        primals_322,
        bmm_61,
        div_58,
        primals_307,
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

    grid = (C, H, triton.cdiv(N, BLOCK_N))
    _relative_position_softmax_bwd_kernel[grid](
        fma_2,
        primals_322,
        bmm_61,
        div_58,
        primals_307,
        out0,
        out1,
        out2,
        fma_2_stride_n=fma_2.stride(0),
        fma_2_stride_c=fma_2.stride(1),
        fma_2_stride_h=fma_2.stride(2),
        fma_2_stride_w=fma_2.stride(3),
        primals_322_stride_h=primals_322.stride(0),
        primals_322_stride_w=primals_322.stride(1),
        bmm_61_stride_m=bmm_61.stride(0),
        bmm_61_stride_h=bmm_61.stride(1),
        bmm_61_stride_w=bmm_61.stride(2),
        div_58_stride_n=div_58.stride(0),
        div_58_stride_c=div_58.stride(1),
        div_58_stride_h=div_58.stride(2),
        div_58_stride_w=div_58.stride(3),
        primals_307_stride_h=primals_307.stride(0),
        primals_307_stride_w=primals_307.stride(1),
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
        num_warps=2,
    )
    return out0, out1, out2


def _resolve_impl(device: torch.device, impl: str) -> str:
    if impl != "auto":
        return impl
    if triton is not None and device.type == "cuda":
        return "triton"
    return "torch"


def oracle_relative_position_scatter_reduce(
    fma_2: torch.Tensor,
    primals_322: torch.Tensor,
    bmm_61: torch.Tensor,
    div_58: torch.Tensor,
    primals_307: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    *,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the complete Repro.forward tuple with source-space scatter reductions."""
    actual_impl = _resolve_impl(fma_2.device, impl)
    if actual_impl == "triton":
        return _oracle_relative_position_scatter_reduce_triton(
            fma_2,
            primals_322,
            bmm_61,
            div_58,
            primals_307,
            _shape_param_0,
            _shape_param_1,
            _shape_param_2,
            _shape_param_3,
        )
    if actual_impl == "torch":
        return _oracle_relative_position_scatter_reduce_torch(
            fma_2,
            primals_322,
            bmm_61,
            div_58,
            primals_307,
            _shape_param_0,
            _shape_param_1,
            _shape_param_2,
            _shape_param_3,
        )
    raise ValueError(f"unknown impl {impl!r}")


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    module.device = lambda *unused_args, **unused_kwargs: device
    out = module.Repro().to(device)(*inputs)
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


def run_check(device: torch.device, impl: str, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    actual_impl = _resolve_impl(device, impl)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_relative_position_scatter_reduce(*inputs, impl=actual_impl)
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

    print(f"Correctness: {'PASS' if ok else 'FAIL'} impl={actual_impl}")
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


def run_bench(device: torch.device, impl: str, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    actual_impl = _resolve_impl(device, impl)
    logical_read_bytes = (N * C * H * W + 2 * NC * H * W) * 4 + 2 * H * W * 8
    logical_write_bytes = (2 * BUCKETS * C + NC * H * W) * 4
    print(
        f"oracle shape: fma_2=f32[{N}, {C}, {H}, {W}], bmm_61=f32[{NC}, {H}, {W}], "
        f"div_58_stride=(38912, 2432, 49, 1) shape={SHAPE_LABEL} device={device}"
    )
    print(f"direct logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_relative_position_scatter_reduce(*inputs, impl=actual_impl)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_relative_position_scatter_reduce(*inputs, impl=actual_impl),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_relative_position_scatter_reduce: {oracle_us:.3f} us "
        f"impl={actual_impl} block_n={BLOCK_N} block_w={BLOCK_W} "
        f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs with Repro.forward")
    parser.add_argument("--bench", action="store_true", help="benchmark the oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, impl=args.impl, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
