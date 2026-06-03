"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full three-output Swin relative-position-bias backward by reducing the original fma source and the fused softmax-backward producer over batch directly into duplicate [169, 32] relative-position buckets while writing the required [4096, 49, 49] softmax-backward side output with contiguous target strides, whereas Inductor currently emits separate generic batch reductions, permute/reshape copies, index_put scatter kernels, and pointwise/row-reduction kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not recognize duplicate-index relative-position index_put(accumulate=True) as a structured scatter-reduce consumer that can share a softmax-backward producer with a materialized side output; the fix is SCATTER_REDUCE: add a structured relative-position scatter-reduce lowering that fuses batch reductions with indexed accumulation and emits the complete side-output tuple from the same producer."""
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


REPRO_ID = "sum_sum_sum_443ecc71f23f"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_9a1b5a84"

N = 128
C = 32
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
        fma_ptr,
        index0_ptr,
        bmm_ptr,
        div_ptr,
        index1_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        fma_stride_n: tl.constexpr,
        fma_stride_c: tl.constexpr,
        fma_stride_h: tl.constexpr,
        fma_stride_w: tl.constexpr,
        index0_stride_h: tl.constexpr,
        index0_stride_w: tl.constexpr,
        bmm_stride_m: tl.constexpr,
        bmm_stride_h: tl.constexpr,
        bmm_stride_w: tl.constexpr,
        div_stride_n: tl.constexpr,
        div_stride_c: tl.constexpr,
        div_stride_h: tl.constexpr,
        div_stride_w: tl.constexpr,
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
        n_mask = n_offsets < N_
        w_mask = w_offsets < W_
        mask = n_mask & w_mask
        m_offsets = n_offsets * C_ + c

        fma_offsets = (
            n_offsets * fma_stride_n
            + c * fma_stride_c
            + h * fma_stride_h
            + w_offsets * fma_stride_w
        )
        fma_vals = tl.load(fma_ptr + fma_offsets, mask=mask, other=0.0).to(tl.float32)
        partial0 = tl.sum(tl.where(mask, fma_vals, 0.0), axis=0)

        bmm_offsets = m_offsets * bmm_stride_m + h * bmm_stride_h + w_offsets * bmm_stride_w
        div_offsets = (
            n_offsets * div_stride_n
            + c * div_stride_c
            + h * div_stride_h
            + w_offsets * div_stride_w
        )
        bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0).to(tl.float32)
        div_vals = tl.load(div_ptr + div_offsets, mask=mask, other=0.0).to(tl.float32)
        mul_vals = bmm_vals * div_vals
        row_sum = tl.sum(tl.where(mask, mul_vals, 0.0), axis=1)
        fma_bwd_vals = mul_vals - div_vals * row_sum[:, None]

        out2_offsets = m_offsets * out2_stride_m + h * out2_stride_h + w_offsets * out2_stride_w
        tl.store(out2_ptr + out2_offsets, fma_bwd_vals, mask=mask)
        partial1 = tl.sum(tl.where(mask, fma_bwd_vals, 0.0), axis=0)

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
    fma: torch.Tensor,
    primals_353: torch.Tensor,
    bmm_53: torch.Tensor,
    div_64: torch.Tensor,
    primals_339: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    if fma.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    expected = [
        (tuple(fma.shape), (N, C, H, W), "fma"),
        (tuple(primals_353.shape), (H, W), "primals_353"),
        (tuple(bmm_53.shape), (NC, H, W), "bmm_53"),
        (tuple(div_64.shape), (N, C, H, W), "div_64"),
        (tuple(primals_339.shape), (H, W), "primals_339"),
    ]
    for got, want, name in expected:
        if got != want:
            raise ValueError(f"unexpected {name} shape: got={got} expected={want}")

    if fma.dtype != torch.float32 or bmm_53.dtype != torch.float32 or div_64.dtype != torch.float32:
        raise ValueError("expected float32 source tensors")
    if primals_353.dtype != torch.int64 or primals_339.dtype != torch.int64:
        raise ValueError("expected int64 relative-position indices")
    if list(_shape_param_0) != [H * W, C] or list(_shape_param_2) != [H * W, C]:
        raise ValueError("unexpected scatter view shape parameter")
    if list(_shape_param_1) != [N, C, H, W] or list(_shape_param_3) != [NC, H, W]:
        raise ValueError("unexpected softmax-backward view shape parameter")


def oracle_relative_position_scatter_reduce(
    fma: torch.Tensor,
    primals_353: torch.Tensor,
    bmm_53: torch.Tensor,
    div_64: torch.Tensor,
    primals_339: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the complete Repro.forward tuple with fused source-space scatter reductions."""
    _validate_inputs(
        fma,
        primals_353,
        bmm_53,
        div_64,
        primals_339,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )

    out0 = torch.empty_strided((BUCKETS, C), (C, 1), device=fma.device, dtype=fma.dtype)
    out1 = torch.empty_strided((BUCKETS, C), (C, 1), device=fma.device, dtype=fma.dtype)
    out2 = torch.empty_strided((NC, H, W), (H * W, W, 1), device=fma.device, dtype=fma.dtype)

    _zero_pair_kernel[(triton.cdiv(BUCKETS * C, ZERO_BLOCK),)](
        out0,
        out1,
        total=BUCKETS * C,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )

    grid = (C, H, triton.cdiv(N, BLOCK_N))
    _relative_position_softmax_bwd_kernel[grid](
        fma,
        primals_353,
        bmm_53,
        div_64,
        primals_339,
        out0,
        out1,
        out2,
        fma_stride_n=fma.stride(0),
        fma_stride_c=fma.stride(1),
        fma_stride_h=fma.stride(2),
        fma_stride_w=fma.stride(3),
        index0_stride_h=primals_353.stride(0),
        index0_stride_w=primals_353.stride(1),
        bmm_stride_m=bmm_53.stride(0),
        bmm_stride_h=bmm_53.stride(1),
        bmm_stride_w=bmm_53.stride(2),
        div_stride_n=div_64.stride(0),
        div_stride_c=div_64.stride(1),
        div_stride_h=div_64.stride(2),
        div_stride_w=div_64.stride(3),
        index1_stride_h=primals_339.stride(0),
        index1_stride_w=primals_339.stride(1),
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


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    if device.type != "cuda":
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
        f"oracle shape: fma=f32[{N}, {C}, {H}, {W}], bmm=f32[{NC}, {H}, {W}] "
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
