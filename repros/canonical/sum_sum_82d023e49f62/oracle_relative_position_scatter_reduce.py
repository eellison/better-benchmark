"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Swin relative-position softmax-backward return tuple, writing the full `[8192, 49, 49]` softmax-backward side output while batch-reducing and duplicate-index scatter-adding the same producer into the `[169, 16]` bias table, whereas Inductor currently materializes the row-reduced `fma` tensor and schedules the batch reduction, permute/view, and `index_put(accumulate=True)` scatter as separate generic work; Inductor cannot do this today because scheduler/codegen does not recognize relative-position duplicate-index `index_put(accumulate=True)` as a structured scatter-reduce that can share the softmax-backward producer with a required materialized side output; the fix is SCATTER_REDUCE: add a structured relative-position scatter-reduce lowering that fuses row softmax-backward production, batch accumulation into buckets, and full side-output stores."""
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


REPRO_ID = "sum_sum_82d023e49f62"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_001_f07fc870"

N = 512
C = 16
H = 49
W = 49
NC = N * C
BUCKETS = 169
BLOCK_N = 64
BLOCK_W = 64

sys.path.insert(0, str(REPO_ROOT))


if triton is not None:

    @triton.jit
    def _softmax_bwd_relative_position_kernel(
        bmm_ptr,
        softmax_ptr,
        index_ptr,
        out0_ptr,
        out1_ptr,
        bmm_stride_m: tl.constexpr,
        bmm_stride_h: tl.constexpr,
        bmm_stride_w: tl.constexpr,
        softmax_stride_n: tl.constexpr,
        softmax_stride_c: tl.constexpr,
        softmax_stride_h: tl.constexpr,
        softmax_stride_w: tl.constexpr,
        index_stride_h: tl.constexpr,
        index_stride_w: tl.constexpr,
        out0_stride_bucket: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out1_stride_m: tl.constexpr,
        out1_stride_h: tl.constexpr,
        out1_stride_w: tl.constexpr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        H_: tl.constexpr,
        W_: tl.constexpr,
        BUCKETS_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_W_: tl.constexpr,
    ):
        n_block = tl.program_id(0)
        c = tl.program_id(1)
        h = tl.program_id(2)

        n_offsets = n_block * BLOCK_N_ + tl.arange(0, BLOCK_N_)[:, None]
        w_offsets = tl.arange(0, BLOCK_W_)[None, :]
        n_mask = n_offsets < N_
        w_mask = w_offsets < W_
        mask = n_mask & w_mask

        m_offsets = n_offsets * C_ + c
        bmm_offsets = (
            m_offsets * bmm_stride_m
            + h * bmm_stride_h
            + w_offsets * bmm_stride_w
        )
        softmax_offsets = (
            n_offsets * softmax_stride_n
            + c * softmax_stride_c
            + h * softmax_stride_h
            + w_offsets * softmax_stride_w
        )

        bmm = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0).to(tl.float32)
        softmax = tl.load(softmax_ptr + softmax_offsets, mask=mask, other=0.0).to(tl.float32)
        product = bmm * softmax
        row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None]
        fma = product - softmax * row_sum

        out1_offsets = (
            m_offsets * out1_stride_m
            + h * out1_stride_h
            + w_offsets * out1_stride_w
        )
        tl.store(out1_ptr + out1_offsets, fma, mask=mask)

        partial = tl.sum(tl.where(n_mask, fma, 0.0), axis=0)
        flat_w = tl.arange(0, BLOCK_W_)
        flat_w_mask = flat_w < W_
        bucket = tl.load(
            index_ptr + h * index_stride_h + flat_w * index_stride_w,
            mask=flat_w_mask,
            other=0,
        ).to(tl.int64)
        bucket = tl.where(bucket < 0, bucket + BUCKETS_, bucket)
        scatter_mask = flat_w_mask & (bucket >= 0) & (bucket < BUCKETS_)
        tl.atomic_add(
            out0_ptr + bucket * out0_stride_bucket + c * out0_stride_c,
            partial,
            sem="relaxed",
            mask=scatter_mask,
        )


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
    bmm_77: torch.Tensor,
    arg234_1: torch.Tensor,
    arg38_1: torch.Tensor,
    full_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> None:
    expected = [
        (tuple(bmm_77.shape), (NC, H, W), "bmm_77"),
        (tuple(arg234_1.shape), (N, C, H, W), "arg234_1"),
        (tuple(arg38_1.shape), (H, W), "arg38_1"),
        (tuple(full_1.shape), (BUCKETS, C), "full_1"),
    ]
    for got, want, name in expected:
        if got != want:
            raise ValueError(f"unexpected {name} shape: got={got} expected={want}")
    if bmm_77.dtype != torch.float32 or arg234_1.dtype != torch.float32 or full_1.dtype != torch.float32:
        raise ValueError("expected float32 data tensors")
    if arg38_1.dtype != torch.int64:
        raise ValueError("expected int64 relative-position indices")
    if list(_shape_param_0) != [N, C, H, W]:
        raise ValueError("unexpected softmax view shape parameter")
    if list(_shape_param_1) != [H * W, C]:
        raise ValueError("unexpected scatter value view shape parameter")
    if list(_shape_param_2) != [NC, H, W]:
        raise ValueError("unexpected side-output view shape parameter")


def oracle_torch(
    bmm_77: torch.Tensor,
    arg234_1: torch.Tensor,
    arg38_1: torch.Tensor,
    full_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-equivalent PyTorch oracle for the complete Repro.forward tuple."""
    _validate_inputs(
        bmm_77,
        arg234_1,
        arg38_1,
        full_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )
    bmm_view = torch.ops.aten.view.default(bmm_77, _shape_param_0)
    product = torch.ops.aten.mul.Tensor(bmm_view, arg234_1)
    row_sum = torch.ops.aten.sum.dim_IntList(product, [-1], True)
    fma = torch.ops.prims.fma.default(
        torch.ops.aten.neg.default(arg234_1),
        row_sum,
        product,
    )
    scatter_values = torch.ops.aten.view.default(
        torch.ops.aten.permute.default(
            torch.ops.aten.squeeze.dim(
                torch.ops.aten.sum.dim_IntList(fma, [0], True),
                0,
            ),
            [1, 2, 0],
        ),
        _shape_param_1,
    )
    out0 = torch.ops.aten.index_put.default(
        full_1,
        [torch.ops.aten.view.default(arg38_1, [-1])],
        scatter_values,
        True,
    )
    out1 = torch.ops.aten.view.default(fma, _shape_param_2)
    return out0, out1


def oracle_triton(
    bmm_77: torch.Tensor,
    arg234_1: torch.Tensor,
    arg38_1: torch.Tensor,
    full_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton oracle that shares the softmax-backward producer."""
    _validate_inputs(
        bmm_77,
        arg234_1,
        arg38_1,
        full_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )
    if triton is None:
        raise RuntimeError("triton is not available")
    if bmm_77.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    out0 = full_1.clone()
    out1 = torch.empty_strided(
        tuple(bmm_77.shape),
        tuple(bmm_77.stride()),
        device=bmm_77.device,
        dtype=bmm_77.dtype,
    )

    grid = (triton.cdiv(N, BLOCK_N), C, H)
    _softmax_bwd_relative_position_kernel[grid](
        bmm_77,
        arg234_1,
        arg38_1,
        out0,
        out1,
        bmm_stride_m=bmm_77.stride(0),
        bmm_stride_h=bmm_77.stride(1),
        bmm_stride_w=bmm_77.stride(2),
        softmax_stride_n=arg234_1.stride(0),
        softmax_stride_c=arg234_1.stride(1),
        softmax_stride_h=arg234_1.stride(2),
        softmax_stride_w=arg234_1.stride(3),
        index_stride_h=arg38_1.stride(0),
        index_stride_w=arg38_1.stride(1),
        out0_stride_bucket=out0.stride(0),
        out0_stride_c=out0.stride(1),
        out1_stride_m=out1.stride(0),
        out1_stride_h=out1.stride(1),
        out1_stride_w=out1.stride(2),
        N_=N,
        C_=C,
        H_=H,
        W_=W,
        BUCKETS_=BUCKETS,
        BLOCK_N_=BLOCK_N,
        BLOCK_W_=BLOCK_W,
        num_warps=8,
    )
    return out0, out1


def oracle_relative_position_scatter_reduce(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    bmm_77 = inputs[0]
    if not isinstance(bmm_77, torch.Tensor):
        raise TypeError("first input must be a tensor")
    if impl == "auto":
        impl = "triton" if bmm_77.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


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


def run_check(device: torch.device, impl: str, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_relative_position_scatter_reduce(*inputs, impl=impl)
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


def run_bench(device: torch.device, impl: str, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    actual_impl = impl
    if actual_impl == "auto":
        actual_impl = "triton" if device.type == "cuda" and triton is not None else "torch"

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
        f"impl={actual_impl} shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs and strides to repro.py")
    parser.add_argument("--bench", action="store_true", help="time the oracle")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-3)
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
