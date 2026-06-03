"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle covers the full `sum_sum_sum_4bb798eb542e` repro by reducing `fma` over batch and scatter-adding by `arg173_1`, computing the full `fma_default = bmm_5.view(128, 32, 49, 49) * arg419_1 - arg419_1 * sum_lastdim(...)` side output, and reducing/scatter-adding that second branch by `arg166_1`. Inductor currently lowers both duplicate-index `index_put(accumulate=True)` operations and the required materialized `fma_default.view(4096, 49, 49)` as ordinary tensor work, so it cannot reuse a source-space tile to both emit the side output and accumulate the scatter reductions. The fix is SCATTER_REDUCE: add an indexed scatter-reduce lowering that can accumulate duplicate spatial indices while also producing the required materialized view from the same pointwise/reduction tile.
"""
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


REPRO_ID = "sum_sum_sum_4bb798eb542e"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_001_9a1b5a84"

N = 128
C = 32
H = 49
W = 49
OUT = 169
BLOCK_N = 32
BLOCK_W = 64

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

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(device: torch.device) -> tuple[object, ...]:
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
    fma: torch.Tensor,
    arg173_1: torch.Tensor,
    bmm_5: torch.Tensor,
    arg419_1: torch.Tensor,
    arg166_1: torch.Tensor,
    _shape_param_0: list[int],
    _shape_param_1: list[int],
    _shape_param_2: list[int],
    _shape_param_3: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sum_fma = fma.sum(dim=0)
    scatter_values = sum_fma.permute(1, 2, 0).reshape(_shape_param_0)
    out0 = torch.zeros((OUT, C), device=fma.device, dtype=fma.dtype)
    out0 = torch.ops.aten.index_put.default(
        out0, [arg173_1.reshape(-1)], scatter_values, True
    )

    bmm_view = bmm_5.view(_shape_param_1)
    mul = bmm_view * arg419_1
    row_sum = mul.sum(dim=-1, keepdim=True)
    fma_default = torch.ops.prims.fma.default(-arg419_1, row_sum, mul)

    sum_fma_default = fma_default.sum(dim=0)
    scatter_values_1 = sum_fma_default.permute(1, 2, 0).reshape(_shape_param_2)
    out1 = torch.zeros((OUT, C), device=fma.device, dtype=fma.dtype)
    out1 = torch.ops.aten.index_put.default(
        out1, [arg166_1.reshape(-1)], scatter_values_1, True
    )
    return out0, out1, fma_default.view(_shape_param_3)


if triton is not None:

    @triton.jit
    def _fma_sum_scatter_kernel(
        fma_ptr,
        index_ptr,
        out_ptr,
        BLOCK_N_: tl.constexpr,
        BLOCK_W_: tl.constexpr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        H_: tl.constexpr,
        W_: tl.constexpr,
    ):
        c = tl.program_id(0)
        h = tl.program_id(1)
        n_block = tl.program_id(2)

        n_offsets = n_block * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        w_offsets = tl.arange(0, BLOCK_W_)
        spatial = h * W_ + w_offsets

        valid = (n_offsets[:, None] < N_) & (w_offsets[None, :] < W_)
        flat = (
            n_offsets[:, None] * (C_ * H_ * W_)
            + c * (H_ * W_)
            + spatial[None, :]
        )
        vals = tl.load(fma_ptr + flat, mask=valid, other=0.0).to(tl.float32)
        partial = tl.sum(vals, axis=0)

        dest = tl.load(index_ptr + spatial, mask=w_offsets < W_, other=0).to(tl.int64)
        tl.atomic_add(out_ptr + dest * C_ + c, partial, mask=w_offsets < W_)


    @triton.jit
    def _fma_default_store_scatter_kernel(
        bmm_ptr,
        arg419_ptr,
        index_ptr,
        out_ptr,
        side_ptr,
        BLOCK_N_: tl.constexpr,
        BLOCK_W_: tl.constexpr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        H_: tl.constexpr,
        W_: tl.constexpr,
    ):
        c = tl.program_id(0)
        h = tl.program_id(1)
        n_block = tl.program_id(2)

        n_offsets = n_block * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        w_offsets = tl.arange(0, BLOCK_W_)
        spatial = h * W_ + w_offsets

        valid = (n_offsets[:, None] < N_) & (w_offsets[None, :] < W_)
        flat = (
            n_offsets[:, None] * (C_ * H_ * W_)
            + c * (H_ * W_)
            + spatial[None, :]
        )
        bmm = tl.load(bmm_ptr + flat, mask=valid, other=0.0).to(tl.float32)
        arg419 = tl.load(arg419_ptr + flat, mask=valid, other=0.0).to(tl.float32)
        mul = bmm * arg419
        row_sum = tl.sum(tl.where(valid, mul, 0.0), axis=1)
        fma_default = mul - arg419 * row_sum[:, None]
        fma_default = tl.where(valid, fma_default, 0.0)

        tl.store(side_ptr + flat, fma_default, mask=valid)

        partial = tl.sum(fma_default, axis=0)
        dest = tl.load(index_ptr + spatial, mask=w_offsets < W_, other=0).to(tl.int64)
        tl.atomic_add(out_ptr + dest * C_ + c, partial, mask=w_offsets < W_)


def oracle_triton(
    fma: torch.Tensor,
    arg173_1: torch.Tensor,
    bmm_5: torch.Tensor,
    arg419_1: torch.Tensor,
    arg166_1: torch.Tensor,
    _shape_param_0: list[int],
    _shape_param_1: list[int],
    _shape_param_2: list[int],
    _shape_param_3: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if fma.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    out0 = torch.zeros((OUT, C), device=fma.device, dtype=fma.dtype)
    out1 = torch.zeros((OUT, C), device=fma.device, dtype=fma.dtype)
    side = torch.empty((N * C, H, W), device=fma.device, dtype=fma.dtype)

    grid = (C, H, triton.cdiv(N, BLOCK_N))
    _fma_sum_scatter_kernel[grid](
        fma,
        arg173_1,
        out0,
        BLOCK_N_=BLOCK_N,
        BLOCK_W_=BLOCK_W,
        N_=N,
        C_=C,
        H_=H,
        W_=W,
        num_warps=2,
    )
    _fma_default_store_scatter_kernel[grid](
        bmm_5,
        arg419_1,
        arg166_1,
        out1,
        side,
        BLOCK_N_=BLOCK_N,
        BLOCK_W_=BLOCK_W,
        N_=N,
        C_=C,
        H_=H,
        W_=W,
        num_warps=2,
    )
    return out0, out1, side


def oracle_structured_scatter_reduce(
    fma: torch.Tensor,
    arg173_1: torch.Tensor,
    bmm_5: torch.Tensor,
    arg419_1: torch.Tensor,
    arg166_1: torch.Tensor,
    _shape_param_0: list[int],
    _shape_param_1: list[int],
    _shape_param_2: list[int],
    _shape_param_3: list[int],
    *,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if impl == "auto":
        impl = "triton" if fma.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(
            fma,
            arg173_1,
            bmm_5,
            arg419_1,
            arg166_1,
            _shape_param_0,
            _shape_param_1,
            _shape_param_2,
            _shape_param_3,
        )
    if impl == "torch":
        return oracle_torch(
            fma,
            arg173_1,
            bmm_5,
            arg419_1,
            arg166_1,
            _shape_param_0,
            _shape_param_1,
            _shape_param_2,
            _shape_param_3,
        )
    raise ValueError(f"unknown impl: {impl}")


class OracleModule(torch.nn.Module):
    def __init__(self, impl: str = "torch") -> None:
        super().__init__()
        self.impl = impl

    def forward(self, *inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return oracle_structured_scatter_reduce(*inputs, impl=self.impl)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    return model(*inputs)


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
        actual = oracle_structured_scatter_reduce(*inputs, impl=impl)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == ref.stride()
        ok = ok and value_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"expected_stride={ref.stride()} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={value_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


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


def _compile_with_config(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    config: dict[str, object],
    device: torch.device,
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        with torch.no_grad():
            compiled(*inputs)
            synchronize(device)
    return compiled


def run_bench(device: torch.device, impl: str, warmup: int, rep: int, no_compile: bool) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    actual_impl = impl
    if actual_impl == "auto":
        actual_impl = "triton" if device.type == "cuda" and triton is not None else "torch"

    logical_read_bytes = (N * C * H * W * 4) + 2 * (N * C * H * W * 4)
    logical_write_bytes = 2 * (OUT * C * 4) + (N * C * H * W * 4)
    print(
        f"oracle shape: fma=f32[{N}, {C}, {H}, {W}], "
        f"bmm_5=f32[{N * C}, {H}, {W}], arg419=f32[{N}, {C}, {H}, {W}]"
    )
    print(f"direct logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_structured_scatter_reduce(*inputs, impl=actual_impl)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_structured_scatter_reduce(*inputs, impl=actual_impl),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_structured_scatter_reduce: {oracle_us:.3f} us "
        f"impl={actual_impl} shape={SHAPE_LABEL} device={device}"
    )

    if no_compile:
        return

    if device.type != "cuda":
        print("torch.compile config timings skipped: CUDA is required")
        return

    module = _load_repro_module()
    print("torch.compile full repro timings:")
    for label, config in COMPILE_CONFIGS:
        try:
            compiled = _compile_with_config(module.Repro().to(device), inputs, config, device)
            compiled_us = benchmark(lambda: compiled(*inputs), device, warmup, rep)
            print(f"torch.compile {label}: {compiled_us:.3f} us")
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile coordinate-descent and combo config baselines",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, impl=args.impl, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(
            device=device,
            impl=args.impl,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    main()
