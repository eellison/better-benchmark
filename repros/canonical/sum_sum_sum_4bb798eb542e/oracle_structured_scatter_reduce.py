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



from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

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
