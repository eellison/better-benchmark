"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle materializes the full `[64, 128, 23923]` zero-padded `slice_scatter` side output and accumulates the sibling masked `[128]` channel sum from the same source-tile traversal over `getitem_9`, whereas Inductor currently schedules `aten.full`/`aten.slice_scatter` and `where(arg31_1, full_1, getitem_9).sum([0, 2])` as separate generic pointwise, scatter, and reduction work that rereads the source and mask; Inductor cannot do this today because its scheduler/codegen has no structured scatter-reduce template that keeps a materialized zero-fill slice-scatter side output and a source-space masked reduction epilogue in one fused producer; the fix is SCATTER_REDUCE: add a structured `slice_scatter` lowering that emits the padded side-output stores while accumulating sibling source-space reductions directly from the scattered tile."""
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



from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_28b0b21169d4"
SHAPE_LABEL = "torchbench_demucs_train_003_28f5572f"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 128
W = 23212
PAD_LEFT = 355
PAD_RIGHT = 356
PADDED_W = W + PAD_LEFT + PAD_RIGHT

BLOCK_C = 2
BLOCK_M = 1024
N_TILES = math.ceil((N * PADDED_W) / BLOCK_M)
FINAL_BLOCK = 2048



if triton is not None:

    @triton.jit
    def _slice_scatter_reduce_kernel(
        src_ptr,
        mask_ptr,
        full_ptr,
        padded_out_ptr,
        partial_ptr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        PAD_LEFT_: tl.constexpr,
        PADDED_W_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        N_TILES_: tl.constexpr,
    ):
        channel_block = tl.program_id(0)
        tile = tl.program_id(1)

        c_offsets = channel_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        m_offsets = tile * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        valid_m = m_offsets < (64 * PADDED_W_)

        n_idx = m_offsets // PADDED_W_
        padded_w_idx = m_offsets - n_idx * PADDED_W_
        in_source = (
            valid_m
            & (padded_w_idx >= PAD_LEFT_)
            & (padded_w_idx < (PAD_LEFT_ + W_))
        )
        source_w_idx = padded_w_idx - PAD_LEFT_

        c = c_offsets[:, None]
        n = n_idx[None, :]
        source_w = source_w_idx[None, :]
        padded_w = padded_w_idx[None, :]

        valid = (c_offsets[:, None] < C_) & valid_m[None, :]
        source_mask = (c_offsets[:, None] < C_) & in_source[None, :]

        src_offsets = n * (C_ * W_) + c * W_ + source_w
        padded_offsets = n * (C_ * PADDED_W_) + c * PADDED_W_ + padded_w

        src_vals = tl.load(src_ptr + src_offsets, mask=source_mask, other=0.0).to(
            tl.float32
        )
        mask_vals = tl.load(mask_ptr + src_offsets, mask=source_mask, other=0)
        full_val = tl.load(full_ptr).to(tl.float32)

        tl.store(
            padded_out_ptr + padded_offsets,
            tl.where(source_mask, src_vals, 0.0),
            mask=valid,
        )

        reduce_vals = tl.where(mask_vals, full_val, src_vals)
        reduce_vals = tl.where(source_mask, reduce_vals, 0.0)
        partial = tl.sum(reduce_vals, axis=1)
        tl.store(
            partial_ptr + c_offsets * N_TILES_ + tile,
            partial,
            mask=c_offsets < C_,
        )

    @triton.jit
    def _finalize_sum_kernel(
        partial_ptr,
        reduced_out_ptr,
        C_: tl.constexpr,
        N_TILES_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        FINAL_BLOCK_: tl.constexpr,
    ):
        channel_block = tl.program_id(0)
        c_offsets = channel_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        tile_offsets = tl.arange(0, FINAL_BLOCK_)
        active = (c_offsets[:, None] < C_) & (tile_offsets[None, :] < N_TILES_)
        values = tl.load(
            partial_ptr + c_offsets[:, None] * N_TILES_ + tile_offsets[None, :],
            mask=active,
            other=0.0,
        ).to(tl.float32)
        reduced = tl.sum(values, axis=1)
        tl.store(reduced_out_ptr + c_offsets, reduced, mask=c_offsets < C_)


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


def oracle_torch(
    getitem_9: torch.Tensor,
    arg31_1: torch.Tensor,
    full_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.empty(
        (N, C, PADDED_W),
        device=getitem_9.device,
        dtype=getitem_9.dtype,
    )
    padded[:, :, :PAD_LEFT].zero_()
    padded[:, :, PAD_LEFT : PAD_LEFT + W] = getitem_9
    padded[:, :, PAD_LEFT + W :].zero_()
    reduced = torch.where(arg31_1, full_1, getitem_9).sum(dim=(0, 2))
    return padded, reduced


def oracle_triton(
    getitem_9: torch.Tensor,
    arg31_1: torch.Tensor,
    full_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if getitem_9.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")
    if getitem_9.shape != (N, C, W):
        raise ValueError(f"unexpected getitem_9 shape: {tuple(getitem_9.shape)}")
    if arg31_1.shape != (N, C, W):
        raise ValueError(f"unexpected arg31_1 shape: {tuple(arg31_1.shape)}")
    if getitem_9.dtype != torch.float32 or arg31_1.dtype != torch.bool:
        raise ValueError(
            f"unexpected dtypes: getitem_9={getitem_9.dtype}, arg31_1={arg31_1.dtype}"
        )
    if not getitem_9.is_contiguous() or not arg31_1.is_contiguous():
        raise ValueError("oracle expects the captured contiguous input layouts")

    padded = torch.empty((N, C, PADDED_W), device=getitem_9.device, dtype=torch.float32)
    partial = torch.empty((C, N_TILES), device=getitem_9.device, dtype=torch.float32)
    reduced = torch.empty((C,), device=getitem_9.device, dtype=torch.float32)

    grid = (triton.cdiv(C, BLOCK_C), N_TILES)
    _slice_scatter_reduce_kernel[grid](
        getitem_9,
        arg31_1,
        full_1,
        padded,
        partial,
        C_=C,
        W_=W,
        PAD_LEFT_=PAD_LEFT,
        PADDED_W_=PADDED_W,
        BLOCK_C_=BLOCK_C,
        BLOCK_M_=BLOCK_M,
        N_TILES_=N_TILES,
        num_warps=8,
    )
    _finalize_sum_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial,
        reduced,
        C_=C,
        N_TILES_=N_TILES,
        BLOCK_C_=BLOCK_C,
        FINAL_BLOCK_=FINAL_BLOCK,
        num_warps=8,
    )
    return padded, reduced


def oracle_slice_scatter_reduce(
    getitem_9: torch.Tensor,
    arg31_1: torch.Tensor,
    full_1: torch.Tensor,
    *,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    if impl == "auto":
        impl = "triton" if getitem_9.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(getitem_9, arg31_1, full_1)
    if impl == "torch":
        return oracle_torch(getitem_9, arg31_1, full_1)
    raise ValueError(f"unknown impl: {impl}")


class OracleModule(torch.nn.Module):
    def __init__(self, impl: str = "auto") -> None:
        super().__init__()
        self.impl = impl

    def forward(self, *inputs: object) -> tuple[torch.Tensor, torch.Tensor]:
        getitem_9, arg31_1, full_1 = inputs
        return oracle_slice_scatter_reduce(getitem_9, arg31_1, full_1, impl=self.impl)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    return model(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the full repro.py return tuple against the oracle",
    )
    parser.add_argument("--bench", action="store_true", help="benchmark the oracle")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=5e-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    return parser.parse_args()


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
