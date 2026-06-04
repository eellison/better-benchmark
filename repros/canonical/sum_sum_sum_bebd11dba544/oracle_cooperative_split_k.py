"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete BEiT layer-norm-backward tail returned by Repro.forward, including the two original-input channel reductions, the dependent row-local 768-wide reductions, residual add, `arg130_1`/`arg7_1` epilogues, returned `[768, 25216]` transposed side output, and final channel sum from that side output, whereas Inductor currently schedules the row reductions, dependent pointwise layer-norm-backward expression, residual add, sibling channel reductions, and permuted side output as separate generic reduction/pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that shares row-local scalar reductions with multiple compatible column accumulators and a required materialized side store; the fix is COOPERATIVE_SPLIT_K: add an Inductor split-row multi-output reduction template for layer-norm-backward tails that keeps row scalars in the producer, emits the transposed side-output backing store, and finalizes all channel reductions from shared row-tile partials."""
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


REPRO_ID = "sum_sum_sum_bebd11dba544"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_beit_base_patch16_224_train_001_d76bd38f"

BATCH = 128
TOKENS = 197
CHANNELS = 768
ROWS = BATCH * TOKENS

TILE_ROWS = 8
TILE_CHANNELS = 1024
FINAL_BLOCK_CHANNELS = 16
FINAL_BLOCK_TILES = 256



def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def make_inputs(device: torch.device) -> tuple[object, ...]:
    module = _load_repro_module()
    moved: list[object] = []
    for value in module.make_inputs():
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_88,
        arg12_1,
        arg131_1,
        arg331_1,
        add_42,
        arg7_1,
        arg130_1,
        *_shape_params,
    ) = inputs

    return (
        mm_88.reshape(ROWS, CHANNELS).contiguous(),
        arg12_1.contiguous(),
        arg131_1.reshape(ROWS, CHANNELS).contiguous(),
        arg331_1.reshape(ROWS).contiguous(),
        add_42.reshape(ROWS, CHANNELS).contiguous(),
        arg7_1.contiguous(),
        arg130_1.reshape(ROWS, CHANNELS).contiguous(),
    )


def oracle_torch(
    mm_88: torch.Tensor,
    arg12_1: torch.Tensor,
    arg131_1: torch.Tensor,
    arg331_1: torch.Tensor,
    add_42: torch.Tensor,
    arg7_1: torch.Tensor,
    arg130_1: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = mm_88.reshape(BATCH, TOKENS, CHANNELS)
    weighted = x * arg12_1
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * arg131_1).sum(dim=2, keepdim=True)
    ln_grad = arg331_1 * (weighted * CHANNELS - row_sum - arg131_1 * row_dot)
    add_value = add_42 + ln_grad

    out_x_rhs = (x * arg131_1).sum(dim=(0, 1))
    out_x = x.sum(dim=(0, 1))
    out_add_arg130 = (add_value * arg130_1.reshape(BATCH, TOKENS, CHANNELS)).sum(
        dim=(0, 1),
        keepdim=True,
    ).reshape(CHANNELS)
    scaled_md = (add_value * arg7_1).reshape(ROWS, CHANNELS)
    out_scaled = scaled_md.sum(dim=0, keepdim=True).reshape(CHANNELS)
    return out_x_rhs, out_x, out_add_arg130, scaled_md.permute(1, 0), out_scaled


if triton is not None:

    @triton.jit
    def _row_tile_partial_kernel(
        x_ptr,
        weight_ptr,
        rhs_ptr,
        gate_ptr,
        residual_ptr,
        out_weight_ptr,
        arg130_ptr,
        out_scaled_md_ptr,
        partial_x_rhs_ptr,
        partial_x_ptr,
        partial_add_arg130_ptr,
        partial_out_scaled_ptr,
        ROWS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        tile_row = tl.program_id(0)
        rows = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        channels = tl.arange(0, BLOCK_CHANNELS)
        row_mask = rows < ROWS_
        channel_mask = channels < CHANNELS_
        mask = row_mask[:, None] & channel_mask[None, :]
        offsets = rows[:, None] * CHANNELS_ + channels[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        arg130 = tl.load(arg130_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(
            tl.float32
        )
        out_weight = tl.load(
            out_weight_ptr + channels,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(gate_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        weighted = x * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
        ln_grad = gate[:, None] * (
            weighted * CHANNELS_ - row_sum[:, None] - rhs * row_dot[:, None]
        )
        add_value = residual + ln_grad
        scaled = add_value * out_weight[None, :]

        tl.store(out_scaled_md_ptr + offsets, scaled, mask=mask)

        partial_offsets = tile_row * CHANNELS_ + channels
        tl.store(
            partial_x_rhs_ptr + partial_offsets,
            tl.sum(tl.where(mask, x * rhs, 0.0), axis=0),
            mask=channel_mask,
        )
        tl.store(
            partial_x_ptr + partial_offsets,
            tl.sum(tl.where(mask, x, 0.0), axis=0),
            mask=channel_mask,
        )
        tl.store(
            partial_add_arg130_ptr + partial_offsets,
            tl.sum(tl.where(mask, add_value * arg130, 0.0), axis=0),
            mask=channel_mask,
        )
        tl.store(
            partial_out_scaled_ptr + partial_offsets,
            tl.sum(tl.where(mask, scaled, 0.0), axis=0),
            mask=channel_mask,
        )

    @triton.jit
    def _finalize_partials_kernel(
        partial_x_rhs_ptr,
        partial_x_ptr,
        partial_add_arg130_ptr,
        partial_out_scaled_ptr,
        out_x_rhs_ptr,
        out_x_ptr,
        out_add_arg130_ptr,
        out_scaled_sum_ptr,
        NUM_ROW_TILES: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        channel_mask = channels < CHANNELS_
        tile_offsets = tl.arange(0, BLOCK_TILES)

        acc_x_rhs = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
        acc_add_arg130 = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
        acc_out_scaled = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)

        for tile_base in range(0, NUM_ROW_TILES, BLOCK_TILES):
            tiles = tile_base + tile_offsets
            mask = (tiles[:, None] < NUM_ROW_TILES) & channel_mask[None, :]
            offsets = tiles[:, None] * CHANNELS_ + channels[None, :]

            acc_x_rhs += tl.sum(
                tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(
                    tl.float32
                ),
                axis=0,
            )
            acc_x += tl.sum(
                tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(
                    tl.float32
                ),
                axis=0,
            )
            acc_add_arg130 += tl.sum(
                tl.load(partial_add_arg130_ptr + offsets, mask=mask, other=0.0).to(
                    tl.float32
                ),
                axis=0,
            )
            acc_out_scaled += tl.sum(
                tl.load(partial_out_scaled_ptr + offsets, mask=mask, other=0.0).to(
                    tl.float32
                ),
                axis=0,
            )

        tl.store(out_x_rhs_ptr + channels, acc_x_rhs, mask=channel_mask)
        tl.store(out_x_ptr + channels, acc_x, mask=channel_mask)
        tl.store(out_add_arg130_ptr + channels, acc_add_arg130, mask=channel_mask)
        tl.store(out_scaled_sum_ptr + channels, acc_out_scaled, mask=channel_mask)


def oracle_triton_prepared(
    x_md: torch.Tensor,
    weight_d: torch.Tensor,
    rhs_md: torch.Tensor,
    gate_m: torch.Tensor,
    residual_md: torch.Tensor,
    out_weight_d: torch.Tensor,
    arg130_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if x_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_md.shape == (ROWS, CHANNELS)
    assert weight_d.shape == (CHANNELS,)
    assert rhs_md.shape == (ROWS, CHANNELS)
    assert gate_m.shape == (ROWS,)
    assert residual_md.shape == (ROWS, CHANNELS)
    assert out_weight_d.shape == (CHANNELS,)
    assert arg130_md.shape == (ROWS, CHANNELS)
    assert x_md.is_contiguous()
    assert weight_d.is_contiguous()
    assert rhs_md.is_contiguous()
    assert gate_m.is_contiguous()
    assert residual_md.is_contiguous()
    assert out_weight_d.is_contiguous()
    assert arg130_md.is_contiguous()

    device = x_md.device
    num_row_tiles = triton.cdiv(ROWS, TILE_ROWS)
    partials = torch.empty(
        (4, num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    vector_outputs = torch.empty((4, CHANNELS), device=device, dtype=torch.float32)
    out_scaled_md = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)

    _row_tile_partial_kernel[(num_row_tiles,)](
        x_md,
        weight_d,
        rhs_md,
        gate_m,
        residual_md,
        out_weight_d,
        arg130_md,
        out_scaled_md,
        partials[0],
        partials[1],
        partials[2],
        partials[3],
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=8,
    )

    _finalize_partials_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partials[0],
        partials[1],
        partials[2],
        partials[3],
        vector_outputs[0],
        vector_outputs[1],
        vector_outputs[2],
        vector_outputs[3],
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return (
        vector_outputs[0],
        vector_outputs[1],
        vector_outputs[2],
        out_scaled_md.permute(1, 0),
        vector_outputs[3],
    )


def oracle_triton(
    *inputs: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def oracle_full(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    if impl == "auto":
        impl = (
            "triton"
            if first_tensor.device.type == "cuda" and triton is not None
            else "torch"
        )
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, impl: str, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = _as_tuple(oracle_full(*inputs, impl=impl))
        synchronize(device)

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        shape_ok = got.shape == ref.shape
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs, max_rel = math.inf, math.inf
            value_ok = False
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        output_ok = shape_ok and value_ok and dtype_ok and stride_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} allclose={value_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


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


def run_bench(device: torch.device, impl: str, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    actual_impl = impl
    if actual_impl == "auto":
        actual_impl = "triton" if device.type == "cuda" and triton is not None else "torch"

    with torch.no_grad():
        if actual_impl == "triton":
            oracle_inputs = prepare_oracle_inputs(*inputs)
            oracle_triton_prepared(*oracle_inputs)
            synchronize(device)
            oracle_us = benchmark(
                lambda: oracle_triton_prepared(*oracle_inputs),
                device,
                warmup,
                rep,
            )
        else:
            oracle_full(*inputs, impl=actual_impl)
            synchronize(device)
            oracle_us = benchmark(
                lambda: oracle_full(*inputs, impl=actual_impl),
                device,
                warmup,
                rep,
            )

    print(
        f"oracle_full cooperative split-k BEiT LN tuple: {oracle_us:.3f} us "
        f"impl={actual_impl} shape={SHAPE_LABEL} device={device}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(
        device=device,
        impl=args.impl,
        rtol=args.rtol,
        atol=args.atol,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
