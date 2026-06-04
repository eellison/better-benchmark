"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete BEiT layer-norm-backward `Repro.forward` tuple by row-tiling the `[128*197, 768]` producer, sharing the hidden-dimension row reductions, accumulating `sum(x * norm)`, `sum(x)`, `sum(add * addmm)`, writing the returned transposed `add * gamma2` side output, and accumulating that side output's `[768]` sum, whereas Inductor currently schedules the row reductions, layer-norm-backward pointwise epilogue, addmm/gamma side paths, transpose materialization, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local scalars, a required materialized side output, and multiple dependent column accumulators in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: add a split-row multi-output reduction template that fuses the row-summary producer with side-output stores and finalizes all compatible channel partials together."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_fb068b7d821b"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_beit_base_patch16_224_train_d76bd38f"

BATCH = 128
TOKENS = 197
CHANNELS = 768
ROWS = BATCH * TOKENS

TILE_ROWS = 8
TILE_CHANNELS = 1024
FINAL_BLOCK_CHANNELS = 16
FINAL_BLOCK_TILES = 256

sys.path.insert(0, str(REPO_ROOT))


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


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda() if isinstance(value, torch.Tensor) else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_88,
        primals_24,
        mul_9,
        div_23,
        add_128,
        primals_16,
        addmm_3,
        *_shape_params,
    ) = inputs

    return (
        mm_88.reshape(ROWS, CHANNELS).contiguous(),
        primals_24.contiguous(),
        mul_9.reshape(ROWS, CHANNELS).contiguous(),
        div_23.reshape(ROWS).contiguous(),
        add_128.reshape(ROWS, CHANNELS).contiguous(),
        primals_16.contiguous(),
        addmm_3.reshape(ROWS, CHANNELS).contiguous(),
    )


@triton.jit
def _row_tile_kernel(
    x_ptr,
    gamma1_ptr,
    norm_ptr,
    div_ptr,
    residual_ptr,
    gamma2_ptr,
    addmm_ptr,
    partial_x_norm_ptr,
    partial_x_ptr,
    partial_add_addmm_ptr,
    partial_scaled_ptr,
    scaled_md_ptr,
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
    norm = tl.load(norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma1 = tl.load(gamma1_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    gamma2 = tl.load(gamma2_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    div = tl.load(div_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    weighted = x * gamma1[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * norm, 0.0), axis=1)
    ln_grad = div[:, None] * (
        weighted * CHANNELS_ - row_sum[:, None] - norm * row_dot[:, None]
    )
    add_value = residual + ln_grad
    scaled = add_value * gamma2[None, :]

    tl.store(scaled_md_ptr + offsets, scaled, mask=mask)

    partial_offsets = tile_row * CHANNELS_ + channels
    tl.store(
        partial_x_norm_ptr + partial_offsets,
        tl.sum(tl.where(mask, x * norm, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_x_ptr + partial_offsets,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_add_addmm_ptr + partial_offsets,
        tl.sum(tl.where(mask, add_value * addmm, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_scaled_ptr + partial_offsets,
        tl.sum(tl.where(mask, scaled, 0.0), axis=0),
        mask=channel_mask,
    )


@triton.jit
def _finalize_partials_kernel(
    partial_x_norm_ptr,
    partial_x_ptr,
    partial_add_addmm_ptr,
    partial_scaled_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    out_add_addmm_ptr,
    out_scaled_ptr,
    NUM_ROW_TILES: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channels < CHANNELS_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_x_norm = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_add_addmm = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_scaled = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)

    for tile_base in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tiles = tile_base + tile_offsets
        mask = (tiles[:, None] < NUM_ROW_TILES) & channel_mask[None, :]
        offsets = tiles[:, None] * CHANNELS_ + channels[None, :]
        acc_x_norm += tl.sum(
            tl.load(partial_x_norm_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_add_addmm += tl.sum(
            tl.load(partial_add_addmm_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        acc_scaled += tl.sum(
            tl.load(partial_scaled_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )

    tl.store(out_x_norm_ptr + channels, acc_x_norm, mask=channel_mask)
    tl.store(out_x_ptr + channels, acc_x, mask=channel_mask)
    tl.store(out_add_addmm_ptr + channels, acc_add_addmm, mask=channel_mask)
    tl.store(out_scaled_ptr + channels, acc_scaled, mask=channel_mask)


def oracle_triton_prepared(
    x_md: torch.Tensor,
    gamma1_d: torch.Tensor,
    norm_md: torch.Tensor,
    div_m: torch.Tensor,
    residual_md: torch.Tensor,
    gamma2_d: torch.Tensor,
    addmm_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_md.shape == (ROWS, CHANNELS)
    assert gamma1_d.shape == (CHANNELS,)
    assert norm_md.shape == (ROWS, CHANNELS)
    assert div_m.shape == (ROWS,)
    assert residual_md.shape == (ROWS, CHANNELS)
    assert gamma2_d.shape == (CHANNELS,)
    assert addmm_md.shape == (ROWS, CHANNELS)
    assert x_md.is_contiguous()
    assert gamma1_d.is_contiguous()
    assert norm_md.is_contiguous()
    assert div_m.is_contiguous()
    assert residual_md.is_contiguous()
    assert gamma2_d.is_contiguous()
    assert addmm_md.is_contiguous()

    device = x_md.device
    num_row_tiles = triton.cdiv(ROWS, TILE_ROWS)
    partials = torch.empty(
        (4, num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    vector_outputs = torch.empty((4, CHANNELS), device=device, dtype=torch.float32)
    scaled_md = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)

    _row_tile_kernel[(num_row_tiles,)](
        x_md,
        gamma1_d,
        norm_md,
        div_m,
        residual_md,
        gamma2_d,
        addmm_md,
        partials[0],
        partials[1],
        partials[2],
        partials[3],
        scaled_md,
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
        scaled_md.t(),
        vector_outputs[3],
    )


def oracle_full(
    *inputs: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = _as_tuple(oracle_full(*inputs))
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        output_ok = value_ok and dtype_ok and stride_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(warmup: int, rep: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)
    with torch.no_grad():
        oracle_triton_prepared(*oracle_inputs)
        torch.cuda.synchronize()
        oracle_ms = triton.testing.do_bench(
            lambda: oracle_triton_prepared(*oracle_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        )

    logical_read_bytes = (
        ROWS * CHANNELS * 4 * 4
        + ROWS * 4
        + CHANNELS * 4 * 2
    )
    logical_write_bytes = ROWS * CHANNELS * 4 + CHANNELS * 4 * 4
    print(
        f"oracle_full cooperative split-k BEiT layernorm tail: {oracle_ms * 1000.0:.3f} us "
        f"shape={SHAPE_LABEL} warmup={warmup} rep={rep}"
    )
    print(f"logical traffic: {(logical_read_bytes + logical_write_bytes) / 1.0e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare against the full Repro.forward tuple")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
