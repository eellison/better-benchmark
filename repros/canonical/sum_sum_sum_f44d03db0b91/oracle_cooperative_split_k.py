"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete BEiT layer-norm-backward tail returned by Repro.forward by row-tiling the `[128*197, 768]` producer, computing each row's hidden-dimension sums once, and cooperatively accumulating the two global channel reductions plus the class-token and patch-token reductions from the same Triton pass, whereas Inductor currently schedules the row reductions, dependent layer-norm-backward pointwise expression, residual add, slice/permute/reshape view path, and sibling reductions as separate generic reduction/pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that combines row-local scalar reductions with multiple compatible batch/token/channel reductions and view-equivalent epilogues in one coordinated producer/finalizer; the fix is COOPERATIVE_SPLIT_K: add an Inductor split-row multi-output reduction template for layer-norm-backward tails that shares row scalars, emits partial accumulators for class/patch/global channel reductions, and finalizes the full return tuple without materializing the intermediate gradient tensor."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_f44d03db0b91"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_beit_base_patch16_224_train_7c3a611f"

BATCH = 128
TOKENS = 197
CHANNELS = 768
ROWS = BATCH * TOKENS
INV_CHANNELS = 1.0 / CHANNELS

TILE_ROWS = 16
TILE_CHANNELS = 1024
FINAL_BLOCK_CHANNELS = 16
FINAL_BLOCK_TILES = 1024



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
        mm_96,
        primals_6,
        cat,
        getitem_1,
        rsqrt,
        add_132,
        *_shape_params,
    ) = inputs

    return (
        mm_96.reshape(ROWS, CHANNELS).contiguous(),
        primals_6.contiguous(),
        cat.reshape(ROWS, CHANNELS).contiguous(),
        getitem_1.reshape(ROWS).contiguous(),
        rsqrt.reshape(ROWS).contiguous(),
        add_132.reshape(ROWS, CHANNELS).contiguous(),
    )


@triton.jit
def _row_tile_partial_kernel(
    x_ptr,
    weight_ptr,
    source_ptr,
    mean_ptr,
    rsqrt_ptr,
    residual_ptr,
    partial_x_norm_ptr,
    partial_x_ptr,
    partial_cls_ptr,
    partial_patch_ptr,
    ROWS_: tl.constexpr,
    TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    INV_CHANNELS_: tl.constexpr,
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
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    norm = (source - mean[:, None]) * rsqrt[:, None]
    weighted = x * weight[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * norm, 0.0), axis=1)
    ln_grad = rsqrt[:, None] * INV_CHANNELS_ * (
        weighted * CHANNELS_ - row_sum[:, None] - norm * row_dot[:, None]
    )
    add_value = residual + ln_grad

    token = rows % TOKENS_
    cls_mask = mask & (token[:, None] == 0)
    patch_mask = mask & (token[:, None] != 0)

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
        partial_cls_ptr + partial_offsets,
        tl.sum(tl.where(cls_mask, add_value, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_patch_ptr + partial_offsets,
        tl.sum(tl.where(patch_mask, add_value, 0.0), axis=0),
        mask=channel_mask,
    )


@triton.jit
def _finalize_partials_kernel(
    partial_x_norm_ptr,
    partial_x_ptr,
    partial_cls_ptr,
    partial_patch_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    out_cls_ptr,
    out_patch_ptr,
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
    acc_cls = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_patch = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)

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
        acc_cls += tl.sum(
            tl.load(partial_cls_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_patch += tl.sum(
            tl.load(partial_patch_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )

    tl.store(out_x_norm_ptr + channels, acc_x_norm, mask=channel_mask)
    tl.store(out_x_ptr + channels, acc_x, mask=channel_mask)
    tl.store(out_cls_ptr + channels, acc_cls, mask=channel_mask)
    tl.store(out_patch_ptr + channels, acc_patch, mask=channel_mask)


def oracle_triton_prepared(
    x_md: torch.Tensor,
    weight_d: torch.Tensor,
    source_md: torch.Tensor,
    mean_m: torch.Tensor,
    rsqrt_m: torch.Tensor,
    residual_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_md.shape == (ROWS, CHANNELS)
    assert weight_d.shape == (CHANNELS,)
    assert source_md.shape == (ROWS, CHANNELS)
    assert mean_m.shape == (ROWS,)
    assert rsqrt_m.shape == (ROWS,)
    assert residual_md.shape == (ROWS, CHANNELS)
    assert x_md.is_contiguous()
    assert weight_d.is_contiguous()
    assert source_md.is_contiguous()
    assert mean_m.is_contiguous()
    assert rsqrt_m.is_contiguous()
    assert residual_md.is_contiguous()

    device = x_md.device
    num_row_tiles = triton.cdiv(ROWS, TILE_ROWS)
    partials = torch.empty(
        (4, num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    vector_outputs = torch.empty((4, CHANNELS), device=device, dtype=torch.float32)

    _row_tile_partial_kernel[(num_row_tiles,)](
        x_md,
        weight_d,
        source_md,
        mean_m,
        rsqrt_m,
        residual_md,
        partials[0],
        partials[1],
        partials[2],
        partials[3],
        ROWS_=ROWS,
        TOKENS_=TOKENS,
        CHANNELS_=CHANNELS,
        INV_CHANNELS_=INV_CHANNELS,
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

    cls_sum = vector_outputs[2].view(1, 1, CHANNELS)
    return vector_outputs[0], vector_outputs[1], cls_sum, vector_outputs[3]


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _compile_with_config(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    config: dict[str, object],
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(rep: int, warmup: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)
    with torch.no_grad():
        oracle_triton_prepared(*oracle_inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_triton_prepared(*oracle_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(
        f"oracle_full cooperative split-k BEiT LN reductions: {oracle_us:.3f} us "
        f"shape={SHAPE_LABEL}"
    )

    if no_compile:
        return

    module = _load_repro_module()
    compile_configs = [
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
    for label, config in compile_configs:
        model = module.Repro().cuda()
        with torch.no_grad():
            compiled = _compile_with_config(model, inputs, config)
            compiled_us = triton.testing.do_bench(
                lambda: compiled(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            ) * 1000.0
        print(f"torch.compile {label}: {compiled_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
