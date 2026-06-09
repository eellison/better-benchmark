"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete BEiT layer-norm-backward `Repro.forward` tuple by row-tiling the `[128*197, 768]` producer, sharing the hidden-dimension row reductions, accumulating `sum(x * norm)`, `sum(x)`, `sum(add * addmm)`, writing the returned transposed `add * gamma2` side output, and accumulating that side output's `[768]` sum, whereas Inductor currently schedules the row reductions, layer-norm-backward pointwise epilogue, addmm/gamma side paths, transpose materialization, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local scalars, a required materialized side output, and multiple dependent column accumulators in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: add a split-row multi-output reduction template that fuses the row-summary producer with side-output stores and finalizes all compatible channel partials together."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


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


@oracle_impl(hardware="H100", shapes="(T([25216, 768], f32), T([768], f32), T([128, 197, 768], f32), T([128, 197, 1], f32), T([128, 197, 768], f32), T([768], f32), T([25216, 768], f32), S([128, 197, 768]), S([128, 197, 768]), S([768]), S([25216, 768]), S([768]))")
def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return oracle_full(*inputs)


def main() -> None:
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
