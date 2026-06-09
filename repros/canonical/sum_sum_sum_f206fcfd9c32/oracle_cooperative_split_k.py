"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete BERT layer-norm-backward return tuple for `sum_sum_sum_f206fcfd9c32` by row-tiling the summed three-matmul producer, reducing the two pre-mask `[768]` column outputs, writing the doubly dropout-masked `[768, 16384]` transposed side output, and cooperatively accumulating that side output's `[768]` column sum from the same producer, whereas Inductor currently schedules the three-input add, row reductions, layer-norm backward arithmetic, two mask epilogues, transpose layout, and sibling column reductions as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reductions, a dependent full-tensor side store, and multiple compatible column accumulators tied to a multi-input pointwise producer in one coordinated schedule; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across token-row tiles, fuse the summed producer and dropout-transpose side store, and finalize all sibling column partials together."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps CPU-only checks usable.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_f206fcfd9c32"

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
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_bert_pytorch_train_001_1f6612ba"

BATCH = 128
SEQ = 128
ROWS = BATCH * SEQ
CHANNELS = 768
INV_CHANNELS = 1.0 / CHANNELS
EPS = 1.0e-6
ROW_BACKWARD_SCALE = 0.002607561929595828
DROPOUT_SCALE = 1.1111111111111112

TILE_ROWS = 4
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


def make_inputs(device: torch.device) -> tuple[object, ...]:
    return (
        torch.randn((ROWS, CHANNELS), device=device, dtype=torch.float32),
        torch.randn((ROWS, CHANNELS), device=device, dtype=torch.float32),
        torch.randn((ROWS, CHANNELS), device=device, dtype=torch.float32),
        torch.randn((CHANNELS,), device=device, dtype=torch.float32),
        torch.randn((BATCH, SEQ, CHANNELS), device=device, dtype=torch.float32),
        torch.randn((BATCH, SEQ, 1), device=device, dtype=torch.float32),
        torch.randn((BATCH, SEQ, CHANNELS), device=device, dtype=torch.float32),
        torch.randn((), device=device, dtype=torch.float32),
        torch.randint(0, 2, (BATCH, SEQ, CHANNELS), device=device, dtype=torch.bool),
        torch.randint(0, 2, (BATCH, SEQ, CHANNELS), device=device, dtype=torch.bool),
        [BATCH, SEQ, CHANNELS],
        [BATCH, SEQ, CHANNELS],
        [BATCH, SEQ, CHANNELS],
        [CHANNELS],
        [CHANNELS],
        [BATCH, SEQ, CHANNELS],
        [ROWS, CHANNELS],
        [CHANNELS],
    )


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_torch(
    mm_130: torch.Tensor,
    mm_132: torch.Tensor,
    mm_134: torch.Tensor,
    arg10_1: torch.Tensor,
    arg120_1: torch.Tensor,
    arg119_1: torch.Tensor,
    add_126: torch.Tensor,
    full_1: torch.Tensor,
    arg118_1: torch.Tensor,
    arg117_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (
        mm_130.view(_shape_param_0)
        + mm_132.view(_shape_param_1)
        + mm_134.view(_shape_param_2)
    )
    sum_x = x.sum(dim=(0, 1), keepdim=True).view(_shape_param_3)

    denom = arg119_1 + EPS
    x_over_denom = x / denom
    x_gamma_over_denom = x_over_denom * arg10_1
    sum_x_dy_over_denom = (x_over_denom * arg120_1).sum(
        dim=(0, 1),
        keepdim=True,
    ).view(_shape_param_4)

    row_neg_x_gamma_sum = (-x_gamma_over_denom).sum(dim=2, keepdim=True)
    row_neg_x_gamma_dy_denom2_sum = (
        -x_gamma_over_denom * arg120_1 / denom
    ).sum(dim=2, keepdim=True)
    row_coef = torch.where(
        arg119_1 == 0,
        full_1,
        row_neg_x_gamma_dy_denom2_sum / (arg119_1 * 2.0),
    ) * ROW_BACKWARD_SCALE

    out = add_126 + x_gamma_over_denom
    out = out + row_coef * arg120_1
    out = out + row_neg_x_gamma_sum.expand(_shape_param_5) / CHANNELS
    out = out * arg118_1.to(torch.float32) * DROPOUT_SCALE
    out = out * arg117_1.to(torch.float32) * DROPOUT_SCALE

    out_flat = out.view(_shape_param_6)
    out_transposed = out_flat.permute(1, 0)
    sum_out = out_flat.sum(dim=0, keepdim=True).view(_shape_param_7)
    return sum_x, sum_x_dy_over_denom, out_transposed, sum_out


if triton is not None:

    @triton.jit
    def _row_tile_store_and_reduce_kernel(
        mm0_ptr,
        mm1_ptr,
        mm2_ptr,
        gamma_ptr,
        dy_ptr,
        denom_base_ptr,
        residual_ptr,
        full_scalar_ptr,
        keep0_ptr,
        keep1_ptr,
        partial_sum_x_ptr,
        partial_sum_x_dy_over_denom_ptr,
        partial_sum_out_ptr,
        out_md_ptr,
        ROWS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        INV_CHANNELS_: tl.constexpr,
        EPS_: tl.constexpr,
        ROW_BACKWARD_SCALE_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
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

        mm0 = tl.load(mm0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mm1 = tl.load(mm1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mm2 = tl.load(mm2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = mm0 + mm1 + mm2
        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        keep0 = tl.load(keep0_ptr + offsets, mask=mask, other=0).to(tl.float32)
        keep1 = tl.load(keep1_ptr + offsets, mask=mask, other=0).to(tl.float32)
        gamma = tl.load(gamma_ptr + channels, mask=channel_mask, other=0.0).to(
            tl.float32
        )
        denom_base = tl.load(denom_base_ptr + rows, mask=row_mask, other=0.0).to(
            tl.float32
        )
        full_scalar = tl.load(full_scalar_ptr + 0).to(tl.float32)
        denom = denom_base + EPS_

        x_over_denom = x / denom[:, None]
        x_gamma_over_denom = x_over_denom * gamma[None, :]
        row_neg_x_gamma_sum = -tl.sum(
            tl.where(mask, x_gamma_over_denom, 0.0),
            axis=1,
        )
        row_neg_x_gamma_dy_denom2_sum = -tl.sum(
            tl.where(mask, x_gamma_over_denom * dy / denom[:, None], 0.0),
            axis=1,
        )
        row_coef = tl.where(
            denom_base == 0.0,
            full_scalar,
            row_neg_x_gamma_dy_denom2_sum / (denom_base * 2.0),
        ) * ROW_BACKWARD_SCALE_

        out = (
            residual
            + x_gamma_over_denom
            + row_coef[:, None] * dy
            + row_neg_x_gamma_sum[:, None] * INV_CHANNELS_
        )
        out = out * keep0 * DROPOUT_SCALE_
        out = out * keep1 * DROPOUT_SCALE_
        tl.store(out_md_ptr + offsets, out, mask=mask)

        partial_offsets = tile_row * CHANNELS_ + channels
        sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        sum_x_dy_over_denom = tl.sum(tl.where(mask, x_over_denom * dy, 0.0), axis=0)
        sum_out = tl.sum(tl.where(mask, out, 0.0), axis=0)
        tl.store(partial_sum_x_ptr + partial_offsets, sum_x, mask=channel_mask)
        tl.store(
            partial_sum_x_dy_over_denom_ptr + partial_offsets,
            sum_x_dy_over_denom,
            mask=channel_mask,
        )
        tl.store(partial_sum_out_ptr + partial_offsets, sum_out, mask=channel_mask)


    @triton.jit
    def _finalize_column_sums_kernel(
        partial_sum_x_ptr,
        partial_sum_x_dy_over_denom_ptr,
        partial_sum_out_ptr,
        out_sum_x_ptr,
        out_sum_x_dy_over_denom_ptr,
        out_sum_out_ptr,
        NUM_ROW_TILES: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        channel_mask = channels < CHANNELS_
        tile_offsets = tl.arange(0, BLOCK_TILES)

        acc_sum_x = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
        acc_sum_x_dy_over_denom = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
        acc_sum_out = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
        for tile_base in range(0, NUM_ROW_TILES, BLOCK_TILES):
            tiles = tile_base + tile_offsets
            mask = (tiles[:, None] < NUM_ROW_TILES) & channel_mask[None, :]
            offsets = tiles[:, None] * CHANNELS_ + channels[None, :]
            acc_sum_x += tl.sum(
                tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0),
                axis=0,
            )
            acc_sum_x_dy_over_denom += tl.sum(
                tl.load(
                    partial_sum_x_dy_over_denom_ptr + offsets,
                    mask=mask,
                    other=0.0,
                ),
                axis=0,
            )
            acc_sum_out += tl.sum(
                tl.load(partial_sum_out_ptr + offsets, mask=mask, other=0.0),
                axis=0,
            )

        tl.store(out_sum_x_ptr + channels, acc_sum_x, mask=channel_mask)
        tl.store(
            out_sum_x_dy_over_denom_ptr + channels,
            acc_sum_x_dy_over_denom,
            mask=channel_mask,
        )
        tl.store(out_sum_out_ptr + channels, acc_sum_out, mask=channel_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_130,
        mm_132,
        mm_134,
        arg10_1,
        arg120_1,
        arg119_1,
        add_126,
        full_1,
        arg118_1,
        arg117_1,
        *_shape_params,
    ) = inputs

    return (
        mm_130.reshape(ROWS, CHANNELS).contiguous(),
        mm_132.reshape(ROWS, CHANNELS).contiguous(),
        mm_134.reshape(ROWS, CHANNELS).contiguous(),
        arg10_1.contiguous(),
        arg120_1.reshape(ROWS, CHANNELS).contiguous(),
        arg119_1.reshape(ROWS).contiguous(),
        add_126.reshape(ROWS, CHANNELS).contiguous(),
        full_1.contiguous(),
        arg118_1.reshape(ROWS, CHANNELS).contiguous(),
        arg117_1.reshape(ROWS, CHANNELS).contiguous(),
    )


def oracle_triton_prepared(
    mm0_md: torch.Tensor,
    mm1_md: torch.Tensor,
    mm2_md: torch.Tensor,
    gamma_d: torch.Tensor,
    dy_md: torch.Tensor,
    denom_base_m: torch.Tensor,
    residual_md: torch.Tensor,
    full_scalar: torch.Tensor,
    keep0_md: torch.Tensor,
    keep1_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm0_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm0_md.shape == (ROWS, CHANNELS)
    assert mm1_md.shape == (ROWS, CHANNELS)
    assert mm2_md.shape == (ROWS, CHANNELS)
    assert gamma_d.shape == (CHANNELS,)
    assert dy_md.shape == (ROWS, CHANNELS)
    assert denom_base_m.shape == (ROWS,)
    assert residual_md.shape == (ROWS, CHANNELS)
    assert full_scalar.shape == ()
    assert keep0_md.shape == (ROWS, CHANNELS)
    assert keep1_md.shape == (ROWS, CHANNELS)
    assert mm0_md.is_contiguous()
    assert mm1_md.is_contiguous()
    assert mm2_md.is_contiguous()
    assert gamma_d.is_contiguous()
    assert dy_md.is_contiguous()
    assert denom_base_m.is_contiguous()
    assert residual_md.is_contiguous()
    assert keep0_md.is_contiguous()
    assert keep1_md.is_contiguous()

    device = mm0_md.device
    num_row_tiles = triton.cdiv(ROWS, TILE_ROWS)
    partial_sum_x = torch.empty((num_row_tiles, CHANNELS), device=device, dtype=torch.float32)
    partial_sum_x_dy_over_denom = torch.empty(
        (num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    partial_sum_out = torch.empty((num_row_tiles, CHANNELS), device=device, dtype=torch.float32)
    out_md = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)

    _row_tile_store_and_reduce_kernel[(num_row_tiles,)](
        mm0_md,
        mm1_md,
        mm2_md,
        gamma_d,
        dy_md,
        denom_base_m,
        residual_md,
        full_scalar,
        keep0_md,
        keep1_md,
        partial_sum_x,
        partial_sum_x_dy_over_denom,
        partial_sum_out,
        out_md,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        INV_CHANNELS_=INV_CHANNELS,
        EPS_=EPS,
        ROW_BACKWARD_SCALE_=ROW_BACKWARD_SCALE,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=8,
    )

    sum_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    sum_x_dy_over_denom = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partial_sum_x,
        partial_sum_x_dy_over_denom,
        partial_sum_out,
        sum_x,
        sum_x_dy_over_denom,
        sum_out,
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return sum_x, sum_x_dy_over_denom, out_md.t(), sum_out


def oracle_triton(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def oracle_full(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    if impl == "auto":
        impl = "triton" if first_tensor.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


@oracle_impl(hardware="H100", shapes="(T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([768], f32), T([128, 128, 768], f32), T([128, 128, 1], f32), T([128, 128, 768], f32), T([], f32), T([128, 128, 768], b8), T([128, 128, 768], b8), S([128, 128, 768]), S([128, 128, 768]), S([128, 128, 768]), S([768]), S([768]), S([128, 128, 768]), S([16384, 768]), S([768]))")
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
