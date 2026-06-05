"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete MobileViT layer-norm-backward return tuple by folding the patch clone/reshape/permute chain into a row-tiled producer, sharing each 144-wide row reduction, writing the returned `[144, 131072]` transposed gradient side output, and cooperatively accumulating the two pre-normalization `[144]` column sums plus the final gradient `[144]` sum, whereas Inductor currently schedules the three clone/view/permute stages, hidden-dimension reductions, dependent pointwise gradient expression, transposed side output, and sibling `sum([0, 1])`/`sum([0])` reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that combines layout-changing row producers, row-local layer-norm scalars, a required transposed side store, and multiple sibling column accumulators in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible column reductions across MobileViT patch rows, compute the patch layout mapping inside the producer, store side outputs directly in their requested layout, and finalize all column partials together."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps CPU-only syntax checks usable.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_48033561d121"

from oracle_harness import (
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
SHAPE_LABEL = "timm_mobilevit_s_train_a4886f9d"

BATCH = 128
CHANNELS = 144
IMAGE_H = 32
IMAGE_W = 32
PATCH_SIDE = 2
PATCH_AREA = PATCH_SIDE * PATCH_SIDE
PATCHES_PER_IMAGE = (IMAGE_H // PATCH_SIDE) * (IMAGE_W // PATCH_SIDE)
PATCH_COLS = IMAGE_W // PATCH_SIDE
PATCH_ROWS = BATCH * PATCH_AREA * PATCHES_PER_IMAGE

TILE_ROWS = 64
BLOCK_CHANNELS = 256
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


def oracle_torch(
    getitem_245: torch.Tensor,
    primals_129: torch.Tensor,
    mul_127: torch.Tensor,
    div_51: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clone_default = getitem_245.clone(memory_format=torch.contiguous_format)
    reshape_default = clone_default.reshape(_shape_param_0)
    permute_default = reshape_default.permute([0, 2, 1, 3])

    clone_default_1 = permute_default.clone(memory_format=torch.contiguous_format)
    reshape_default_1 = clone_default_1.reshape(_shape_param_1)
    permute_default_1 = reshape_default_1.permute([0, 3, 2, 1])

    clone_default_2 = permute_default_1.clone(memory_format=torch.contiguous_format)
    x = clone_default_2.reshape(_shape_param_2)

    weighted = x * primals_129
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * mul_127).sum(dim=2, keepdim=True)
    grad = div_51 * (weighted * CHANNELS - row_sum - mul_127 * row_dot)

    out0 = (x * mul_127).sum(dim=(0, 1))
    out1 = x.sum(dim=(0, 1))
    out2 = grad.reshape(_shape_param_3).permute([1, 0])
    out3 = grad.reshape(_shape_param_3).sum(dim=0, keepdim=True).reshape(
        _shape_param_4
    )
    return out0, out1, out2, out3


if triton is not None:

    @triton.jit
    def _mobilevit_row_tile_kernel(
        src_ptr,
        gamma_ptr,
        xhat_ptr,
        div_ptr,
        partial_x_xhat_ptr,
        partial_x_ptr,
        partial_grad_ptr,
        out_grad_t_ptr,
        ROWS: tl.constexpr,
        CHANNELS_: tl.constexpr,
        PATCHES_PER_IMAGE_: tl.constexpr,
        PATCH_COLS_: tl.constexpr,
        SRC_STRIDE_B: tl.constexpr,
        SRC_STRIDE_C: tl.constexpr,
        SRC_STRIDE_H: tl.constexpr,
        SRC_STRIDE_W: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        tile_row = tl.program_id(0)
        rows = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_CHANNELS_)
        row_mask = rows < ROWS
        col_mask = cols < CHANNELS_
        mask = row_mask[:, None] & col_mask[None, :]

        patch = rows % PATCHES_PER_IMAGE_
        batch_patch = rows // PATCHES_PER_IMAGE_
        patch_lane = batch_patch % 4
        batch = batch_patch // 4
        src_h = (patch // PATCH_COLS_) * 2 + (patch_lane // 2)
        src_w = (patch % PATCH_COLS_) * 2 + (patch_lane % 2)

        src_offsets = (
            batch[:, None] * SRC_STRIDE_B
            + cols[None, :] * SRC_STRIDE_C
            + src_h[:, None] * SRC_STRIDE_H
            + src_w[:, None] * SRC_STRIDE_W
        )
        row_col_offsets = rows[:, None] * CHANNELS_ + cols[None, :]

        x = tl.load(src_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + row_col_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        inv_std = tl.load(div_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        weighted = x * gamma[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * xhat, 0.0), axis=1)
        grad = inv_std[:, None] * (
            weighted * CHANNELS_ - row_sum[:, None] - xhat * row_dot[:, None]
        )

        tl.store(out_grad_t_ptr + row_col_offsets, grad, mask=mask)

        partial_offsets = tile_row * CHANNELS_ + cols
        sum_x_xhat = tl.sum(tl.where(mask, x * xhat, 0.0), axis=0)
        sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        sum_grad = tl.sum(tl.where(mask, grad, 0.0), axis=0)
        tl.store(partial_x_xhat_ptr + partial_offsets, sum_x_xhat, mask=col_mask)
        tl.store(partial_x_ptr + partial_offsets, sum_x, mask=col_mask)
        tl.store(partial_grad_ptr + partial_offsets, sum_grad, mask=col_mask)


    @triton.jit
    def _finalize_column_sums_kernel(
        partial_x_xhat_ptr,
        partial_x_ptr,
        partial_grad_ptr,
        out_x_xhat_ptr,
        out_x_ptr,
        out_grad_ptr,
        NUM_ROW_TILES: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
        col_mask = cols < CHANNELS_
        tile_offsets = tl.arange(0, BLOCK_TILES)

        acc_x_xhat = tl.zeros((BLOCK_CHANNELS_,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_CHANNELS_,), dtype=tl.float32)
        acc_grad = tl.zeros((BLOCK_CHANNELS_,), dtype=tl.float32)

        for tile_start in range(0, NUM_ROW_TILES, BLOCK_TILES):
            tiles = tile_start + tile_offsets
            mask = (tiles[:, None] < NUM_ROW_TILES) & col_mask[None, :]
            offsets = tiles[:, None] * CHANNELS_ + cols[None, :]

            x_xhat = tl.load(
                partial_x_xhat_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            grad = tl.load(partial_grad_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            acc_x_xhat += tl.sum(x_xhat, axis=0)
            acc_x += tl.sum(x, axis=0)
            acc_grad += tl.sum(grad, axis=0)

        tl.store(out_x_xhat_ptr + cols, acc_x_xhat, mask=col_mask)
        tl.store(out_x_ptr + cols, acc_x, mask=col_mask)
        tl.store(out_grad_ptr + cols, acc_grad, mask=col_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    getitem_245, primals_129, mul_127, div_51, *_shape_params = inputs
    return (
        getitem_245,
        primals_129,
        mul_127,
        div_51,
    )


def oracle_triton_prepared(
    getitem_245: torch.Tensor,
    primals_129: torch.Tensor,
    mul_127: torch.Tensor,
    div_51: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if getitem_245.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert getitem_245.shape == (BATCH, CHANNELS, IMAGE_H, IMAGE_W)
    assert primals_129.shape == (CHANNELS,)
    assert mul_127.shape == (BATCH * PATCH_AREA, PATCHES_PER_IMAGE, CHANNELS)
    assert div_51.shape == (BATCH * PATCH_AREA, PATCHES_PER_IMAGE, 1)
    assert primals_129.is_contiguous()
    assert mul_127.is_contiguous()
    assert div_51.is_contiguous()

    device = getitem_245.device
    num_row_tiles = triton.cdiv(PATCH_ROWS, TILE_ROWS)
    partial_x_xhat = torch.empty(
        (num_row_tiles, CHANNELS), device=device, dtype=torch.float32
    )
    partial_x = torch.empty(
        (num_row_tiles, CHANNELS), device=device, dtype=torch.float32
    )
    partial_grad = torch.empty(
        (num_row_tiles, CHANNELS), device=device, dtype=torch.float32
    )
    out_grad_t = torch.empty_strided(
        (CHANNELS, PATCH_ROWS),
        (1, CHANNELS),
        device=device,
        dtype=torch.float32,
    )

    _mobilevit_row_tile_kernel[(num_row_tiles,)](
        getitem_245,
        primals_129,
        mul_127,
        div_51,
        partial_x_xhat,
        partial_x,
        partial_grad,
        out_grad_t,
        ROWS=PATCH_ROWS,
        CHANNELS_=CHANNELS,
        PATCHES_PER_IMAGE_=PATCHES_PER_IMAGE,
        PATCH_COLS_=PATCH_COLS,
        SRC_STRIDE_B=getitem_245.stride(0),
        SRC_STRIDE_C=getitem_245.stride(1),
        SRC_STRIDE_H=getitem_245.stride(2),
        SRC_STRIDE_W=getitem_245.stride(3),
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS_=BLOCK_CHANNELS,
        num_warps=8,
    )

    out_x_xhat = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_grad = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partial_x_xhat,
        partial_x,
        partial_grad,
        out_x_xhat,
        out_x,
        out_grad,
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS_=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return out_x_xhat, out_x, out_grad_t, out_grad


def oracle_triton(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def oracle_full(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
