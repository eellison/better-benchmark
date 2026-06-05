"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileViT 2x2 patch layout rewrite directly from the original NCHW tensor, fuses the per-row hidden reductions with the dependent full `[512,256,144]` tensor, and returns the two layout-reduction outputs plus the aliased `[144,131072]` side view and its `[144]` reduction, whereas Inductor lowers the clone/view/permute/clone/view/permute/clone layout pipeline, row reductions, pointwise dependent tensor, side-output materialization, and column reductions as generic scheduled pieces; Inductor cannot do this today because its reduction templates do not recognize this MobileViT patch-unfold multi-output reduction with a required full side output sharing storage with the final reduction producer; the fix is NEW_PATTERN: add a MobileViT layout-reduction template that maps rows back to source NCHW indices and emits the row reductions, dependent side tensor, and column partials in one fused lowering."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _layout_side_and_partials_kernel(
        x_ptr,
        weight_ptr,
        arg240_ptr,
        arg413_ptr,
        side_base_ptr,
        partial0_ptr,
        partial1_ptr,
        partial3_ptr,
        stride_n: tl.constexpr,
        stride_c: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        patches: tl.constexpr,
        width: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        tile = tl.program_id(0)
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        cols = tl.arange(0, BLOCK_C)[None, :]
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask

        patch_group = rows // patches
        patch = rows - patch_group * patches
        lane = patch_group % 4
        batch = patch_group // 4
        half_width = width // 2
        h = (patch // half_width) * 2 + lane // 2
        w = (patch - (patch // half_width) * half_width) * 2 + lane % 2
        x_offsets = batch * stride_n + cols * stride_c + h * stride_h + w * stride_w

        v = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        arg240 = tl.load(arg240_ptr + rows * hidden + cols, mask=mask, other=0.0).to(tl.float32)
        weighted = v * weight

        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)[:, None]
        row_sum_weighted_arg240 = tl.sum(tl.where(mask, weighted * arg240, 0.0), axis=1)[:, None]
        arg413 = tl.load(arg413_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        side = arg413 * (weighted * hidden - row_sum - arg240 * row_sum_weighted_arg240)
        tl.store(side_base_ptr + rows * hidden + cols, side, mask=mask)

        partial_cols = tl.arange(0, BLOCK_C)
        partial_mask = partial_cols < hidden
        partial0 = tl.sum(tl.where(mask, v * arg240, 0.0), axis=0)
        partial1 = tl.sum(tl.where(mask, v, 0.0), axis=0)
        partial3 = tl.sum(tl.where(mask, side, 0.0), axis=0)
        partial_offsets = tile * hidden + partial_cols
        tl.store(partial0_ptr + partial_offsets, partial0, mask=partial_mask)
        tl.store(partial1_ptr + partial_offsets, partial1, mask=partial_mask)
        tl.store(partial3_ptr + partial_offsets, partial3, mask=partial_mask)

    @triton.jit
    def _finalize_partials_kernel(
        partial0_ptr,
        partial1_ptr,
        partial3_ptr,
        out0_ptr,
        out1_ptr,
        out3_ptr,
        hidden: tl.constexpr,
        n_tiles: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        c = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_TILES)
        acc0 = tl.zeros([BLOCK_TILES], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_TILES], dtype=tl.float32)
        acc3 = tl.zeros([BLOCK_TILES], dtype=tl.float32)

        for start in range(0, n_tiles, BLOCK_TILES):
            tile_offsets = start + offsets
            mask = tile_offsets < n_tiles
            partial_offsets = tile_offsets * hidden + c
            acc0 += tl.load(partial0_ptr + partial_offsets, mask=mask, other=0.0)
            acc1 += tl.load(partial1_ptr + partial_offsets, mask=mask, other=0.0)
            acc3 += tl.load(partial3_ptr + partial_offsets, mask=mask, other=0.0)

        tl.store(out0_ptr + c, tl.sum(acc0, axis=0))
        tl.store(out1_ptr + c, tl.sum(acc1, axis=0))
        tl.store(out3_ptr + c, tl.sum(acc3, axis=0))


def _shape_tuple(value):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs):
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects 9 inputs, got {len(inputs)}")

    x, weight, arg240, arg413, shape0, shape1, shape2, shape3, shape4 = inputs
    tensor_inputs = (x, weight, arg240, arg413)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first four inputs must be tensors")

    shape0 = _shape_tuple(shape0)
    shape1 = _shape_tuple(shape1)
    shape2 = _shape_tuple(shape2)
    shape3 = _shape_tuple(shape3)
    shape4 = _shape_tuple(shape4)
    batch, hidden, height, width = (int(dim) for dim in x.shape)

    expected_shape0 = (batch * hidden * (height // 2), 2, width // 2, 2)
    expected_shape1 = (batch, hidden, (height // 2) * (width // 2), 4)
    expected_shape2 = (batch * 4, (height // 2) * (width // 2), hidden)
    expected_shape3 = (batch * 4 * (height // 2) * (width // 2), hidden)
    expected_shape4 = (hidden,)
    expected = (expected_shape0, expected_shape1, expected_shape2, expected_shape3, expected_shape4)
    actual = (shape0, shape1, shape2, shape3, shape4)

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"MobileViT patch layout expects even H/W, got {(height, width)}")
    if actual != expected:
        raise ValueError(f"shape parameters do not match MobileViT layout: actual={actual}, expected={expected}")
    if tuple(weight.shape) != (hidden,):
        raise ValueError(f"expected weight shape {(hidden,)}, got {tuple(weight.shape)}")
    if tuple(arg240.shape) != expected_shape2:
        raise ValueError(f"expected arg240 shape {expected_shape2}, got {tuple(arg240.shape)}")
    if tuple(arg413.shape) != (expected_shape2[0], expected_shape2[1], 1):
        raise ValueError(f"expected arg413 shape {(expected_shape2[0], expected_shape2[1], 1)}, got {tuple(arg413.shape)}")
    return x, weight, arg240, arg413, shape0, shape1, shape2, shape3, shape4


def _torch_full_scope(inputs):
    getitem_76, arg60_1, arg240_1, arg413_1, shape0, shape1, shape2, shape3, shape4 = _validate_inputs(inputs)
    clone_default = torch.ops.aten.clone.default(getitem_76, memory_format=torch.contiguous_format)
    view_default = torch.ops.aten.view.default(clone_default, shape0)
    permute_default = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3])
    clone_default_1 = torch.ops.aten.clone.default(permute_default, memory_format=torch.contiguous_format)
    view_default_1 = torch.ops.aten.view.default(clone_default_1, shape1)
    permute_default_1 = torch.ops.aten.permute.default(view_default_1, [0, 3, 2, 1])
    clone_default_2 = torch.ops.aten.clone.default(permute_default_1, memory_format=torch.contiguous_format)
    view_default_2 = torch.ops.aten.view.default(clone_default_2, shape2)
    mul_tensor = torch.ops.aten.mul.Tensor(view_default_2, arg60_1)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, int(shape4[0]))
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, arg240_1)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(arg240_1, sum_dim_int_list_1)
    sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list)
    sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(arg413_1, sub_tensor_1)
    mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_2, arg240_1)
    sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1])
    sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(view_default_2, [0, 1])
    view_default_3 = torch.ops.aten.view.default(mul_tensor_4, shape3)
    permute_default_2 = torch.ops.aten.permute.default(view_default_3, [1, 0])
    sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True)
    view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_4, shape4)
    return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default_2, view_default_4)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    x, weight, arg240, arg413, _shape0, _shape1, shape2, shape3, _shape4 = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_full_scope(inputs)

    total_rows, hidden = shape3
    _patch_groups, patches, _hidden = shape2
    block_m = 64
    block_c = 1 << (hidden - 1).bit_length()
    n_tiles = triton.cdiv(total_rows, block_m)

    side_base = torch.empty(shape3, device=x.device, dtype=x.dtype)
    partial0 = torch.empty((n_tiles, hidden), device=x.device, dtype=torch.float32)
    partial1 = torch.empty((n_tiles, hidden), device=x.device, dtype=torch.float32)
    partial3 = torch.empty((n_tiles, hidden), device=x.device, dtype=torch.float32)
    out0 = torch.empty((hidden,), device=x.device, dtype=x.dtype)
    out1 = torch.empty((hidden,), device=x.device, dtype=x.dtype)
    out3 = torch.empty((hidden,), device=x.device, dtype=x.dtype)

    _layout_side_and_partials_kernel[(n_tiles,)](
        x,
        weight,
        arg240,
        arg413,
        side_base,
        partial0,
        partial1,
        partial3,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        total_rows,
        hidden,
        patches,
        int(x.shape[3]),
        block_m,
        block_c,
        num_warps=8,
    )
    _finalize_partials_kernel[(hidden,)](
        partial0,
        partial1,
        partial3,
        out0,
        out1,
        out3,
        hidden,
        n_tiles,
        1024,
        num_warps=8,
    )
    return (out0, out1, side_base.permute(1, 0), out3)


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
