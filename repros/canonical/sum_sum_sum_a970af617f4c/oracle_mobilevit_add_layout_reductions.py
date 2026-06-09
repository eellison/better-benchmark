"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileViT backward scope by fusing the hidden-dimension row reductions, the two sibling channel reductions, the dependent residual add, and the final 2x2 patch layout restore directly into the returned f32[128,144,32,32] tensor, whereas Inductor currently schedules the row reductions, column reductions, residual add, and reshape/permute/clone layout pipeline as generic separate regions; Inductor cannot do this today because its scheduler/codegen does not recognize this MobileViT patch-fold reduction with a dependent materialized side output as one reusable full-scope lowering; the fix is NEW_PATTERN: add a MobileViT patch-fold reduction template that maps flattened patch rows to NCHW stores while sharing the row summaries and channel-reduction partials."""
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
    oracle_impl,
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
    def _fused_add_layout_and_partials_kernel(
        mm_ptr,
        weight_ptr,
        mul119_ptr,
        div55_ptr,
        add318_ptr,
        out_nchw_ptr,
        partial0_ptr,
        partial1_ptr,
        mm_stride0: tl.constexpr,
        mm_stride1: tl.constexpr,
        mul_stride0: tl.constexpr,
        mul_stride1: tl.constexpr,
        mul_stride2: tl.constexpr,
        div_stride0: tl.constexpr,
        div_stride1: tl.constexpr,
        add_stride0: tl.constexpr,
        add_stride1: tl.constexpr,
        add_stride2: tl.constexpr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        patches: tl.constexpr,
        out_h: tl.constexpr,
        out_w: tl.constexpr,
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
        n = patch_group // 4
        patch_w = out_w // 2
        h = (patch // patch_w) * 2 + lane // 2
        w = (patch - (patch // patch_w) * patch_w) * 2 + lane % 2

        mm_offsets = rows * mm_stride0 + cols * mm_stride1
        mul_offsets = patch_group * mul_stride0 + patch * mul_stride1 + cols * mul_stride2
        div_offsets = patch_group * div_stride0 + patch * div_stride1
        add_offsets = patch_group * add_stride0 + patch * add_stride1 + cols * add_stride2
        out_offsets = ((n * hidden + cols) * out_h + h) * out_w + w

        x = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        mul119 = tl.load(mul119_ptr + mul_offsets, mask=mask, other=0.0).to(tl.float32)
        div55 = tl.load(div55_ptr + div_offsets, mask=row_mask, other=0.0).to(tl.float32)
        add318 = tl.load(add318_ptr + add_offsets, mask=mask, other=0.0).to(tl.float32)

        weighted = x * weight
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)[:, None]
        row_mul_sum = tl.sum(tl.where(mask, weighted * mul119, 0.0), axis=1)[:, None]
        side = div55 * (weighted * hidden - row_sum - mul119 * row_mul_sum)
        out = add318 + side
        tl.store(out_nchw_ptr + out_offsets, out, mask=mask)

        partial_cols = tl.arange(0, BLOCK_C)
        partial_mask = partial_cols < hidden
        partial0 = tl.sum(tl.where(mask, x * mul119, 0.0), axis=0)
        partial1 = tl.sum(tl.where(mask, x, 0.0), axis=0)
        partial_offsets = tile * hidden + partial_cols
        tl.store(partial0_ptr + partial_offsets, partial0, mask=partial_mask)
        tl.store(partial1_ptr + partial_offsets, partial1, mask=partial_mask)

    @triton.jit
    def _finalize_partials_kernel(
        partial0_ptr,
        partial1_ptr,
        out0_ptr,
        out1_ptr,
        hidden: tl.constexpr,
        n_tiles: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        c = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_TILES)
        acc0 = tl.zeros([BLOCK_TILES], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_TILES], dtype=tl.float32)

        for start in range(0, n_tiles, BLOCK_TILES):
            tile_offsets = start + offsets
            mask = tile_offsets < n_tiles
            partial_offsets = tile_offsets * hidden + c
            acc0 += tl.load(partial0_ptr + partial_offsets, mask=mask, other=0.0)
            acc1 += tl.load(partial1_ptr + partial_offsets, mask=mask, other=0.0)

        tl.store(out0_ptr + c, tl.sum(acc0, axis=0))
        tl.store(out1_ptr + c, tl.sum(acc1, axis=0))


def _shape_tuple(value):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs):
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects 9 inputs, got {len(inputs)}")

    mm_72, primals_105, mul_119, div_55, add_318, shape0, shape1, shape2, shape3 = inputs
    tensor_inputs = (mm_72, primals_105, mul_119, div_55, add_318)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first five inputs must be tensors")

    shape0 = _shape_tuple(shape0)
    shape1 = _shape_tuple(shape1)
    shape2 = _shape_tuple(shape2)
    shape3 = _shape_tuple(shape3)
    total_rows, hidden = (int(dim) for dim in mm_72.shape)

    if len(shape0) != 3 or len(shape1) != 4 or len(shape2) != 4 or len(shape3) != 4:
        raise ValueError(f"unexpected shape parameter ranks: {(shape0, shape1, shape2, shape3)}")
    if shape0 != tuple(mul_119.shape) or shape0 != tuple(add_318.shape):
        raise ValueError(
            f"expected mul_119/add_318 shape {shape0}, got {tuple(mul_119.shape)} and {tuple(add_318.shape)}"
        )
    if shape0[0] * shape0[1] != total_rows or shape0[2] != hidden:
        raise ValueError(f"shape0={shape0} does not match mm_72 shape {tuple(mm_72.shape)}")
    if tuple(primals_105.shape) != (hidden,):
        raise ValueError(f"expected primals_105 shape {(hidden,)}, got {tuple(primals_105.shape)}")
    if tuple(div_55.shape) != (shape0[0], shape0[1], 1):
        raise ValueError(f"expected div_55 shape {(shape0[0], shape0[1], 1)}, got {tuple(div_55.shape)}")

    batch, patch_area, patches, shape1_hidden = shape1
    out_batch, out_hidden, out_h, out_w = shape3
    if shape1_hidden != hidden or patches != shape0[1] or batch * patch_area != shape0[0]:
        raise ValueError(f"shape1={shape1} is not compatible with shape0={shape0}")
    if patch_area != 4:
        raise ValueError(f"expected 2x2 MobileViT patch area, got {patch_area}")
    if (out_batch, out_hidden) != (batch, hidden) or out_h % 2 != 0 or out_w % 2 != 0:
        raise ValueError(f"shape3={shape3} is not compatible with shape1={shape1}")
    if patches != (out_h // 2) * (out_w // 2):
        raise ValueError(f"patch count {patches} does not match output spatial shape {shape3}")

    expected_shape2 = (batch * hidden * (out_h // 2), out_w // 2, 2, 2)
    if shape2 != expected_shape2:
        raise ValueError(f"shape2={shape2} does not match expected {expected_shape2}")

    return mm_72, primals_105, mul_119, div_55, add_318, shape0, shape1, shape2, shape3


def _torch_full_scope(inputs):
    mm_72, primals_105, mul_119, div_55, add_318, shape0, shape1, shape2, shape3 = _validate_inputs(inputs)
    reshape_default = torch.ops.aten.reshape.default(mm_72, shape0)
    mul_tensor = torch.ops.aten.mul.Tensor(reshape_default, primals_105)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, shape0[2])
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, mul_119)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_119, sum_dim_int_list_1)
    sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list)
    sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(div_55, sub_tensor_1)
    mul_tensor_5 = torch.ops.aten.mul.Tensor(reshape_default, mul_119)
    sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1])
    sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1])
    add_tensor = torch.ops.aten.add.Tensor(add_318, mul_tensor_4)
    reshape_default_1 = torch.ops.aten.reshape.default(add_tensor, shape1)
    permute_default = torch.ops.aten.permute.default(reshape_default_1, [0, 3, 2, 1])
    clone_default = torch.ops.aten.clone.default(permute_default, memory_format=torch.contiguous_format)
    reshape_default_2 = torch.ops.aten.reshape.default(clone_default, shape2)
    permute_default_1 = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3])
    clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format=torch.contiguous_format)
    reshape_default_3 = torch.ops.aten.reshape.default(clone_default_1, shape3)
    return (sum_dim_int_list_2, sum_dim_int_list_3, reshape_default_3)


@oracle_impl(hardware="H100", shapes="(T([131072, 144], f32), T([144], f32), T([512, 256, 144], f32), T([512, 256, 1], f32), T([512, 256, 144], f32), S([512, 256, 144]), S([128, 4, 256, 144]), S([294912, 16, 2, 2]), S([128, 144, 32, 32]))")
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
    mm_72, primals_105, mul_119, div_55, add_318, shape0, shape1, _shape2, shape3 = _validate_inputs(inputs)
    if triton is None or not mm_72.is_cuda:
        return _torch_full_scope(inputs)

    total_rows, hidden = (int(dim) for dim in mm_72.shape)
    batch, _patch_area, patches, _hidden = shape1
    _out_batch, _out_hidden, out_h, out_w = shape3
    block_m = 16
    block_c = 1 << (hidden - 1).bit_length()
    n_tiles = triton.cdiv(total_rows, block_m)

    out0 = torch.empty((hidden,), device=mm_72.device, dtype=mm_72.dtype)
    out1 = torch.empty((hidden,), device=mm_72.device, dtype=mm_72.dtype)
    out_nchw = torch.empty(shape3, device=mm_72.device, dtype=mm_72.dtype)
    partial0 = torch.empty((n_tiles, hidden), device=mm_72.device, dtype=torch.float32)
    partial1 = torch.empty((n_tiles, hidden), device=mm_72.device, dtype=torch.float32)

    _fused_add_layout_and_partials_kernel[(n_tiles,)](
        mm_72,
        primals_105,
        mul_119,
        div_55,
        add_318,
        out_nchw,
        partial0,
        partial1,
        mm_72.stride(0),
        mm_72.stride(1),
        mul_119.stride(0),
        mul_119.stride(1),
        mul_119.stride(2),
        div_55.stride(0),
        div_55.stride(1),
        add_318.stride(0),
        add_318.stride(1),
        add_318.stride(2),
        total_rows,
        hidden,
        patches,
        out_h,
        out_w,
        block_m,
        block_c,
        num_warps=4,
    )
    _finalize_partials_kernel[(hidden,)](
        partial0,
        partial1,
        out0,
        out1,
        hidden,
        n_tiles,
        4096,
        num_warps=8,
    )
    return (out0, out1, out_nchw)


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
