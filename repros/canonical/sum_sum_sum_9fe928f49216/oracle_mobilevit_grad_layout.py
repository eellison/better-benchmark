"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileViT backward layout tail by sharing the 144-wide row reductions from `mm_72.view(512, 256, 144) * arg48_1`, accumulating the two returned `[144]` input reductions, and writing the dependent `add_75 + arg417_1 * (...)` tensor directly into the final `[128, 144, 32, 32]` 2x2 patch layout, whereas Inductor currently schedules the row reductions, sibling column reductions, dependent pointwise tensor, and clone/view/permute/clone/view/permute/clone layout chain as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen does not recognize this MobileViT patch-unfold reduction/layout pattern with shared row summaries and a required full layout-changing side output; the fix is NEW_PATTERN: add a MobileViT layout-reduction template that maps patch rows to NCHW coordinates while emitting shared row reductions, column partials, and the dependent final-layout tensor in one planned lowering."""
from __future__ import annotations

import argparse
import sys
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
    get_shape_key,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _layout_side_and_partials_kernel(
        mm_ptr,
        weight_ptr,
        arg214_ptr,
        arg417_ptr,
        add_ptr,
        out_ptr,
        partial0_ptr,
        partial1_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        patches: tl.constexpr,
        height: tl.constexpr,
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
        row_offsets = rows * hidden + cols

        mm = tl.load(mm_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        arg214 = tl.load(arg214_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        add = tl.load(add_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)

        weighted = mm * weight
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)[:, None]
        row_arg214_sum = tl.sum(tl.where(mask, weighted * arg214, 0.0), axis=1)[:, None]
        row_scale = tl.load(arg417_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        side = add + row_scale * (weighted * hidden - row_sum - arg214 * row_arg214_sum)

        group = rows // patches
        patch = rows - group * patches
        lane = group - (group // 4) * 4
        batch = group // 4
        half_width = width // 2
        out_h = (patch // half_width) * 2 + lane // 2
        out_w = (patch - (patch // half_width) * half_width) * 2 + lane - (lane // 2) * 2
        out_offsets = ((batch * hidden + cols) * height + out_h) * width + out_w
        tl.store(out_ptr + out_offsets, side, mask=mask)

        partial_cols = tl.arange(0, BLOCK_C)
        partial_mask = partial_cols < hidden
        partial0 = tl.sum(tl.where(mask, mm * arg214, 0.0), axis=0)
        partial1 = tl.sum(tl.where(mask, mm, 0.0), axis=0)
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

    mm_72, arg48, arg214, arg417, add_75, shape0, shape1, shape2, shape3 = inputs
    tensor_inputs = (mm_72, arg48, arg214, arg417, add_75)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first five inputs must be tensors")

    shape0 = _shape_tuple(shape0)
    shape1 = _shape_tuple(shape1)
    shape2 = _shape_tuple(shape2)
    shape3 = _shape_tuple(shape3)
    groups, patches, hidden = shape0
    batch, lanes, shape1_patches, shape1_hidden = shape1
    out_batch, out_hidden, out_height, out_width = shape3

    expected_shape1 = (groups // 4, 4, patches, hidden)
    expected_shape2 = (batch * hidden * (out_height // 2), out_width // 2, 2, 2)
    if shape1 != expected_shape1:
        raise ValueError(f"unexpected shape1 {shape1}, expected {expected_shape1}")
    if shape2 != expected_shape2:
        raise ValueError(f"unexpected shape2 {shape2}, expected {expected_shape2}")
    if batch != out_batch or hidden != out_hidden or lanes != 4:
        raise ValueError(f"inconsistent MobileViT shapes: shape1={shape1}, shape3={shape3}")
    if shape1_patches != patches or shape1_hidden != hidden:
        raise ValueError(f"inconsistent patch/hidden shapes: shape0={shape0}, shape1={shape1}")
    if out_height % 2 != 0 or out_width % 2 != 0:
        raise ValueError(f"MobileViT patch layout expects even H/W, got {(out_height, out_width)}")
    if patches != (out_height // 2) * (out_width // 2):
        raise ValueError(f"patch count {patches} does not match output shape {shape3}")
    if groups != batch * 4:
        raise ValueError(f"group count {groups} does not match batch/lanes in {shape1}")

    expected_tensors = {
        "mm_72": (mm_72, (groups * patches, hidden), torch.float32),
        "arg48_1": (arg48, (hidden,), torch.float32),
        "arg214_1": (arg214, (groups, patches, hidden), torch.float32),
        "arg417_1": (arg417, (groups, patches, 1), torch.float32),
        "add_75": (add_75, (groups, patches, hidden), torch.float32),
    }
    for name, (tensor, expected_shape, expected_dtype) in expected_tensors.items():
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} expected shape={expected_shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != expected_dtype:
            raise ValueError(f"{name} expected dtype={expected_dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for this oracle")

    return mm_72, arg48, arg214, arg417, add_75, shape0, shape1, shape2, shape3


def _torch_full_scope(inputs):
    mm_72, arg48_1, arg214_1, arg417_1, add_75, shape0, shape1, shape2, shape3 = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(mm_72, shape0)
    mul_tensor = torch.ops.aten.mul.Tensor(view_default, arg48_1)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, int(shape0[2]))
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, arg214_1)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(arg214_1, sum_dim_int_list_1)
    sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list)
    sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(arg417_1, sub_tensor_1)
    mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default, arg214_1)
    sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1])
    sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(view_default, [0, 1])
    add_tensor = torch.ops.aten.add.Tensor(add_75, mul_tensor_4)
    view_default_1 = torch.ops.aten.view.default(add_tensor, shape1)
    permute_default = torch.ops.aten.permute.default(view_default_1, [0, 3, 2, 1])
    clone_default = torch.ops.aten.clone.default(permute_default, memory_format=torch.contiguous_format)
    view_default_2 = torch.ops.aten.view.default(clone_default, shape2)
    permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3])
    clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format=torch.contiguous_format)
    view_default_3 = torch.ops.aten.view.default(clone_default_1, shape3)
    return (sum_dim_int_list_2, sum_dim_int_list_3, view_default_3)


def oracle_forward(inputs):
    """Run the full compiled computation scope for the oracle."""
    mm_72, arg48, arg214, arg417, add_75, shape0, _shape1, _shape2, shape3 = _validate_inputs(inputs)
    if triton is None or not mm_72.is_cuda:
        return _torch_full_scope(inputs)

    groups, patches, hidden = shape0
    batch, _out_hidden, out_height, out_width = shape3
    total_rows = groups * patches
    block_m = 32
    block_c = 1 << (hidden - 1).bit_length()
    n_tiles = triton.cdiv(total_rows, block_m)

    out0 = torch.empty((hidden,), device=mm_72.device, dtype=mm_72.dtype)
    out1 = torch.empty((hidden,), device=mm_72.device, dtype=mm_72.dtype)
    out2 = torch.empty((batch, hidden, out_height, out_width), device=mm_72.device, dtype=mm_72.dtype)
    partial0 = torch.empty((n_tiles, hidden), device=mm_72.device, dtype=torch.float32)
    partial1 = torch.empty((n_tiles, hidden), device=mm_72.device, dtype=torch.float32)

    _layout_side_and_partials_kernel[(n_tiles,)](
        mm_72,
        arg48,
        arg214,
        arg417,
        add_75,
        out2,
        partial0,
        partial1,
        total_rows,
        hidden,
        patches,
        out_height,
        out_width,
        block_m,
        block_c,
        num_warps=8,
    )
    _finalize_partials_kernel[(hidden,)](
        partial0,
        partial1,
        out0,
        out1,
        hidden,
        n_tiles,
        1024,
        num_warps=8,
    )
    return (out0, out1, out2)


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
