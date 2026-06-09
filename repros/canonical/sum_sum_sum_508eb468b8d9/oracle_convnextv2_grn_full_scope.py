"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 GRN backward-style scope from Repro.forward by folding the input add, the strided NCHW-to-NHWC logical loads, the broadcasted `(arg81 - arg82) * arg83` producer, the two per-pixel channel reductions, and all three returned f32[80] sibling reductions into one shared partial-reduction pass plus a small finalizer, whereas Inductor schedules the add/permutes, row reductions, pointwise GRN-gradient algebra, layout permute, and final channel reductions as generic regions over large logical intermediates; Inductor cannot do this today because its scheduler does not form one multi-output reduction plan across reductions with different axes that share strided producers and dependent row summaries; the fix is SCHEDULER_FUSION: teach the reduction scheduler to recognize this GRN DAG, compute the row summaries once, and reuse them while accumulating all compatible channel partials."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
BATCH = 128
CHANNELS = 80
HEIGHT = 56
WIDTH = 56
HW = HEIGHT * WIDTH
TOTAL_ROWS = BATCH * HW
BLOCK_M = 64
BLOCK_C = 128
BLOCK_TILES = 1024

if triton is not None:

    @triton.jit
    def _partials_kernel(
        add77_ptr,
        getitem132_ptr,
        weight_ptr,
        arg81_ptr,
        arg82_ptr,
        arg83_ptr,
        partial0_ptr,
        partial1_ptr,
        partial2_ptr,
        add_stride_n: tl.constexpr,
        add_stride_c: tl.constexpr,
        add_stride_h: tl.constexpr,
        add_stride_w: tl.constexpr,
        getitem_stride_n: tl.constexpr,
        getitem_stride_c: tl.constexpr,
        getitem_stride_h: tl.constexpr,
        getitem_stride_w: tl.constexpr,
        arg81_stride_n: tl.constexpr,
        arg81_stride_c: tl.constexpr,
        arg81_stride_h: tl.constexpr,
        arg81_stride_w: tl.constexpr,
        arg82_stride_n: tl.constexpr,
        arg82_stride_h: tl.constexpr,
        arg82_stride_w: tl.constexpr,
        arg83_stride_n: tl.constexpr,
        arg83_stride_h: tl.constexpr,
        arg83_stride_w: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        tile = tl.program_id(0)
        c_offsets = tl.arange(0, BLOCK_C_)[:, None]
        rows = tile * BLOCK_M_ + tl.arange(0, BLOCK_M_)[None, :]
        n = rows // 3136
        hw = rows - n * 3136
        h = hw // 56
        w = hw - h * 56

        row_mask = rows < 401408
        c_mask = c_offsets < 80
        mask = row_mask & c_mask

        add_offsets = (
            n * add_stride_n
            + c_offsets * add_stride_c
            + h * add_stride_h
            + w * add_stride_w
        )
        getitem_offsets = (
            n * getitem_stride_n
            + c_offsets * getitem_stride_c
            + h * getitem_stride_h
            + w * getitem_stride_w
        )
        arg81_offsets = (
            n * arg81_stride_n
            + c_offsets * arg81_stride_c
            + h * arg81_stride_h
            + w * arg81_stride_w
        )
        row_offsets_arg82 = n * arg82_stride_n + h * arg82_stride_h + w * arg82_stride_w
        row_offsets_arg83 = n * arg83_stride_n + h * arg83_stride_h + w * arg83_stride_w

        add77 = tl.load(add77_ptr + add_offsets, mask=mask, other=0.0).to(tl.float32)
        getitem132 = tl.load(getitem132_ptr + getitem_offsets, mask=mask, other=0.0).to(tl.float32)
        x = add77 + getitem132
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        arg81 = tl.load(arg81_ptr + arg81_offsets, mask=mask, other=0.0).to(tl.float32)
        arg82 = tl.load(arg82_ptr + row_offsets_arg82, mask=row_mask, other=0.0).to(tl.float32)
        arg83 = tl.load(arg83_ptr + row_offsets_arg83, mask=row_mask, other=0.0).to(tl.float32)

        mul2 = (arg81 - arg82) * arg83
        weighted = x * weight
        row_weighted_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_weighted_mul2_sum = tl.sum(tl.where(mask, weighted * mul2, 0.0), axis=0)
        side = (arg83 * 0.0125) * (
            weighted * 80.0 - row_weighted_sum[None, :] - mul2 * row_weighted_mul2_sum[None, :]
        )

        partial0 = tl.sum(tl.where(mask, x * mul2, 0.0), axis=1)
        partial1 = tl.sum(tl.where(mask, x, 0.0), axis=1)
        partial2 = tl.sum(tl.where(mask, side, 0.0), axis=1)

        partial_c = tl.arange(0, BLOCK_C_)
        partial_offsets = tile * 80 + partial_c
        partial_mask = partial_c < 80
        tl.store(partial0_ptr + partial_offsets, partial0, mask=partial_mask)
        tl.store(partial1_ptr + partial_offsets, partial1, mask=partial_mask)
        tl.store(partial2_ptr + partial_offsets, partial2, mask=partial_mask)

    @triton.jit
    def _finalize_partials_kernel(
        partial0_ptr,
        partial1_ptr,
        partial2_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        n_tiles: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        c = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_TILES_)
        acc0 = tl.zeros([BLOCK_TILES_], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_TILES_], dtype=tl.float32)
        acc2 = tl.zeros([BLOCK_TILES_], dtype=tl.float32)

        for start in range(0, n_tiles, BLOCK_TILES_):
            tile_offsets = start + offsets
            mask = tile_offsets < n_tiles
            partial_offsets = tile_offsets * 80 + c
            acc0 += tl.load(partial0_ptr + partial_offsets, mask=mask, other=0.0)
            acc1 += tl.load(partial1_ptr + partial_offsets, mask=mask, other=0.0)
            acc2 += tl.load(partial2_ptr + partial_offsets, mask=mask, other=0.0)

        tl.store(out0_ptr + c, tl.sum(acc0, axis=0))
        tl.store(out1_ptr + c, tl.sum(acc1, axis=0))
        tl.store(out2_ptr + c, tl.sum(acc2, axis=0))


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    add77, getitem132, weight, arg81, arg82, arg83 = inputs
    add77 = _require_tensor("add_77", add77, (BATCH, CHANNELS, HEIGHT, WIDTH), torch.float32)
    getitem132 = _require_tensor(
        "getitem_132",
        getitem132,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        torch.float32,
    )
    weight = _require_tensor("arg2_1", weight, (CHANNELS,), torch.float32)
    arg81 = _require_tensor("arg81_1", arg81, (BATCH, CHANNELS, HEIGHT, WIDTH), torch.float32)
    arg82 = _require_tensor("arg82_1", arg82, (BATCH, HEIGHT, WIDTH, 1), torch.float32)
    arg83 = _require_tensor("arg83_1", arg83, (BATCH, HEIGHT, WIDTH, 1), torch.float32)

    tensor_inputs = (add77, getitem132, weight, arg81, arg82, arg83)
    if add77.is_cuda:
        if not all(t.is_cuda for t in tensor_inputs):
            raise ValueError("all tensor inputs must be CUDA tensors when add_77 is CUDA")
        if any(t.device != add77.device for t in tensor_inputs):
            raise ValueError("all tensor inputs must be on the same CUDA device")
    return add77, getitem132, weight, arg81, arg82, arg83


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]):
    add77, getitem132, weight, arg81, arg82, arg83 = _validate_inputs(inputs)
    add_tensor = torch.ops.aten.add.Tensor(add77, getitem132)
    permute_default = torch.ops.aten.permute.default(add_tensor, [0, 2, 3, 1])
    mul_tensor = torch.ops.aten.mul.Tensor(permute_default, weight)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 80)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
    permute_default_1 = torch.ops.aten.permute.default(arg81, [0, 2, 3, 1])
    sub_tensor = torch.ops.aten.sub.Tensor(permute_default_1, arg82)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, arg83)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1)
    sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list)
    sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4)
    div_tensor = torch.ops.aten.div.Tensor(arg83, 80)
    mul_tensor_5 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2)
    mul_tensor_6 = torch.ops.aten.mul.Tensor(permute_default, mul_tensor_2)
    sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1, 2])
    sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(permute_default, [0, 1, 2])
    permute_default_2 = torch.ops.aten.permute.default(mul_tensor_5, [0, 3, 1, 2])
    sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(permute_default_2, [0, 2, 3])
    return (sum_dim_int_list_2, sum_dim_int_list_3, sum_dim_int_list_4)


@oracle_impl(hardware="H100", shapes="(T([128, 80, 56, 56], f32), T([128, 80, 56, 56], f32), T([80], f32), T([128, 80, 56, 56], f32), T([128, 56, 56, 1], f32), T([128, 56, 56, 1], f32))")
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
    add77, getitem132, weight, arg81, arg82, arg83 = _validate_inputs(inputs)
    if triton is None or not add77.is_cuda:
        return _torch_full_scope(inputs)

    n_tiles = triton.cdiv(TOTAL_ROWS, BLOCK_M)
    partial0 = torch.empty((n_tiles, CHANNELS), device=add77.device, dtype=torch.float32)
    partial1 = torch.empty((n_tiles, CHANNELS), device=add77.device, dtype=torch.float32)
    partial2 = torch.empty((n_tiles, CHANNELS), device=add77.device, dtype=torch.float32)
    out0 = torch.empty((CHANNELS,), device=add77.device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=add77.device, dtype=torch.float32)
    out2 = torch.empty((CHANNELS,), device=add77.device, dtype=torch.float32)

    _partials_kernel[(n_tiles,)](
        add77,
        getitem132,
        weight,
        arg81,
        arg82,
        arg83,
        partial0,
        partial1,
        partial2,
        add77.stride(0),
        add77.stride(1),
        add77.stride(2),
        add77.stride(3),
        getitem132.stride(0),
        getitem132.stride(1),
        getitem132.stride(2),
        getitem132.stride(3),
        arg81.stride(0),
        arg81.stride(1),
        arg81.stride(2),
        arg81.stride(3),
        arg82.stride(0),
        arg82.stride(1),
        arg82.stride(2),
        arg83.stride(0),
        arg83.stride(1),
        arg83.stride(2),
        BLOCK_M_=BLOCK_M,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _finalize_partials_kernel[(CHANNELS,)](
        partial0,
        partial1,
        partial2,
        out0,
        out1,
        out2,
        n_tiles,
        BLOCK_TILES,
        num_warps=8,
        num_stages=3,
    )
    return (out0, out1, out2)


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
