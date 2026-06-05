"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 GRN backward-style multi-output reduction from Repro.forward, including the NCHW-to-NHWC logical layout, per-pixel channel reductions, sibling channel reductions, and final scaled epilogue reduction in one fused partial-reduction pass plus a small finalizer, whereas Inductor currently lowers the permute, row reductions, pointwise GRN-gradient algebra, layout permute, and channel reductions as generic scheduled regions; Inductor cannot do this today because its scheduler does not form one multi-output reduction plan across reductions with different axes that share a strided layout producer and dependent row summaries; the fix is SCHEDULER_FUSION: teach the reduction scheduler to fuse this GRN DAG by computing channel row summaries once and reusing them for all channel partials and the final reduction."""
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
        x_ptr,
        weight_ptr,
        arg102_ptr,
        scale_ptr,
        partial0_ptr,
        partial1_ptr,
        partial2_ptr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        arg102_stride_n: tl.constexpr,
        arg102_stride_h: tl.constexpr,
        arg102_stride_w: tl.constexpr,
        arg102_stride_c: tl.constexpr,
        scale_stride_n: tl.constexpr,
        scale_stride_h: tl.constexpr,
        scale_stride_w: tl.constexpr,
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

        x_offsets = (
            n * x_stride_n
            + c_offsets * x_stride_c
            + h * x_stride_h
            + w * x_stride_w
        )
        arg102_offsets = (
            n * arg102_stride_n
            + h * arg102_stride_h
            + w * arg102_stride_w
            + c_offsets * arg102_stride_c
        )
        scale_offsets = n * scale_stride_n + h * scale_stride_h + w * scale_stride_w

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        arg102 = tl.load(arg102_ptr + arg102_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + scale_offsets, mask=row_mask, other=0.0).to(tl.float32)

        weighted = x * weight
        row_weighted_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_weighted_arg_sum = tl.sum(tl.where(mask, weighted * arg102, 0.0), axis=0)
        side = scale * (weighted * 80.0 - row_weighted_sum[None, :] - arg102 * row_weighted_arg_sum[None, :])

        partial0 = tl.sum(tl.where(mask, x * arg102, 0.0), axis=1)
        partial1 = tl.sum(tl.where(mask, x, 0.0), axis=1)
        partial2 = tl.sum(tl.where(mask, side, 0.0), axis=1)

        partial_offsets = tile * 80 + tl.arange(0, BLOCK_C_)
        final_c_mask = tl.arange(0, BLOCK_C_) < 80
        tl.store(partial0_ptr + partial_offsets, partial0, mask=final_c_mask)
        tl.store(partial1_ptr + partial_offsets, partial1, mask=final_c_mask)
        tl.store(partial2_ptr + partial_offsets, partial2, mask=final_c_mask)

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    x, weight, arg102, scale = inputs
    x = _require_tensor("getitem_114", x, (BATCH, CHANNELS, HEIGHT, WIDTH), torch.float32)
    weight = _require_tensor("arg13_1", weight, (CHANNELS,), torch.float32)
    arg102 = _require_tensor("arg102_1", arg102, (BATCH, HEIGHT, WIDTH, CHANNELS), torch.float32)
    scale = _require_tensor("arg221_1", scale, (BATCH, HEIGHT, WIDTH, 1), torch.float32)

    if x.is_cuda and not (weight.is_cuda and arg102.is_cuda and scale.is_cuda):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if x.is_cuda and any(t.device != x.device for t in (weight, arg102, scale)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, weight, arg102, scale


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]):
    x, weight, arg102, scale = _validate_inputs(inputs)
    permute_default = torch.ops.aten.permute.default(x, [0, 2, 3, 1])
    mul_tensor = torch.ops.aten.mul.Tensor(permute_default, weight)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 80)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, arg102)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True)
    mul_tensor_3 = torch.ops.aten.mul.Tensor(arg102, sum_dim_int_list_1)
    sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list)
    sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3)
    mul_tensor_4 = torch.ops.aten.mul.Tensor(scale, sub_tensor_1)
    mul_tensor_5 = torch.ops.aten.mul.Tensor(permute_default, arg102)
    sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1, 2])
    sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(permute_default, [0, 1, 2])
    permute_default_1 = torch.ops.aten.permute.default(mul_tensor_4, [0, 3, 1, 2])
    sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(permute_default_1, [0, 2, 3])
    return (sum_dim_int_list_2, sum_dim_int_list_3, sum_dim_int_list_4)


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
    x, weight, arg102, scale = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_full_scope(inputs)

    n_tiles = triton.cdiv(TOTAL_ROWS, BLOCK_M)
    partial0 = torch.empty((n_tiles, CHANNELS), device=x.device, dtype=torch.float32)
    partial1 = torch.empty((n_tiles, CHANNELS), device=x.device, dtype=torch.float32)
    partial2 = torch.empty((n_tiles, CHANNELS), device=x.device, dtype=torch.float32)
    out0 = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    out2 = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)

    _partials_kernel[(n_tiles,)](
        x,
        weight,
        arg102,
        scale,
        partial0,
        partial1,
        partial2,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        arg102.stride(0),
        arg102.stride(1),
        arg102.stride(2),
        arg102.stride(3),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
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
