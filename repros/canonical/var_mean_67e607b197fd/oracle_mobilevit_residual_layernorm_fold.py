"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileViT residual-add LayerNorm and patch-fold return with Triton kernels, including the `[8192, C] -> [512, patches, C]` view, residual add, fp32 `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 affine LayerNorm, and both contiguous patch-layout rewrites into the final `[128, C, H, W]` tensor, whereas tuned Inductor already lands in the same memory-traffic envelope for the required normalization and layout materialization; Inductor cannot materially improve this local repro through scheduler fusion, split-K, recompute fusion, or algebraic elimination because the remaining work is dominated by mandatory activation/residual/affine reads, row reduction, normalized intermediate movement, and final contiguous layout stores; the fix is BANDWIDTH_BOUND: record this as an at-floor MobileViT residual LayerNorm fold unless broader normalization or layout-copy improvements move both paths."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

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


EPS = 1.0e-5
PATCH_LANES = 4
PATCH_H = 2
PATCH_W = 2
ROW_BLOCK = 4


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _mobilevit_residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        norm_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        rows = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask

        offsets = rows * hidden + cols
        x = (
            tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
            + tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        ).to(tl.float32)

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias
        tl.store(norm_ptr + offsets, y, mask=mask)

    @triton.jit
    def _mobilevit_patch_fold_kernel(
        norm_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        hidden: tl.constexpr,
        patches: tl.constexpr,
        num_patch_w: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        patch_lanes: tl.constexpr,
        patch_w_size: tl.constexpr,
        block_n: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_n + tl.arange(0, block_n)
        mask = offsets < n_elements

        w = offsets % width
        tmp = offsets // width
        h = tmp % height
        tmp = tmp // height
        c = tmp % hidden
        batch = tmp // hidden

        lane = (h % 2) * patch_w_size + (w % 2)
        patch = (h // 2) * num_patch_w + (w // 2)
        row = (batch * patch_lanes + lane) * patches + patch
        values = tl.load(norm_ptr + row * hidden + c, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, values, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    addmm, residual, weight, bias, shape0, shape1, shape2, shape3 = inputs
    tensor_inputs = (addmm, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")
    for index, value in enumerate(tensor_inputs):
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} must be torch.float32, got {value.dtype}")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    shape2_tuple = _shape_tuple(shape2)
    shape3_tuple = _shape_tuple(shape3)

    if len(shape3_tuple) != 4:
        raise ValueError(f"final output shape must be 4D, got {shape3_tuple}")
    batch, hidden, height, width = shape3_tuple
    if height % PATCH_H != 0 or width % PATCH_W != 0:
        raise ValueError(f"expected even output H/W, got {(height, width)}")
    num_patch_h = height // PATCH_H
    num_patch_w = width // PATCH_W
    patches = num_patch_h * num_patch_w
    total_rows = batch * PATCH_LANES * patches

    expected_shape0 = (batch * PATCH_LANES, patches, hidden)
    expected_shape1 = (batch, PATCH_LANES, patches, -1)
    expected_shape1_resolved = (batch, PATCH_LANES, patches, hidden)
    expected_shape2 = (batch * hidden * num_patch_h, num_patch_w, PATCH_H, PATCH_W)
    if shape0_tuple != expected_shape0:
        raise ValueError(f"shape0 {shape0_tuple} != {expected_shape0}")
    if shape1_tuple not in (expected_shape1, expected_shape1_resolved):
        raise ValueError(f"shape1 {shape1_tuple} != {expected_shape1} or {expected_shape1_resolved}")
    if shape2_tuple != expected_shape2:
        raise ValueError(f"shape2 {shape2_tuple} != {expected_shape2}")

    expected_tensor_shapes = (
        (total_rows, hidden),
        expected_shape0,
        (hidden,),
        (hidden,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_tensor_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    return addmm, residual, weight, bias, shape0_tuple, shape1_tuple, shape2_tuple, shape3_tuple


@oracle_impl(hardware="H100", shapes="(T([8192, 240], f32), T([512, 16, 240], f32), T([240], f32), T([240], f32), S([512, 16, 240]), S([128, 4, 16, -1]), S([122880, 4, 2, 2]), S([128, 240, 8, 8]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward residual LayerNorm plus patch-fold layout."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    addmm, residual, weight, bias, _shape0, _shape1, _shape2, shape3 = _validate_inputs(inputs)
    batch, hidden, height, width = shape3
    patches = (height // PATCH_H) * (width // PATCH_W)
    total_rows = batch * PATCH_LANES * patches
    block_h = 1 << (hidden - 1).bit_length()

    output = torch.empty_strided(
        shape3,
        (hidden * height * width, height * width, width, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    norm = torch.empty((total_rows, hidden), device=addmm.device, dtype=torch.float32)
    _mobilevit_residual_layernorm_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
        addmm,
        residual,
        weight,
        bias,
        norm,
        total_rows,
        hidden,
        EPS,
        block_h,
        ROW_BLOCK,
        num_warps=1,
        num_stages=3,
    )
    _mobilevit_patch_fold_kernel[(triton.cdiv(output.numel(), 256),)](
        norm,
        output,
        output.numel(),
        hidden,
        patches,
        width // PATCH_W,
        height,
        width,
        PATCH_LANES,
        PATCH_W,
        256,
        num_warps=4,
        num_stages=3,
    )
    return output


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
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
