"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileViT reverse patch-layout LayerNorm scope with Triton, including residual add, fp32 var_mean(correction=0, keepdim=True) over hidden dim 240, eps=1e-5 rsqrt, affine weight/bias, final contiguous [128,240,8,8] NCHW layout, and the sibling [512,16,1] invstd/240 output, whereas tuned Inductor already reaches the same required row-normalization plus layout-canonicalization traffic envelope for this shape; Inductor cannot materially improve it through a local norm-template rewrite because the remaining cost is the required input/residual/affine reads, one fixed-hidden reduction, side-output store, and either strided final stores or a coalesced layout pass; the fix is BANDWIDTH_BOUND: record this repro as at-floor unless a broader norm-layout memory-traffic or launch-overhead optimization moves both implementations."""
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


ROWS = 8192
GROUPS = 512
BATCH = 128
HIDDEN = 240
PATCHES = 16
PATCH_LANES = 4
HEIGHT = 8
WIDTH = 8
SPATIAL = HEIGHT * WIDTH
EPS = 1.0e-5
BLOCK_H = 256
ROW_BLOCK = 4
LAYOUT_BLOCK = 256

if triton is not None:

    @triton.jit
    def _mobilevit_reverse_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        row_out_ptr,
        side_out_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        patches: tl.constexpr,
        patch_lanes: tl.constexpr,
        width: tl.constexpr,
        spatial: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        rows = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask

        flat_offsets = rows * hidden + cols
        addmm = tl.load(addmm_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
        x = addmm + residual

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(row_out_ptr + flat_offsets, y, mask=mask)
        tl.store(side_out_ptr + rows, invstd / hidden, mask=row_mask)

    @triton.jit
    def _fold_mobilevit_layout_kernel(
        row_out_ptr,
        out_ptr,
        total: tl.constexpr,
        hidden: tl.constexpr,
        patches: tl.constexpr,
        patch_lanes: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < total

        w = offsets % width
        tmp = offsets // width
        h = tmp % height
        tmp = tmp // height
        channel = tmp % hidden
        batch = tmp // hidden

        lane = (h % 2) * 2 + (w % 2)
        patch = (h // 2) * (width // 2) + (w // 2)
        row = (batch * patch_lanes + lane) * patches + patch
        src_offsets = row * hidden + channel

        y = tl.load(row_out_ptr + src_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, y, mask=mask)


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

    expected_tensor_shapes = (
        (ROWS, HIDDEN),
        (GROUPS, PATCHES, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_tensor_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} must be torch.float32, got {value.dtype}")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    shape2_tuple = _shape_tuple(shape2)
    shape3_tuple = _shape_tuple(shape3)
    expected_shapes = (
        (GROUPS, PATCHES, HIDDEN),
        (BATCH, PATCH_LANES, PATCHES, -1),
        (BATCH * HIDDEN * PATCH_LANES, PATCH_LANES, 2, 2),
        (BATCH, HIDDEN, HEIGHT, WIDTH),
    )
    actual_shapes = (shape0_tuple, shape1_tuple, shape2_tuple, shape3_tuple)
    if actual_shapes != expected_shapes:
        raise ValueError(f"shape parameters {actual_shapes} != {expected_shapes}")

    return addmm, residual, weight, bias, shape0_tuple, shape1_tuple, shape2_tuple, shape3_tuple


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, residual, weight, bias, shape0, shape1, shape2, shape3 = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(addmm, shape0)
    add_tensor = torch.ops.aten.add.Tensor(residual, view_default)
    var, mean = torch.ops.aten.var_mean.correction(add_tensor, [2], correction=0, keepdim=True)
    rsqrt = torch.ops.aten.rsqrt.default(var + EPS)
    normalized = (add_tensor - mean) * rsqrt
    affine = normalized * weight + bias
    view_default_1 = torch.ops.aten.view.default(affine, shape1)
    permute_default = torch.ops.aten.permute.default(view_default_1, [0, 3, 2, 1])
    clone_default = torch.ops.aten.clone.default(permute_default, memory_format=torch.contiguous_format)
    view_default_2 = torch.ops.aten.view.default(clone_default, shape2)
    permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3])
    clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format=torch.contiguous_format)
    return torch.ops.aten.view.default(clone_default_1, shape3), rsqrt / HIDDEN


@oracle_impl(hardware="H100", shapes="(T([8192, 240], f32), T([512, 16, 240], f32), T([240], f32), T([240], f32), S([512, 16, 240]), S([128, 4, 16, -1]), S([122880, 4, 2, 2]), S([128, 240, 8, 8]))")
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
    addmm, residual, weight, bias, _shape0, _shape1, _shape2, shape3 = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    row_output = torch.empty_strided(
        (GROUPS, PATCHES, HIDDEN),
        (PATCHES * HIDDEN, HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    side_output = torch.empty_strided(
        (GROUPS, PATCHES, 1),
        (PATCHES, 1, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    output = torch.empty_strided(
        shape3,
        (HIDDEN * SPATIAL, SPATIAL, WIDTH, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(ROWS, ROW_BLOCK),)
    _mobilevit_reverse_layernorm_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        row_output,
        side_output,
        ROWS,
        HIDDEN,
        PATCHES,
        PATCH_LANES,
        WIDTH,
        SPATIAL,
        EPS,
        BLOCK_H,
        ROW_BLOCK,
        num_warps=8,
        num_stages=2,
    )
    _fold_mobilevit_layout_kernel[(triton.cdiv(ROWS * HIDDEN, LAYOUT_BLOCK),)](
        row_output,
        output,
        ROWS * HIDDEN,
        HIDDEN,
        PATCHES,
        PATCH_LANES,
        HEIGHT,
        WIDTH,
        LAYOUT_BLOCK,
        num_warps=4,
        num_stages=2,
    )
    return output, side_output


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
