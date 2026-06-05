"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileViT patch-layout rewrite and fp32 hidden-dim LayerNorm in one row Triton kernel, including the input clone, view/permute/contiguous/view/permute/contiguous/view layout semantics, var_mean(correction=0, keepdim=True) over hidden dim 240 with eps=1e-5, f32 affine weight/bias, and final [8192, 240] view, whereas Inductor lowers the generic clone/view/permute/normalization graph without recognizing that each final row is a fixed spatial 2x2 patch lane over channels; Inductor cannot do this today because its scheduler/codegen pattern library has no MobileViT patch-unfold LayerNorm template that fuses the rearrangement indexing into the row-normalization kernel; the fix is NEW_PATTERN: add a MobileViT patch LayerNorm lowering that maps output rows directly to original NCHW logical indices and emits the reduction plus affine epilogue in one kernel."""
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


if triton is not None:

    @triton.jit
    def _mobilevit_patch_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        stride_n: tl.constexpr,
        stride_c: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        width: tl.constexpr,
        patches: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
        cols = tl.arange(0, BLOCK_H)[None, :]
        col_mask = cols < hidden
        row_mask = rows < total_rows
        mask = row_mask & col_mask

        patch = rows % patches
        patch_group = rows // patches
        patch_lane = patch_group % 4
        batch = patch_group // 4
        half_width = width // 2
        patch_h = patch // half_width
        patch_w = patch % half_width

        h = patch_h * 2 + patch_lane // 2
        w = patch_w * 2 + patch_lane % 2
        x_offsets = batch * stride_n + cols * stride_c + h * stride_h + w * stride_w

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias
        tl.store(out_ptr + rows * hidden + cols, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    x, weight, bias, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("first three inputs must be tensors")
    if x.ndim != 4:
        raise ValueError(f"expected 4D input, got shape {tuple(x.shape)}")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("weight and bias must be 1D tensors")

    shape0 = _shape_tuple(shape0)
    shape1 = _shape_tuple(shape1)
    shape2 = _shape_tuple(shape2)
    shape3 = _shape_tuple(shape3)

    batch, hidden, height, width = (int(dim) for dim in x.shape)
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"MobileViT patch layout expects even H/W, got {(height, width)}")
    patches = (height // 2) * (width // 2)
    expected_shape0 = (batch * hidden * (height // 2), 2, width // 2, 2)
    expected_shape1 = (batch, hidden, patches, 4)
    expected_shape2 = (batch * 4, patches, hidden)
    expected_shape3 = (batch * 4 * patches, hidden)
    expected = (expected_shape0, expected_shape1, expected_shape2, expected_shape3)
    actual = (shape0, shape1, shape2, shape3)
    if actual != expected:
        raise ValueError(f"shape parameters do not match MobileViT patch layout: actual={actual}, expected={expected}")
    if tuple(weight.shape) != (hidden,) or tuple(bias.shape) != (hidden,):
        raise ValueError(f"expected weight/bias shape {(hidden,)}, got {tuple(weight.shape)} and {tuple(bias.shape)}")
    return x, weight, bias, shape0, shape1, shape2, shape3


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    x, weight, bias, shape0, shape1, shape2, shape3 = _validate_inputs(inputs)
    clone_default = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
    view_default = torch.ops.aten.view.default(clone_default, shape0)
    permute_default = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3])
    clone_default_1 = torch.ops.aten.clone.default(permute_default, memory_format=torch.contiguous_format)
    view_default_1 = torch.ops.aten.view.default(clone_default_1, shape1)
    permute_default_1 = torch.ops.aten.permute.default(view_default_1, [0, 3, 2, 1])
    clone_default_2 = torch.ops.aten.clone.default(permute_default_1, memory_format=torch.contiguous_format)
    view_default_2 = torch.ops.aten.view.default(clone_default_2, shape2)
    var, mean = torch.ops.aten.var_mean.correction(view_default_2, [2], correction=0, keepdim=True)
    normalized = (view_default_2 - mean) * torch.ops.aten.rsqrt.default(var + 1.0e-5)
    return torch.ops.aten.view.default(normalized * weight + bias, shape3)


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
    x, weight, bias, _shape0, _shape1, shape2, shape3 = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_full_scope(inputs)

    total_rows, hidden = shape3
    _patch_groups, patches, _hidden = shape2
    block_h = 1 << (hidden - 1).bit_length()
    row_block = 4
    output = torch.empty(shape3, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(total_rows, row_block),)
    _mobilevit_patch_layernorm_kernel[grid](
        x,
        weight,
        bias,
        output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        total_rows,
        hidden,
        int(x.shape[3]),
        patches,
        1.0e-5,
        block_h,
        row_block,
        num_warps=8,
    )
    return output


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
