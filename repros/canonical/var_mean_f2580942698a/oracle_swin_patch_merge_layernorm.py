"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin patch-merge residual add, 2x2 window-layout clone/view, population var_mean LayerNorm over the flattened patch channels, affine epilogue, and final contiguous output in one Triton row kernel, whereas Inductor currently materializes the residual add and patch-merge clone/view before scheduling a generic var_mean LayerNorm and affine pointwise epilogue; Inductor cannot do this today because its scheduler/codegen pattern library has no Swin patch-merge LayerNorm template that sinks the 2x2 layout transform and residual/addmm loads into the normalization row schedule; the fix is NEW_PATTERN: add a guarded Swin patch-merge LayerNorm lowering that maps output rows directly to source spatial rows and emits residual add, reduction, affine, and final contiguous stores in one kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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


EPS = 1.0e-5


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _swin_patch_merge_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        addmm_s0: tl.constexpr,
        addmm_s1: tl.constexpr,
        residual_s0: tl.constexpr,
        residual_s1: tl.constexpr,
        residual_s2: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        channels: tl.constexpr,
        grid_h: tl.constexpr,
        grid_w: tl.constexpr,
        inner_h: tl.constexpr,
        inner_w: tl.constexpr,
        hidden_out: tl.constexpr,
        total_rows: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        col_mask = cols < hidden_out
        mask = (row < total_rows) & col_mask

        batch_idx = row // (grid_h * grid_w)
        patch = row - batch_idx * grid_h * grid_w
        patch_h = patch // grid_w
        patch_w = patch - patch_h * grid_w

        channel = cols % channels
        patch_col = cols // channels
        inner_h_idx = patch_col % inner_h
        inner_w_idx = patch_col // inner_h
        src_h = patch_h * inner_h + inner_h_idx
        src_w = patch_w * inner_w + inner_w_idx
        spatial = src_h * width + src_w
        addmm_row = batch_idx * height * width + spatial

        addmm_offsets = addmm_row * addmm_s0 + channel * addmm_s1
        residual_offsets = batch_idx * residual_s0 + spatial * residual_s1 + channel * residual_s2
        x = (
            tl.load(
                addmm_ptr + addmm_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            + tl.load(
                residual_ptr + residual_offsets,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
        )

        x_for_reduce = tl.where(col_mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden_out
        centered = x - mean
        variance = tl.sum(tl.where(col_mask, centered * centered, 0.0), axis=0) / hidden_out
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y = centered * invstd * weight + bias

        out_offsets = row * out_s0 + cols * out_s1
        tl.store(out_ptr + out_offsets, y, mask=mask)


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
    tuple[int, int, int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int],
]:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects 9 inputs, got {len(inputs)}")

    addmm_87, residual, weight, bias, shape0, shape1, shape2, shape3, shape4 = inputs

    tensor_inputs = (addmm_87, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")
    if not all(value.dtype == torch.float32 for value in tensor_inputs):
        raise TypeError("all tensor inputs must be torch.float32")
    if addmm_87.ndim != 2 or residual.ndim != 3 or weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("expected addmm[rows, channels], residual[batch, spatial, channels], and 1D affine tensors")
    if not all(value.device == addmm_87.device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same device")
    if not addmm_87.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("all tensor inputs must be contiguous")

    shape0 = _shape_tuple(shape0)
    shape1 = _shape_tuple(shape1)
    shape2 = _shape_tuple(shape2)
    shape3 = _shape_tuple(shape3)
    shape4 = _shape_tuple(shape4)

    batch, spatial_size, channels = (int(dim) for dim in residual.shape)
    if tuple(addmm_87.shape) != (batch * spatial_size, channels):
        raise ValueError(f"addmm_87 shape {tuple(addmm_87.shape)} does not match residual shape {tuple(residual.shape)}")
    if shape0 != (batch, spatial_size, channels):
        raise ValueError(f"unexpected addmm view shape: {shape0!r}")
    if len(shape1) != 4 or shape1[0] != batch or shape1[3] != channels:
        raise ValueError(f"unexpected spatial view shape: {shape1!r}")

    height, width = shape1[1], shape1[2]
    if height * width != spatial_size:
        raise ValueError(f"shape1 spatial product {height * width} != residual spatial size {spatial_size}")
    if len(shape2) != 6 or shape2[0] != batch or shape2[5] != channels:
        raise ValueError(f"unexpected patch view shape: {shape2!r}")

    grid_h, inner_h, grid_w, inner_w = shape2[1], shape2[2], shape2[3], shape2[4]
    if grid_h * inner_h != height or grid_w * inner_w != width:
        raise ValueError(f"patch view {shape2!r} does not tile spatial shape {(height, width)}")

    hidden_out = inner_h * inner_w * channels
    if shape3 != (batch, grid_h, grid_w, hidden_out):
        raise ValueError(f"unexpected normalized view shape: {shape3!r}")
    if shape4 != (batch * grid_h * grid_w, hidden_out):
        raise ValueError(f"unexpected output shape: {shape4!r}")
    if tuple(weight.shape) != (hidden_out,) or tuple(bias.shape) != (hidden_out,):
        raise ValueError(f"expected affine tensors of shape {(hidden_out,)}, got {tuple(weight.shape)} and {tuple(bias.shape)}")

    return addmm_87, residual, weight, bias, shape0, shape1, shape2, shape3, shape4


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm_87, residual, weight, bias, shape0, shape1, shape2, shape3, shape4 = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(addmm_87, shape0)
    add_tensor = torch.ops.aten.add.Tensor(residual, view_default)
    view_default_1 = torch.ops.aten.view.default(add_tensor, shape1)
    view_default_2 = torch.ops.aten.view.default(view_default_1, shape2)
    permute_default = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 4, 2, 5])
    clone_default = torch.ops.aten.clone.default(permute_default, memory_format=torch.contiguous_format)
    view_default_3 = torch.ops.aten.view.default(clone_default, shape3)
    variance, mean = torch.ops.aten.var_mean.correction(
        view_default_3,
        [3],
        correction=0,
        keepdim=True,
    )
    normalized = (view_default_3 - mean) * torch.ops.aten.rsqrt.default(variance + EPS)
    affine = normalized * weight + bias
    return torch.ops.aten.view.default(affine, shape4)


@oracle_impl(hardware="H100", shapes="(T([25088, 512], f32), T([128, 196, 512], f32), T([2048], f32), T([2048], f32), S([128, 196, 512]), S([128, 14, 14, 512]), S([128, 7, 2, 7, 2, 512]), S([128, 7, 7, 2048]), S([6272, 2048]))")
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
    addmm_87, residual, weight, bias, _shape0, shape1, shape2, _shape3, shape4 = _validate_inputs(inputs)
    if triton is None or not addmm_87.is_cuda:
        return _torch_full_scope(inputs)

    batch, height, width, channels = shape1
    _batch, grid_h, inner_h, grid_w, inner_w, _channels = shape2
    total_rows, hidden_out = shape4
    block_h = 1 << (hidden_out - 1).bit_length()

    output = torch.empty_strided(
        shape4,
        (hidden_out, 1),
        device=addmm_87.device,
        dtype=addmm_87.dtype,
    )
    _swin_patch_merge_layernorm_kernel[(total_rows,)](
        addmm_87,
        residual,
        weight,
        bias,
        output,
        addmm_87.stride(0),
        addmm_87.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        output.stride(0),
        output.stride(1),
        height,
        width,
        channels,
        grid_h,
        grid_w,
        inner_h,
        inner_w,
        hidden_out,
        total_rows,
        EPS,
        block_h,
        num_warps=8,
        num_stages=3,
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
