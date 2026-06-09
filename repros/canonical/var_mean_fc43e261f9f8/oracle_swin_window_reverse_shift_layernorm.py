"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin window-reverse, cyclic spatial shift by 11 on both 14x14 axes, residual add, fp32 hidden-size-512 population LayerNorm, affine epilogue, and final contiguous [25088,512] flatten directly from the windowed source and residual tensors, whereas Inductor currently materializes the window-reverse clone, generated-index gather, residual add, generic var_mean LayerNorm, affine pointwise, and final view as separate scheduled work; Inductor cannot do this today because its scheduler/codegen pattern library has no Swin window-reverse shifted residual LayerNorm template that sinks the layout/index loads into the normalization row schedule; the fix is NEW_PATTERN: add a guarded Swin window-reverse shifted LayerNorm lowering that maps output rows directly to source window rows and emits residual add, reduction, affine, and final layout stores in one kernel."""
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
SHIFT = 11


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _swin_window_reverse_shift_layernorm_kernel(
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
        residual_s3: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        batch: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        hidden: tl.constexpr,
        grid_h: tl.constexpr,
        grid_w: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        shift: tl.constexpr,
        eps: tl.constexpr,
        total_rows: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = (row < total_rows) & (cols < hidden)

        out_w = row % width
        tmp = row // width
        out_h = tmp % height
        batch_idx = tmp // height

        src_h = (out_h + shift) % height
        src_w = (out_w + shift) % width
        window_row = src_h // window_h
        window_col = src_w // window_w
        inner_h = src_h - window_row * window_h
        inner_w = src_w - window_col * window_w

        addmm_row = (
            ((batch_idx * grid_h + window_row) * grid_w + window_col)
            * (window_h * window_w)
            + inner_h * window_w
            + inner_w
        )

        addmm_offsets = addmm_row * addmm_s0 + cols * addmm_s1
        residual_offsets = (
            batch_idx * residual_s0
            + out_h * residual_s1
            + out_w * residual_s2
            + cols * residual_s3
        )
        x = (
            tl.load(addmm_ptr + addmm_offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
        )

        x_for_reduce = tl.where(cols < hidden, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(cols < hidden, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        out_offsets = row * out_s0 + cols * out_s1
        tl.store(out_ptr + out_offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _dim_matches(actual: int, expected: int) -> bool:
    return actual == -1 or actual == expected


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
    tuple[int, int, int],
    tuple[int, int],
]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        addmm_85,
        residual,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
    ) = inputs

    tensor_inputs = (addmm_85, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensor inputs are required for the Triton oracle")
    if not all(value.dtype == torch.float32 for value in tensor_inputs):
        raise TypeError("all tensor inputs must be torch.float32")
    if addmm_85.ndim != 2 or residual.ndim != 4 or weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("expected addmm[rows, hidden], residual[batch, height, width, hidden], and 1D affine tensors")
    if not all(value.is_contiguous() for value in tensor_inputs):
        raise ValueError("all tensor inputs must be contiguous")

    shape0 = _shape_tuple(shape0)
    shape1 = _shape_tuple(shape1)
    shape2 = _shape_tuple(shape2)
    shape3 = _shape_tuple(shape3)
    shape4 = _shape_tuple(shape4)
    shape5 = _shape_tuple(shape5)

    batch, height, width, hidden = (int(dim) for dim in residual.shape)
    total_rows = batch * height * width
    if tuple(addmm_85.shape) != (total_rows, hidden):
        raise ValueError(f"addmm_85 shape {tuple(addmm_85.shape)} does not match residual shape {tuple(residual.shape)}")
    if tuple(weight.shape) != (hidden,) or tuple(bias.shape) != (hidden,):
        raise ValueError(f"expected affine tensors of shape {(hidden,)}, got {tuple(weight.shape)} and {tuple(bias.shape)}")

    if len(shape0) != 3 or shape0[2] != hidden:
        raise ValueError(f"unexpected first window view shape: {shape0!r}")
    num_windows, window_area, _hidden = shape0
    if len(shape1) != 4 or shape1[3] != hidden:
        raise ValueError(f"unexpected 4D window view shape: {shape1!r}")
    if not _dim_matches(shape1[0], num_windows):
        raise ValueError(f"shape1 first dim {shape1[0]} does not match number of windows {num_windows}")
    window_h, window_w = shape1[1], shape1[2]
    if window_h * window_w != window_area:
        raise ValueError(f"window area {window_area} does not match window shape {(window_h, window_w)}")

    if len(shape2) != 6 or shape2[5] != hidden:
        raise ValueError(f"unexpected reverse-window view shape: {shape2!r}")
    if not _dim_matches(shape2[0], batch):
        raise ValueError(f"shape2 batch dim {shape2[0]} does not match residual batch {batch}")
    grid_h, grid_w = shape2[1], shape2[2]
    if shape2[3] != window_h or shape2[4] != window_w:
        raise ValueError(f"shape2 window dims {shape2[3:5]} do not match {(window_h, window_w)}")
    if num_windows != batch * grid_h * grid_w:
        raise ValueError(f"number of windows {num_windows} does not match batch/grid {(batch, grid_h, grid_w)}")
    if grid_h * window_h != height or grid_w * window_w != width:
        raise ValueError(f"window grid {shape2!r} does not tile residual spatial {(height, width)}")

    if len(shape3) != 4 or not _dim_matches(shape3[0], batch) or shape3[1:] != (height, width, hidden):
        raise ValueError(f"unexpected reverse-window output view shape: {shape3!r}")
    if len(shape4) != 3 or not _dim_matches(shape4[0], batch) or not _dim_matches(shape4[1], height * width) or shape4[2] != hidden:
        raise ValueError(f"unexpected layernorm view shape: {shape4!r}")
    if shape5 != (total_rows, hidden):
        raise ValueError(f"unexpected output shape: {shape5!r}")

    return addmm_85, residual, weight, bias, shape0, shape1, shape2, shape3, shape4, shape5


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm_85, residual, weight, bias, shape0, shape1, shape2, shape3, shape4, shape5 = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(addmm_85, shape0)
    view_default_1 = torch.ops.aten.view.default(view_default, shape1)
    view_default_2 = torch.ops.aten.view.default(view_default_1, shape2)
    permute_default = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5])
    clone_default = torch.ops.aten.clone.default(permute_default, memory_format=torch.contiguous_format)
    view_default_3 = torch.ops.aten.view.default(clone_default, shape3)
    shift_h = torch.ops.aten.fmod.Scalar(
        torch.ops.aten.add.Tensor(
            torch.ops.prims.iota.default(
                view_default_3.shape[1],
                start=0,
                step=1,
                dtype=torch.int64,
                device=addmm_85.device,
                requires_grad=False,
            ),
            SHIFT,
        ),
        view_default_3.shape[1],
    )
    index_tensor = torch.ops.aten.index.Tensor(view_default_3, [None, shift_h])
    shift_w = torch.ops.aten.fmod.Scalar(
        torch.ops.aten.add.Tensor(
            torch.ops.prims.iota.default(
                view_default_3.shape[2],
                start=0,
                step=1,
                dtype=torch.int64,
                device=addmm_85.device,
                requires_grad=False,
            ),
            SHIFT,
        ),
        view_default_3.shape[2],
    )
    index_tensor_1 = torch.ops.aten.index.Tensor(index_tensor, [None, None, shift_w])
    add_tensor = torch.ops.aten.add.Tensor(residual, index_tensor_1)
    view_default_4 = torch.ops.aten.view.default(add_tensor, shape4)
    var, mean = torch.ops.aten.var_mean.correction(view_default_4, [2], correction=0, keepdim=True)
    normalized = (view_default_4 - mean) * torch.ops.aten.rsqrt.default(var + EPS)
    affine = normalized * weight + bias
    return torch.ops.aten.view.default(affine, shape5)


@oracle_impl(hardware="H100", shapes="(T([25088, 512], f32), T([128, 14, 14, 512], f32), T([512], f32), T([512], f32), S([512, 49, 512]), S([-1, 7, 7, 512]), S([-1, 2, 2, 7, 7, 512]), S([-1, 14, 14, 512]), S([128, -1, 512]), S([25088, 512]))")
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
    addmm_85, residual, weight, bias, _shape0, shape1, shape2, _shape3, _shape4, shape5 = _validate_inputs(inputs)
    if triton is None:
        return _torch_full_scope(inputs)

    batch, height, width, hidden = (int(dim) for dim in residual.shape)
    _num_windows, window_h, window_w, _hidden = shape1
    _batch, grid_h, grid_w, _window_h, _window_w, _hidden = shape2
    total_rows, out_hidden = shape5
    if out_hidden != hidden:
        raise ValueError(f"output hidden {out_hidden} does not match input hidden {hidden}")

    block_h = 1 << (hidden - 1).bit_length()
    output = torch.empty_strided(shape5, (hidden, 1), device=addmm_85.device, dtype=addmm_85.dtype)
    _swin_window_reverse_shift_layernorm_kernel[(total_rows,)](
        addmm_85,
        residual,
        weight,
        bias,
        output,
        addmm_85.stride(0),
        addmm_85.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        residual.stride(3),
        output.stride(0),
        output.stride(1),
        batch,
        height,
        width,
        hidden,
        grid_h,
        grid_w,
        window_h,
        window_w,
        SHIFT,
        EPS,
        total_rows,
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
