"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the hidden-size-512 population var_mean LayerNorm affine and fixed Swin 7x7 window-partition clone in one output-contiguous Triton row kernel, whereas Inductor fuses the same normalization/layout scope but traverses source-contiguous rows and emits strided window-layout stores from the normalization epilogue; Inductor cannot do this today because its normalization scheduler preserves producer-contiguous row order when sinking deterministic reshape/permute/clone consumers instead of cost-modeling an output-contiguous remap; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to choose between source-contiguous and output-contiguous epilogue traversal for fixed layout clones."""
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


EPS = 1.0e-5
DTYPE = torch.float32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 8}, num_warps=8, num_stages=3),
        ],
        key=["total_rows", "hidden"],
    )
    @triton.jit
    def _swin_window_layernorm_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        grid_h: tl.constexpr,
        grid_w: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        out_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, BLOCK_H)
        row_mask = out_rows < total_rows
        col_mask = cols < hidden

        batch = out_rows // (height * width)
        within_image = out_rows - batch * height * width
        window_id = within_image // (window_h * window_w)
        position = within_image - window_id * (window_h * window_w)
        block_h = window_id // grid_w
        block_w = window_id - block_h * grid_w
        inner_h = position // window_w
        inner_w = position - inner_h * window_w
        src_h = block_h * window_h + inner_h
        src_w = block_w * window_w + inner_w
        src_rows = batch * height * width + src_h * width + src_w

        input_offsets = src_rows[:, None] * hidden + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        x = tl.load(
            input_ptr + input_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1) / hidden
        centered = tl.where(mask, x - mean[:, None], 0.0)
        variance = tl.sum(centered * centered, axis=1) / hidden
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
        output = centered * invstd[:, None] * weight[None, :] + bias[None, :]

        output_offsets = out_rows[:, None] * hidden + cols[None, :]
        tl.store(output_ptr + output_offsets, output, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...] | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int, int],
    tuple[int, int, int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int],
    tuple[int, int],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects eight inputs, got {len(inputs)}")

    input_tensor, weight, bias, shape0, shape1, shape2, shape3, shape4 = inputs
    shape0 = _shape_tuple(shape0)
    shape1 = _shape_tuple(shape1)
    shape2 = _shape_tuple(shape2)
    shape3 = _shape_tuple(shape3)
    shape4 = _shape_tuple(shape4)

    if len(shape0) != 4:
        raise ValueError(f"unexpected input view shape: {shape0!r}")
    batch, height, width, hidden = shape0
    total_rows = batch * height * width

    input_tensor = _require_tensor("mm_1", input_tensor, (total_rows, hidden), DTYPE)
    weight = _require_tensor("arg69_1", weight, (hidden,), DTYPE)
    bias = _require_tensor("arg70_1", bias, (hidden,), DTYPE)

    if len(shape1) != 6:
        raise ValueError(f"unexpected window reshape shape: {shape1!r}")
    if shape1[0] != batch or shape1[5] != hidden:
        raise ValueError(f"window reshape shape does not preserve batch/hidden: {shape1!r}")
    grid_h, window_h, grid_w, window_w = shape1[1], shape1[2], shape1[3], shape1[4]
    if grid_h * window_h != height or grid_w * window_w != width:
        raise ValueError(f"window shape {shape1!r} does not tile {(height, width)}")
    if height % window_h != 0 or width % window_w != 0:
        raise ValueError(f"window {(window_h, window_w)} must divide spatial {(height, width)}")
    if shape2 != (-1, window_h, window_w, hidden):
        raise ValueError(f"unexpected first flatten reshape shape: {shape2!r}")
    if shape3 != (-1, window_h * window_w, hidden):
        raise ValueError(f"unexpected second flatten reshape shape: {shape3!r}")
    if shape4 != (total_rows, hidden):
        raise ValueError(f"unexpected output shape: {shape4!r}")

    tensors = (weight, bias)
    device = input_tensor.device
    for index, tensor in enumerate(tensors, start=1):
        if tensor.device != device:
            raise ValueError(f"input {index} device {tensor.device} != {device}")

    return input_tensor, weight, bias, shape0, shape1, shape2, shape3, shape4


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    input_tensor, weight, bias, shape0, shape1, shape2, shape3, shape4 = _validate_inputs(inputs)
    x = torch.ops.aten.view.default(input_tensor, shape0)
    variance, mean = torch.ops.aten.var_mean.correction(x, [3], correction=0, keepdim=True)
    x = torch.ops.aten.sub.Tensor(x, mean)
    x = torch.ops.aten.mul.Tensor(x, torch.ops.aten.rsqrt.default(variance + EPS))
    x = torch.ops.aten.mul.Tensor(x, weight)
    x = torch.ops.aten.add.Tensor(x, bias)
    x = torch.ops.aten.view.default(x, shape1)
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 2, 4, 5])
    x = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
    x = torch.ops.aten.view.default(x, shape2)
    x = torch.ops.aten.view.default(x, shape3)
    return torch.ops.aten.view.default(x, shape4)


@oracle_impl(hardware="H100", shapes="(T([25088, 512], f32), T([512], f32), T([512], f32), S([128, 14, 14, 512]), S([128, 2, 7, 2, 7, 512]), S([-1, 7, 7, 512]), S([-1, 49, 512]), S([25088, 512]))")
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
    input_tensor, weight, bias, shape0, shape1, _shape2, _shape3, shape4 = _validate_inputs(inputs)
    if triton is None or not input_tensor.is_cuda:
        return _torch_full_scope(inputs)

    batch, height, width, hidden = shape0
    grid_h, window_h, grid_w, window_w = shape1[1], shape1[2], shape1[3], shape1[4]
    total_rows = shape4[0]
    block_h = 1 << (hidden - 1).bit_length()
    output = torch.empty_strided(shape4, (hidden, 1), device=input_tensor.device, dtype=DTYPE)
    grid = lambda meta: (triton.cdiv(total_rows, meta["ROW_BLOCK"]),)
    _swin_window_layernorm_kernel[grid](
        input_tensor,
        weight,
        bias,
        output,
        total_rows=total_rows,
        height=height,
        width=width,
        window_h=window_h,
        window_w=window_w,
        grid_h=grid_h,
        grid_w=grid_w,
        hidden=hidden,
        eps=EPS,
        BLOCK_H=block_h,
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
