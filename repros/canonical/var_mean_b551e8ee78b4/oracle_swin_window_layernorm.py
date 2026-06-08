"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin channel LayerNorm plus 7x7 window partition/permute/clone/final flatten scope in one Triton row-reduction kernel that writes the affine-normalized rows directly in final window order, whereas Inductor currently lowers the decomposed reshape/var_mean/affine/permute/clone/view graph as a generic normalization followed by separate layout materialization; Inductor cannot do this today because its scheduler/codegen pattern library has no Swin window-partition LayerNorm template that sinks the static window-layout store into the normalization row schedule; the fix is NEW_PATTERN: add a guarded Swin window LayerNorm-partition lowering that maps final window rows directly to source spatial rows and emits reduction, affine, and layout epilogue in one kernel."""
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
ROW_BLOCK = 4


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _swin_window_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        batch: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        hidden: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        grid_h: tl.constexpr,
        grid_w: tl.constexpr,
        total_rows: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ROW_BLK: tl.constexpr,
    ):
        out_row = tl.program_id(0) * ROW_BLK + tl.arange(0, ROW_BLK)[:, None]
        cols = tl.arange(0, BLOCK_H)[None, :]
        row_mask = out_row < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask

        inner_w = out_row % window_w
        tmp = out_row // window_w
        inner_h = tmp % window_h
        tmp = tmp // window_h
        window_col = tmp % grid_w
        tmp = tmp // grid_w
        window_row = tmp % grid_h
        batch_idx = tmp // grid_h

        src_h = window_row * window_h + inner_h
        src_w = window_col * window_w + inner_w
        src_row = batch_idx * height * width + src_h * width + src_w

        x = tl.load(
            x_ptr + src_row * x_s0 + cols * x_s1,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = (tl.sum(x_for_reduce, axis=1) / hidden)[:, None]
        centered = x - mean
        variance = (
            tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden
        )[:, None]
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

        tl.store(out_ptr + out_row * out_s0 + cols * out_s1, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


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
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    x, weight, bias, shape0, shape1, shape2, shape3, shape4 = inputs
    tensor_inputs = (x, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensor inputs are required for the Triton oracle")
    if not all(value.dtype == DTYPE for value in tensor_inputs):
        dtypes = [value.dtype for value in tensor_inputs]
        raise TypeError(f"all tensor inputs must be torch.float32, got {dtypes}")
    if x.ndim != 2 or weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("expected x[rows, hidden] and 1D affine tensors")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError(
            "all tensor inputs must be contiguous, got "
            f"x={tuple(x.stride())}, weight={tuple(weight.stride())}, bias={tuple(bias.stride())}"
        )

    shape0 = _shape_tuple(shape0)
    shape1 = _shape_tuple(shape1)
    shape2 = _shape_tuple(shape2)
    shape3 = _shape_tuple(shape3)
    shape4 = _shape_tuple(shape4)

    rows, hidden = (int(dim) for dim in x.shape)
    if tuple(weight.shape) != (hidden,) or tuple(bias.shape) != (hidden,):
        raise ValueError(
            f"expected affine tensors of shape {(hidden,)}, got {tuple(weight.shape)} and {tuple(bias.shape)}"
        )
    if len(shape0) != 4 or shape0[3] != hidden:
        raise ValueError(f"unexpected 4D reshape parameter: {shape0!r}")

    batch, height, width, shape_hidden = shape0
    if shape_hidden != hidden or batch * height * width != rows:
        raise ValueError(f"shape0 {shape0!r} is incompatible with x shape {tuple(x.shape)}")
    if len(shape1) != 6 or shape1[0] != batch or shape1[5] != hidden:
        raise ValueError(f"unexpected window reshape parameter: {shape1!r}")

    grid_h, window_h, grid_w, window_w = shape1[1], shape1[2], shape1[3], shape1[4]
    if grid_h * window_h != height or grid_w * window_w != width:
        raise ValueError(f"window shape {shape1!r} does not tile {(height, width)}")
    if shape2 != (-1, window_h, window_w, hidden):
        raise ValueError(f"unexpected first flatten view shape: {shape2!r}")
    if shape3 != (-1, window_h * window_w, hidden):
        raise ValueError(f"unexpected second flatten view shape: {shape3!r}")
    if shape4 != (rows, hidden):
        raise ValueError(f"unexpected output shape: {shape4!r}")

    return x, weight, bias, shape0, shape1, shape2, shape3, shape4


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    x, weight, bias, shape0, shape1, shape2, shape3, shape4 = _validate_inputs(inputs)
    reshaped = torch.ops.aten.reshape.default(x, shape0)
    var, mean = torch.ops.aten.var_mean.correction(
        reshaped,
        [3],
        correction=0,
        keepdim=True,
    )
    normalized = (reshaped - mean) * torch.ops.aten.rsqrt.default(var + EPS)
    affine = normalized * weight + bias
    windows = torch.ops.aten.reshape.default(affine, shape1)
    permuted = torch.ops.aten.permute.default(windows, [0, 1, 3, 2, 4, 5])
    contiguous = torch.ops.aten.clone.default(permuted, memory_format=torch.contiguous_format)
    return torch.ops.aten.reshape.default(
        torch.ops.aten.reshape.default(
            torch.ops.aten.reshape.default(contiguous, shape2),
            shape3,
        ),
        shape4,
    )


def oracle_forward(inputs):
    """Run the complete LayerNorm plus Swin window-partition repro scope."""
    x, weight, bias, shape0, shape1, _shape2, _shape3, shape4 = _validate_inputs(inputs)
    if triton is None:
        return _torch_full_scope(inputs)

    batch, height, width, hidden = shape0
    _batch, grid_h, window_h, grid_w, window_w, _hidden = shape1
    total_rows, out_hidden = shape4
    if out_hidden != hidden:
        raise ValueError(f"output hidden {out_hidden} does not match input hidden {hidden}")

    block_h = _next_power_of_2(hidden)
    output = torch.empty_strided(shape4, (hidden, 1), device=x.device, dtype=x.dtype)
    _swin_window_layernorm_kernel[(triton.cdiv(total_rows, ROW_BLOCK),)](
        x,
        weight,
        bias,
        output,
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        batch,
        height,
        width,
        hidden,
        window_h,
        window_w,
        grid_h,
        grid_w,
        total_rows,
        EPS,
        block_h,
        ROW_BLOCK,
        num_warps=4,
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
