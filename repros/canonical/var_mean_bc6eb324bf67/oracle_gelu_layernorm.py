"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete exact-erf GELU producer and fp32 row LayerNorm affine epilogue with one reusable all-shape Triton row-reduction kernel, whereas Inductor currently reaches the fused computation only through fixed-shape persistent-reduction codegen for this captured graph; Inductor cannot do this today because normalization scheduler fusion does not expose a reusable exact-GELU-producer LayerNorm template across the hidden-size family; the fix is SCHEDULER_FUSION: teach the var_mean/LayerNorm scheduler to inline single-use exact-erf GELU producers into a reusable statistics and output epilogue row-kernel template."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden", "block_h"],
    )
    @triton.jit
    def _gelu_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        rows_total: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = row_ids[:, None] < rows_total
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        erf_term = libdevice.erf(x * 0.7071067811865476) + 1.0
        gelu = (x * 0.5) * erf_term

        gelu_for_reduce = tl.where(mask, gelu, 0.0)
        mean = tl.sum(gelu_for_reduce, axis=1) / hidden
        centered = gelu - mean[:, None]
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden
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
        y = centered * invstd[:, None] * weight + bias
        tl.store(out_ptr + offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _resolve_view_shape(shape: Any, numel: int) -> tuple[int, ...]:
    dims = list(_shape_tuple(shape))
    unknown_index = None
    known_product = 1
    for index, dim in enumerate(dims):
        if dim == -1:
            if unknown_index is not None:
                raise ValueError(f"view shape {shape!r} has more than one inferred dimension")
            unknown_index = index
        else:
            known_product *= dim

    if unknown_index is not None:
        if known_product == 0 or numel % known_product != 0:
            raise ValueError(f"view shape {shape!r} is incompatible with {numel} elements")
        dims[unknown_index] = numel // known_product
    elif known_product != numel:
        raise ValueError(f"view shape {shape!r} is incompatible with {numel} elements")

    return tuple(dims)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...], int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    x, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (x, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")
    if x.ndim != 2:
        raise ValueError(f"input 0 must be rank-2, got shape {tuple(x.shape)}")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("affine inputs must be rank-1 tensors")

    rows, hidden = (int(x.shape[0]), int(x.shape[1]))
    if tuple(weight.shape) != (hidden,) or tuple(bias.shape) != (hidden,):
        raise ValueError(f"affine inputs must both have shape ({hidden},)")

    input_shape = _resolve_view_shape(shape0, x.numel())
    output_shape = _resolve_view_shape(shape1, x.numel())
    if len(input_shape) != 3 or input_shape[-1] != hidden:
        raise ValueError(f"first view shape {shape0!r} must be [*, *, {hidden}]")
    if math.prod(input_shape[:-1]) != rows:
        raise ValueError(f"first view shape {shape0!r} is incompatible with input rows={rows}")

    for index, value in enumerate(tensor_inputs):
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if value.device != x.device:
            raise ValueError("all tensor inputs must be on the same device")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    return x, weight, bias, input_shape, output_shape, rows, hidden


def _num_warps(hidden: int) -> int:
    if hidden <= 128:
        return 4
    return 8


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    x, weight, bias, input_shape, output_shape, _, _ = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(x, input_shape)
    erf_arg = torch.ops.aten.mul.Tensor(view_default, 0.7071067811865476)
    erf_term = torch.ops.aten.add.Tensor(torch.ops.aten.erf.default(erf_arg), 1)
    gelu = torch.ops.aten.mul.Tensor(torch.ops.aten.mul.Tensor(view_default, 0.5), erf_term)
    var, mean = torch.ops.aten.var_mean.correction(gelu, [2], correction=0, keepdim=True)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var, EPS))
    normalized = torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(gelu, mean), invstd)
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    return torch.ops.aten.view.default(affine, output_shape)


@oracle_impl(hardware="H100", shapes="(T([8192, 768], f32), T([768], f32), T([768], f32), S([8, 1024, 768]), S([8192, 768]))")
def oracle_forward(inputs):
    """Run the full Repro.forward exact-GELU LayerNorm computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    x, weight, bias, _, output_shape, rows, hidden = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=x.device,
        dtype=x.dtype,
    )
    block_h = triton.next_power_of_2(hidden)
    grid = lambda meta: (triton.cdiv(rows, meta["ROW_BLOCK"]),)
    _gelu_layernorm_kernel[grid](
        x,
        weight,
        bias,
        output,
        rows_total=rows,
        hidden=hidden,
        eps=EPS,
        block_h=block_h,
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
