"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle emits a dedicated row-normalization schedule that fuses the captured contiguous view, NaN-preserving ReLU producer, hidden-size-512 population var_mean, eps=1e-12 rsqrt normalization, affine scale/bias, and final contiguous view into one Triton kernel, whereas Inductor currently reaches this full scope through generic fused reduction codegen for the captured graph; Inductor cannot do this today because the normalization scheduler does not expose a reusable same-layout activation-producing LayerNorm template with a benchmark-gated cost model; the fix is SCHEDULER_FUSION: teach the var_mean/LayerNorm scheduler to sink ReLU-like pointwise producers into a generated row-reduction template while emitting the requested final view layout directly."""
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


EPS = 1.0e-12
DTYPE = torch.float32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _relu_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        mask = (row_ids[:, None] < rows) & (cols[None, :] < hidden)
        offsets = row_ids[:, None] * hidden + cols[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        relu = tl.maximum(x, 0.0, propagate_nan=tl.PropagateNan.ALL)
        col_mask = cols[None, :] < hidden
        relu_for_reduce = tl.where(col_mask, relu, 0.0)

        mean = tl.sum(relu_for_reduce, axis=1) / hidden
        centered = relu - mean[:, None]
        variance = tl.sum(tl.where(col_mask, centered * centered, 0.0), axis=1) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(
            weight_ptr + cols,
            mask=cols < hidden,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=cols < hidden,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        out = (centered * invstd[:, None]) * weight[None, :] + bias[None, :]

        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


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


def _block_m_for_hidden(hidden: int) -> int:
    if hidden <= 256:
        return 2
    return 1


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    x, weight, bias, view_shape, output_shape_value = inputs
    tensor_inputs = (x, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")
    if x.ndim != 2:
        raise ValueError(f"input 0 must be rank-2, got shape {tuple(x.shape)}")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("affine inputs must be rank-1 tensors")

    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    if rows <= 0 or hidden <= 0:
        raise ValueError(f"unexpected input shape {tuple(x.shape)}")
    if hidden > 2048:
        raise ValueError(f"{REPRO_ID} supports hidden size up to 2048, got {hidden}")
    if tuple(weight.shape) != (hidden,) or tuple(bias.shape) != (hidden,):
        raise ValueError(
            f"affine inputs must both have shape ({hidden},), "
            f"got weight={tuple(weight.shape)} bias={tuple(bias.shape)}"
        )

    for index, value in enumerate(tensor_inputs):
        if value.dtype != DTYPE:
            raise TypeError(f"input {index} dtype {value.dtype} != {DTYPE}")
        if value.device != x.device:
            raise ValueError("all tensor inputs must be on the same device")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    input_shape = _resolve_view_shape(view_shape, x.numel())
    output_shape = _resolve_view_shape(output_shape_value, x.numel())
    if len(input_shape) != 3 or input_shape[-1] != hidden:
        raise ValueError(f"first view shape {view_shape!r} must be rank-3 with last dim {hidden}")
    if _numel(input_shape[:-1]) != rows:
        raise ValueError(f"first view shape {view_shape!r} is incompatible with input rows={rows}")

    return x, weight, bias, output_shape, rows, hidden


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    x, weight, bias, output_shape, _, _ = _validate_inputs(inputs)
    view_shape = _resolve_view_shape(inputs[3], x.numel())
    view_default = torch.ops.aten.view.default(x, view_shape)
    relu_default = torch.ops.aten.relu.default(view_default)
    variance, mean = torch.ops.aten.var_mean.correction(
        relu_default,
        [2],
        correction=0,
        keepdim=True,
    )
    centered = torch.ops.aten.sub.Tensor(relu_default, mean)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(centered, invstd)
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    return torch.ops.aten.view.default(affine, output_shape)


@oracle_impl(hardware="H100", shapes="(T([32768, 512], f32), T([512], f32), T([512], f32), S([256, 128, 512]), S([32768, 512]))")
def oracle_forward(inputs):
    """Run the full Repro.forward ReLU plus LayerNorm affine computation.

    SCOPE INVARIANT: accepts the same five inputs as Repro.forward() and
    returns the same single output with the requested final view shape.
    """
    x, weight, bias, output_shape, rows, hidden = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=x.device,
        dtype=DTYPE,
    )
    block_n = triton.next_power_of_2(hidden)
    block_m = _block_m_for_hidden(hidden)
    _relu_layernorm_kernel[(triton.cdiv(rows, block_m),)](
        x,
        weight,
        bias,
        output,
        rows=rows,
        hidden=hidden,
        eps=EPS,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8 if block_n >= 1024 else 4,
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
