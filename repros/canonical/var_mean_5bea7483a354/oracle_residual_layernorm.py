"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full residual-add LayerNorm scope in one shape-specialized Triton row kernel, preserving Inductor's generated fp32 correction=0 mean then centered-variance order, eps-before-rsqrt, libdevice.rsqrt, affine epilogue, and final contiguous view, whereas Inductor already emits a single fused persistent row-reduction kernel for the same graph; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, recompute fusion, or a new pattern because the required residual/addmm/affine reads, one hidden-size-768 reduction, rsqrt, and output store dominate the one-kernel workload; the fix is BANDWIDTH_BOUND: record it as an at-floor LayerNorm case unless broader normalization-template, launch-overhead, or memory-traffic work moves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
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


EPS = 1.0e-6


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
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["total_rows", "hidden"],
    )
    @triton.jit
    def _residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, BLOCK_H)[None, :]
        row_mask = row_ids[:, None] < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        addmm = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = residual + addmm

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = (tl.sum(x_for_reduce, axis=1) / hidden)[:, None]
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden
        invstd = libdevice.rsqrt(variance + eps)[:, None]

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
        out = centered * invstd * weight + bias
        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError(f"expected positive hidden size, got {value}")
    return 1 << (value - 1).bit_length()


def _resolve_view_shape(shape: tuple[int, ...], numel: int) -> tuple[int, ...]:
    if not shape:
        if numel != 1:
            raise ValueError(f"scalar view is incompatible with numel={numel}")
        return shape

    unknown_index = None
    known_product = 1
    for index, dim in enumerate(shape):
        if dim == -1:
            if unknown_index is not None:
                raise ValueError(f"only one inferred dimension is allowed, got {shape}")
            unknown_index = index
        else:
            if dim < 0:
                raise ValueError(f"invalid negative dimension {dim} in shape {shape}")
            known_product *= dim

    if unknown_index is None:
        if known_product != numel:
            raise ValueError(f"view shape {shape} is incompatible with numel={numel}")
        return shape

    if known_product == 0 or numel % known_product != 0:
        raise ValueError(f"cannot infer view shape {shape} for numel={numel}")
    resolved = list(shape)
    resolved[unknown_index] = numel // known_product
    return tuple(resolved)


def _require_f32_contiguous(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, tuple[int, ...]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm, residual, weight, bias, base_shape_arg, output_shape_arg = inputs
    addmm_t = _require_f32_contiguous("addmm_45", addmm)
    residual_t = _require_f32_contiguous("add_77", residual)
    weight_t = _require_f32_contiguous("arg144_1", weight)
    bias_t = _require_f32_contiguous("arg145_1", bias)

    if addmm_t.ndim != 2:
        raise ValueError(f"addmm_45 must be rank-2, got shape={tuple(addmm_t.shape)}")
    if residual_t.ndim != 3:
        raise ValueError(f"add_77 must be rank-3, got shape={tuple(residual_t.shape)}")

    device = addmm_t.device
    for name, tensor in (
        ("add_77", residual_t),
        ("arg144_1", weight_t),
        ("arg145_1", bias_t),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    total_rows = int(addmm_t.shape[0])
    hidden = int(addmm_t.shape[1])
    base_shape = _shape_tuple(base_shape_arg)
    output_shape = _resolve_view_shape(_shape_tuple(output_shape_arg), addmm_t.numel())

    if tuple(residual_t.shape) != base_shape:
        raise ValueError(
            f"view shape parameter {base_shape_arg!r} does not match residual shape "
            f"{tuple(residual_t.shape)}"
        )
    if len(base_shape) != 3 or base_shape[0] * base_shape[1] != total_rows or base_shape[2] != hidden:
        raise ValueError(
            f"base view shape {base_shape} is incompatible with addmm shape {tuple(addmm_t.shape)}"
        )
    if residual_t.numel() != addmm_t.numel():
        raise ValueError("addmm_45 and add_77 must have the same flattened numel")
    if tuple(weight_t.shape) != (hidden,) or tuple(bias_t.shape) != (hidden,):
        raise ValueError(
            f"weight/bias shapes must both be {(hidden,)}, got {tuple(weight_t.shape)} "
            f"and {tuple(bias_t.shape)}"
        )
    if not output_shape:
        raise ValueError("final view shape must be rank-1 or greater")
    if output_shape[-1] != hidden:
        raise ValueError(f"final view shape {output_shape} must end with hidden={hidden}")
    if _numel(output_shape) != addmm_t.numel():
        raise ValueError(f"final view shape {output_shape} is incompatible with addmm numel")

    return addmm_t, residual_t, weight_t, bias_t, total_rows, hidden, output_shape


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias, _, _ = inputs
    reshape_default = torch.ops.aten.reshape.default(addmm, _shape_tuple(inputs[4]))
    add_tensor = torch.ops.aten.add.Tensor(residual, reshape_default)
    variance, mean = torch.ops.aten.var_mean.correction(
        add_tensor,
        [2],
        correction=0,
        keepdim=True,
    )
    centered = torch.ops.aten.sub.Tensor(add_tensor, mean)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(centered, invstd)
    scaled = torch.ops.aten.mul.Tensor(normalized, weight)
    shifted = torch.ops.aten.add.Tensor(scaled, bias)
    return torch.ops.aten.reshape.default(shifted, _shape_tuple(inputs[5]))


@oracle_impl(hardware="H100", shapes="(T([25344, 768], f32), T([128, 198, 768], f32), T([768], f32), T([768], f32), S([128, 198, 768]), S([25344, 768]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward computation with a fused Triton LayerNorm kernel.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single f32 output tensor with the final contiguous view layout.
    """
    addmm, residual, weight, bias, total_rows, hidden, output_shape = _validate_inputs(inputs)
    if triton is None or libdevice is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=addmm.device,
        dtype=torch.float32,
    )
    block_h = _next_power_of_2(hidden)
    grid = lambda meta: (triton.cdiv(total_rows, meta["ROW_BLOCK"]),)
    _residual_layernorm_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        total_rows=total_rows,
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
