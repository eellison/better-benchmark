"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full residual-add LayerNorm scope in one shape-specialized Triton row kernel and returns all three final views as aliases of one contiguous normalized allocation, whereas Inductor lowers the decomposed view/add/var_mean/affine/view graph through its generic normalization template and metadata handling; Inductor cannot do this today because norm-template canonicalization does not recognize this fixed-hidden residual LayerNorm with multiple alias-only view outputs as a dedicated semantic pattern; the fix is NEW_PATTERN: add a guarded residual LayerNorm-alias lowering that folds the residual add into the row reduction, emits the affine epilogue directly, and returns metadata-only aliases from the same output storage."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
EPS = 1.0e-12
BLOCK_HIDDEN_768 = 1024
BLOCK_HIDDEN_4096 = 4096

if triton is not None:

    @triton.jit
    def _residual_layernorm_aliases_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * row_block + tl.arange(0, row_block)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = row_ids[:, None] < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        addmm = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = addmm + residual

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


def _require_f32_cuda_contiguous(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    addmm, residual, weight, bias, base_shape_arg, shape1_arg, shape2_arg, shape3_arg = inputs
    addmm_t = _require_f32_cuda_contiguous("addmm_66", addmm)
    residual_t = _require_f32_cuda_contiguous("add_98", residual)
    weight_t = _require_f32_cuda_contiguous("arg24_1", weight)
    bias_t = _require_f32_cuda_contiguous("arg25_1", bias)

    if addmm_t.ndim != 2:
        raise ValueError(f"addmm_66 must be rank-2, got shape={tuple(addmm_t.shape)}")
    if residual_t.ndim != 3:
        raise ValueError(f"add_98 must be rank-3, got shape={tuple(residual_t.shape)}")

    total_rows = int(addmm_t.shape[0])
    hidden = int(addmm_t.shape[1])
    base_shape = _shape_tuple(base_shape_arg)
    output_shapes = (
        _shape_tuple(shape1_arg),
        _shape_tuple(shape2_arg),
        _shape_tuple(shape3_arg),
    )

    if tuple(residual_t.shape) != base_shape:
        raise ValueError(
            f"view shape parameter {base_shape_arg!r} does not match residual shape "
            f"{tuple(residual_t.shape)}"
        )
    if len(base_shape) != 3 or base_shape[0] * base_shape[1] != total_rows or base_shape[2] != hidden:
        raise ValueError(
            f"base view shape {base_shape} is incompatible with addmm shape {tuple(addmm_t.shape)}"
        )
    if tuple(weight_t.shape) != (hidden,) or tuple(bias_t.shape) != (hidden,):
        raise ValueError(
            f"weight/bias shapes must both be {(hidden,)}, got {tuple(weight_t.shape)} "
            f"and {tuple(bias_t.shape)}"
        )
    for index, shape in enumerate(output_shapes, start=1):
        if shape != (total_rows, hidden):
            raise ValueError(f"unexpected output view shape {index}: {shape!r}")
    if any(value.device != addmm_t.device for value in (residual_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if hidden not in (768, 4096):
        raise ValueError(f"oracle is specialized for hidden size 768 or 4096, got {hidden}")

    return addmm_t, residual_t, weight_t, bias_t, base_shape, output_shapes


@oracle_impl(hardware="H100", shapes="(T([4096, 4096], f32), T([8, 512, 4096], f32), T([4096], f32), T([4096], f32), S([8, 512, 4096]), S([4096, 4096]), S([4096, 4096]), S([4096, 4096]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same three f32 2D view outputs, all aliasing one normalized base allocation.
    """
    if triton is None or libdevice is None:
        raise RuntimeError("Triton and Inductor libdevice are required for this oracle")

    addmm, residual, weight, bias, base_shape, output_shapes = _validate_inputs(inputs)
    total_rows = int(addmm.shape[0])
    hidden = int(addmm.shape[1])
    block_h = BLOCK_HIDDEN_4096 if hidden == 4096 else BLOCK_HIDDEN_768
    row_block = 1 if hidden == 4096 else 2
    num_warps = 8 if hidden == 4096 else 4

    base = torch.empty_strided(
        base_shape,
        _contiguous_stride(base_shape),
        device=addmm.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(total_rows, row_block),)
    _residual_layernorm_aliases_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        base,
        total_rows=total_rows,
        hidden=hidden,
        eps=EPS,
        block_h=block_h,
        row_block=row_block,
        num_warps=num_warps,
        num_stages=3,
    )
    return tuple(base.view(shape) for shape in output_shapes)


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
