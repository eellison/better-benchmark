"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle collapses the captured Swin window-reverse reshape/permute chain whose interchanged grid dimensions are both size one, then computes the residual add and hidden-size-1024 population var_mean LayerNorm affine epilogue directly into the final [6272, 1024] layout in one Triton row kernel, whereas Inductor currently carries the rank-changing window-reverse view/permutation through generic normalization scheduling instead of canonicalizing it to the same row map before fusion; Inductor cannot do this today because its view/permute simplifier does not prove singleton-dimension Swin window-reverse layouts equivalent to identity across dynamic -1 reshape parameters before the var_mean scheduler runs; the fix is ALGEBRAIC_ELIMINATION: canonicalize singleton-grid window-reverse reshape/permute chains to an identity row layout before LayerNorm fusion and codegen."""
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


BATCH = 128
WINDOW_H = 7
WINDOW_W = 7
TOKENS = WINDOW_H * WINDOW_W
CHANNELS = 1024
ROWS = BATCH * TOKENS
EPS = 1.0e-5

ADDMM_SHAPE = (ROWS, CHANNELS)
RESIDUAL_SHAPE = (BATCH, WINDOW_H, WINDOW_W, CHANNELS)
AFFINE_SHAPE = (CHANNELS,)
OUTPUT_SHAPE = (ROWS, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)

SHAPE_PARAMS = (
    (BATCH, TOKENS, CHANNELS),
    (-1, WINDOW_H, WINDOW_W, CHANNELS),
    (-1, 1, 1, WINDOW_H, WINDOW_W, CHANNELS),
    (-1, WINDOW_H, WINDOW_W, CHANNELS),
    (BATCH, -1, CHANNELS),
    OUTPUT_SHAPE,
)


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
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["total_rows"],
    )
    @triton.jit
    def _singleton_window_reverse_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, BLOCK_C)
        row_mask = rows < total_rows
        col_mask = cols < channels
        mask = row_mask[:, None] & col_mask[None, :]

        offsets = rows[:, None] * channels + cols[None, :]
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
        mean = tl.sum(x_for_reduce, axis=1) / channels
        centered = x - mean[:, None]
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / channels
        inv_std = tl.rsqrt(variance + eps)

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
        output = centered * inv_std[:, None] * weight[None, :] + bias[None, :]

        tl.store(output_ptr + offsets, output, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects ten inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_93", inputs[0], ADDMM_SHAPE, torch.float32, OUTPUT_STRIDE)
    residual = _require_tensor(
        "view_631",
        inputs[1],
        RESIDUAL_SHAPE,
        torch.float32,
        (TOKENS * CHANNELS, WINDOW_W * CHANNELS, CHANNELS, 1),
    )
    weight = _require_tensor("arg355_1", inputs[2], AFFINE_SHAPE, torch.float32, (1,))
    bias = _require_tensor("arg356_1", inputs[3], AFFINE_SHAPE, torch.float32, (1,))

    actual_shapes = tuple(_shape_tuple(shape_param) for shape_param in inputs[4:])
    if actual_shapes != SHAPE_PARAMS:
        raise ValueError(f"shape parameters {actual_shapes} != {SHAPE_PARAMS}")

    device = addmm.device
    for name, tensor in (
        ("view_631", residual),
        ("arg355_1", weight),
        ("arg356_1", bias),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    return addmm, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias = _validate_inputs(inputs)
    reshape_default = torch.ops.aten.reshape.default(addmm, _shape_tuple(inputs[4]))
    reshape_default_1 = torch.ops.aten.reshape.default(reshape_default, _shape_tuple(inputs[5]))
    reshape_default_2 = torch.ops.aten.reshape.default(reshape_default_1, _shape_tuple(inputs[6]))
    permute_default = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5])
    reshape_default_3 = torch.ops.aten.reshape.default(permute_default, _shape_tuple(inputs[7]))
    add_tensor = torch.ops.aten.add.Tensor(residual, reshape_default_3)
    reshape_default_4 = torch.ops.aten.reshape.default(add_tensor, _shape_tuple(inputs[8]))
    variance, mean = torch.ops.aten.var_mean.correction(
        reshape_default_4,
        [2],
        correction=0,
        keepdim=True,
    )
    centered = torch.ops.aten.sub.Tensor(reshape_default_4, mean)
    inv_std = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(centered, inv_std)
    scaled = torch.ops.aten.mul.Tensor(normalized, weight)
    shifted = torch.ops.aten.add.Tensor(scaled, bias)
    return torch.ops.aten.reshape.default(shifted, _shape_tuple(inputs[9]))


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
    addmm, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _singleton_window_reverse_layernorm_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        total_rows=ROWS,
        channels=CHANNELS,
        eps=EPS,
        BLOCK_C=CHANNELS,
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
