"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin singleton-grid reverse-window residual LayerNorm scope in one Triton row kernel, preserving the captured view `[6272,1024] -> [128,49,1024] -> [128,7,7,1024] -> [128,1,1,7,7,1024]`, permute `[0,1,3,2,4,5]`, final view, residual-first add with `view_629`, fp32 correction=0 centered-variance `var_mean`, generated `libdevice.rsqrt(var + 1e-5)`, affine scale/bias, and contiguous final `[6272,1024]` output, whereas Inductor already measures at the same floor for this singleton-window layout path and fixed hidden-size normalization; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, or recompute fusion because the required residual/addmm reads, one 1024-wide row reduction, affine reads, rsqrt, and output store dominate; the fix is BANDWIDTH_BOUND: record this as an at-floor Swin reverse-window LayerNorm case unless broader normalization-template or memory-traffic work moves both implementations."""
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
RESIDUAL_STRIDE = (TOKENS * CHANNELS, WINDOW_W * CHANNELS, CHANNELS, 1)

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

if triton is not None and libdevice is not None:

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
    def _swin_reverse_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, BLOCK_C)
        row_mask = rows < total_rows
        col_mask = cols < channels
        mask = row_mask[:, None] & col_mask[None, :]

        token = rows % (window_h * window_w)
        batch = rows // (window_h * window_w)
        h = token // window_w
        w = token - h * window_w

        # This is the captured reverse-window path with singleton grid dims:
        # [B, 1, 1, 7, 7, C] -> permute [0, 1, 3, 2, 4, 5] -> [B, 7, 7, C].
        source_row = (
            ((((batch * 1 + 0) * window_h + h) * 1 + 0) * window_w + w)
        )
        addmm_offsets = source_row[:, None] * channels + cols[None, :]
        residual_offsets = (
            batch[:, None] * (window_h * window_w * channels)
            + h[:, None] * (window_w * channels)
            + w[:, None] * channels
            + cols[None, :]
        )

        reverse_window = tl.load(
            addmm_ptr + addmm_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + residual_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = residual + reverse_window

        mean = tl.sum(tl.where(mask, x, 0.0), axis=1)[:, None] / channels
        centered = x - mean
        variance = (
            tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None]
            / channels
        )
        invstd = libdevice.rsqrt(variance + eps)

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
        normalized = centered * invstd
        scaled = normalized * weight[None, :]
        shifted = scaled + bias[None, :]

        output_offsets = rows[:, None] * channels + cols[None, :]
        tl.store(output_ptr + output_offsets, shifted, mask=mask)


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

    addmm = _require_tensor(
        "addmm_93",
        inputs[0],
        ADDMM_SHAPE,
        torch.float32,
        OUTPUT_STRIDE,
    )
    residual = _require_tensor(
        "view_629",
        inputs[1],
        RESIDUAL_SHAPE,
        torch.float32,
        RESIDUAL_STRIDE,
    )
    weight = _require_tensor("arg355_1", inputs[2], AFFINE_SHAPE, torch.float32, (1,))
    bias = _require_tensor("arg356_1", inputs[3], AFFINE_SHAPE, torch.float32, (1,))

    actual_shapes = tuple(_shape_tuple(shape_param) for shape_param in inputs[4:])
    if actual_shapes != SHAPE_PARAMS:
        raise ValueError(f"shape parameters {actual_shapes} != {SHAPE_PARAMS}")

    device = addmm.device
    for name, tensor in (
        ("view_629", residual),
        ("arg355_1", weight),
        ("arg356_1", bias),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    return addmm, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(addmm, _shape_tuple(inputs[4]))
    view_default_1 = torch.ops.aten.view.default(view_default, _shape_tuple(inputs[5]))
    view_default_2 = torch.ops.aten.view.default(view_default_1, _shape_tuple(inputs[6]))
    permute_default = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5])
    view_default_3 = torch.ops.aten.view.default(permute_default, _shape_tuple(inputs[7]))
    add_tensor = torch.ops.aten.add.Tensor(residual, view_default_3)
    view_default_4 = torch.ops.aten.view.default(add_tensor, _shape_tuple(inputs[8]))
    variance, mean = torch.ops.aten.var_mean.correction(
        view_default_4,
        [2],
        correction=0,
        keepdim=True,
    )
    centered = torch.ops.aten.sub.Tensor(view_default_4, mean)
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(centered, invstd)
    scaled = torch.ops.aten.mul.Tensor(normalized, weight)
    shifted = torch.ops.aten.add.Tensor(scaled, bias)
    return torch.ops.aten.view.default(shifted, _shape_tuple(inputs[9]))


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
    if triton is None or libdevice is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_reverse_layernorm_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        total_rows=ROWS,
        channels=CHANNELS,
        window_h=WINDOW_H,
        window_w=WINDOW_W,
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
