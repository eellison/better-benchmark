"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin window-reverse residual LayerNorm scope in one shape-specialized Triton row kernel and keeps each 512-channel row tile live from the window gather through variance, affine, and final flatten store, whereas Inductor currently emits one fused generic var_mean kernel that reloads the window-reversed residual row for the normalization epilogue after Welford statistics; Inductor cannot do this today because its normalization scheduler cannot retain a nontrivial reshape/permute/clone producer tile across the row-statistics pass and affine epilogue; the fix is SCHEDULER_FUSION: extend the fixed-hidden LayerNorm template to inline window-reverse producers and carry the row tile into the epilogue when the producer is single-use."""
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

BATCH = 128
HEIGHT = 14
WIDTH = 14
PATCHES = HEIGHT * WIDTH
HIDDEN = 512
ROWS = BATCH * PATCHES
WINDOW = 7
WINDOW_AREA = WINDOW * WINDOW
WINDOWS_H = HEIGHT // WINDOW
WINDOWS_W = WIDTH // WINDOW
WINDOWS_PER_IMAGE = WINDOWS_H * WINDOWS_W
EPS = 1.0e-5

ADDMM_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
WEIGHT_SHAPE = (HIDDEN,)
OUTPUT_SHAPE = (ROWS, HIDDEN)
EXPECTED_SHAPE_PARAMS = (
    (WINDOWS_PER_IMAGE * BATCH, WINDOW_AREA, HIDDEN),
    (-1, WINDOW, WINDOW, HIDDEN),
    (-1, WINDOWS_H, WINDOWS_W, WINDOW, WINDOW, HIDDEN),
    (-1, HEIGHT, WIDTH, HIDDEN),
    (BATCH, -1, HIDDEN),
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
            triton.Config({"block_m": 1}, num_warps=8, num_stages=3),
            triton.Config({"block_m": 2}, num_warps=8, num_stages=3),
            triton.Config({"block_m": 4}, num_warps=8, num_stages=3),
            triton.Config({"block_m": 8}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _swin_window_residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_h)
        row_cols = rows[:, None] * hidden + cols[None, :]

        patch = rows % 196
        batch = rows // 196
        h = patch // 14
        w = patch - h * 14
        window_h = h // 7
        window_w = w // 7
        inner_h = h - window_h * 7
        inner_w = w - window_w * 7
        addmm_row = batch * 196 + (window_h * 2 + window_w) * 49 + inner_h * 7 + inner_w
        addmm_cols = addmm_row[:, None] * hidden + cols[None, :]

        residual = tl.load(residual_ptr + row_cols)
        projected = tl.load(addmm_ptr + addmm_cols)
        x = residual + projected

        mean = tl.sum(x, axis=1)[:, None] / 512.0
        centered = x - mean
        variance = tl.sum(centered * centered, axis=1)[:, None] / 512.0
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols)
        bias = tl.load(bias_ptr + cols)
        out = centered * invstd * weight[None, :] + bias[None, :]
        tl.store(out_ptr + row_cols, out)


def _shape_tuple(name: str, value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a shape sequence, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_81", inputs[0], ADDMM_SHAPE, torch.float32)
    residual = _require_tensor("view_547", inputs[1], RESIDUAL_SHAPE, torch.float32)
    weight = _require_tensor("arg309_1", inputs[2], WEIGHT_SHAPE, torch.float32)
    bias = _require_tensor("arg310_1", inputs[3], WEIGHT_SHAPE, torch.float32)

    if residual.device != addmm.device or weight.device != addmm.device or bias.device != addmm.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    shape_params = tuple(
        _shape_tuple(f"_shape_param_{index}", value)
        for index, value in enumerate(inputs[4:])
    )
    if shape_params != EXPECTED_SHAPE_PARAMS:
        raise ValueError(
            f"unexpected shape params {shape_params!r}, expected {EXPECTED_SHAPE_PARAMS!r}"
        )
    return addmm, residual, weight, bias, shape_params[-1]


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
    if triton is None:
        raise RuntimeError("Triton is required for oracle_swin_window_layernorm.py")

    addmm, residual, weight, bias, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        (HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["block_m"]),)
    _swin_window_residual_layernorm_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        hidden=HIDDEN,
        eps=EPS,
        block_h=HIDDEN,
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
