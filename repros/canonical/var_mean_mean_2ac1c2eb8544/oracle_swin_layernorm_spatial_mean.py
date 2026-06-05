"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin residual-add, hidden-dimension variance/mean, layernorm affine, and 7x7 spatial mean by staging only per-token layernorm statistics and reducing directly to the final [128,1024] output, whereas Inductor currently materializes the normalized [128,49,1024] layernorm result before launching a separate orthogonal spatial mean reduction; Inductor cannot do this today because its reduction scheduler cannot pipeline a hidden-axis normalization producer into a later spatial-axis mean consumer while virtualizing the normalized intermediate; the fix is SCHEDULER_FUSION: teach Inductor to fuse producer layernorm reductions with downstream reductions over independent axes using a small stats side buffer or equivalent recompute schedule."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


BATCH = 128
WINDOW = 49
HIDDEN = 1024
ROWS = BATCH * WINDOW
EPS = 1.0e-5
BLOCK_HIDDEN = 1024
BLOCK_FEATURE = 64


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _row_stats_kernel(
        addmm_ptr,
        residual_ptr,
        mean_ptr,
        invstd_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_H)
        mask = offsets < hidden
        flat_offsets = row * hidden + offsets

        x = (
            tl.load(addmm_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
        )
        sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        sum_x2 = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        mean = sum_x / hidden
        var = tl.maximum(sum_x2 / hidden - mean * mean, 0.0)

        tl.store(mean_ptr + row, mean)
        tl.store(invstd_ptr + row, tl.rsqrt(var + eps))

    @triton.jit
    def _spatial_mean_affine_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        window: tl.constexpr,
        hidden: tl.constexpr,
        BLOCK_FEATURE: tl.constexpr,
    ):
        batch = tl.program_id(0)
        feature_block = tl.program_id(1)
        features = feature_block * BLOCK_FEATURE + tl.arange(0, BLOCK_FEATURE)
        feature_mask = features < hidden

        acc = tl.zeros((BLOCK_FEATURE,), tl.float32)
        for token in tl.static_range(0, 49):
            row = batch * window + token
            offsets = row * hidden + features
            x = (
                tl.load(addmm_ptr + offsets, mask=feature_mask, other=0.0).to(tl.float32)
                + tl.load(residual_ptr + offsets, mask=feature_mask, other=0.0).to(tl.float32)
            )
            mean = tl.load(mean_ptr + row).to(tl.float32)
            invstd = tl.load(invstd_ptr + row).to(tl.float32)
            acc += (x - mean) * invstd

        weight = tl.load(weight_ptr + features, mask=feature_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + features, mask=feature_mask, other=0.0).to(tl.float32)
        out = acc * (1.0 / window) * weight + bias
        tl.store(out_ptr + batch * hidden + features, out, mask=feature_mask)


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _expect_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a shape list/tuple, got {type(value)!r}")
    actual = tuple(int(dim) for dim in value)
    if actual != expected:
        raise ValueError(f"{name} is {actual}, expected {expected}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    addmm, residual, weight, bias, shape0, shape1 = inputs
    addmm = _expect_tensor("addmm_95", addmm, (ROWS, HIDDEN), (HIDDEN, 1))
    residual = _expect_tensor(
        "view_650",
        residual,
        (BATCH, WINDOW, HIDDEN),
        (WINDOW * HIDDEN, HIDDEN, 1),
    )
    weight = _expect_tensor("arg361_1", weight, (HIDDEN,), (1,))
    bias = _expect_tensor("arg362_1", bias, (HIDDEN,), (1,))
    _expect_shape("_shape_param_0", shape0, (BATCH, WINDOW, HIDDEN))
    _expect_shape("_shape_param_1", shape1, (BATCH, 7, 7, HIDDEN))

    device = addmm.device
    if residual.device != device or weight.device != device or bias.device != device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm, residual, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_swin_layernorm_spatial_mean.py")

    addmm, residual, weight, bias = _validate_inputs(inputs)
    means = torch.empty((ROWS,), device=addmm.device, dtype=torch.float32)
    invstd = torch.empty((ROWS,), device=addmm.device, dtype=torch.float32)
    out = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )

    _row_stats_kernel[(ROWS,)](
        addmm,
        residual,
        means,
        invstd,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_H=BLOCK_HIDDEN,
        num_warps=8,
        num_stages=2,
    )
    _spatial_mean_affine_kernel[(BATCH, triton.cdiv(HIDDEN, BLOCK_FEATURE))](
        addmm,
        residual,
        weight,
        bias,
        means,
        invstd,
        out,
        window=WINDOW,
        hidden=HIDDEN,
        BLOCK_FEATURE=BLOCK_FEATURE,
        num_warps=1,
        num_stages=3,
    )
    return out


# --- CLI entry point ---
def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
