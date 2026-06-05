"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full ConvNeXtV2 exact-GELU plus GRN backward reduction fragment by staging shared per-batch/per-channel spatial statistics for the three returned f32 [2560] reductions, whereas Inductor lowers the same graph as generic pointwise/reduction pieces with repeated GELU/GRN math around reductions with different axes; Inductor cannot do this today because its scheduler does not form a single multi-output reduction plan when channel reductions, per-(N,C) spatial reductions, and a dependent per-N channel reduction all share the same channels-last producers; the fix is SCHEDULER_FUSION: recognize this reduction DAG, compute the shared spatial statistics once, and fuse the downstream channel reductions through a small template."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

BATCH = 128
CHANNELS = 2560
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
CONV_STRIDE_N = 125440
CONV_STRIDE_C = 1
CONV_STRIDE_H = 17920
CONV_STRIDE_W = 2560
NC = BATCH * CHANNELS

BLOCK_C_STATS = 64
BLOCK_HW = 64
BLOCK_C_FINAL = 128
BLOCK_N = 128
BLOCK_C_ALL = 4096
RSQRT2 = 0.7071067811865476
INV_SQRT_2PI = 0.3989422804014327


if triton is not None:

    @triton.jit
    def _spatial_stats_kernel(
        conv_ptr,
        getitem_ptr,
        stats_ptr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        n = tl.program_id(1)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        h_offsets = hw_offsets // 7
        w_offsets = hw_offsets - h_offsets * 7
        offsets = (
            n * 125440
            + h_offsets[:, None] * 17920
            + w_offsets[:, None] * 2560
            + c_offsets[None, :]
        )
        mask = (c_offsets[None, :] < 2560) & (hw_offsets[:, None] < 49)

        x = tl.load(conv_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(getitem_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        erf_plus_one = tl.math.erf(x * 0.7071067811865476) + 1.0
        gelu = (0.5 * x) * erf_plus_one
        gelu_grad = 0.5 * erf_plus_one + x * tl.exp(-0.5 * x * x) * 0.3989422804014327

        r = tl.sum(tl.where(mask, g * gelu, 0.0), axis=0)
        g_sum = tl.sum(tl.where(mask, g, 0.0), axis=0)
        b = tl.sum(tl.where(mask, g * gelu_grad, 0.0), axis=0)
        t = tl.sum(tl.where(mask, gelu * gelu_grad, 0.0), axis=0)

        nc_offsets = n * 2560 + c_offsets
        c_mask = c_offsets < 2560
        tl.store(stats_ptr + nc_offsets, r, mask=c_mask)
        tl.store(stats_ptr + 327680 + nc_offsets, g_sum, mask=c_mask)
        tl.store(stats_ptr + 655360 + nc_offsets, b, mask=c_mask)
        tl.store(stats_ptr + 983040 + nc_offsets, t, mask=c_mask)

    @triton.jit
    def _mean_backprop_kernel(
        stats_ptr,
        pow_ptr,
        add_ptr,
        weight_ptr,
        mean_ptr,
        BLOCK_C: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_offsets = tl.arange(0, BLOCK_C)
        mask = c_offsets < 2560

        r = tl.load(stats_ptr + n * 2560 + c_offsets, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(pow_ptr + n * 2560 + c_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
        add = tl.load(add_ptr + n).to(tl.float32)
        div = p / add
        mean = tl.sum(tl.where(mask, -weight * r * (div / add), 0.0), axis=0) * (1.0 / 2560.0)
        tl.store(mean_ptr + n, mean)

    @triton.jit
    def _finalize_outputs_kernel(
        stats_ptr,
        pow_ptr,
        add_ptr,
        weight_ptr,
        mean_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        BLOCK_C: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        n_offsets = tl.arange(0, BLOCK_N_)
        c_mask = c_offsets < 2560

        nc_offsets = n_offsets[:, None] * 2560 + c_offsets[None, :]
        mask = (n_offsets[:, None] < 128) & c_mask[None, :]

        r = tl.load(stats_ptr + nc_offsets, mask=mask, other=0.0).to(tl.float32)
        g_sum = tl.load(stats_ptr + 327680 + nc_offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(stats_ptr + 655360 + nc_offsets, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(stats_ptr + 983040 + nc_offsets, mask=mask, other=0.0).to(tl.float32)

        p = tl.load(pow_ptr + nc_offsets, mask=mask, other=0.0).to(tl.float32)
        add = tl.load(add_ptr + n_offsets, mask=n_offsets < 128, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + n_offsets, mask=n_offsets < 128, other=0.0).to(tl.float32)

        div = p / add[:, None]
        out0_terms = r * div
        safe_t_over_p = tl.where(p == 0.0, 0.0, t / p)
        out2_terms = (
            (1.0 + weight[None, :] * div) * b
            + (weight[None, :] * r / add[:, None] + mean[:, None]) * safe_t_over_p
        )

        out0 = tl.sum(tl.where(mask, out0_terms, 0.0), axis=0)
        out1 = tl.sum(tl.where(mask, g_sum, 0.0), axis=0)
        out2 = tl.sum(tl.where(mask, out2_terms, 0.0), axis=0)

        tl.store(out0_ptr + c_offsets, out0, mask=c_mask)
        tl.store(out1_ptr + c_offsets, out1, mask=c_mask)
        tl.store(out2_ptr + c_offsets, out2, mask=c_mask)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in value)
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    convolution_44, pow_28, add_89, getitem_38, primals_155, shape0, shape1, shape2 = inputs
    conv = _require_tensor(
        "convolution_44",
        convolution_44,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        torch.float32,
        (CONV_STRIDE_N, CONV_STRIDE_C, CONV_STRIDE_H, CONV_STRIDE_W),
    )
    pow_t = _require_tensor(
        "pow_28",
        pow_28,
        (BATCH, CHANNELS, 1, 1),
        torch.float32,
        (CHANNELS, 1, 1, 1),
    )
    add_t = _require_tensor(
        "add_89",
        add_89,
        (BATCH, 1, 1, 1),
        torch.float32,
        (1, 1, 1, 1),
    )
    getitem = _require_tensor(
        "getitem_38",
        getitem_38,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        torch.float32,
        (CONV_STRIDE_N, CONV_STRIDE_C, CONV_STRIDE_H, CONV_STRIDE_W),
    )
    weight = _require_tensor(
        "primals_155",
        primals_155,
        (CHANNELS,),
        torch.float32,
        (1,),
    )
    _require_shape("_shape_param_0", shape0, (CHANNELS,))
    _require_shape("_shape_param_1", shape1, (CHANNELS,))
    _require_shape("_shape_param_2", shape2, (BATCH, CHANNELS, 1, 1))

    if any(t.device != conv.device for t in (pow_t, add_t, getitem, weight)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return conv, pow_t, add_t, getitem, weight


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
        raise RuntimeError("Triton is required for oracle_convnextv2_grn_reductions.py")

    conv, pow_t, add_t, getitem, weight = _validate_inputs(inputs)
    device = conv.device
    stats = torch.empty_strided(
        (4, BATCH, CHANNELS),
        (NC, CHANNELS, 1),
        device=device,
        dtype=torch.float32,
    )
    mean = torch.empty((BATCH,), device=device, dtype=torch.float32)
    out0 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out2 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    _spatial_stats_kernel[(triton.cdiv(CHANNELS, BLOCK_C_STATS), BATCH)](
        conv,
        getitem,
        stats,
        BLOCK_C=BLOCK_C_STATS,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )
    _mean_backprop_kernel[(BATCH,)](
        stats,
        pow_t,
        add_t,
        weight,
        mean,
        BLOCK_C=BLOCK_C_ALL,
        num_warps=8,
        num_stages=3,
    )
    _finalize_outputs_kernel[(triton.cdiv(CHANNELS, BLOCK_C_FINAL),)](
        stats,
        pow_t,
        add_t,
        weight,
        mean,
        out0,
        out1,
        out2,
        BLOCK_C=BLOCK_C_FINAL,
        BLOCK_N_=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )
    return (out0, out1, out2)


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
