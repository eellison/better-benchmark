"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete channels-last ConvNeXtV2 exact-GELU plus GRN forward scope as the canonical three-stage norm template: reduce exact-GELU squared over each spatial tile, reduce the resulting per-(N,C) L2 norms across channels, then recompute exact-GELU in the final affine residual output store with the original output layout; the full-scope harness reports this oracle at floor against torch.compile, and Inductor already emits a comparable compact side-output reduction and epilogue decomposition for this case; there is no confirmed scheduler-fusion, scatter-reduce, cooperative split-K, algebraic-elimination, or new-pattern gap left in this measured repro; the fix is BANDWIDTH_BOUND: record this as an at-floor GRN norm-template case unless broader pointwise/reduction codegen or launch-overhead improvements move both implementations."""
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

RSQRT2 = 0.7071067811865476
EPS = 1.0e-6


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _choose_block_c(hw_size: int, channels: int) -> int:
    if hw_size <= 64:
        return min(64, _next_power_of_2(channels))
    if hw_size <= 256:
        return min(16, _next_power_of_2(channels))
    if hw_size <= 1024:
        return min(4, _next_power_of_2(channels))
    return 1


if triton is not None:

    @triton.jit
    def _gelu_spatial_sumsq_channels_last_kernel(
        x_ptr,
        sumsq_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
        hw_offsets = tl.arange(0, BLOCK_HW)[:, None]

        c_valid = c_offsets < channels
        hw_valid = hw_offsets < hw_size
        offsets = batch * channels * hw_size + hw_offsets * channels + c_offsets
        mask = hw_valid & c_valid

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gelu = (0.5 * x) * (tl.math.erf(x * 0.7071067811865476) + 1.0)
        sum_sq = tl.sum(tl.where(mask, gelu * gelu, 0.0), axis=0)

        c_store = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        tl.store(sumsq_ptr + batch * channels + c_store, sum_sq, mask=c_store < channels)

    @triton.jit
    def _channel_norm_sum_kernel(
        sumsq_ptr,
        norm_sum_ptr,
        channels: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_C)
        mask = offsets < channels
        sumsq = tl.load(sumsq_ptr + batch * channels + offsets, mask=mask, other=0.0).to(tl.float32)
        total = tl.sum(tl.where(mask, tl.sqrt(sumsq), 0.0), axis=0)
        tl.store(norm_sum_ptr + batch, total)

    @triton.jit
    def _grn_channels_last_output_kernel(
        x_ptr,
        sumsq_ptr,
        norm_sum_ptr,
        bias_ptr,
        weight_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
        mask = offsets < total

        channel = offsets % channels
        batch = offsets // (channels * hw_size)
        nc_index = batch * channels + channel

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gelu = (0.5 * x) * (tl.math.erf(x * 0.7071067811865476) + 1.0)
        sumsq = tl.load(sumsq_ptr + nc_index, mask=mask, other=0.0).to(tl.float32)
        norm_sum = tl.load(norm_sum_ptr + batch, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        mean = norm_sum / channels
        ratio = tl.sqrt(sumsq) / (mean + 1.0e-6)
        out = bias + gelu * (1.0 + weight * ratio)
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if stride is not None and tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    x, bias, weight = inputs
    x_t = _require_tensor("convolution_44", x)
    if x_t.ndim != 4:
        raise ValueError(f"convolution_44 must be 4D, got shape {tuple(x_t.shape)}")
    batch, channels, height, width = tuple(int(dim) for dim in x_t.shape)
    expected_x_stride = (channels * height * width, 1, width * channels, channels)
    if tuple(x_t.stride()) != expected_x_stride:
        raise ValueError(f"convolution_44 has stride {tuple(x_t.stride())}, expected {expected_x_stride}")
    if x_t.storage_offset() != 0:
        raise ValueError("convolution_44 must have storage_offset() == 0")

    bias_t = _require_tensor("primals_154", bias, (channels,), (1,))
    weight_t = _require_tensor("primals_155", weight, (channels,), (1,))
    if batch <= 0 or channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"unexpected empty shape {tuple(x_t.shape)}")
    if x_t.is_cuda and any(t.device != x_t.device for t in (bias_t, weight_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return x_t, bias_t, weight_t


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    x, bias, weight = _validate_inputs(inputs)
    gelu = torch.ops.aten.mul.Tensor(
        torch.ops.aten.mul.Tensor(x, 0.5),
        torch.ops.aten.add.Tensor(
            torch.ops.aten.erf.default(torch.ops.aten.mul.Tensor(x, RSQRT2)),
            1,
        ),
    )
    sumsq = torch.ops.aten.sum.dim_IntList(torch.ops.aten.pow.Tensor_Scalar(gelu, 2), [2, 3], True)
    norms = torch.ops.aten.pow.Tensor_Scalar(sumsq, 0.5)
    denom = torch.ops.aten.add.Tensor(torch.ops.aten.mean.dim(norms, [1], True), EPS)
    ratio = torch.ops.aten.div.Tensor(norms, denom)
    out = torch.ops.aten.add.Tensor(
        gelu,
        torch.ops.aten.addcmul.default(
            torch.ops.aten.reshape.default(bias, [1, -1, 1, 1]),
            torch.ops.aten.reshape.default(weight, [1, -1, 1, 1]),
            torch.ops.aten.mul.Tensor(gelu, ratio),
        ),
    )
    return out


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
    x, bias, weight = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_full_scope(inputs)

    batch, channels, height, width = tuple(int(dim) for dim in x.shape)
    hw_size = height * width
    total = x.numel()
    block_hw = _next_power_of_2(hw_size)
    block_c = _choose_block_c(hw_size, channels)
    n_channel_blocks = triton.cdiv(channels, block_c)
    mean_block = _next_power_of_2(channels)

    sumsq = torch.empty_strided((batch, channels), (channels, 1), device=x.device, dtype=torch.float32)
    norm_sums = torch.empty_strided((batch,), (1,), device=x.device, dtype=torch.float32)
    output = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.float32)

    _gelu_spatial_sumsq_channels_last_kernel[(batch, n_channel_blocks)](
        x,
        sumsq,
        channels=channels,
        hw_size=hw_size,
        BLOCK_HW=block_hw,
        BLOCK_C=block_c,
        num_warps=8 if block_hw >= 1024 else 4,
        num_stages=3,
    )
    _channel_norm_sum_kernel[(batch,)](
        sumsq,
        norm_sums,
        channels=channels,
        BLOCK_C=mean_block,
        num_warps=8,
    )
    _grn_channels_last_output_kernel[(triton.cdiv(total, 1024),)](
        x,
        sumsq,
        norm_sums,
        bias,
        weight,
        output,
        total=total,
        channels=channels,
        hw_size=hw_size,
        BLOCK_ELEMS=1024,
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
