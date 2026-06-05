"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full captured channels-last BN-inference affine, NaN-preserving ReLU, 3x3 stride-2 maxpool stencil, and padded 3x3 avgpool output for the final f32[128,192,35,35] tensor with the BN/ReLU producer fused into the maxpool layout-indexing kernel, whereas Inductor currently lowers the channels-last pointwise producer and stencil consumers through generic scheduling with avoidable intermediate traffic; Inductor cannot do this today because the scheduler does not sink a broadcast channel affine through a channels-last low-memory maxpool stencil and onward into the full pool pipeline while preserving the output layout; the fix is SCHEDULER_FUSION: teach the scheduler a guarded channels-last BN-affine/ReLU-to-pooling stencil fusion and benchmark it against the generic pool lowering."""
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
CHANNELS = 192
H_IN = 71
W_IN = 71
H_OUT = 35
W_OUT = 35
OUT_HW = H_OUT * W_OUT
EPS = 0.001

INPUT_SHAPE = (BATCH, CHANNELS, H_IN, W_IN)
INPUT_STRIDE = (CHANNELS * H_IN * W_IN, 1, W_IN * CHANNELS, CHANNELS)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
OUTPUT_SHAPE = (BATCH, CHANNELS, H_OUT, W_OUT)
OUTPUT_STRIDE = (CHANNELS * OUT_HW, 1, W_OUT * CHANNELS, CHANNELS)

BLOCK_C = 32
BLOCK_S = 16


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _bn_relu_maxpool_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        pool_ptr,
        CHANNELS_: tl.constexpr,
        H_IN_: tl.constexpr,
        W_IN_: tl.constexpr,
        W_OUT_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        EPS_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_S_: tl.constexpr,
    ):
        batch = tl.program_id(0)
        channel_block = tl.program_id(1)
        spatial_block = tl.program_id(2)

        c_offsets = channel_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        s_offsets = spatial_block * BLOCK_S_ + tl.arange(0, BLOCK_S_)
        c_active = c_offsets < CHANNELS_
        s_active = s_offsets < OUT_HW_

        out_h = s_offsets // W_OUT_
        out_w = s_offsets - out_h * W_OUT_

        mean = tl.load(mean_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)
        inv_std = 1.0 / tl.sqrt(var + EPS_)
        weight = tl.load(weight_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_active, other=0.0).to(tl.float32)

        best = tl.full((BLOCK_S_, BLOCK_C_), -float("inf"), tl.float32)
        input_batch_base = batch * (CHANNELS_ * H_IN_ * W_IN_)

        for kh in tl.static_range(0, 3):
            in_h = out_h * 2 + kh
            for kw in tl.static_range(0, 3):
                in_w = out_w * 2 + kw
                load_offsets = (
                    input_batch_base
                    + in_h[:, None] * (W_IN_ * CHANNELS_)
                    + in_w[:, None] * CHANNELS_
                    + c_offsets[None, :]
                )
                active = s_active[:, None] & c_active[None, :]
                x = tl.load(x_ptr + load_offsets, mask=active, other=0.0).to(tl.float32)
                normalized = (x - mean[None, :]) * inv_std[None, :]
                affine = normalized * weight[None, :] + bias[None, :]
                relu = tl.where(affine != affine, affine, tl.maximum(affine, 0.0))
                take = active & ((relu > best) | (relu != relu))
                best = tl.where(take, relu, best)

        pool_offsets = (
            batch * (CHANNELS_ * OUT_HW_)
            + s_offsets[:, None] * CHANNELS_
            + c_offsets[None, :]
        )
        tl.store(pool_ptr + pool_offsets, best, mask=s_active[:, None] & c_active[None, :])


    @triton.jit
    def _avgpool_kernel(
        pool_ptr,
        out_ptr,
        CHANNELS_: tl.constexpr,
        H_OUT_: tl.constexpr,
        W_OUT_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_S_: tl.constexpr,
    ):
        batch = tl.program_id(0)
        channel_block = tl.program_id(1)
        spatial_block = tl.program_id(2)

        c_offsets = channel_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        s_offsets = spatial_block * BLOCK_S_ + tl.arange(0, BLOCK_S_)
        c_active = c_offsets < CHANNELS_
        s_active = s_offsets < OUT_HW_

        out_h = s_offsets // W_OUT_
        out_w = s_offsets - out_h * W_OUT_
        batch_base = batch * (CHANNELS_ * OUT_HW_)

        acc = tl.zeros((BLOCK_S_, BLOCK_C_), tl.float32)
        for kh in tl.static_range(0, 3):
            pool_h = out_h + kh - 1
            h_valid = (pool_h >= 0) & (pool_h < H_OUT_)
            for kw in tl.static_range(0, 3):
                pool_w = out_w + kw - 1
                valid = s_active & h_valid & (pool_w >= 0) & (pool_w < W_OUT_)
                load_offsets = (
                    batch_base
                    + (pool_h[:, None] * W_OUT_ + pool_w[:, None]) * CHANNELS_
                    + c_offsets[None, :]
                )
                value = tl.load(
                    pool_ptr + load_offsets,
                    mask=valid[:, None] & c_active[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += value

        out_offsets = batch_base + s_offsets[:, None] * CHANNELS_ + c_offsets[None, :]
        tl.store(
            out_ptr + out_offsets,
            acc * (1.0 / 9.0),
            mask=s_active[:, None] & c_active[None, :],
        )


def _require_f32_tensor(
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
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    mean = _require_f32_tensor("arg22_1", inputs[0], STAT_SHAPE, STAT_STRIDE)
    x = _require_f32_tensor("convolution_4", inputs[1], INPUT_SHAPE, INPUT_STRIDE)
    var = _require_f32_tensor("arg23_1", inputs[2], STAT_SHAPE, STAT_STRIDE)
    weight = _require_f32_tensor("arg24_1", inputs[3], STAT_SHAPE, STAT_STRIDE)
    bias = _require_f32_tensor("arg25_1", inputs[4], STAT_SHAPE, STAT_STRIDE)

    device = x.device
    if any(t.device != device for t in (mean, var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return mean, x, var, weight, bias


def _torch_oracle(
    mean: torch.Tensor,
    x: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    inv_std = torch.reciprocal(torch.sqrt(var + EPS))
    y = (x - mean.view(1, CHANNELS, 1, 1)) * inv_std.view(1, CHANNELS, 1, 1)
    y = torch.relu(y * weight.view(1, CHANNELS, 1, 1) + bias.view(1, CHANNELS, 1, 1))
    pool = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        y,
        [3, 3],
        [2, 2],
        [0, 0],
        [1, 1],
        False,
    )[0]
    return torch.nn.functional.avg_pool2d(pool, kernel_size=3, stride=1, padding=1)


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
    mean, x, var, weight, bias = _validate_inputs(inputs)
    if not x.is_cuda:
        return _torch_oracle(mean, x, var, weight, bias)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    pool = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )

    grid = (BATCH, triton.cdiv(CHANNELS, BLOCK_C), triton.cdiv(OUT_HW, BLOCK_S))
    _bn_relu_maxpool_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        pool,
        CHANNELS_=CHANNELS,
        H_IN_=H_IN,
        W_IN_=W_IN,
        W_OUT_=W_OUT,
        OUT_HW_=OUT_HW,
        EPS_=EPS,
        BLOCK_C_=BLOCK_C,
        BLOCK_S_=BLOCK_S,
        num_warps=8,
        num_stages=3,
    )
    _avgpool_kernel[grid](
        pool,
        out,
        CHANNELS_=CHANNELS,
        H_OUT_=H_OUT,
        W_OUT_=W_OUT,
        OUT_HW_=OUT_HW,
        BLOCK_C_=BLOCK_C,
        BLOCK_S_=BLOCK_S,
        num_warps=8,
        num_stages=3,
    )
    return out


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
