"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete EfficientNet training-BatchNorm, running-stat side effects, affine SiLU, and bottom/right zero-pad scope for f32 `[128,240,28,28]` using a split-K channel-statistics reduction plus a coalesced padded epilogue, whereas Inductor currently lowers the decomposed `var_mean`/running-stat update/affine/SiLU/pad graph through generic normalization and pointwise schedules; Inductor cannot do this today because its BN-training lowering does not select a cooperative split-K statistics template that feeds mutable running-stat returns and a padded activation store; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-training stats schedule that splits the long `N*H*W` reduction, finalizes running-stat side effects, and feeds the exact padded SiLU epilogue."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


BATCH = 128
CHANNELS = 240
HEIGHT = 28
WIDTH = 28
HW = HEIGHT * WIDTH
OUT_HEIGHT = 29
OUT_WIDTH = 29
OUT_HW = OUT_HEIGHT * OUT_WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW
TOTAL_OUTPUTS = BATCH * CHANNELS * OUT_HW
EPS = 1.0e-3
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.00000996502277
STAT_BLOCK = 4096
STAT_BLOCKS = triton.cdiv(ELEMENTS_PER_CHANNEL, STAT_BLOCK) if triton is not None else 25
FINAL_BLOCK = 32
EPILOGUE_BLOCK = 1024
INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
VECTOR_SHAPE = (CHANNELS,)
VECTOR_STRIDE = (1,)
OUTPUT_SHAPE = (BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH)
OUTPUT_STRIDE = (CHANNELS * OUT_HW, OUT_HW, OUT_WIDTH, 1)
STATS_SHAPE = (2, CHANNELS)
STATS_STRIDE = (CHANNELS, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_stats_kernel(
        x_ptr,
        partial_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        stat_blocks: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        channel = tl.program_id(1)
        offsets = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < elements_per_channel

        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        x_offsets = (n_idx * channels + channel) * hw_size + hw_idx

        vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        part_sum = tl.sum(vals, axis=0)
        part_sum_sq = tl.sum(vals * vals, axis=0)
        out_offset = channel * stat_blocks + block_id
        tl.store(partial_ptr + out_offset, part_sum)
        tl.store(partial_ptr + channels * stat_blocks + out_offset, part_sum_sq)

    @triton.jit
    def _finalize_stats_kernel(
        partial_ptr,
        running_mean_ptr,
        running_var_ptr,
        stats_ptr,
        channels: tl.constexpr,
        elements_per_channel: tl.constexpr,
        stat_blocks: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_P)
        mask = offsets < stat_blocks
        base = channel * stat_blocks + offsets

        sums = tl.load(partial_ptr + base, mask=mask, other=0.0).to(tl.float32)
        sums_sq = tl.load(partial_ptr + channels * stat_blocks + base, mask=mask, other=0.0).to(tl.float32)
        total_sum = tl.sum(sums, axis=0)
        total_sum_sq = tl.sum(sums_sq, axis=0)
        mean = total_sum / elements_per_channel
        var = total_sum_sq / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )
        tl.store(stats_ptr + channel, mean)
        tl.store(stats_ptr + channels + channel, invstd)

    @triton.jit
    def _bn_silu_pad_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        stats_ptr,
        out_ptr,
        channels: tl.constexpr,
        in_width: tl.constexpr,
        in_hw: tl.constexpr,
        out_width: tl.constexpr,
        out_hw: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        nc = tl.program_id(0)
        n_idx = nc // channels
        channel = nc - n_idx * channels
        offsets = tl.arange(0, BLOCK_HW)
        mask = offsets < out_hw
        out_h = offsets // out_width
        out_w = offsets - out_h * out_width
        valid_input = mask & (out_h < in_width) & (out_w < in_width)

        x_offsets = (n_idx * channels + channel) * in_hw + out_h * in_width + out_w
        x = tl.load(x_ptr + x_offsets, mask=valid_input, other=0.0).to(tl.float32)

        mean = tl.load(stats_ptr + channel).to(tl.float32)
        invstd = tl.load(stats_ptr + channels + channel).to(tl.float32)
        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)

        affine = (x - mean) * invstd * weight + bias
        silu = affine * tl.sigmoid(affine)
        padded = tl.where(valid_input, silu, 0.0)
        tl.store(out_ptr + nc * out_hw + offsets, padded, mask=mask)


def _expect_f32_tensor(
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
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    x = _expect_f32_tensor("convolution_25", inputs[0], INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg113_1", inputs[1], VECTOR_SHAPE, VECTOR_STRIDE)
    running_var = _expect_f32_tensor("arg114_1", inputs[2], VECTOR_SHAPE, VECTOR_STRIDE)
    weight = _expect_f32_tensor("arg115_1", inputs[3], VECTOR_SHAPE, VECTOR_STRIDE)
    bias = _expect_f32_tensor("arg116_1", inputs[4], VECTOR_SHAPE, VECTOR_STRIDE)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    affine = (x - mean) * invstd * weight[None, :, None, None] + bias[None, :, None, None]
    silu = affine / (torch.exp(-affine) + 1.0)
    padded = torch.ops.aten.constant_pad_nd.default(silu, [0, 1, 0, 1], 0.0)
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
    running_var.copy_(running_var * (1.0 - MOMENTUM) + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM)
    return padded, running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([128, 240, 28, 28], f32), T([240], f32), T([240], f32), T([240], f32), T([240], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    partial = torch.empty_strided(
        (2, CHANNELS, STAT_BLOCKS),
        (CHANNELS * STAT_BLOCKS, STAT_BLOCKS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    stats = torch.empty_strided(STATS_SHAPE, STATS_STRIDE, device=x.device, dtype=torch.float32)
    out = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=torch.float32)

    _partial_stats_kernel[(STAT_BLOCKS, CHANNELS)](
        x,
        partial,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        stat_blocks=STAT_BLOCKS,
        BLOCK_M=STAT_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _finalize_stats_kernel[(CHANNELS,)](
        partial,
        running_mean,
        running_var,
        stats,
        channels=CHANNELS,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        stat_blocks=STAT_BLOCKS,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_P=FINAL_BLOCK,
        num_warps=1,
        num_stages=3,
    )
    _bn_silu_pad_kernel[(BATCH * CHANNELS,)](
        x,
        weight,
        bias,
        stats,
        out,
        channels=CHANNELS,
        in_width=WIDTH,
        in_hw=HW,
        out_width=OUT_WIDTH,
        out_hw=OUT_HW,
        BLOCK_HW=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return out, running_mean, running_var


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(item.clone() if isinstance(item, torch.Tensor) else item for item in inputs)


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _check_oracle_cloned_inputs(
    oracle_fn,
    instance,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
    skip_stochastic: bool = True,
) -> bool:
    del skip_stochastic
    eager_inputs = _clone_inputs(inputs)
    oracle_inputs = _clone_inputs(inputs)

    with torch.no_grad():
        eager = instance(*eager_inputs)
        actual = oracle_fn(oracle_inputs)
        if isinstance(oracle_inputs[0], torch.Tensor) and oracle_inputs[0].is_cuda:
            torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    actual_list = _normalize_outputs(actual)
    if len(actual_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    alias_input_indices = (None, 1, 2)
    all_pass = True
    for i, (expected, observed) in enumerate(zip(eager_list, actual_list)):
        metadata_ok = (
            expected.shape == observed.shape
            and expected.dtype == observed.dtype
            and expected.stride() == observed.stride()
            and expected.device == observed.device
        )
        if not metadata_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH "
                f"oracle=(shape={list(observed.shape)} stride={observed.stride()} dtype={observed.dtype}) "
                f"eager=(shape={list(expected.shape)} stride={expected.stride()} dtype={expected.dtype})"
            )
            all_pass = False
            continue

        alias_index = alias_input_indices[i]
        if alias_index is not None:
            expected_alias = expected.data_ptr() == eager_inputs[alias_index].data_ptr()
            observed_alias = observed.data_ptr() == oracle_inputs[alias_index].data_ptr()
            if expected_alias != observed_alias or not observed_alias:
                print(
                    f"  output {i}: SCOPE_MISMATCH alias "
                    f"oracle={observed_alias} eager={expected_alias}"
                )
                all_pass = False
                continue

        if expected.is_floating_point():
            expected_f32 = expected.float()
            observed_f32 = observed.float()
            expected_nan = torch.isnan(expected_f32)
            observed_nan = torch.isnan(observed_f32)
            nan_mask_ok = torch.equal(expected_nan, observed_nan)
            finite = ~(expected_nan | observed_nan)
            if finite.any():
                max_diff = (expected_f32[finite] - observed_f32[finite]).abs().max().item()
                values_ok = torch.allclose(
                    expected_f32[finite],
                    observed_f32[finite],
                    atol=atol,
                    rtol=rtol,
                )
            else:
                max_diff = 0.0
                values_ok = True
            ok = nan_mask_ok and values_ok
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected.shape)} dtype={expected.dtype} stride={expected.stride()} "
                f"max_finite_diff={max_diff:.2e} nan_count={int(expected_nan.sum().item())})"
            )
        else:
            ok = torch.equal(expected, observed)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected.dtype})")
        all_pass = all_pass and bool(ok)

    return all_pass


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
        ok = _check_oracle_cloned_inputs(
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
