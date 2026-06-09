"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete EfficientNet training-BatchNorm plus affine SiLU and constant-pad scope by splitting the per-channel population var_mean reduction over N*H*W, finalizing mean/invstd and the two running-stat copy_ updates, then writing the padded activation directly, whereas Inductor lowers this norm-template-canonicalization case through its generic BN-training reduction and padded epilogue schedule; Inductor cannot do this today because the normalization scheduler lacks a cooperative split-K BN-training statistics template that also feeds mutable running-stat side effects and a fused padded SiLU epilogue; the fix is COOPERATIVE_SPLIT_K: add a guarded split-K training-BN template with direct output-space epilogue support for padding consumers."""
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


N = 128
CHANNELS = 672
HEIGHT = 14
WIDTH = 14
HW = HEIGHT * WIDTH
OUT_HEIGHT = 17
OUT_WIDTH = 17
OUT_HW = OUT_HEIGHT * OUT_WIDTH
ELEMENTS_PER_CHANNEL = N * HW
OUTPUT_ELEMENTS = N * CHANNELS * OUT_HW
EPS = 1.0e-3
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361
STAT_BLOCK_K = 1024
STAT_BLOCKS = 25
STAT_BLOCK_C = 4
FINAL_BLOCK_P = 32
EPILOGUE_BLOCK_C = 2
EPILOGUE_BLOCK_HW = 512


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _partial_stats_kernel(
        x_ptr,
        partial_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        stat_blocks: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        block_k = tl.program_id(0)
        block_c = tl.program_id(1)
        c_offsets = block_c * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
        k_offsets = block_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        n_idx = k_offsets // hw_size
        hw_idx = k_offsets - n_idx * hw_size
        x_offsets = (n_idx * channels + c_offsets) * hw_size + hw_idx
        mask = (c_offsets < channels) & (k_offsets < elements_per_channel)

        vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        sums = tl.sum(vals, axis=1)
        sums_sq = tl.sum(vals * vals, axis=1)
        c_vec = block_c * BLOCK_C + tl.arange(0, BLOCK_C)
        out_offsets = c_vec * stat_blocks + block_k
        out_mask = c_vec < channels
        tl.store(partial_ptr + out_offsets, sums, mask=out_mask)
        tl.store(partial_ptr + channels * stat_blocks + out_offsets, sums_sq, mask=out_mask)

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
        BLOCK_C: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        block_c = tl.program_id(0)
        c_offsets = block_c * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
        p_offsets = tl.arange(0, BLOCK_P)[None, :]
        mask = (c_offsets < channels) & (p_offsets < stat_blocks)
        partial_offsets = c_offsets * stat_blocks + p_offsets

        sums = tl.load(partial_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        sums_sq = tl.load(
            partial_ptr + channels * stat_blocks + partial_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        total_sum = tl.sum(sums, axis=1)
        total_sum_sq = tl.sum(sums_sq, axis=1)
        c_vec = block_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_vec < channels
        mean = total_sum / elements_per_channel
        var = total_sum_sq / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + c_vec, mask=c_mask, other=0.0).to(tl.float32)
        old_var = tl.load(running_var_ptr + c_vec, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(running_mean_ptr + c_vec, old_mean * (1.0 - momentum) + mean * momentum, mask=c_mask)
        tl.store(
            running_var_ptr + c_vec,
            old_var * (1.0 - momentum) + var * running_var_correction * momentum,
            mask=c_mask,
        )
        tl.store(stats_ptr + c_vec, mean, mask=c_mask)
        tl.store(stats_ptr + channels + c_vec, invstd, mask=c_mask)

    @triton.jit
    def _bn_silu_pad_epilogue_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        stats_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        width: tl.constexpr,
        out_hw_size: tl.constexpr,
        out_width: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n_idx = tl.program_id(0)
        block_c = tl.program_id(1)
        c_offsets = block_c * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
        out_pix = tl.arange(0, BLOCK_HW)[None, :]
        out_h = out_pix // out_width
        out_w = out_pix - out_h * out_width
        interior = (out_h >= 1) & (out_h <= 14) & (out_w >= 1) & (out_w <= 14)
        c_mask = c_offsets < channels
        mask = c_mask & (out_pix < out_hw_size)

        input_h = out_h - 1
        input_w = out_w - 1
        input_pix = input_h * width + input_w
        x_offsets = (n_idx * channels + c_offsets) * hw_size + input_pix
        x = tl.load(x_ptr + x_offsets, mask=mask & interior, other=0.0).to(tl.float32)

        c_vec = block_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mean = tl.load(stats_ptr + c_vec, mask=c_vec < channels, other=0.0).to(tl.float32)[:, None]
        invstd = tl.load(stats_ptr + channels + c_vec, mask=c_vec < channels, other=0.0).to(tl.float32)[:, None]
        weight = tl.load(weight_ptr + c_vec, mask=c_vec < channels, other=0.0).to(tl.float32)[:, None]
        bias = tl.load(bias_ptr + c_vec, mask=c_vec < channels, other=0.0).to(tl.float32)[:, None]

        y = (x - mean) * invstd * weight + bias
        silu = y / (tl.exp(-y) + 1.0)
        result = tl.where(interior, silu, 0.0)
        out_offsets = (n_idx * channels + c_offsets) * out_hw_size + out_pix
        tl.store(out_ptr + out_offsets, result, mask=mask)


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
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    x, running_mean, running_var, weight, bias = inputs
    x_t = _expect_f32_tensor(
        "convolution_55",
        x,
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    vector_shape = (CHANNELS,)
    vector_stride = (1,)
    running_mean_t = _expect_f32_tensor("arg245_1", running_mean, vector_shape, vector_stride)
    running_var_t = _expect_f32_tensor("arg246_1", running_var, vector_shape, vector_stride)
    weight_t = _expect_f32_tensor("arg247_1", weight, vector_shape, vector_stride)
    bias_t = _expect_f32_tensor("arg248_1", bias, vector_shape, vector_stride)

    if any(t.device != x_t.device for t in (running_mean_t, running_var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same device")
    return x_t, running_mean_t, running_var_t, weight_t, bias_t


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    y = (x - mean) * invstd
    y = y * weight[None, :, None, None] + bias[None, :, None, None]
    out = y / (torch.exp(-y) + 1.0)
    out = torch.nn.functional.pad(out, (1, 2, 1, 2))
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
    running_var.copy_(running_var * (1.0 - MOMENTUM) + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM)
    return out, running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([128, 672, 14, 14], f32), T([672], f32), T([672], f32), T([672], f32), T([672], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward computation.

    The returned running mean and variance are the same input tensors after the
    in-place copy_ updates, matching the captured training-BN side effects.
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
    stats = torch.empty_strided(
        (2, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        (N, CHANNELS, OUT_HEIGHT, OUT_WIDTH),
        (CHANNELS * OUT_HW, OUT_HW, OUT_WIDTH, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _partial_stats_kernel[(STAT_BLOCKS, triton.cdiv(CHANNELS, STAT_BLOCK_C))](
        x,
        partial,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        stat_blocks=STAT_BLOCKS,
        BLOCK_C=STAT_BLOCK_C,
        BLOCK_K=STAT_BLOCK_K,
        num_warps=8,
        num_stages=3,
    )
    _finalize_stats_kernel[(triton.cdiv(CHANNELS, STAT_BLOCK_C),)](
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
        BLOCK_C=STAT_BLOCK_C,
        BLOCK_P=FINAL_BLOCK_P,
        num_warps=1,
        num_stages=3,
    )
    _bn_silu_pad_epilogue_kernel[(N, triton.cdiv(CHANNELS, EPILOGUE_BLOCK_C))](
        x,
        weight,
        bias,
        stats,
        out,
        channels=CHANNELS,
        hw_size=HW,
        width=WIDTH,
        out_hw_size=OUT_HW,
        out_width=OUT_WIDTH,
        BLOCK_C=EPILOGUE_BLOCK_C,
        BLOCK_HW=EPILOGUE_BLOCK_HW,
        num_warps=8,
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
        if any(isinstance(item, torch.Tensor) and item.is_cuda for item in oracle_inputs):
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
