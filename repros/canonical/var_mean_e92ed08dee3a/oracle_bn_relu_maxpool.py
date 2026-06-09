"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete VoVNet training batchnorm var_mean, running-stat copy_ updates, affine ReLU, and low-memory 3x3 stride-2 ceil-mode maxpool-with-offsets scope with a channel statistics kernel plus a fused normalization-to-pool stencil that writes only final pooled values and int8 offsets, whereas Inductor materializes the normalized/ReLU activation before the pooling stencil and handles the mutable running-stat outputs as separate generic work; Inductor cannot do this today because scheduler fusion does not inline a reduction producer with mutation side outputs into a multi-output low-memory maxpool consumer while preserving ceil-mode offset semantics and tiny 7x7 output tiling; the fix is SCHEDULER_FUSION: teach the scheduler a training-BN-affine-ReLU-to-low-memory-maxpool template with explicit running-stat copy_ epilogues."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

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


N_BATCH = 32
CHANNELS = 768
HEIGHT = 14
WIDTH = 14
OUT_HEIGHT = 7
OUT_WIDTH = 7
HW = HEIGHT * WIDTH
ELEMS_PER_CHANNEL = N_BATCH * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871
STATS_BLOCK = 8192
POOL_BLOCK_C = 4
POOL_BLOCK_OUT = 64
INPUT_SHAPE = (N_BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
OUTPUT_SHAPE = (N_BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH)
OUTPUT_STRIDE = (CHANNELS * OUT_HEIGHT * OUT_WIDTH, OUT_HEIGHT * OUT_WIDTH, OUT_WIDTH, 1)
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _bn_stats_update_kernel(
        x_ptr,
        mean_ptr,
        inv_std_ptr,
        running_mean_ptr,
        running_var_ptr,
        elems_per_channel: tl.constexpr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_M)
        active = offsets < elems_per_channel

        batch = offsets // hw
        spatial = offsets - batch * hw
        flat = batch * (channels * hw) + channel * hw + spatial
        x = tl.load(x_ptr + flat, mask=active, other=0.0).to(tl.float32)

        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
        mean = sum_x / elems_per_channel
        var = sum_x2 / elems_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        inv_std = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(mean_ptr + channel, mean)
        tl.store(inv_std_ptr + channel, inv_std)
        tl.store(running_mean_ptr + channel, old_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )

    @triton.jit
    def _bn_relu_maxpool_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        inv_std_ptr,
        values_ptr,
        offsets_ptr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        out_height: tl.constexpr,
        out_width: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        batch = tl.program_id(0)
        channel_block = tl.program_id(1)
        out_block = tl.program_id(2)

        channel_offsets = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
        out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        out_active = out_offsets < (out_height * out_width)
        channel_active = channel_offsets < channels
        out_h = out_offsets // out_width
        out_w = out_offsets - out_h * out_width

        input_base = batch * (channels * hw) + channel_offsets[:, None] * hw
        mean = tl.load(mean_ptr + channel_offsets, mask=channel_active, other=0.0).to(tl.float32)
        inv_std = tl.load(inv_std_ptr + channel_offsets, mask=channel_active, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel_offsets, mask=channel_active, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=channel_active, other=0.0).to(tl.float32)

        best = tl.full((BLOCK_C, BLOCK_OUT), -float("inf"), tl.float32)
        best_offset = tl.zeros((BLOCK_C, BLOCK_OUT), tl.int32)

        for kh in tl.static_range(0, 3):
            in_h = out_h * 2 + kh
            valid_h = in_h < height
            load_h = tl.minimum(in_h, height - 1)
            for kw in tl.static_range(0, 3):
                in_w = out_w * 2 + kw
                valid_out = out_active & valid_h & (in_w < width)
                load_w = tl.minimum(in_w, width - 1)
                valid = channel_active[:, None] & valid_out[None, :]
                x = tl.load(
                    x_ptr + input_base + load_h[None, :] * width + load_w[None, :],
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)

                affine = (x - mean[:, None]) * inv_std[:, None] * weight[:, None] + bias[:, None]
                relu = tl.where((affine > 0.0) | (affine != affine), affine, 0.0)
                take = valid & ((relu > best) | (relu != relu))
                best = tl.where(take, relu, best)
                best_offset = tl.where(take, kh * 3 + kw, best_offset)

        out_hw = out_height * out_width
        out_base = batch * (channels * out_hw) + channel_offsets[:, None] * out_hw + out_offsets[None, :]
        store_mask = channel_active[:, None] & out_active[None, :]
        tl.store(values_ptr + out_base, best, mask=store_mask)
        tl.store(offsets_ptr + out_base, best_offset.to(tl.int8), mask=store_mask)


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

    x = _require_f32_tensor("convolution_26", inputs[0], INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _require_f32_tensor("arg159_1", inputs[1], STAT_SHAPE, STAT_STRIDE)
    running_var = _require_f32_tensor("arg160_1", inputs[2], STAT_SHAPE, STAT_STRIDE)
    weight = _require_f32_tensor("arg161_1", inputs[3], STAT_SHAPE, STAT_STRIDE)
    bias = _require_f32_tensor("arg162_1", inputs[4], STAT_SHAPE, STAT_STRIDE)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias


def _torch_oracle(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    inv_std = torch.rsqrt(var + EPS)
    affine = (x - mean) * inv_std * weight.view(1, CHANNELS, 1, 1) + bias.view(1, CHANNELS, 1, 1)
    relu = torch.relu(affine)
    values, offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu,
        [3, 3],
        [2, 2],
        [0, 0],
        [1, 1],
        True,
    )

    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    next_running_mean = running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM
    next_running_var = running_var * (1.0 - MOMENTUM) + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM
    running_mean.copy_(next_running_mean)
    running_var.copy_(next_running_var)
    return values, offsets, running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([32, 768, 14, 14], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward computation.

    The pooled values and int8 offsets are materialized directly from the
    normalized affine expression. Returned running stats are the mutated input
    tensors, matching the captured `copy_` output aliases.
    """
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if not x.is_cuda:
        return _torch_oracle(x, running_mean, running_var, weight, bias)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    mean = torch.empty_strided(STAT_SHAPE, STAT_STRIDE, device=x.device, dtype=torch.float32)
    inv_std = torch.empty_strided(STAT_SHAPE, STAT_STRIDE, device=x.device, dtype=torch.float32)
    values = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=torch.float32)
    offsets = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=torch.int8)

    _bn_stats_update_kernel[(CHANNELS,)](
        x,
        mean,
        inv_std,
        running_mean,
        running_var,
        elems_per_channel=ELEMS_PER_CHANNEL,
        channels=CHANNELS,
        hw=HW,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_M=STATS_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _bn_relu_maxpool_kernel[(
        N_BATCH,
        triton.cdiv(CHANNELS, POOL_BLOCK_C),
        triton.cdiv(OUT_HEIGHT * OUT_WIDTH, POOL_BLOCK_OUT),
    )](
        x,
        weight,
        bias,
        mean,
        inv_std,
        values,
        offsets,
        channels=CHANNELS,
        height=HEIGHT,
        width=WIDTH,
        out_height=OUT_HEIGHT,
        out_width=OUT_WIDTH,
        hw=HW,
        BLOCK_C=POOL_BLOCK_C,
        BLOCK_OUT=POOL_BLOCK_OUT,
        num_warps=4,
        num_stages=3,
    )
    return values, offsets, running_mean, running_var


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for item in inputs:
        cloned.append(item.clone() if isinstance(item, torch.Tensor) else item)
    return tuple(cloned)


def _tensor_outputs(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in value:
            result.extend(_tensor_outputs(item))
        return result
    return []


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    eager_inputs = _clone_inputs(inputs)
    oracle_inputs = _clone_inputs(inputs)

    with torch.no_grad():
        eager = instance(*eager_inputs)
        oracle_out = oracle_forward(oracle_inputs)
        if isinstance(oracle_inputs[0], torch.Tensor) and oracle_inputs[0].is_cuda:
            torch.cuda.synchronize()

    eager_list = _tensor_outputs(eager)
    oracle_list = _tensor_outputs(oracle_out)
    if len(eager_list) != len(oracle_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        layout_ok = (
            expected.shape == actual.shape
            and expected.dtype == actual.dtype
            and expected.stride() == actual.stride()
        )
        if not layout_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH "
                f"shape oracle={tuple(actual.shape)} eager={tuple(expected.shape)} "
                f"dtype oracle={actual.dtype} eager={expected.dtype} "
                f"stride oracle={actual.stride()} eager={expected.stride()}"
            )
            all_pass = False
            continue

        if expected.is_floating_point():
            expected_f32 = expected.float()
            actual_f32 = actual.float()
            max_diff = (expected_f32 - actual_f32).abs().max().item()
            values_ok = torch.allclose(expected_f32, actual_f32, atol=atol, rtol=rtol)
            print(
                f"  output {i}: {'PASS' if values_ok else 'FAIL'} "
                f"(shape={list(expected.shape)} dtype={expected.dtype} "
                f"stride={expected.stride()} max_diff={max_diff:.2e})"
            )
        else:
            values_ok = torch.equal(expected, actual)
            print(
                f"  output {i}: {'PASS' if values_ok else 'FAIL'} "
                f"(exact, shape={list(expected.shape)} dtype={expected.dtype} "
                f"stride={expected.stride()})"
            )
        all_pass = all_pass and bool(values_ok)

    alias_ok = (
        isinstance(oracle_out, tuple)
        and len(oracle_out) == 4
        and oracle_out[2] is oracle_inputs[1]
        and oracle_out[3] is oracle_inputs[2]
    )
    print(f"  running-stat aliases: {'PASS' if alias_ok else 'FAIL'}")
    return all_pass and alias_ok


# --- CLI entry point ---
def main() -> None:
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
        ok = _run_check(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
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
            floor_status = "true_floor" if result["status"] == "GOOD" else (
                "at_floor" if result["status"] == "AT_FLOOR" else "not_true_floor"
            )
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {floor_status}")


if __name__ == "__main__":
    main()
