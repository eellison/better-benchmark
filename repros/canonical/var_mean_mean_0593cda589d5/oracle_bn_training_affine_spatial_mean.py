"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete GhostNet training-BatchNorm running-stat update, affine normalization, and returned 7x7 spatial mean by carrying per-batch/channel row sums from the statistics pass and forming `mean_hw((x - mean) * invstd * weight + bias)` algebraically, whereas Inductor currently lowers the normalized affine tensor and following spatial mean as generic producer/consumer work around the `var_mean` result; Inductor cannot do this today because its algebraic simplifier does not rewrite affine spatial means across training-BatchNorm statistics with mutable `copy_` side outputs; the fix is ALGEBRAIC_ELIMINATION: add a guarded BatchNorm-spatial-mean rewrite that reuses row sums, preserves running-stat aliases, and skips the elementwise normalized activation."""
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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "ALGEBRAIC_ELIMINATION"
ACTIONABLE = True

N = 512
CHANNELS = 672
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
TOTAL_ROWS = N * CHANNELS
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361
ROW_RBLOCK = 64
FINAL_C_BLOCK = 8
FINAL_N_BLOCK = 512


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 32}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 64}, num_warps=8, num_stages=3),
            triton.Config({"XBLOCK": 128}, num_warps=8, num_stages=3),
        ],
        key=["total_rows"],
    )
    @triton.jit
    def _row_sums_kernel(
        x_ptr,
        row_sum_ptr,
        row_sum2_ptr,
        total_rows: tl.constexpr,
        hw_size: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        hw_offsets = tl.arange(0, RBLOCK)[None, :]
        mask = (row_offsets < total_rows) & (hw_offsets < hw_size)
        x = tl.load(x_ptr + row_offsets * hw_size + hw_offsets, mask=mask, other=0.0).to(tl.float32)
        row_sum = tl.sum(x, axis=1)[:, None]
        row_sum2 = tl.sum(x * x, axis=1)[:, None]
        tl.store(row_sum_ptr + row_offsets, row_sum, mask=row_offsets < total_rows)
        tl.store(row_sum2_ptr + row_offsets, row_sum2, mask=row_offsets < total_rows)

    @triton.jit
    def _finalize_and_output_kernel(
        row_sum_ptr,
        row_sum2_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        channels: tl.constexpr,
        batch: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
        n_offsets = tl.arange(0, BLOCK_N)[None, :]
        mask = (c_offsets < channels) & (n_offsets < batch)
        row_offsets = n_offsets * channels + c_offsets

        row_sum = tl.load(row_sum_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        row_sum2 = tl.load(row_sum2_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        total_sum = tl.sum(row_sum, axis=1)[:, None]
        total_sum2 = tl.sum(row_sum2, axis=1)[:, None]
        mean = total_sum / elements_per_channel
        var = total_sum2 / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_running_mean = tl.load(running_mean_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)
        tl.store(
            running_mean_ptr + c_offsets,
            old_running_mean * (1.0 - momentum) + mean * momentum,
            mask=c_offsets < channels,
        )
        tl.store(
            running_var_ptr + c_offsets,
            old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
            mask=c_offsets < channels,
        )

        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)
        row_mean = row_sum / hw_size
        out = (row_mean - mean) * invstd
        out = out * weight + bias
        tl.store(out_ptr + row_offsets, out, mask=mask)


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


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    convolution_66, arg367_1, arg368_1, arg369_1, arg370_1 = inputs
    x = _expect_f32_tensor(
        "convolution_66",
        convolution_66,
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    running_mean = _expect_f32_tensor("arg367_1", arg367_1, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("arg368_1", arg368_1, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("arg369_1", arg369_1, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("arg370_1", arg370_1, (CHANNELS,), (1,))

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    y = (x - mean) * invstd
    y = y * weight[None, :, None, None] + bias[None, :, None, None]
    out = y.mean(dim=(2, 3), keepdim=True)
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean.squeeze((0, 2, 3)) * MOMENTUM)
    running_var.copy_(
        running_var * (1.0 - MOMENTUM)
        + var.squeeze((0, 2, 3)) * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    return out, running_mean, running_var


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward scope, including running-stat mutations."""
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    row_sum = torch.empty_strided((N, CHANNELS), (CHANNELS, 1), device=x.device, dtype=torch.float32)
    row_sum2 = torch.empty_strided((N, CHANNELS), (CHANNELS, 1), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(
        (N, CHANNELS, 1, 1),
        (CHANNELS, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _row_sums_kernel[lambda meta: (triton.cdiv(TOTAL_ROWS, meta["XBLOCK"]),)](
        x,
        row_sum,
        row_sum2,
        total_rows=TOTAL_ROWS,
        hw_size=HW,
        RBLOCK=ROW_RBLOCK,
    )
    _finalize_and_output_kernel[(triton.cdiv(CHANNELS, FINAL_C_BLOCK),)](
        row_sum,
        row_sum2,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        channels=CHANNELS,
        batch=N,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_C=FINAL_C_BLOCK,
        BLOCK_N=FINAL_N_BLOCK,
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
        expected = instance(*eager_inputs)
        actual = oracle_forward(oracle_inputs)
        if isinstance(oracle_inputs[0], torch.Tensor) and oracle_inputs[0].is_cuda:
            torch.cuda.synchronize()

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(expected_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    all_pass = True
    alias_input_indices = (None, 1, 2)
    for i, (expected_tensor, actual_tensor) in enumerate(zip(expected_list, actual_list)):
        if expected_tensor.shape != actual_tensor.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual_tensor.shape)} "
                f"eager={list(expected_tensor.shape)}"
            )
            all_pass = False
            continue
        if expected_tensor.dtype != actual_tensor.dtype:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={actual_tensor.dtype} "
                f"eager={expected_tensor.dtype}"
            )
            all_pass = False
            continue
        if expected_tensor.stride() != actual_tensor.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={actual_tensor.stride()} "
                f"eager={expected_tensor.stride()}"
            )
            all_pass = False
            continue

        alias_index = alias_input_indices[i]
        if alias_index is not None:
            expected_alias = expected_tensor.data_ptr() == eager_inputs[alias_index].data_ptr()
            actual_alias = actual_tensor.data_ptr() == oracle_inputs[alias_index].data_ptr()
            if expected_alias != actual_alias or not actual_alias:
                print(
                    f"  output {i}: SCOPE_MISMATCH alias oracle={actual_alias} "
                    f"eager={expected_alias}"
                )
                all_pass = False
                continue

        max_diff = (expected_tensor.float() - actual_tensor.float()).abs().max().item()
        ok = torch.allclose(expected_tensor.float(), actual_tensor.float(), atol=atol, rtol=rtol)
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype} "
            f"stride={expected_tensor.stride()} max_diff={max_diff:.2e})"
        )
        all_pass = all_pass and bool(ok)

    return all_pass


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
    parser.add_argument("--no-skip-stochastic", action="store_true", help="Accepted for template compatibility")
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
        ok = _run_check(instance, inputs, atol=args.atol, rtol=args.rtol)
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
            floor_status = "true_floor" if result["status"] == "GOOD" else (
                "at_floor" if result["status"] == "AT_FLOOR" else "not_true_floor"
            )
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {floor_status}")
            print(f"actionable: {'yes' if ACTIONABLE and result['status'] == 'GOOD' else 'no'}")


if __name__ == "__main__":
    main()
