"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet training-BatchNorm, running-stat copy_ side effects, affine ReLU, and returned 7x7 spatial mean with explicit Triton kernels matching the tuned two-kernel full-scope schedule, whereas Inductor already emits an equivalent Welford stats/update kernel plus a direct pooled ReLU writer for this shape; Inductor cannot materially improve this repro today through local fusion because the remaining time is dominated by the required channel-stat read, second input read for the nonlinear ReLU spatial mean, output/running-stat stores, and two launches; the fix is BANDWIDTH_BOUND: record this as an at-floor norm-template case unless broader normalization or launch-overhead work moves the family."""
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


CLASSIFICATION = "BANDWIDTH_BOUND"
ACTIONABLE = False

N = 512
CHANNELS = 960
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
TOTAL_ROWS = N * CHANNELS
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361
ROW_RBLOCK = 64
STATS_XBLOCK = 2
STATS_RBLOCK = 1024
OUTPUT_XBLOCK = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice

    @triton.jit
    def _stats_update_welford_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        stats_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < channels
        r_base = tl.arange(0, RBLOCK)[None, :]
        mean_acc = tl.zeros([XBLOCK, RBLOCK], tl.float32)
        m2_acc = tl.zeros([XBLOCK, RBLOCK], tl.float32)
        weight_acc = tl.zeros([XBLOCK, RBLOCK], tl.float32)

        for r_start in tl.range(0, elements_per_channel, RBLOCK):
            r_index = r_start + r_base
            r_mask = r_index < elements_per_channel
            hw = r_index % hw_size
            n_idx = r_index // hw_size
            vals = tl.load(
                x_ptr + (hw + hw_size * xindex + channels * hw_size * n_idx),
                mask=r_mask & xmask,
                eviction_policy="evict_first",
                other=0.0,
            )
            vals = tl.broadcast_to(vals, [XBLOCK, RBLOCK])
            mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
                vals, mean_acc, m2_acc, weight_acc, r_start == 0
            )
            mean_acc = tl.where(r_mask & xmask, mean_next, mean_acc)
            m2_acc = tl.where(r_mask & xmask, m2_next, m2_acc)
            weight_acc = tl.where(r_mask & xmask, weight_next, weight_acc)

        mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
        mean = mean[:, None]
        m2 = m2[:, None]
        var = m2 / elements_per_channel

        old_running_mean = tl.load(running_mean_ptr + xindex, mask=xmask, other=0.0)
        old_running_var = tl.load(running_var_ptr + xindex, mask=xmask, other=0.0)
        new_running_mean = mean * momentum + old_running_mean * (1.0 - momentum)
        new_running_var = var * running_var_correction * momentum + old_running_var * (1.0 - momentum)

        tl.store(stats_ptr + xindex, mean, mask=xmask)
        tl.store(stats_ptr + channels + xindex, m2, mask=xmask)
        tl.store(running_mean_ptr + xindex, new_running_mean, mask=xmask)
        tl.store(running_var_ptr + xindex, new_running_var, mask=xmask)

    @triton.jit
    def _relu_spatial_mean_m2_kernel(
        out_ptr,
        x_ptr,
        mean_ptr,
        m2_ptr,
        weight_ptr,
        bias_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        total_rows: tl.constexpr,
        eps: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        hw_offsets = tl.arange(0, RBLOCK)[None, :]
        hw_mask = hw_offsets < hw_size
        channel = row_offsets % channels

        x = tl.load(
            x_ptr + (hw_offsets + hw_size * row_offsets),
            mask=hw_mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        mean = tl.load(mean_ptr + channel, eviction_policy="evict_last")
        m2 = tl.load(m2_ptr + channel, eviction_policy="evict_last")
        weight = tl.load(weight_ptr + channel, eviction_policy="evict_last")
        bias = tl.load(bias_ptr + channel, eviction_policy="evict_last")

        y = x - mean
        invstd = libdevice.rsqrt(m2 / elements_per_channel + eps)
        y = y * invstd
        y = y * weight + bias
        y = triton_helpers.maximum(tl.full([1, 1], 0, tl.int32), y)
        y = tl.where(hw_mask, y, 0.0)
        pooled = tl.sum(y, axis=1)[:, None] / hw_size
        tl.store(out_ptr + row_offsets, pooled)


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

    convolution_93, arg505_1, arg506_1, arg507_1, arg508_1 = inputs
    x = _expect_f32_tensor(
        "convolution_93",
        convolution_93,
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    running_mean = _expect_f32_tensor("arg505_1", arg505_1, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("arg506_1", arg506_1, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("arg507_1", arg507_1, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("arg508_1", arg508_1, (CHANNELS,), (1,))

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
    pooled = torch.relu(y).mean(dim=(-1, -2), keepdim=True)
    out = torch.as_strided(
        pooled,
        (N, CHANNELS, 1, 1),
        (CHANNELS, 1, CHANNELS, CHANNELS),
    )
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    running_mean.copy_(mean_1d * MOMENTUM + running_mean * (1.0 - MOMENTUM))
    running_var.copy_(var_1d * RUNNING_VAR_CORRECTION * MOMENTUM + running_var * (1.0 - MOMENTUM))
    return out, running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([512, 960, 7, 7], f32), T([960], f32), T([960], f32), T([960], f32), T([960], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope, including running-stat mutations."""
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    stats = torch.empty_strided((2, CHANNELS), (CHANNELS, 1), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(
        (N, CHANNELS, 1, 1),
        (CHANNELS, 1, CHANNELS, CHANNELS),
        device=x.device,
        dtype=torch.float32,
    )

    _stats_update_welford_kernel[(triton.cdiv(CHANNELS, STATS_XBLOCK),)](
        x,
        running_mean,
        running_var,
        stats,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        XBLOCK=STATS_XBLOCK,
        RBLOCK=STATS_RBLOCK,
        num_warps=16,
        num_stages=1,
    )
    _relu_spatial_mean_m2_kernel[(triton.cdiv(TOTAL_ROWS, OUTPUT_XBLOCK),)](
        out,
        x,
        stats,
        stats[1],
        weight,
        bias,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        total_rows=TOTAL_ROWS,
        eps=EPS,
        XBLOCK=OUTPUT_XBLOCK,
        RBLOCK=ROW_RBLOCK,
        num_warps=2,
        num_stages=1,
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

        if not expected_tensor.is_floating_point():
            ok = torch.equal(expected_tensor, actual_tensor)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected_tensor.dtype})")
            all_pass = all_pass and bool(ok)
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
            print(f"actionable: {'yes' if ACTIONABLE and result['status'] == 'GOOD' else 'no'}")


if __name__ == "__main__":
    main()
