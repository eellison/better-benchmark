"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileViT training-BatchNorm, running-stat copy_ side effects, affine SiLU, and final spatial mean in one channel-specialized Triton kernel that returns only the required `[128,640]` pooled tensor plus the two mutated running-stat aliases, whereas Inductor lowers the captured var_mean/normalization/SiLU/mean graph through its generic normalization schedule; Inductor cannot do this today because norm-template canonicalization does not keep mutable training-BN side outputs and the downstream spatial reduction inside one per-channel template; the fix is SCHEDULER_FUSION: teach norm-template canonicalization to fuse training-BN statistics, mutable running-stat returns, affine activation, and small spatial pooling consumers into one channel-tiled lowering."""
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
CHANNELS = 640
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001220852154804
BLOCK_N = 128
BLOCK_HW = 64
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _bn_silu_spatial_mean_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        n_offsets = tl.arange(0, BLOCK_N_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        hw_mask = hw_offsets < hw_size
        offsets = (n_offsets[:, None] * channels + channel) * hw_size + hw_offsets[None, :]

        x = tl.load(x_ptr + offsets, mask=hw_mask[None, :], other=0.0).to(tl.float32)
        row_sum = tl.sum(x, axis=1)
        row_sum_sq = tl.sum(x * x, axis=1)
        sum_x = tl.sum(row_sum, axis=0)
        sum_x_sq = tl.sum(row_sum_sq, axis=0)
        mean = sum_x / elements_per_channel
        var = tl.maximum(sum_x_sq / elements_per_channel - mean * mean, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_running_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + channel).to(tl.float32)
        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)

        tl.store(running_mean_ptr + channel, old_running_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )

        y = (x - mean) * invstd
        y = y * weight + bias
        silu = y / (tl.exp(-y) + 1.0)
        silu = tl.where(hw_mask[None, :], silu, 0.0)
        pooled = tl.sum(silu, axis=1) / hw_size
        tl.store(out_ptr + n_offsets * channels + channel, pooled)


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
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    convolution_34, arg306_1, arg307_1, arg308_1, arg309_1, shape_param = inputs
    x = _expect_f32_tensor(
        "convolution_34",
        convolution_34,
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    running_mean = _expect_f32_tensor("arg306_1", arg306_1, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("arg307_1", arg307_1, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("arg308_1", arg308_1, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("arg309_1", arg309_1, (CHANNELS,), (1,))

    if tuple(shape_param) != (N, CHANNELS):
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")

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
    pooled = (y / (torch.exp(-y) + 1.0)).mean(dim=(-1, -2), keepdim=True)
    view_default = torch.as_strided(
        pooled,
        (N, CHANNELS, 1, 1),
        (CHANNELS, 1, CHANNELS, CHANNELS),
    ).view(N, CHANNELS)
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    running_mean.copy_(mean_1d * MOMENTUM + running_mean * (1.0 - MOMENTUM))
    running_var.copy_(
        var_1d * RUNNING_VAR_CORRECTION * MOMENTUM
        + running_var * (1.0 - MOMENTUM)
    )
    return view_default, running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([128, 640, 8, 8], f32), T([640], f32), T([640], f32), T([640], f32), T([640], f32), S([128, 640]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope, including copy_ side effects."""
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    view_default = torch.empty_strided(
        (N, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _bn_silu_spatial_mean_kernel[(CHANNELS,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        view_default,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_N_=BLOCK_N,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )
    return view_default, running_mean, running_var


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for item in inputs:
        cloned.append(item.clone() if isinstance(item, torch.Tensor) else item)
    return tuple(cloned)


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
        oracle_out = oracle_fn(oracle_inputs)
        if isinstance(oracle_inputs[0], torch.Tensor) and oracle_inputs[0].is_cuda:
            torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    alias_input_indices = (None, 1, 2)
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        if expected.shape != actual.shape:
            print(f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual.shape)} eager={list(expected.shape)}")
            all_pass = False
            continue
        if expected.dtype != actual.dtype:
            print(f"  output {i}: SCOPE_MISMATCH dtype oracle={actual.dtype} eager={expected.dtype}")
            all_pass = False
            continue
        if expected.stride() != actual.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={actual.stride()} "
                f"eager={expected.stride()}"
            )
            all_pass = False
            continue

        alias_index = alias_input_indices[i]
        if alias_index is not None:
            expected_alias = expected.data_ptr() == eager_inputs[alias_index].data_ptr()
            actual_alias = actual.data_ptr() == oracle_inputs[alias_index].data_ptr()
            if expected_alias != actual_alias or not actual_alias:
                print(
                    f"  output {i}: SCOPE_MISMATCH alias oracle={actual_alias} "
                    f"eager={expected_alias}"
                )
                all_pass = False
                continue

        if expected.is_floating_point():
            max_diff = (expected.float() - actual.float()).abs().max().item()
            ok = torch.allclose(expected.float(), actual.float(), atol=atol, rtol=rtol)
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected.shape)} dtype={expected.dtype} "
                f"stride={expected.stride()} max_diff={max_diff:.2e})"
            )
        else:
            ok = torch.equal(expected, actual)
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
            floor_status = "true_floor" if result["status"] == "GOOD" else (
                "at_floor" if result["status"] == "AT_FLOOR" else "not_true_floor"
            )
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {floor_status}")


if __name__ == "__main__":
    main()
