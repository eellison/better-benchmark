"""Gap diagnosis (classification: RECOMPUTE_FUSION): this oracle computes the complete Visformer chained training-BatchNorm scope with one channel-tiled kernel for the first per-channel var/mean/running-stat update and one channel-tiled kernel that recomputes the first BN affine plus residual directly inside the second var/mean reduction and final affine output store, whereas Inductor currently treats the two norm-template reductions and the intermediate affine/residual producer as separately scheduled regions; Inductor cannot do this today because its normalization scheduler does not thread a first BN epilogue expression as a recomputed producer for a dependent second BN reduction with mutable running-stat side outputs; the fix is RECOMPUTE_FUSION: allow chained BN-training templates to recompute cheap first-epilogue pointwise work inside the dependent reduction/output schedule instead of materializing or separately scheduling the intermediate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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
C = 768
H = 7
W = 7
HW = H * W
ELEMENTS_PER_CHANNEL = N * HW
TOTAL_ELEMENTS = N * C * HW
EPS = 1.0e-5
MOMENTUM = 0.1
OLD_WEIGHT = 0.9
RUNNING_VAR_CORRECTION = 1.0001594642002871
BLOCK_K = 8192
CLASSIFICATION = "RECOMPUTE_FUSION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _first_bn_stats_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        mean_ptr,
        invstd_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        old_weight: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < elements_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        x_offsets = n_idx * channels * hw_size + channel * hw_size + hw_idx

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        total_sum = tl.sum(x, axis=0)
        total_sum_sq = tl.sum(x * x, axis=0)
        mean = total_sum / elements_per_channel
        var = total_sum_sq / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * old_weight + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * old_weight + var * running_var_correction * momentum,
        )
        tl.store(mean_ptr + channel, mean)
        tl.store(invstd_ptr + channel, invstd)

    @triton.jit
    def _second_bn_recompute_output_kernel(
        x_ptr,
        residual_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight1_ptr,
        bias1_ptr,
        weight2_ptr,
        bias2_ptr,
        mean1_ptr,
        invstd1_ptr,
        invstd2_ptr,
        mean2_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        old_weight: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < elements_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        x_offsets = n_idx * channels * hw_size + channel * hw_size + hw_idx
        residual_offsets = channel * hw_size + hw_idx

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
        mean1 = tl.load(mean1_ptr + channel).to(tl.float32)
        invstd1 = tl.load(invstd1_ptr + channel).to(tl.float32)
        weight1 = tl.load(weight1_ptr + channel).to(tl.float32)
        bias1 = tl.load(bias1_ptr + channel).to(tl.float32)

        first_bn = (x - mean1) * invstd1
        first_affine = first_bn * weight1 + bias1
        y1 = first_affine + residual
        y1_for_stats = tl.where(mask, y1, 0.0)

        total_sum = tl.sum(y1_for_stats, axis=0)
        total_sum_sq = tl.sum(y1_for_stats * y1_for_stats, axis=0)
        mean2 = total_sum / elements_per_channel
        var2 = total_sum_sq / elements_per_channel - mean2 * mean2
        var2 = tl.maximum(var2, 0.0)
        invstd2 = tl.rsqrt(var2 + eps)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * old_weight + mean2 * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * old_weight + var2 * running_var_correction * momentum,
        )
        tl.store(invstd2_ptr + channel, invstd2)
        tl.store(mean2_ptr + channel, mean2)

        weight2 = tl.load(weight2_ptr + channel).to(tl.float32)
        bias2 = tl.load(bias2_ptr + channel).to(tl.float32)
        out = (y1 - mean2) * invstd2
        out = out * weight2 + bias2
        tl.store(out_ptr + x_offsets, out, mask=mask)


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
) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        x,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        residual,
        running_mean2,
        running_var2,
        weight2,
        bias2,
    ) = inputs

    activation_stride = (C * HW, HW, W, 1)
    residual_stride = (C * HW, HW, W, 1)
    tensors = (
        _expect_f32_tensor("convolution_40", x, (N, C, H, W), activation_stride),
        _expect_f32_tensor("arg138_1", running_mean1, (C,), (1,)),
        _expect_f32_tensor("arg139_1", running_var1, (C,), (1,)),
        _expect_f32_tensor("arg140_1", weight1, (C,), (1,)),
        _expect_f32_tensor("arg141_1", bias1, (C,), (1,)),
        _expect_f32_tensor("arg142_1", residual, (1, C, H, W), residual_stride),
        _expect_f32_tensor("arg144_1", running_mean2, (C,), (1,)),
        _expect_f32_tensor("arg145_1", running_var2, (C,), (1,)),
        _expect_f32_tensor("arg146_1", weight2, (C,), (1,)),
        _expect_f32_tensor("arg147_1", bias2, (C,), (1,)),
    )
    device = tensors[0].device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")
    return tensors


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    (
        x,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        residual,
        running_mean2,
        running_var2,
        weight2,
        bias2,
    ) = _validate_inputs(inputs)

    var1, mean1 = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd1 = torch.rsqrt(var1 + EPS)
    y1 = (x - mean1) * invstd1
    y1 = y1 * weight1[None, :, None, None] + bias1[None, :, None, None]
    y1 = y1 + residual

    mean1_1d = mean1.squeeze((0, 2, 3))
    var1_1d = var1.squeeze((0, 2, 3))
    running_mean1.copy_(running_mean1 * OLD_WEIGHT + mean1_1d * MOMENTUM)
    running_var1.copy_(running_var1 * OLD_WEIGHT + var1_1d * RUNNING_VAR_CORRECTION * MOMENTUM)

    var2, mean2 = torch.var_mean(y1, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd2_4d = torch.rsqrt(var2 + EPS)
    out = (y1 - mean2) * invstd2_4d
    out = out * weight2[None, :, None, None] + bias2[None, :, None, None]

    mean2_1d = mean2.squeeze((0, 2, 3))
    var2_1d = var2.squeeze((0, 2, 3))
    invstd2 = invstd2_4d.squeeze((0, 2, 3))
    running_mean2.copy_(running_mean2 * OLD_WEIGHT + mean2_1d * MOMENTUM)
    running_var2.copy_(running_var2 * OLD_WEIGHT + var2_1d * RUNNING_VAR_CORRECTION * MOMENTUM)
    return invstd2, out, mean2_1d[None, :, None, None], running_mean1, running_var1, running_mean2, running_var2


@oracle_impl(hardware="H100", shapes="(T([128, 768, 7, 7], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([1, 768, 7, 7], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full chained training-BatchNorm repro scope.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same seven outputs, including the in-place running-stat copy_ aliases.
    """
    (
        x,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        residual,
        running_mean2,
        running_var2,
        weight2,
        bias2,
    ) = _validate_inputs(inputs)

    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    mean1 = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    invstd1 = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    invstd2 = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    mean2 = torch.empty_strided((1, C, 1, 1), (C, 1, 1, 1), device=x.device, dtype=torch.float32)
    out = torch.empty_strided((N, C, H, W), (C * HW, HW, W, 1), device=x.device, dtype=torch.float32)

    _first_bn_stats_kernel[(C,)](
        x,
        running_mean1,
        running_var1,
        mean1,
        invstd1,
        channels=C,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        old_weight=OLD_WEIGHT,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )
    _second_bn_recompute_output_kernel[(C,)](
        x,
        residual,
        running_mean2,
        running_var2,
        weight1,
        bias1,
        weight2,
        bias2,
        mean1,
        invstd1,
        invstd2,
        mean2,
        out,
        channels=C,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        old_weight=OLD_WEIGHT,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )

    return invstd2, out, mean2, running_mean1, running_var1, running_mean2, running_var2


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            cloned.append(item.clone())
        else:
            cloned.append(item)
    return tuple(cloned)


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, (tuple, list)):
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
        eager = instance(*eager_inputs)
        oracle_out = oracle_forward(oracle_inputs)
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
    for i, (eager_tensor, oracle_tensor) in enumerate(zip(eager_list, oracle_list)):
        shape_ok = eager_tensor.shape == oracle_tensor.shape
        dtype_ok = eager_tensor.dtype == oracle_tensor.dtype
        stride_ok = eager_tensor.stride() == oracle_tensor.stride()
        if not shape_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(oracle_tensor.shape)} "
                f"eager={list(eager_tensor.shape)}"
            )
            all_pass = False
            continue
        if not dtype_ok:
            print(f"  output {i}: SCOPE_MISMATCH dtype oracle={oracle_tensor.dtype} eager={eager_tensor.dtype}")
            all_pass = False
            continue
        if not stride_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={oracle_tensor.stride()} "
                f"eager={eager_tensor.stride()}"
            )
            all_pass = False
            continue

        if eager_tensor.is_floating_point():
            eager_f32 = eager_tensor.float()
            oracle_f32 = oracle_tensor.float()
            max_diff = (eager_f32 - oracle_f32).abs().max().item()
            values_ok = torch.allclose(eager_f32, oracle_f32, atol=atol, rtol=rtol)
            print(
                f"  output {i}: {'PASS' if values_ok else 'FAIL'} "
                f"(shape={list(eager_tensor.shape)} dtype={eager_tensor.dtype} "
                f"stride={eager_tensor.stride()} max_diff={max_diff:.2e})"
            )
        else:
            values_ok = torch.equal(eager_tensor, oracle_tensor)
            print(
                f"  output {i}: {'PASS' if values_ok else 'FAIL'} "
                f"(exact, shape={list(eager_tensor.shape)} dtype={eager_tensor.dtype} "
                f"stride={eager_tensor.stride()})"
            )
        all_pass = all_pass and bool(values_ok)

    alias_ok = (
        isinstance(oracle_out, tuple)
        and len(oracle_out) == 7
        and oracle_out[3] is oracle_inputs[1]
        and oracle_out[4] is oracle_inputs[2]
        and oracle_out[5] is oracle_inputs[6]
        and oracle_out[6] is oracle_inputs[7]
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
