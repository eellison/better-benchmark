"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Shufflenet training-BatchNorm plus skip-branch channel shuffle scope, including the BN var_mean reduction, running mean/variance copy_ side effects, affine ReLU epilogue, cat/view/permute/clone/view layout transform, and split view return aliases by writing the skip branch and BN branch directly into the final shuffled storage, whereas Inductor lowers the decomposed BN, mutable running-stat updates, concat, permutation clone, and split as generic reduction and layout work; Inductor cannot do this today because its scheduler does not preserve the fixed channel-shuffle consumer across a training normalization producer with side-effect outputs and an already-strided sibling input; the fix is SCHEDULER_FUSION: teach the BN-training/layout scheduler to fuse running-stat epilogues with direct channel-shuffle stores while preserving split-view aliases."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


N = 512
C = 232
H = 7
W = 7
HW = H * W
K = N * HW
OUT_C = C * 2
EPS = 1.0e-5
MOMENTUM = 0.1
OLD_WEIGHT = 0.9
RUNNING_VAR_CORRECTION = 1.0000398612827361
BLOCK_K = 32768


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _bn_stats_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        mean_ptr,
        invstd_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        k_size: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        old_weight: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < k_size
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size

        x_offsets = n_idx * channels * hw_size + channel * hw_size + hw_idx
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=0) / k_size
        centered = x - mean
        var = tl.sum(centered * centered, axis=0) / k_size
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        tl.store(mean_ptr + channel, mean)
        tl.store(invstd_ptr + channel, invstd)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * old_weight + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * old_weight + var * running_var_correction * momentum,
        )

    @triton.jit
    def _shuffle_output_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        skip_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        out_channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        hw_idx = offsets % hw_size
        channel = (offsets // hw_size) % channels
        n_idx = offsets // (channels * hw_size)

        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * invstd * weight + bias
        y = tl.where(y != y, y, tl.maximum(y, 0.0))

        skip_offsets = n_idx * out_channels * hw_size + channel * hw_size + hw_idx
        skip = tl.load(skip_ptr + skip_offsets, mask=mask, other=0.0).to(tl.float32)

        out_base = n_idx * out_channels * hw_size + channel * 2 * hw_size + hw_idx
        tl.store(out_ptr + out_base, skip, mask=mask)
        tl.store(out_ptr + out_base + hw_size, y, mask=mask)

def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _expect_tensor(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        convolution_51,
        arg309_1,
        arg310_1,
        arg311_1,
        arg312_1,
        getitem_122,
        shape0,
        shape1,
    ) = inputs

    activation_stride = (C * HW, HW, W, 1)
    shuffled_half_stride = (OUT_C * HW, HW, W, 1)
    conv = _expect_tensor("convolution_51", convolution_51, (N, C, H, W), activation_stride)
    running_mean = _expect_tensor("arg309_1", arg309_1, (C,), (1,))
    running_var = _expect_tensor("arg310_1", arg310_1, (C,), (1,))
    weight = _expect_tensor("arg311_1", arg311_1, (C,), (1,))
    bias = _expect_tensor("arg312_1", arg312_1, (C,), (1,))
    skip = _expect_tensor("getitem_122", getitem_122, (N, C, H, W), shuffled_half_stride)

    tensors = (conv, running_mean, running_var, weight, bias, skip)
    device = conv.device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")

    if _shape_tuple(shape0) != (N, 2, C, H, W):
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != (N, OUT_C, H, W):
        raise ValueError(f"unexpected second view shape parameter: {shape1!r}")

    return tensors


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    (
        conv,
        running_mean,
        running_var,
        weight,
        bias,
        skip,
        shape0,
        shape1,
    ) = inputs
    var, mean = torch.var_mean(conv, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    y = torch.relu((conv - mean) * invstd * weight[None, :, None, None] + bias[None, :, None, None])
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    new_mean = running_mean * OLD_WEIGHT + mean_1d * MOMENTUM
    new_var = running_var * OLD_WEIGHT + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM
    shuffled = (
        torch.cat([skip, y], dim=1)
        .view(_shape_tuple(shape0))
        .permute(0, 2, 1, 3, 4)
        .clone(memory_format=torch.contiguous_format)
        .view(_shape_tuple(shape1))
    )
    first, second = torch.split(shuffled, C, dim=1)
    running_mean.copy_(new_mean)
    running_var.copy_(new_var)
    return second, running_mean, running_var, first


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the complete BN, running-stat update, channel-shuffle, split-return scope."""
    (
        conv,
        running_mean,
        running_var,
        weight,
        bias,
        skip,
    ) = _validate_inputs(inputs)

    if triton is None or not conv.is_cuda:
        return _torch_reference(inputs)

    shuffled = torch.empty_strided(
        (N, OUT_C, H, W),
        (OUT_C * HW, HW, W, 1),
        device=conv.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided((C,), (1,), device=conv.device, dtype=torch.float32)
    invstd = torch.empty_strided((C,), (1,), device=conv.device, dtype=torch.float32)

    _bn_stats_kernel[(C,)](
        conv,
        running_mean,
        running_var,
        mean,
        invstd,
        channels=C,
        hw_size=HW,
        k_size=K,
        eps=EPS,
        momentum=MOMENTUM,
        old_weight=OLD_WEIGHT,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )
    _shuffle_output_kernel[(triton.cdiv(N * C * HW, 256),)](
        conv,
        weight,
        bias,
        skip,
        mean,
        invstd,
        shuffled,
        total=N * C * HW,
        channels=C,
        out_channels=OUT_C,
        hw_size=HW,
        BLOCK=256,
        num_warps=4,
        num_stages=3,
    )

    first = shuffled[:, :C, :, :]
    second = shuffled[:, C:, :, :]
    return second, running_mean, running_var, first


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            copy = torch.empty_strided(
                tuple(item.shape),
                tuple(item.stride()),
                device=item.device,
                dtype=item.dtype,
            )
            copy.copy_(item)
            cloned.append(copy)
        else:
            cloned.append(item)
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


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


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
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        shape_ok = expected.shape == actual.shape
        dtype_ok = expected.dtype == actual.dtype
        stride_ok = expected.stride() == actual.stride()
        if not shape_ok or not dtype_ok or not stride_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH "
                f"shape oracle={list(actual.shape)} eager={list(expected.shape)} "
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
        and oracle_out[1] is oracle_inputs[1]
        and oracle_out[2] is oracle_inputs[2]
        and _same_storage(oracle_out[0], oracle_out[3])
        and oracle_out[3].storage_offset() == 0
        and oracle_out[0].storage_offset() - oracle_out[3].storage_offset() == C * HW
    )
    print(f"  returned aliases: {'PASS' if alias_ok else 'FAIL'}")
    return all_pass and alias_ok


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
