"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Shufflenet dual training-BatchNorm block, including both channelwise var_mean reductions, running mean/variance copy_ updates, affine ReLU epilogues, cat/view/permute/clone/view channel shuffle, and split return contract by writing the two branches directly into the final shuffled layout, whereas Inductor lowers the decomposed BN, concat, layout shuffle, and split graph through generic reduction and layout scheduling; Inductor cannot do this today because its scheduler does not preserve the semantic channel-shuffle consumer across sibling BN-training reductions with mutable running-stat outputs; the fix is SCHEDULER_FUSION: teach the BN-training/layout scheduler to fuse fixed branch reductions with direct channel-shuffle stores and side-effect running-stat returns."""
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
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _bn_relu_shuffle_branch_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        branch: tl.constexpr,
        channels: tl.constexpr,
        out_channels: tl.constexpr,
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

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * old_weight + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * old_weight + var * running_var_correction * momentum,
        )

        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        y = centered * invstd * weight + bias
        y = tl.where(y != y, y, tl.maximum(y, 0.0))

        shuffled_channel = channel * 2 + branch
        out_offsets = n_idx * out_channels * hw_size + shuffled_channel * hw_size + hw_idx
        tl.store(out_ptr + out_offsets, y, mask=mask)


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
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        convolution_42,
        arg255_1,
        arg256_1,
        arg257_1,
        arg258_1,
        convolution_45,
        arg273_1,
        arg274_1,
        arg275_1,
        arg276_1,
        shape0,
        shape1,
    ) = inputs

    activation_stride = (C * HW, HW, W, 1)
    conv0 = _expect_tensor("convolution_42", convolution_42, (N, C, H, W), activation_stride)
    run_mean0 = _expect_tensor("arg255_1", arg255_1, (C,), (1,))
    run_var0 = _expect_tensor("arg256_1", arg256_1, (C,), (1,))
    weight0 = _expect_tensor("arg257_1", arg257_1, (C,), (1,))
    bias0 = _expect_tensor("arg258_1", arg258_1, (C,), (1,))
    conv1 = _expect_tensor("convolution_45", convolution_45, (N, C, H, W), activation_stride)
    run_mean1 = _expect_tensor("arg273_1", arg273_1, (C,), (1,))
    run_var1 = _expect_tensor("arg274_1", arg274_1, (C,), (1,))
    weight1 = _expect_tensor("arg275_1", arg275_1, (C,), (1,))
    bias1 = _expect_tensor("arg276_1", arg276_1, (C,), (1,))

    tensors = (conv0, run_mean0, run_var0, weight0, bias0, conv1, run_mean1, run_var1, weight1, bias1)
    device = conv0.device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")

    if _shape_tuple(shape0) != (N, 2, C, H, W):
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != (N, OUT_C, H, W):
        raise ValueError(f"unexpected second view shape parameter: {shape1!r}")

    return tensors


@oracle_impl(hardware="H100", shapes="(T([512, 232, 7, 7], f32), T([232], f32), T([232], f32), T([232], f32), T([232], f32), T([512, 232, 7, 7], f32), T([232], f32), T([232], f32), T([232], f32), T([232], f32), S([512, 2, 232, 7, 7]), S([512, 464, 7, 7]))")
def oracle_forward(inputs):
    """Run the complete dual-BN, channel-shuffle, split-return repro scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        conv0,
        run_mean0,
        run_var0,
        weight0,
        bias0,
        conv1,
        run_mean1,
        run_var1,
        weight1,
        bias1,
    ) = _validate_inputs(inputs)

    shuffled = torch.empty_strided(
        (N, OUT_C, H, W),
        (OUT_C * HW, HW, W, 1),
        device=conv0.device,
        dtype=torch.float32,
    )

    _bn_relu_shuffle_branch_kernel[(C,)](
        conv0,
        run_mean0,
        run_var0,
        weight0,
        bias0,
        shuffled,
        branch=0,
        channels=C,
        out_channels=OUT_C,
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
    _bn_relu_shuffle_branch_kernel[(C,)](
        conv1,
        run_mean1,
        run_var1,
        weight1,
        bias1,
        shuffled,
        branch=1,
        channels=C,
        out_channels=OUT_C,
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

    first = shuffled[:, :C, :, :]
    second = shuffled[:, C:, :, :]
    return second, run_mean0, run_var0, run_mean1, run_var1, first


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
