"""
Oracle for pointwise_cf3acd87ba9e: fused BN-affine + ReLU + channel shuffle.

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes both
ShuffleNet branches and writes each element directly into the final
channel-shuffled storage, so the returned split outputs are views of the same
layout that Inductor produces after cat -> view -> permute -> clone -> split.
Inductor currently fuses each branch into the cat producer, but then materializes
that unshuffled cat buffer and launches a second pointwise clone/permute kernel
because scheduler fusion does not propagate the consumer's layout transform and
split offsets back into the producer store indexing. The actionable Inductor fix
is scheduler/codegen fusion through reshape/permute/clone/split-only layout
chains: sink the final output indexing into the producer and store the fused
pointwise result directly in the consumer layout.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "pointwise_cf3acd87ba9e"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 512
CHANNELS = 232
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
OUT_CHANNELS = 2 * CHANNELS
BRANCH_NUMEL = BATCH * CHANNELS * HW

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


@triton.jit
def _fused_bn_relu_channel_shuffle_kernel(
    mean_a_ptr,
    conv_a_ptr,
    var_a_ptr,
    weight_a_ptr,
    bias_a_ptr,
    mean_b_ptr,
    conv_b_ptr,
    var_b_ptr,
    weight_b_ptr,
    bias_b_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements

    hw = offsets % 49
    src_c = (offsets // 49) % 232
    batch = offsets // (232 * 49)

    src_offset = batch * (232 * 49) + src_c * 49 + hw

    a_val = tl.load(conv_a_ptr + src_offset, mask=mask, other=0.0).to(tl.float32)
    a_mean = tl.load(mean_a_ptr + src_c, mask=mask, other=0.0).to(tl.float32)
    a_var = tl.load(var_a_ptr + src_c, mask=mask, other=0.0).to(tl.float32)
    a_weight = tl.load(weight_a_ptr + src_c, mask=mask, other=0.0).to(tl.float32)
    a_bias = tl.load(bias_a_ptr + src_c, mask=mask, other=0.0).to(tl.float32)
    a_out = (a_val - a_mean) * (1.0 / tl.sqrt(a_var + 0.00001)) * a_weight + a_bias
    a_out = tl.maximum(a_out, 0.0)

    b_val = tl.load(conv_b_ptr + src_offset, mask=mask, other=0.0).to(tl.float32)
    b_mean = tl.load(mean_b_ptr + src_c, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.load(var_b_ptr + src_c, mask=mask, other=0.0).to(tl.float32)
    b_weight = tl.load(weight_b_ptr + src_c, mask=mask, other=0.0).to(tl.float32)
    b_bias = tl.load(bias_b_ptr + src_c, mask=mask, other=0.0).to(tl.float32)
    b_out = (b_val - b_mean) * (1.0 / tl.sqrt(b_var + 0.00001)) * b_weight + b_bias
    b_out = tl.maximum(b_out, 0.0)

    out_offset = batch * (464 * 49) + (2 * src_c) * 49 + hw
    tl.store(out_ptr + out_offset, a_out, mask=mask)
    tl.store(out_ptr + out_offset + 49, b_out, mask=mask)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(positive_variance: bool = True) -> tuple:
    module = _load_repro_module()
    inputs = list(module.make_inputs())
    if positive_variance:
        inputs[2] = inputs[2].abs() + 0.1
        inputs[7] = inputs[7].abs() + 0.1
    return tuple(inputs)


def triton_fused_channel_shuffle(
    arg212_1: torch.Tensor,
    convolution_42: torch.Tensor,
    arg213_1: torch.Tensor,
    arg214_1: torch.Tensor,
    arg215_1: torch.Tensor,
    arg227_1: torch.Tensor,
    convolution_45: torch.Tensor,
    arg228_1: torch.Tensor,
    arg229_1: torch.Tensor,
    arg230_1: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert convolution_42.shape == (BATCH, CHANNELS, HEIGHT, WIDTH)
    assert convolution_45.shape == (BATCH, CHANNELS, HEIGHT, WIDTH)
    assert convolution_42.is_contiguous()
    assert convolution_45.is_contiguous()

    shuffled = torch.empty(
        (BATCH, OUT_CHANNELS, HEIGHT, WIDTH),
        device=convolution_42.device,
        dtype=convolution_42.dtype,
    )
    block = 256
    grid = (triton.cdiv(BRANCH_NUMEL, block),)
    _fused_bn_relu_channel_shuffle_kernel[grid](
        arg212_1,
        convolution_42,
        arg213_1,
        arg214_1,
        arg215_1,
        arg227_1,
        convolution_45,
        arg228_1,
        arg229_1,
        arg230_1,
        shuffled,
        BRANCH_NUMEL,
        block=block,
    )
    return shuffled[:, CHANNELS:, :, :], shuffled[:, :CHANNELS, :, :]


def max_abs_diff(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> float:
    max_diff = 0.0
    for actual_tensor, expected_tensor in zip(actual, expected):
        diff = (actual_tensor.float() - expected_tensor.float()).abs()
        finite = diff[torch.isfinite(diff)]
        if finite.numel():
            max_diff = max(max_diff, finite.max().item())
    return max_diff


def allclose_with_nan(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], rtol: float, atol: float) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        torch.allclose(a.float(), e.float(), rtol=rtol, atol=atol, equal_nan=True)
        for a, e in zip(actual, expected)
    )


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = make_inputs(positive_variance=True)
    model = module.Repro().cuda()

    with torch.no_grad():
        expected = model(*inputs)
        actual = triton_fused_channel_shuffle(*inputs)
        torch.cuda.synchronize()

    ok = allclose_with_nan(actual, expected, rtol=rtol, atol=atol)
    diff = max_abs_diff(actual, expected)
    print(f"check max_abs={diff:.6e} allclose={ok}")
    print(f"expected strides: {[tuple(t.stride()) for t in expected]}")
    print(f"oracle strides:   {[tuple(t.stride()) for t in actual]}")
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _bench_cuda(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _compile_with_config(module, inputs: tuple, config: dict[str, object], warmup: int):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = module.Repro().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(warmup: int, rep: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    inputs = make_inputs(positive_variance=True)
    module = _load_repro_module()
    dense_read_bytes = 2 * BATCH * CHANNELS * HW * 2
    dense_write_bytes = BATCH * OUT_CHANNELS * HW * 2
    floor_bytes = dense_read_bytes + dense_write_bytes

    print(
        "oracle shape: "
        f"two f16[{BATCH}, {CHANNELS}, {HEIGHT}, {WIDTH}] inputs -> "
        f"shuffled f16[{BATCH}, {OUT_CHANNELS}, {HEIGHT}, {WIDTH}] storage"
    )
    print(f"direct read+write traffic: {floor_bytes / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda(lambda: triton_fused_channel_shuffle(*inputs), warmup=warmup, rep=rep)

    oracle_bw = floor_bytes / (oracle_us * 1e-6) / 1e12
    print(f"oracle fused channel shuffle: {oracle_us:.3f} us ({oracle_bw:.3f} TB/s direct bytes)")

    if no_compile:
        return

    print("torch.compile full repro timings include the current cat materialization and clone/permute kernel")
    for label, config in COMPILE_CONFIGS:
        try:
            compiled = _compile_with_config(module, inputs, config, warmup)
            compiled_us = _bench_cuda(lambda: compiled(*inputs), warmup=warmup, rep=rep)
            print(f"torch.compile {label}: {compiled_us:.3f} us")
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile config baselines")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(args.rtol, args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
