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

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


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


@oracle_impl(hardware="H100", shapes="(T([232], f16), T([512, 232, 7, 7], f16), T([232], f16), T([232], f16), T([232], f16), T([232], f16), T([512, 232, 7, 7], f16), T([232], f16), T([232], f16), T([232], f16), S([512, 2, 232, 7, 7]), S([512, 464, 7, 7]))")
def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return triton_fused_channel_shuffle(*inputs)


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
