"""
Oracle for pointwise_7a4083b7e3d1

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full batch-norm-affine-ReLU pointwise region with a channel-blocked Triton kernel that loads each fp16 mean/variance/weight/bias value once per batch-channel tile, computes the f32 inverse standard deviation once per channel in that tile, broadcasts it over the 7x7 spatial block, rounds the affine result to fp16, and applies fp16 ReLU before storing the required contiguous fp16 output, whereas Inductor emits a generic flat pointwise kernel that repeats the variance sqrt/reciprocal and scalar broadcasts for every output element; Inductor cannot do this today because its pointwise scheduler does not hoist channel-invariant subexpressions out of the inner element loop or form a per-channel tiled affine epilogue for NCHW batch-norm inference patterns; the fix is NEW_PATTERN: add loop-invariant scalar hoisting or a BN-inference affine pointwise template that computes per-channel normalization scalars once per tile while preserving the final fp16 cast and ReLU semantics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 128
H = 7
W = 7
HW = H * W


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _bn_relu_kernel(
        mean_ptr,
        conv_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_block = tl.program_id(1)

        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.arange(0, BLOCK_HW)
        c_mask = c_offsets < C_
        hw_mask = hw_offsets < HW_
        mask = c_mask[:, None] & hw_mask[None, :]

        base = n * C_ * HW_
        offsets = base + c_offsets[:, None] * HW_ + hw_offsets[None, :]

        conv = tl.load(conv_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        invstd = tl.rsqrt(var + 1.0e-5)
        normalized = (conv - mean[:, None]) * invstd[:, None]
        affine = normalized * weight[:, None] + bias[:, None]

        affine_fp16 = affine.to(tl.float16)
        relu = tl.where(
            affine_fp16 != affine_fp16,
            affine_fp16,
            tl.maximum(affine_fp16, 0.0),
        )
        tl.store(out_ptr + offsets, relu, mask=mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_relu.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, conv, var, weight, bias = inputs
    names = ("mean", "conv", "var", "weight", "bias")
    for name, value in zip(names, inputs):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if not value.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if value.dtype != torch.float16:
            raise ValueError(f"{name} must have dtype torch.float16, got {value.dtype}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if tuple(mean.shape) != (C,):
        raise ValueError(f"unexpected mean shape: {tuple(mean.shape)}")
    if tuple(conv.shape) != (N, C, H, W):
        raise ValueError(f"unexpected conv shape: {tuple(conv.shape)}")
    if tuple(var.shape) != (C,) or tuple(weight.shape) != (C,) or tuple(bias.shape) != (C,):
        raise ValueError("var, weight, and bias must all have shape [128]")
    return mean, conv, var, weight, bias


def oracle_forward(inputs):
    """Run the full Repro.forward scope with the Triton BN-affine-ReLU kernel."""
    mean, conv, var, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=conv.device,
        dtype=torch.float16,
    )
    grid = lambda meta: (N, triton.cdiv(C, meta["BLOCK_C"]))
    _bn_relu_kernel[grid](mean, conv, var, weight, bias, out, C_=C, HW_=HW)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify correctness against eager Repro",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark oracle vs torch.compile",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for correctness check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for correctness check",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations for benchmark",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=200,
        help="Repetitions for benchmark",
    )
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Benchmark across all shapes from shapes.txt",
    )
    parser.add_argument(
        "--show-hw",
        action="store_true",
        help="Print GPU hardware info and exit",
    )
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


if __name__ == "__main__":
    main()
