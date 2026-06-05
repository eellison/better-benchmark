"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ResNeSt radix-2 split-attention forward returned by Repro.forward, including fp32 softmax over the convolution logits, the fp16 cast boundary, broadcast payload multiply, and radix branch sum into the final [32, 64, 56, 56] output in one Triton kernel, whereas Inductor currently schedules the decomposed view/permute/amax/exp/sum/div/cast, broadcast multiply, and branch reduction as generic separate kernels/materialized intermediates; Inductor cannot do this today because its scheduler cannot sink a small reduction-derived broadcast producer through the surrounding reshape/cast boundary into the consumer branch reduction over the payload layout; the fix is SCHEDULER_FUSION: teach reduction fusion to keep radix softmax weights in registers and generate the fused payload multiply/sum output loop directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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

RADIX = 2
BLOCK_HW = 2048


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _split_attention_sum_kernel(
        convolution_ptr,
        view_ptr,
        out_ptr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        radix: tl.constexpr,
        block_hw: tl.constexpr,
    ):
        tile = tl.program_id(0)
        c = tl.program_id(1)
        n = tl.program_id(2)

        hw: tl.constexpr = height * width
        spatial = tile * block_hw + tl.arange(0, block_hw)
        mask = spatial < hw

        conv_base = n * (radix * channels) + c
        x0 = tl.load(convolution_ptr + conv_base).to(tl.float32)
        x1 = tl.load(convolution_ptr + conv_base + channels).to(tl.float32)
        row_max = tl.maximum(x0, x1)
        exp0 = tl.exp(x0 - row_max)
        exp1 = tl.exp(x1 - row_max)
        denom = exp0 + exp1
        weight0 = (exp0 / denom).to(tl.float16)
        weight1 = (exp1 / denom).to(tl.float16)

        base0 = (n * radix * channels + c) * hw + spatial
        base1 = base0 + channels * hw
        v0 = tl.load(view_ptr + base0, mask=mask, other=0.0)
        v1 = tl.load(view_ptr + base1, mask=mask, other=0.0)
        out = (v0 * weight0 + v1 * weight1).to(tl.float16)

        out_base = (n * channels + c) * hw + spatial
        tl.store(out_ptr + out_base, out, mask=mask)


def _require_f16_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype is not torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if stride is not None and tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    convolution_6, view, shape0, shape1, shape2, shape3 = inputs
    view = _require_f16_tensor("view", view)
    if view.ndim != 5:
        raise ValueError(f"view must be rank 5, got rank {view.ndim}")

    batch, radix, channels, height, width = view.shape
    if radix != RADIX:
        raise ValueError(f"view radix dim must be {RADIX}, got {radix}")
    expected_view_stride = (
        RADIX * channels * height * width,
        channels * height * width,
        height * width,
        width,
        1,
    )
    if tuple(view.stride()) != expected_view_stride:
        raise ValueError(
            f"view has stride {tuple(view.stride())}, expected {expected_view_stride}"
        )

    convolution_6 = _require_f16_tensor(
        "convolution_6",
        convolution_6,
        (batch, RADIX * channels, 1, 1),
        (RADIX * channels, 1, 1, 1),
    )
    if convolution_6.device != view.device:
        raise ValueError("convolution_6 and view must be on the same device")

    expected_shapes = (
        [batch, 1, RADIX, -1],
        [batch, -1],
        [batch, -1, 1, 1],
        [batch, RADIX, channels, 1, 1],
    )
    got_shapes = (list(shape0), list(shape1), list(shape2), list(shape3))
    if got_shapes != expected_shapes:
        raise ValueError(f"unexpected shape params: {got_shapes}")

    return convolution_6, view, batch, channels, height, width


def _torch_oracle(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    convolution_6, view, shape0, shape1, shape2, shape3 = inputs
    logits = convolution_6.view(shape0).permute(0, 2, 1, 3).float()
    shifted = logits - logits.amax([1], keepdim=True)
    exp = shifted.exp()
    weights = (exp / exp.sum([1], keepdim=True)).half()
    weights = weights.view(shape1).view(shape2).view(shape3)
    return (view * weights).sum([1])


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full split-attention softmax, broadcast multiply, and branch sum."""
    convolution_6, view, batch, channels, height, width = _validate_inputs(inputs)
    if not view.is_cuda:
        return _torch_oracle(inputs)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    out = torch.empty_strided(
        (batch, channels, height, width),
        (channels * height * width, height * width, width, 1),
        device=view.device,
        dtype=torch.float16,
    )
    grid = (triton.cdiv(height * width, BLOCK_HW), channels, batch)
    _split_attention_sum_kernel[grid](
        convolution_6,
        view,
        out,
        channels=channels,
        height=height,
        width=width,
        radix=RADIX,
        block_hw=BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )
    return out


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
