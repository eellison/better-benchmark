"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete fp16 ResNeSt split-attention forward scope by forming radix-2 fp32 softmax weights, rounding them to fp16, applying the weighted branch sum, and folding the following 3x3 stride-2 avg_pool2d into one output-tiled Triton kernel, whereas Inductor currently materializes the decomposed softmax, broadcast multiply, branch sum, and avg_pool2d as separate generic scheduled work; Inductor cannot do this today because its scheduler cannot inline a small reduction-derived broadcast producer through a branch reduction into a downstream pooling stencil while preserving the fp16 cast boundary; the fix is SCHEDULER_FUSION: extend reduction/stencil fusion so radix softmax weights and weighted branch sums can be sunk into the avg_pool2d output loop."""
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
POOL_KERNEL = 3
POOL_STRIDE = 2
POOL_PADDING = 1
POOL_DIVISOR = 9.0
BLOCK_OUT = 128


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _split_attention_avgpool_kernel(
        convolution_ptr,
        view_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        out_height: tl.constexpr,
        out_width: tl.constexpr,
        pool_divisor: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < total

        ow = offsets % out_width
        tmp = offsets // out_width
        oh = tmp % out_height
        tmp = tmp // out_height
        c = tmp % channels
        n = tmp // channels

        conv_base = n * (2 * channels) + c
        x0 = tl.load(convolution_ptr + conv_base, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(
            convolution_ptr + conv_base + channels,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        row_max = tl.maximum(x0, x1)
        exp0 = tl.exp(x0 - row_max)
        exp1 = tl.exp(x1 - row_max)
        denom = exp0 + exp1
        weight0 = (exp0 / denom).to(tl.float16)
        weight1 = (exp1 / denom).to(tl.float16)

        branch_stride = channels * height * width
        acc = tl.full((block,), 0.0, tl.float32)
        for kh in tl.static_range(0, 3):
            ih = oh * 2 + kh - 1
            valid_h = (ih >= 0) & (ih < height)
            for kw in tl.static_range(0, 3):
                iw = ow * 2 + kw - 1
                valid = mask & valid_h & (iw >= 0) & (iw < width)
                view0_offset = ((n * 2 * channels + c) * height + ih) * width + iw
                v0 = tl.load(view_ptr + view0_offset, mask=valid, other=0.0)
                v1 = tl.load(
                    view_ptr + view0_offset + branch_stride,
                    mask=valid,
                    other=0.0,
                )
                weighted = (v0 * weight0 + v1 * weight1).to(tl.float16)
                acc += weighted.to(tl.float32)

        tl.store(out_ptr + offsets, acc * (1.0 / pool_divisor), mask=mask)


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


def _out_dim(size: int) -> int:
    return (size + 2 * POOL_PADDING - POOL_KERNEL) // POOL_STRIDE + 1


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    convolution_24, view_15, shape0, shape1, shape2, shape3 = inputs
    view_15 = _require_f16_tensor("view_15", view_15)
    if view_15.ndim != 5:
        raise ValueError(f"view_15 must be rank 5, got rank {view_15.ndim}")
    batch, radix, channels, height, width = view_15.shape
    if radix != RADIX:
        raise ValueError(f"view_15 radix dim must be {RADIX}, got {radix}")
    expected_view_stride = (
        RADIX * channels * height * width,
        channels * height * width,
        height * width,
        width,
        1,
    )
    if tuple(view_15.stride()) != expected_view_stride:
        raise ValueError(
            f"view_15 has stride {tuple(view_15.stride())}, expected {expected_view_stride}"
        )

    convolution_24 = _require_f16_tensor(
        "convolution_24",
        convolution_24,
        (batch, RADIX * channels, 1, 1),
        (RADIX * channels, 1, 1, 1),
    )
    if convolution_24.device != view_15.device:
        raise ValueError("convolution_24 and view_15 must be on the same device")

    expected_shapes = (
        [batch, 1, RADIX, -1],
        [batch, -1],
        [batch, -1, 1, 1],
        [batch, RADIX, channels, 1, 1],
    )
    got_shapes = (list(shape0), list(shape1), list(shape2), list(shape3))
    if got_shapes != expected_shapes:
        raise ValueError(f"unexpected shape params: {got_shapes}")

    out_height = _out_dim(height)
    out_width = _out_dim(width)
    return convolution_24, view_15, batch, channels, out_height, out_width


def _torch_oracle(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    convolution_24, view_15, shape0, shape1, shape2, shape3 = inputs
    logits = convolution_24.view(shape0).permute(0, 2, 1, 3).float()
    logits = logits - logits.amax([1], keepdim=True)
    weights = (logits.exp() / logits.exp().sum([1], keepdim=True)).half()
    weights = weights.view(shape1).view(shape2).view(shape3)
    summed = (view_15 * weights).sum([1])
    return torch.ops.aten.avg_pool2d.default(
        summed,
        [POOL_KERNEL, POOL_KERNEL],
        [POOL_STRIDE, POOL_STRIDE],
        [POOL_PADDING, POOL_PADDING],
    )


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full split-attention softmax, branch sum, and avg-pool scope."""
    convolution_24, view_15, batch, channels, out_height, out_width = _validate_inputs(inputs)
    if not view_15.is_cuda:
        return _torch_oracle(inputs)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    out = torch.empty_strided(
        (batch, channels, out_height, out_width),
        (channels * out_height * out_width, out_height * out_width, out_width, 1),
        device=view_15.device,
        dtype=torch.float16,
    )
    total = out.numel()
    _split_attention_avgpool_kernel[(triton.cdiv(total, BLOCK_OUT),)](
        convolution_24,
        view_15,
        out,
        total=total,
        channels=channels,
        height=view_15.shape[3],
        width=view_15.shape[4],
        out_height=out_height,
        out_width=out_width,
        pool_divisor=POOL_DIVISOR,
        block=BLOCK_OUT,
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
