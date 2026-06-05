"""
Full-scope Triton oracle for amax_sum_sum_b942094d64da.

Gap diagnosis (classification: SCHEDULER_FUSION): the oracle computes the
two-branch softmax weights, branch-weighted spatial tensor, branch sum, and
3x3 stride-2 avg-pool epilogue in one Triton kernel for the exact repro
inputs and output. Inductor lowers the decomposed view/permute/amax/exp/sum/div
softmax, weighted branch sum, and avg_pool2d as generic scheduled work, so it
does not currently fuse the branch reduction with the following spatial pooling
reduction into this one-pass layout-specific kernel.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "amax_sum_sum_b942094d64da"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
QUEUE_PATH = REPO_ROOT / "investigation_results" / "oracle_gap_closure_queue.csv"
HISTORICAL_BEST_COMPILE_US = 17.535999417304993

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
def _softmax_weighted_avgpool_kernel(
    conv_ptr,
    view_ptr,
    out_ptr,
    total: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
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
    x0 = tl.load(conv_ptr + conv_base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(conv_ptr + conv_base + channels, mask=mask, other=0.0).to(tl.float32)
    m = tl.maximum(x0, x1)
    e0 = tl.exp(x0 - m)
    e1 = tl.exp(x1 - m)
    denom = e0 + e1
    w0 = e0 / denom
    w1 = e1 / denom

    acc = tl.full((block,), 0.0, tl.float32)
    branch_stride = channels * height * width

    for kh in tl.static_range(0, 3):
        ih = oh * 2 + kh - 1
        valid_h = (ih >= 0) & (ih < height)
        for kw in tl.static_range(0, 3):
            iw = ow * 2 + kw - 1
            valid = mask & valid_h & (iw >= 0) & (iw < width)
            view0_offset = ((n * 2 * channels + c) * height + ih) * width + iw
            v0 = tl.load(view_ptr + view0_offset, mask=valid, other=0.0).to(tl.float32)
            v1 = tl.load(view_ptr + view0_offset + branch_stride, mask=valid, other=0.0).to(tl.float32)
            acc += v0 * w0 + v1 * w1

    tl.store(out_ptr + offsets, acc * (1.0 / 9.0), mask=mask)


def _launch_oracle(
    convolution_24: torch.Tensor,
    view_15: torch.Tensor,
    out: torch.Tensor,
    *,
    block: int,
) -> torch.Tensor:
    if not convolution_24.is_cuda or not view_15.is_cuda:
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if convolution_24.dtype != torch.float32 or view_15.dtype != torch.float32:
        raise TypeError("expected f32 convolution_24 and view_15 inputs")
    if convolution_24.ndim != 4 or view_15.ndim != 5:
        raise ValueError("unexpected input rank for full-scope oracle")

    batch, branches, channels, height, width = view_15.shape
    if branches != 2:
        raise ValueError(f"expected two branches, got {branches}")
    if convolution_24.shape != (batch, 2 * channels, 1, 1):
        raise ValueError(
            "convolution_24 shape must be [batch, 2 * channels, 1, 1], "
            f"got {tuple(convolution_24.shape)} for view_15 {tuple(view_15.shape)}"
        )
    out_height = (height + 2 - 3) // 2 + 1
    out_width = (width + 2 - 3) // 2 + 1
    if out.shape != (batch, channels, out_height, out_width):
        raise ValueError(f"unexpected output shape {tuple(out.shape)}")
    if not convolution_24.is_contiguous() or not view_15.is_contiguous() or not out.is_contiguous():
        raise ValueError("oracle expects the canonical contiguous repro inputs and output")

    total = out.numel()
    grid = (triton.cdiv(total, block),)
    _softmax_weighted_avgpool_kernel[grid](
        convolution_24,
        view_15,
        out,
        total=total,
        channels=channels,
        height=height,
        width=width,
        out_height=out_height,
        out_width=out_width,
        block=block,
        num_warps=4,
    )
    return out


def oracle_forward(inputs: tuple, *, block: int = 128) -> torch.Tensor:
    """Compute exactly Repro()(*inputs): same inputs, one f32 NCHW output."""
    convolution_24, view_15 = inputs[:2]
    batch, _, channels, height, width = view_15.shape
    out_height = (height + 2 - 3) // 2 + 1
    out_width = (width + 2 - 3) // 2 + 1
    out = torch.empty(
        (batch, channels, out_height, out_width),
        device=view_15.device,
        dtype=view_15.dtype,
    )
    return _launch_oracle(convolution_24, view_15, out, block=block)


def _load_repro_module():
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs() -> tuple:
    module = _load_repro_module()
    return tuple(module.make_inputs())


def get_repro_instance() -> torch.nn.Module:
    module = _load_repro_module()
    return module.Repro()


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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
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
