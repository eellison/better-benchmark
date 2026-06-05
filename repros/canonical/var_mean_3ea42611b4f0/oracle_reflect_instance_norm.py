"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete instance-normalization plus residual add plus reflected two-axis gather returned by `Repro.forward`, reducing `[1,256,64,64]` per channel and writing the final contiguous `[1,256,66,66]` result directly without materializing the normalized `[1,256,64,64]` intermediate or the first `[1,256,66,64]` gather, whereas Inductor currently schedules the norm-template reduction/epilogue and the surrounding unsafe-index layout work as separate generic fused regions; Inductor cannot do this today because its normalization scheduler does not treat small static reflection-index consumers as virtual layout epilogues of the norm result; the fix is SCHEDULER_FUSION: allow fixed affine/indexing layout consumers to fuse into the normalization epilogue or be generated as a direct final-layout store."""
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
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


BATCH = 1
CHANNELS = 256
HEIGHT = 64
WIDTH = 64
PADDED = 66
HW = HEIGHT * WIDTH
OUT_HW = PADDED * PADDED
TOTAL_OUT = BATCH * CHANNELS * OUT_HW
EPS = 1.0e-5
STAT_BLOCK = 4096
OUT_BLOCK = 256
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _channel_stats_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        eps: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_M)
        mask = offsets < hw
        vals = tl.load(x_ptr + channel * hw + offsets, mask=mask, other=0.0).to(tl.float32)
        vals_for_sum = tl.where(mask, vals, 0.0)
        mean = tl.sum(vals_for_sum, axis=0) / hw
        centered = vals - mean
        var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hw
        invstd = tl.rsqrt(tl.maximum(var, 0.0) + eps)
        tl.store(mean_ptr + channel, mean)
        tl.store(invstd_ptr + channel, invstd)

    @triton.jit
    def _reflect_norm_add_kernel(
        x_ptr,
        residual_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        total_out: tl.constexpr,
        hw: tl.constexpr,
        out_hw: tl.constexpr,
        padded: tl.constexpr,
        width: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < total_out

        channel = offsets // out_hw
        spatial = offsets - channel * out_hw
        oh = spatial // padded
        ow = spatial - oh * padded

        ih = tl.where(oh == 0, 1, tl.where(oh == padded - 1, width - 2, oh - 1))
        iw = tl.where(ow == 0, 1, tl.where(ow == padded - 1, width - 2, ow - 1))
        input_offsets = channel * hw + ih * width + iw

        x = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        out = residual + (x - mean) * invstd
        tl.store(out_ptr + offsets, out, mask=mask)


def _expect_f32_contiguous(
    name: str,
    value: Any,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")
    x = _expect_f32_contiguous("convolution_18", inputs[0], (BATCH, CHANNELS, HEIGHT, WIDTH))
    residual = _expect_f32_contiguous("add_23", inputs[1], (BATCH, CHANNELS, HEIGHT, WIDTH))
    if residual.device != x.device:
        raise ValueError("all tensor inputs must be on the same device")
    return x, residual


def oracle_forward(inputs):
    """Run the full Repro.forward computation with shape-specialized Triton kernels."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, residual = _validate_inputs(inputs)
    mean = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    invstd = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(
        (BATCH, CHANNELS, PADDED, PADDED),
        (CHANNELS * OUT_HW, OUT_HW, PADDED, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _channel_stats_kernel[(CHANNELS,)](
        x,
        mean,
        invstd,
        eps=EPS,
        hw=HW,
        BLOCK_M=STAT_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _reflect_norm_add_kernel[(triton.cdiv(TOTAL_OUT, OUT_BLOCK),)](
        x,
        residual,
        mean,
        invstd,
        out,
        total_out=TOTAL_OUT,
        hw=HW,
        out_hw=OUT_HW,
        padded=PADDED,
        width=WIDTH,
        BLOCK_M=OUT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return out


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
