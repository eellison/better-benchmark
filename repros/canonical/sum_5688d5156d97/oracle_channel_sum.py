"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete channel sum with a two-stage Triton reduction that hoists the two [N,C] broadcast operands outside each H/W tile, whereas Inductor lowers the same expression as a generic staged reduction with three kernels and vector-shaped reloads of broadcast operands; Inductor cannot do this today because its reduction scheduler does not form a tile schedule with broadcast-scalar hoisting for this NCHW layout; the fix is SCHEDULER_FUSION: add a specialized schedule for channelwise spatial/batch reductions with small broadcast operands while preserving the per-element f32 multiply, scale, add, and sum."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 256
H = 48
W = 48
HW = H * W
HW_BLOCK = 1024
HW_TILES = 3
FINAL_PARTIALS = N * HW_TILES
PARTIAL_NUMEL = N * C * HW_TILES
SPATIAL_SCALE = 1.0 / HW


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_sum_kernel(
        x_ptr,
        scale_ptr,
        offset_ptr,
        partial_ptr,
        X_BLOCK: tl.constexpr,
    ):
        xindex = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)[:, None]
        xmask = xindex < 98304
        lanes = tl.arange(0, 1024)[None, :]

        hw_tile = xindex % 3
        c = (xindex // 3) % 256
        n = xindex // 768
        hw = hw_tile * 1024 + lanes
        value_mask = xmask & (hw < 2304)

        scale_value = tl.load(
            scale_ptr + n * 256 + c,
            mask=xmask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        offset_value = (
            tl.load(
                offset_ptr + n * 256 + c,
                mask=xmask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            * 0.00043402777777777775
        )
        x = tl.load(
            x_ptr + n * 589824 + c * 2304 + hw,
            mask=value_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        product = x * scale_value
        value = product + offset_value
        value = tl.where(value_mask, value, 0.0)
        partial = tl.sum(value, axis=1)[:, None]
        tl.store(partial_ptr + xindex, partial, mask=xmask)

    @triton.jit
    def _final_sum_kernel(
        partial_ptr,
        out_ptr,
        BLOCK_PARTIAL: tl.constexpr,
    ):
        c = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_PARTIAL)
        mask = offsets < 384
        n = offsets // 3
        hw_tile = offsets - n * 3
        partial = tl.load(
            partial_ptr + n * 768 + c * 3 + hw_tile,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        total = tl.sum(partial, axis=0)
        tl.store(out_ptr + c, total)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 4:
        raise ValueError(f"expected 4 inputs, got {len(inputs)}")

    x, scale, offset, shape_param = inputs
    if list(shape_param) != [N, C, H, W]:
        raise ValueError(f"unexpected shape parameter: {shape_param}")
    for name, tensor in (("x", x), ("scale", scale), ("offset", offset)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be torch.float32, got {tensor.dtype}")

    if tuple(x.shape) != (N, C, H, W) or tuple(x.stride()) != (C * HW, HW, W, 1):
        raise ValueError(f"unexpected x layout: shape={tuple(x.shape)} stride={tuple(x.stride())}")
    expected_small = (N, C, 1, 1)
    expected_small_stride = (C, 1, 1, 1)
    if tuple(scale.shape) != expected_small or tuple(scale.stride()) != expected_small_stride:
        raise ValueError(
            f"unexpected scale layout: shape={tuple(scale.shape)} stride={tuple(scale.stride())}"
        )
    if tuple(offset.shape) != expected_small or tuple(offset.stride()) != expected_small_stride:
        raise ValueError(
            f"unexpected offset layout: shape={tuple(offset.shape)} stride={tuple(offset.stride())}"
        )
    return x, scale, offset


@oracle_impl(hardware="H100", shapes="(T([128, 256, 48, 48], f32), T([128, 256, 1, 1], f32), T([128, 256, 1, 1], f32), S([128, 256, 48, 48]))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation for the captured shape."""
    x, scale, offset = _validate_inputs(inputs)
    partial = torch.empty_strided((PARTIAL_NUMEL,), (1,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    _partial_sum_kernel[(triton.cdiv(PARTIAL_NUMEL, 2),)](
        x,
        scale,
        offset,
        partial,
        X_BLOCK=2,
        num_warps=4,
        num_stages=3,
    )
    _final_sum_kernel[(C,)](
        partial,
        out,
        BLOCK_PARTIAL=512,
        num_warps=1,
        num_stages=1,
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
