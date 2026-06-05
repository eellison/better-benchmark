"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet inference pair of fp32 BN affine stages, intervening fp16 casts/ReLUs, and padded 3x3 stride-2 low-memory maxpool-with-offsets in one shape-specialized Triton stencil that emits the first int8 pool offsets and final fp16 ReLU directly, whereas Inductor currently schedules the BN/ReLU producer, multi-output maxpool stencil, and second BN/ReLU epilogue as generic fused regions with materialized intermediate activations; Inductor cannot do this today because scheduler fusion does not sink pointwise affine producers and consumers through prims low-memory maxpool-with-offsets while preserving fp16 cast, tie, and NaN offset semantics; the fix is SCHEDULER_FUSION: add guarded producer/consumer fusion for BN/ReLU around low-memory maxpool-with-offsets and lower the offsets plus downstream pointwise epilogue from the same loop nest."""
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


N_BATCH = 64
CHANNELS = 64
HEIGHT = 112
WIDTH = 112
OUT_HEIGHT = 56
OUT_WIDTH = 56
HW = HEIGHT * WIDTH
OUT_HW = OUT_HEIGHT * OUT_WIDTH
EPS = 1.0e-5

INPUT_SHAPE = (N_BATCH, CHANNELS, HEIGHT, WIDTH)
PARAM_SHAPE = (CHANNELS,)
OUTPUT_SHAPE = (N_BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
PARAM_STRIDE = (1,)
OUTPUT_STRIDE = (CHANNELS * OUT_HW, OUT_HW, OUT_WIDTH, 1)

BLOCK_C = 4
BLOCK_OUT = 128

if triton is not None:

    @triton.jit
    def _densenet_bn_pool_bn_kernel(
        mean1_ptr,
        conv_ptr,
        inv1_ptr,
        weight1_ptr,
        bias1_ptr,
        mean2_ptr,
        inv2_ptr,
        weight2_ptr,
        bias2_ptr,
        offsets_ptr,
        final_ptr,
        BLOCK_C_: tl.constexpr,
        BLOCK_OUT_: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_block = tl.program_id(1)
        out_block = tl.program_id(2)

        c_offsets = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        out_offsets = out_block * BLOCK_OUT_ + tl.arange(0, BLOCK_OUT_)
        out_h = out_offsets // 56
        out_w = out_offsets - out_h * 56

        c_mask = c_offsets < 64
        out_mask = out_offsets < 3136
        mask = c_mask[:, None] & out_mask[None, :]

        mean1 = tl.load(mean1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        inv1 = tl.load(inv1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        best = tl.full((BLOCK_C_, BLOCK_OUT_), -float("inf"), tl.float32)
        best_offset = tl.zeros((BLOCK_C_, BLOCK_OUT_), tl.int32)
        conv_base = (n * 64 + c_offsets[:, None]) * 12544

        for kh in tl.static_range(0, 3):
            in_h = out_h * 2 + kh - 1
            valid_h = (in_h >= 0) & (in_h < 112)
            for kw in tl.static_range(0, 3):
                in_w = out_w * 2 + kw - 1
                valid = mask & valid_h[None, :] & (in_w[None, :] >= 0) & (in_w[None, :] < 112)
                conv = tl.load(
                    conv_ptr + conv_base + in_h[None, :] * 112 + in_w[None, :],
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)

                bn1 = (conv - mean1[:, None]) * inv1[:, None]
                bn1 = bn1 * weight1[:, None]
                bn1 = bn1 + bias1[:, None]
                bn1_fp16 = bn1.to(tl.float16)
                relu1 = tl.where(
                    bn1_fp16 <= 0.0,
                    0.0,
                    bn1_fp16.to(tl.float32),
                )

                take = valid & ((relu1 > best) | (relu1 != relu1))
                best = tl.where(take, relu1, best)
                best_offset = tl.where(take, kh * 3 + kw, best_offset)

        mean2 = tl.load(mean2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        inv2 = tl.load(inv2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        weight2 = tl.load(weight2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias2 = tl.load(bias2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        bn2 = (best - mean2[:, None]) * inv2[:, None]
        bn2 = bn2 * weight2[:, None]
        bn2 = bn2 + bias2[:, None]
        bn2_fp16 = bn2.to(tl.float16)
        relu2 = tl.where(
            bn2_fp16 <= 0.0,
            0.0,
            bn2_fp16.to(tl.float32),
        )

        flat_out = (n * 64 + c_offsets[:, None]) * 3136 + out_offsets[None, :]
        tl.store(offsets_ptr + flat_out, best_offset.to(tl.int8), mask=mask)
        tl.store(final_ptr + flat_out, relu2, mask=mask)


def _require_f16_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} must have dtype torch.float16, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_densenet_infer_bn_maxpool.py")
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects nine inputs, got {len(inputs)}")

    names = (
        "arg2_1",
        "convolution",
        "arg3_1",
        "arg4_1",
        "arg5_1",
        "arg6_1",
        "arg7_1",
        "arg8_1",
        "arg9_1",
    )
    expected_shapes = (
        PARAM_SHAPE,
        INPUT_SHAPE,
        PARAM_SHAPE,
        PARAM_SHAPE,
        PARAM_SHAPE,
        PARAM_SHAPE,
        PARAM_SHAPE,
        PARAM_SHAPE,
        PARAM_SHAPE,
    )
    expected_strides = (
        PARAM_STRIDE,
        INPUT_STRIDE,
        PARAM_STRIDE,
        PARAM_STRIDE,
        PARAM_STRIDE,
        PARAM_STRIDE,
        PARAM_STRIDE,
        PARAM_STRIDE,
        PARAM_STRIDE,
    )
    tensors = tuple(
        _require_f16_tensor(name, value, shape, stride)
        for name, value, shape, stride in zip(names, inputs, expected_shapes, expected_strides)
    )
    device = tensors[1].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return tensors


def oracle_forward(inputs):
    """Run the full BN/ReLU/maxpool/BN/ReLU scope from Repro.forward."""
    (
        mean1,
        conv,
        var1,
        weight1,
        bias1,
        mean2,
        var2,
        weight2,
        bias2,
    ) = _validate_inputs(inputs)

    first_maxpool_offsets = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=conv.device,
        dtype=torch.int8,
    )
    final_relu = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=conv.device,
        dtype=torch.float16,
    )
    inv1 = torch.reciprocal(torch.sqrt(var1.float() + EPS))
    inv2 = torch.reciprocal(torch.sqrt(var2.float() + EPS))

    grid = (
        N_BATCH,
        triton.cdiv(CHANNELS, BLOCK_C),
        triton.cdiv(OUT_HW, BLOCK_OUT),
    )
    _densenet_bn_pool_bn_kernel[grid](
        mean1,
        conv,
        inv1,
        weight1,
        bias1,
        mean2,
        inv2,
        weight2,
        bias2,
        first_maxpool_offsets,
        final_relu,
        BLOCK_C_=BLOCK_C,
        BLOCK_OUT_=BLOCK_OUT,
        num_warps=8,
        num_stages=3,
    )
    return first_maxpool_offsets, final_relu


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
