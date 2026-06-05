"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Doctr FPN pointwise/layout scope as three fused Triton stencil stages, folding each per-channel BN-affine+ReLU producer into the following align-corners bilinear upsample and residual add while writing only the required 64x64, 128x128, and final 256x256 pyramid tensors, whereas Inductor lowers the decomposed unsqueeze/sub/sqrt/reciprocal/mul/add/ReLU plus twelve _unsafe_index bilinear-gather tensors as separate generic pointwise and materialization regions; Inductor cannot do this today because its scheduler treats the indexed interpolation materializations as fusion barriers and does not sink broadcast channel-affine producers through the multi-level stencil adds; the fix is SCHEDULER_FUSION: allow pointwise channel producers to fuse into affine bilinear-index materialization kernels and chain the pyramid add outputs directly between stencil stages."""
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


# --- Oracle kernel(s) ---

C = 256
H0 = 32
H1 = 64
H2 = 128
H3 = 256
EPS = 1.0e-5

if triton is not None:

    @triton.jit
    def _affine_params_kernel(
        mean32_ptr,
        var32_ptr,
        weight32_ptr,
        bias32_ptr,
        mean64_ptr,
        var64_ptr,
        weight64_ptr,
        bias64_ptr,
        mean128_ptr,
        var128_ptr,
        weight128_ptr,
        bias128_ptr,
        mean256_ptr,
        var256_ptr,
        weight256_ptr,
        bias256_ptr,
        params_ptr,
        block_c: tl.constexpr,
    ):
        c = tl.program_id(0) * block_c + tl.arange(0, block_c)
        mask = c < 256

        mean32 = tl.load(mean32_ptr + c, mask=mask, other=0.0)
        scale32 = tl.load(weight32_ptr + c, mask=mask, other=0.0) / tl.sqrt(
            tl.load(var32_ptr + c, mask=mask, other=0.0) + 1.0e-5
        )
        shift32 = tl.load(bias32_ptr + c, mask=mask, other=0.0) - mean32 * scale32
        tl.store(params_ptr + c, scale32, mask=mask)
        tl.store(params_ptr + 256 + c, shift32, mask=mask)

        mean64 = tl.load(mean64_ptr + c, mask=mask, other=0.0)
        scale64 = tl.load(weight64_ptr + c, mask=mask, other=0.0) / tl.sqrt(
            tl.load(var64_ptr + c, mask=mask, other=0.0) + 1.0e-5
        )
        shift64 = tl.load(bias64_ptr + c, mask=mask, other=0.0) - mean64 * scale64
        tl.store(params_ptr + 512 + c, scale64, mask=mask)
        tl.store(params_ptr + 768 + c, shift64, mask=mask)

        mean128 = tl.load(mean128_ptr + c, mask=mask, other=0.0)
        scale128 = tl.load(weight128_ptr + c, mask=mask, other=0.0) / tl.sqrt(
            tl.load(var128_ptr + c, mask=mask, other=0.0) + 1.0e-5
        )
        shift128 = tl.load(bias128_ptr + c, mask=mask, other=0.0) - mean128 * scale128
        tl.store(params_ptr + 1024 + c, scale128, mask=mask)
        tl.store(params_ptr + 1280 + c, shift128, mask=mask)

        mean256 = tl.load(mean256_ptr + c, mask=mask, other=0.0)
        scale256 = tl.load(weight256_ptr + c, mask=mask, other=0.0) / tl.sqrt(
            tl.load(var256_ptr + c, mask=mask, other=0.0) + 1.0e-5
        )
        shift256 = tl.load(bias256_ptr + c, mask=mask, other=0.0) - mean256 * scale256
        tl.store(params_ptr + 1536 + c, scale256, mask=mask)
        tl.store(params_ptr + 1792 + c, shift256, mask=mask)

    @triton.jit
    def _stage0_32_to_64_kernel(
        params_ptr,
        conv32_ptr,
        conv64_ptr,
        out64_ptr,
        block_h: tl.constexpr,
        block_w: tl.constexpr,
    ):
        channel = tl.program_id(0)
        h_block = tl.program_id(1)
        oh = h_block * block_h + tl.arange(0, block_h)[:, None]
        ow = tl.arange(0, block_w)[None, :]

        scale32 = tl.load(params_ptr + channel)
        shift32 = tl.load(params_ptr + 256 + channel)

        src_h = tl.cast(oh, tl.float32) * 0.49206349206349204
        h_lo = tl.cast(src_h, tl.int32)
        h_hi = tl.minimum(h_lo + 1, 31)
        h_frac = src_h - tl.cast(h_lo, tl.float32)
        h_frac = tl.minimum(tl.maximum(h_frac, 0.0), 1.0)

        src_w = tl.cast(ow, tl.float32) * 0.49206349206349204
        w_lo = tl.cast(src_w, tl.int32)
        w_hi = tl.minimum(w_lo + 1, 31)
        w_frac = src_w - tl.cast(w_lo, tl.float32)
        w_frac = tl.minimum(tl.maximum(w_frac, 0.0), 1.0)

        base32 = channel * 1024
        v_hi_hi = tl.load(conv32_ptr + base32 + h_hi * 32 + w_hi)
        v_hi_lo = tl.load(conv32_ptr + base32 + h_hi * 32 + w_lo)
        v_lo_hi = tl.load(conv32_ptr + base32 + h_lo * 32 + w_hi)
        v_lo_lo = tl.load(conv32_ptr + base32 + h_lo * 32 + w_lo)

        v_hi_hi = v_hi_hi * scale32 + shift32
        v_hi_lo = v_hi_lo * scale32 + shift32
        v_lo_hi = v_lo_hi * scale32 + shift32
        v_lo_lo = v_lo_lo * scale32 + shift32
        v_hi_hi = tl.where(v_hi_hi != v_hi_hi, v_hi_hi, tl.maximum(v_hi_hi, 0.0))
        v_hi_lo = tl.where(v_hi_lo != v_hi_lo, v_hi_lo, tl.maximum(v_hi_lo, 0.0))
        v_lo_hi = tl.where(v_lo_hi != v_lo_hi, v_lo_hi, tl.maximum(v_lo_hi, 0.0))
        v_lo_lo = tl.where(v_lo_lo != v_lo_lo, v_lo_lo, tl.maximum(v_lo_lo, 0.0))

        row_hi = v_hi_lo + (v_hi_hi - v_hi_lo) * w_frac
        row_lo = v_lo_lo + (v_lo_hi - v_lo_lo) * w_frac
        upsampled = row_lo + (row_hi - row_lo) * h_frac

        scale64 = tl.load(params_ptr + 512 + channel)
        shift64 = tl.load(params_ptr + 768 + channel)

        base64 = channel * 4096 + oh * 64 + ow
        level64 = tl.load(conv64_ptr + base64) * scale64 + shift64
        level64 = tl.where(level64 != level64, level64, tl.maximum(level64, 0.0))

        tl.store(out64_ptr + base64, upsampled + level64)

    @triton.jit
    def _stage1_64_to_128_kernel(
        params_ptr,
        in64_ptr,
        conv128_ptr,
        out128_ptr,
        block_h: tl.constexpr,
        block_w: tl.constexpr,
    ):
        channel = tl.program_id(0)
        h_block = tl.program_id(1)
        oh = h_block * block_h + tl.arange(0, block_h)[:, None]
        ow = tl.arange(0, block_w)[None, :]

        src_h = tl.cast(oh, tl.float32) * 0.49606299212598426
        h_lo = tl.cast(src_h, tl.int32)
        h_hi = tl.minimum(h_lo + 1, 63)
        h_frac = src_h - tl.cast(h_lo, tl.float32)
        h_frac = tl.minimum(tl.maximum(h_frac, 0.0), 1.0)

        src_w = tl.cast(ow, tl.float32) * 0.49606299212598426
        w_lo = tl.cast(src_w, tl.int32)
        w_hi = tl.minimum(w_lo + 1, 63)
        w_frac = src_w - tl.cast(w_lo, tl.float32)
        w_frac = tl.minimum(tl.maximum(w_frac, 0.0), 1.0)

        base64 = channel * 4096
        v_hi_hi = tl.load(in64_ptr + base64 + h_hi * 64 + w_hi)
        v_hi_lo = tl.load(in64_ptr + base64 + h_hi * 64 + w_lo)
        v_lo_hi = tl.load(in64_ptr + base64 + h_lo * 64 + w_hi)
        v_lo_lo = tl.load(in64_ptr + base64 + h_lo * 64 + w_lo)

        row_hi = v_hi_lo + (v_hi_hi - v_hi_lo) * w_frac
        row_lo = v_lo_lo + (v_lo_hi - v_lo_lo) * w_frac
        upsampled = row_lo + (row_hi - row_lo) * h_frac

        scale128 = tl.load(params_ptr + 1024 + channel)
        shift128 = tl.load(params_ptr + 1280 + channel)

        base128 = channel * 16384 + oh * 128 + ow
        level128 = tl.load(conv128_ptr + base128) * scale128 + shift128
        level128 = tl.where(level128 != level128, level128, tl.maximum(level128, 0.0))

        tl.store(out128_ptr + base128, upsampled + level128)

    @triton.jit
    def _stage2_128_to_256_kernel(
        params_ptr,
        in128_ptr,
        conv256_ptr,
        out256_ptr,
        block_h: tl.constexpr,
        block_w: tl.constexpr,
    ):
        channel = tl.program_id(0)
        h_block = tl.program_id(1)
        oh = h_block * block_h + tl.arange(0, block_h)[:, None]
        ow = tl.arange(0, block_w)[None, :]

        src_h = tl.cast(oh, tl.float32) * 0.4980392156862745
        h_lo = tl.cast(src_h, tl.int32)
        h_hi = tl.minimum(h_lo + 1, 127)
        h_frac = src_h - tl.cast(h_lo, tl.float32)
        h_frac = tl.minimum(tl.maximum(h_frac, 0.0), 1.0)

        src_w = tl.cast(ow, tl.float32) * 0.4980392156862745
        w_lo = tl.cast(src_w, tl.int32)
        w_hi = tl.minimum(w_lo + 1, 127)
        w_frac = src_w - tl.cast(w_lo, tl.float32)
        w_frac = tl.minimum(tl.maximum(w_frac, 0.0), 1.0)

        base128 = channel * 16384
        v_hi_hi = tl.load(in128_ptr + base128 + h_hi * 128 + w_hi)
        v_hi_lo = tl.load(in128_ptr + base128 + h_hi * 128 + w_lo)
        v_lo_hi = tl.load(in128_ptr + base128 + h_lo * 128 + w_hi)
        v_lo_lo = tl.load(in128_ptr + base128 + h_lo * 128 + w_lo)

        row_hi = v_hi_lo + (v_hi_hi - v_hi_lo) * w_frac
        row_lo = v_lo_lo + (v_lo_hi - v_lo_lo) * w_frac
        upsampled = row_lo + (row_hi - row_lo) * h_frac

        scale256 = tl.load(params_ptr + 1536 + channel)
        shift256 = tl.load(params_ptr + 1792 + channel)

        base256 = channel * 65536 + oh * 256 + ow
        level256 = tl.load(conv256_ptr + base256) * scale256 + shift256
        level256 = tl.where(level256 != level256, level256, tl.maximum(level256, 0.0))

        tl.store(out256_ptr + base256, upsampled + level256)


def _expect_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} dtype {value.dtype} != torch.float32")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")


def _expect_shape_param(name: str, value, expected: tuple[int, ...]) -> None:
    if tuple(value) != expected:
        raise ValueError(f"{name} value {value!r} != {expected}")


def oracle_forward(inputs):
    """Compute the full four-level FPN Repro.forward scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        mean32,
        conv32,
        var32,
        weight32,
        bias32,
        mean64,
        conv64,
        var64,
        weight64,
        bias64,
        mean128,
        conv128,
        var128,
        weight128,
        bias128,
        mean256,
        conv256,
        var256,
        weight256,
        bias256,
        shape64,
        shape128,
        shape256,
    ) = inputs

    for name, value in (
        ("mean32", mean32),
        ("var32", var32),
        ("weight32", weight32),
        ("bias32", bias32),
        ("mean64", mean64),
        ("var64", var64),
        ("weight64", weight64),
        ("bias64", bias64),
        ("mean128", mean128),
        ("var128", var128),
        ("weight128", weight128),
        ("bias128", bias128),
        ("mean256", mean256),
        ("var256", var256),
        ("weight256", weight256),
        ("bias256", bias256),
    ):
        _expect_tensor(name, value, (C,))
    _expect_tensor("conv32", conv32, (1, C, H0, H0))
    _expect_tensor("conv64", conv64, (1, C, H1, H1))
    _expect_tensor("conv128", conv128, (1, C, H2, H2))
    _expect_tensor("conv256", conv256, (1, C, H3, H3))
    _expect_shape_param("shape64", shape64, (H1, 1))
    _expect_shape_param("shape128", shape128, (H2, 1))
    _expect_shape_param("shape256", shape256, (H3, 1))

    stage64 = torch.empty((1, C, H1, H1), device=conv32.device, dtype=torch.float32)
    stage128 = torch.empty((1, C, H2, H2), device=conv32.device, dtype=torch.float32)
    out = torch.empty_like(conv256)
    params = torch.empty((8, C), device=conv32.device, dtype=torch.float32)

    _affine_params_kernel[(triton.cdiv(C, 256),)](
        mean32,
        var32,
        weight32,
        bias32,
        mean64,
        var64,
        weight64,
        bias64,
        mean128,
        var128,
        weight128,
        bias128,
        mean256,
        var256,
        weight256,
        bias256,
        params,
        block_c=256,
        num_warps=8,
    )

    _stage0_32_to_64_kernel[(C, triton.cdiv(H1, 4))](
        params,
        conv32,
        conv64,
        stage64,
        block_h=4,
        block_w=H1,
        num_warps=4,
    )
    _stage1_64_to_128_kernel[(C, triton.cdiv(H2, 4))](
        params,
        stage64,
        conv128,
        stage128,
        block_h=4,
        block_w=H2,
        num_warps=8,
    )
    _stage2_128_to_256_kernel[(C, triton.cdiv(H3, 4))](
        params,
        stage128,
        conv256,
        out,
        block_h=4,
        block_w=H3,
        num_warps=8,
    )
    return out


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
