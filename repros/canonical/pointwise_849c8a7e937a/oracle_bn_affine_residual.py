"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full batchnorm-inference affine plus residual add in one dense layout-aware pointwise Triton kernel with equal-NaN semantics, whereas Inductor already fuses the pointwise graph but uses a slower generic broadcast/indexing schedule for this rank-4 layout; Inductor cannot do this today because its pointwise codegen does not recognize this per-channel affine-residual broadcast as a layout-specialized streaming template with channel parameters reused across spatial elements; the fix is NEW_PATTERN: add a guarded contiguous per-channel affine-residual pointwise template or schedule choice that avoids generic broadcast index overhead."""
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
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


EPS = 1.0e-5
BLOCK_ELEMS = 256


if triton is not None:

    @triton.jit
    def _bn_affine_residual_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        residual_ptr,
        out_ptr,
        total: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        CHANNELS_CONTIG: tl.constexpr,
        BLOCK_ELEMS_: tl.constexpr,
        eps: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ELEMS_ + tl.arange(0, BLOCK_ELEMS_)
        mask = offsets < total

        if CHANNELS_CONTIG:
            c = offsets % C
        else:
            c = (offsets // (H * W)) % C

        mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        out = ((x - mean) * (1.0 / tl.sqrt(var + eps))) * weight + bias + residual
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_cuda_f32_tensor(name, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have zero storage_offset")
    return value


def _validate_inputs(inputs):
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mean, x, var, weight, bias, residual = inputs
    mean = _require_cuda_f32_tensor("arg237_1", mean)
    x = _require_cuda_f32_tensor("convolution_47", x)
    var = _require_cuda_f32_tensor("arg238_1", var)
    weight = _require_cuda_f32_tensor("arg239_1", weight)
    bias = _require_cuda_f32_tensor("arg240_1", bias)
    residual = _require_cuda_f32_tensor("add_98", residual)

    if x.ndim != 4:
        raise ValueError(f"convolution_47 must be rank 4, got shape {tuple(x.shape)}")
    if residual.shape != x.shape:
        raise ValueError(
            f"add_98 shape {tuple(residual.shape)} must match convolution_47 {tuple(x.shape)}"
        )
    if residual.stride() != x.stride():
        raise ValueError(
            f"add_98 stride {tuple(residual.stride())} must match convolution_47 {tuple(x.stride())}"
        )

    _, channels, _, _ = x.shape
    expected_vec_shape = (channels,)
    for name, tensor in (
        ("arg237_1", mean),
        ("arg238_1", var),
        ("arg239_1", weight),
        ("arg240_1", bias),
    ):
        if tuple(tensor.shape) != expected_vec_shape:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}, expected {expected_vec_shape}"
            )
        if tensor.stride() != (1,):
            raise ValueError(f"{name} must be contiguous, got stride {tuple(tensor.stride())}")
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on the same device as convolution_47")

    nchw_stride = (channels * x.shape[2] * x.shape[3], x.shape[2] * x.shape[3], x.shape[3], 1)
    nhwc_stride = (channels * x.shape[2] * x.shape[3], 1, x.shape[3] * channels, channels)
    if tuple(x.stride()) == nchw_stride:
        channels_contig = False
    elif tuple(x.stride()) == nhwc_stride:
        channels_contig = True
    else:
        raise ValueError(f"unsupported dense input stride {tuple(x.stride())}")

    return mean, x, var, weight, bias, residual, channels_contig


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_affine_residual.py")

    mean, x, var, weight, bias, residual, channels_contig = _validate_inputs(inputs)
    _, channels, height, width = x.shape
    total = x.numel()

    output = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.float32)

    _bn_affine_residual_kernel[(triton.cdiv(total, BLOCK_ELEMS),)](
        mean,
        x,
        var,
        weight,
        bias,
        residual,
        output,
        total=total,
        C=channels,
        H=height,
        W=width,
        CHANNELS_CONTIG=channels_contig,
        BLOCK_ELEMS_=BLOCK_ELEMS,
        eps=EPS,
        num_warps=4,
        num_stages=3,
    )
    return output


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
