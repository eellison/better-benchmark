"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes both masked channel reductions and the dependent batchnorm-backward epilogue in one Triton program per channel, producing the full f32[128,64,8,8] tensor and the sibling f32[64] output, whereas Inductor currently schedules this as generic multi-output reduction work with the channel summaries separated from the full-tensor epilogue; Inductor cannot do this today because its scheduler/codegen lacks a cross-dimension reduction fusion template that keeps sibling per-channel accumulators local while emitting a dependent full-size output; the fix is SCHEDULER_FUSION: add a fused channel-reduction-plus-epilogue lowering for batchnorm-backward-shaped reductions with sibling scalar outputs."""
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
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
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

if triton is not None:

    @triton.jit
    def _masked_bn_backward_kernel(
        mm_ptr,
        mask_ptr,
        arg96_ptr,
        arg100_ptr,
        arg97_ptr,
        arg40_ptr,
        out0_ptr,
        out1_ptr,
        BLOCK_M: tl.constexpr,
    ):
        c = tl.program_id(0)
        offsets_m = tl.arange(0, BLOCK_M)
        n = offsets_m // 64
        spatial = offsets_m - n * 64

        nchw_offsets = n * 4096 + c * 64 + spatial
        mm_offsets = n * 64 + c

        mm_value = tl.load(mm_ptr + mm_offsets).to(tl.float32) * 0.015625
        masked = tl.load(mask_ptr + nchw_offsets)
        where_value = tl.where(masked, 0.0, mm_value)

        mean = tl.load(arg100_ptr + c).to(tl.float32)
        centered = tl.load(arg96_ptr + nchw_offsets).to(tl.float32) - mean

        sum_where = tl.sum(where_value, axis=0)
        sum_weighted = tl.sum(where_value * centered, axis=0)

        inv_count = 0.0001220703125
        gamma = tl.load(arg97_ptr + c).to(tl.float32)
        upstream = tl.load(arg40_ptr + c).to(tl.float32)
        channel_mean = sum_where * inv_count
        channel_weight = sum_weighted * inv_count * gamma * gamma
        output_scale = gamma * upstream
        out = ((where_value - centered * channel_weight) - channel_mean) * output_scale

        tl.store(out0_ptr + nchw_offsets, out)
        tl.store(out1_ptr + c, sum_weighted * gamma)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is not available")
    if len(inputs) != 8:
        raise ValueError(f"expected 8 inputs, got {len(inputs)}")

    mm, mask, arg96, arg100, arg97, arg40, shape0, shape1 = inputs
    expected = {
        "mm": (mm, (128, 64), torch.float32),
        "mask": (mask, (128, 64, 8, 8), torch.bool),
        "arg96": (arg96, (128, 64, 8, 8), torch.float32),
        "arg100": (arg100, (1, 64, 1, 1), torch.float32),
        "arg97": (arg97, (64,), torch.float32),
        "arg40": (arg40, (64,), torch.float32),
    }
    for name, (tensor, shape, dtype) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"unexpected {name} shape: got {tuple(tensor.shape)} expected {shape}")
        if tensor.dtype != dtype:
            raise ValueError(f"unexpected {name} dtype: got {tensor.dtype} expected {dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for this captured shape")

    if list(shape0) != [128, 64, 1, 1] or list(shape1) != [128, 64, 8, 8]:
        raise ValueError(f"unexpected shape params: {shape0}, {shape1}")


@oracle_impl(hardware="H100", shapes="(T([128, 64], f32), T([128, 64, 8, 8], b8), T([128, 64, 8, 8], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), S([128, 64, 1, 1]), S([128, 64, 8, 8]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    _validate_inputs(inputs)
    mm, mask, arg96, arg100, arg97, arg40, _shape0, _shape1 = inputs
    out0 = torch.empty_like(arg96)
    out1 = torch.empty_like(arg97)
    _masked_bn_backward_kernel[(64,)](
        mm,
        mask,
        arg96,
        arg100,
        arg97,
        arg40,
        out0,
        out1,
        BLOCK_M=8192,
        num_warps=16,
    )
    return (out0, out1)


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
            # All timing must go through bench_oracle(). Direct do_bench or
            # compiled(*inputs) timing includes dispatch overhead and can invent
            # fake gaps for fast kernels.
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
