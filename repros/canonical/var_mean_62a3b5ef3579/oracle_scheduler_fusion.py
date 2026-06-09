"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle folds the seed lookup and 128-value inductor_random producer into the layer-norm reduction kernel and computes the row's random value at the consumer, whereas Inductor emits a separate pointwise random kernel plus the reduction; Inductor cannot do this today because scheduler fusion materializes the seed-based RNG producer across the reduction boundary; the fix is SCHEDULER_FUSION: allow inlining prims.inductor_random with affine consumer offsets into persistent reductions when the seed argument is explicit and row-deterministic."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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

    @triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"XBLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"XBLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 4}, num_warps=8, num_stages=3),
            triton.Config({"XBLOCK": 4}, num_warps=16, num_stages=3),
            triton.Config({"XBLOCK": 8}, num_warps=8, num_stages=3),
            triton.Config({"XBLOCK": 8}, num_warps=16, num_stages=3),
        ],
        key=["xnumel"],
    )
    @triton.jit
    def oracle_layernorm_kernel(
        out_ptr,
        residual_ptr,
        index_ptr,
        addmm_ptr,
        seed_ptr,
        weight_ptr,
        bias_ptr,
        side_ptr,
        xnumel: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        R0_BLOCK: tl.constexpr = 512
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]

        r0_3 = r0_index
        x4 = xindex
        x0 = xindex % 14
        x1 = (xindex // 14) % 14
        x2 = xindex // 196

        residual = tl.load(
            residual_ptr + (r0_3 + 512 * x4),
            eviction_policy="evict_first",
        )
        index_0 = tl.load(index_ptr + x0, eviction_policy="evict_last")
        index_1 = tl.load(index_ptr + x1, eviction_policy="evict_last")
        seed = tl.load(seed_ptr + 40, eviction_policy="evict_last")
        weight = tl.load(weight_ptr + r0_3, eviction_policy="evict_last")
        bias = tl.load(bias_ptr + r0_3, eviction_policy="evict_last")

        size_14 = tl.full([1, 1], 14, tl.int32)
        index_0_plus = index_0 + size_14
        index_0_neg = index_0 < 0
        index_0_norm = tl.where(index_0_neg, index_0_plus, index_0)
        tl.device_assert(
            (0 <= index_0_norm) & (index_0_norm < 14),
            "index out of bounds: 0 <= index_0_norm < 14",
        )

        index_1_plus = index_1 + size_14
        index_1_neg = index_1 < 0
        index_1_norm = tl.where(index_1_neg, index_1_plus, index_1)
        tl.device_assert(
            (0 <= index_1_norm) & (index_1_norm < 14),
            "index out of bounds: 0 <= index_1_norm < 14",
        )

        addmm = tl.load(
            addmm_ptr
            + (
                r0_3
                + 512 * (index_0_norm % 7)
                + 3584 * (index_1_norm % 7)
                + 25088 * ((index_0_norm // 7) % 2)
                + 50176 * ((index_1_norm // 7) % 2)
                + 100352 * x2
            ),
            eviction_policy="evict_first",
        )

        keep_threshold = tl.full([1, 1], 0.9086956530809402, tl.float32)
        rand_value = tl.rand(seed, x2.to(tl.uint32))
        keep = rand_value < keep_threshold
        keep_f32 = keep.to(tl.float32)
        keep_scale = tl.full([1, 1], 1.1004784678010637, tl.float32)
        scaled_keep = keep_f32 * keep_scale
        dropped = addmm * scaled_keep
        values = residual + dropped

        values_for_mean = tl.broadcast_to(values, [XBLOCK, R0_BLOCK])
        mean_sum = tl.sum(values_for_mean, 1)[:, None].to(tl.float32)
        denom_i32 = tl.full([1, 1], 512, tl.int32)
        denom_f32 = denom_i32.to(tl.float32)
        mean = mean_sum / denom_f32

        centered_for_var = values_for_mean - mean
        squared = centered_for_var * centered_for_var
        squared_broadcast = tl.broadcast_to(squared, [XBLOCK, R0_BLOCK])
        var_sum = tl.sum(squared_broadcast, 1)[:, None].to(tl.float32)
        centered_for_output = values - mean
        denom_512 = tl.full([1, 1], 512.0, tl.float32)
        variance = var_sum / denom_512
        eps = tl.full([1, 1], 1e-05, tl.float32)
        variance_eps = variance + eps
        inv_std = libdevice.rsqrt(variance_eps)
        normalized = centered_for_output * inv_std
        weighted = normalized * weight
        output = weighted + bias
        side_scale = tl.full([1, 1], 0.001953125, tl.float32)
        side = inv_std * side_scale

        tl.store(out_ptr + (r0_3 + 512 * x4), output)
        tl.store(side_ptr + x4, side)


@oracle_impl(hardware="H100", shapes="(T([25088, 512], f32), T([14], i64, gen=Index(14)), T([46], i64), T([128, 14, 14, 512], f32), T([512], f32), T([512], f32), S([512, 49, 512]), S([-1, 7, 7, 512]), S([-1, 2, 2, 7, 7, 512]), S([-1, 14, 14, 512]), S([128, -1, 512]), S([25088, 512]))")
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
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    addmm_85, fmod_5, inductor_seeds, view_571, arg324_1, arg325_1, *_ = inputs

    output = torch.empty_strided(
        (25088, 512),
        (512, 1),
        device=addmm_85.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        (128, 196, 1),
        (196, 1, 1),
        device=addmm_85.device,
        dtype=torch.float32,
    )

    xnumel = 25088
    grid = lambda meta: (triton.cdiv(xnumel, meta["XBLOCK"]),)
    oracle_layernorm_kernel[grid](
        output,
        view_571,
        fmod_5,
        addmm_85,
        inductor_seeds,
        arg324_1,
        arg325_1,
        side,
        xnumel,
    )
    return output, side


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
