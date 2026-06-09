"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete residual MLP reshape/add, class-token drop, token mean, and hidden-size-768 LayerNorm with the same practical two-kernel shape-specialized schedule as tuned Inductor, storing token sums in-place and normalizing the `[128, 768]` buffer, whereas Inductor already emits an equivalent pooled-reduction producer followed by an in-place normalization-template consumer for this fixed scope; Inductor cannot materially improve this today with local scheduler fusion, algebraic elimination, split-K, or recompute fusion because the measured floor is dominated by required reads of the residual and addmm inputs plus the small normalization pass and launch envelope; the fix is BANDWIDTH_BOUND: record this norm-template-canonicalization case as at-floor unless broader memory-bandwidth, launch-overhead, or normalization-template infrastructure moves both implementations together."""
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


if triton is not None:

    @triton.jit
    def _pool_kernel(
        residual_ptr,
        gamma2_ptr,
        addmm_ptr,
        pooled_ptr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        xidx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        ridx = tl.arange(0, RBLOCK)[None, :]
        channel = xidx % 768
        row = xidx // 768

        gamma2 = tl.load(gamma2_ptr + channel, eviction_policy="evict_last").to(tl.float32)
        accum = tl.zeros((XBLOCK, RBLOCK), dtype=tl.float32)
        for roffset in tl.range(0, 196, RBLOCK):
            token = roffset + ridx
            rmask = token < 196
            offsets = 768 + channel + 768 * token + 151296 * row
            addmm = tl.load(
                addmm_ptr + offsets,
                mask=rmask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            residual = tl.load(
                residual_ptr + offsets,
                mask=rmask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            value = residual + gamma2 * addmm
            accum += tl.where(rmask, value, 0.0)

        pooled_sum = tl.sum(accum, axis=1)[:, None]
        tl.store(pooled_ptr + xidx, pooled_sum)

    @triton.jit
    def _layernorm_inplace_kernel(
        in_out_ptr,
        weight_ptr,
        bias_ptr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < 768
        pooled_sum = tl.load(in_out_ptr + row * 768 + offs, mask=mask, other=0.0).to(tl.float32)
        pooled = pooled_sum * (1.0 / 196.0)
        mean = tl.sum(tl.where(mask, pooled, 0.0), axis=0) * (1.0 / 768.0)
        centered = tl.where(mask, pooled - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) * (1.0 / 768.0)
        inv_std = tl.rsqrt(variance + 1.0e-6)

        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        out = centered * inv_std * weight + bias
        tl.store(in_out_ptr + row * 768 + offs, out, mask=mask)


def _oracle_forward_torch(inputs):
    addmm_47, arg213_1, add_79, arg220_1, arg221_1, shape_param = inputs
    reshape_default = torch.ops.aten.reshape.default(addmm_47, shape_param)
    mul_tensor = torch.ops.aten.mul.Tensor(arg213_1, reshape_default)
    add_tensor = torch.ops.aten.add.Tensor(add_79, mul_tensor)
    slice_tensor = torch.ops.aten.slice.Tensor(add_tensor, 1, 1, 9223372036854775807)
    mean_dim = torch.ops.aten.mean.dim(slice_tensor, [1])
    var_mean = torch.ops.aten.var_mean.correction(
        mean_dim, [1], correction=0, keepdim=True
    )
    variance = var_mean[0]
    mean = var_mean[1]
    sub_tensor = torch.ops.aten.sub.Tensor(mean_dim, mean)
    add_tensor_1 = torch.ops.aten.add.Tensor(variance, 1.0e-6)
    rsqrt = torch.ops.aten.rsqrt.default(add_tensor_1)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, arg220_1)
    return torch.ops.aten.add.Tensor(mul_tensor_2, arg221_1)


@oracle_impl(hardware="H100", shapes="(T([25216, 768], f32), T([768], f32), T([128, 197, 768], f32), T([768], f32), T([768], f32), S([128, 197, 768]))")
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
    addmm_47, arg213_1, add_79, arg220_1, arg221_1, shape_param = inputs
    if (
        triton is None
        or not addmm_47.is_cuda
        or tuple(addmm_47.shape) != (25216, 768)
        or tuple(arg213_1.shape) != (768,)
        or tuple(add_79.shape) != (128, 197, 768)
        or tuple(arg220_1.shape) != (768,)
        or tuple(arg221_1.shape) != (768,)
        or tuple(shape_param) != (128, 197, 768)
        or addmm_47.dtype != torch.float32
        or arg213_1.dtype != torch.float32
        or add_79.dtype != torch.float32
        or arg220_1.dtype != torch.float32
        or arg221_1.dtype != torch.float32
    ):
        return _oracle_forward_torch(inputs)

    pooled = torch.empty((128, 768), device=addmm_47.device, dtype=torch.float32)
    _pool_kernel[(triton.cdiv(98304, 64),)](
        add_79,
        arg213_1,
        addmm_47,
        pooled,
        XBLOCK=64,
        RBLOCK=8,
        num_warps=4,
    )
    _layernorm_inplace_kernel[(128,)](
        pooled,
        arg220_1,
        arg221_1,
        BLOCK_C=1024,
        num_warps=8,
    )
    return pooled


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
