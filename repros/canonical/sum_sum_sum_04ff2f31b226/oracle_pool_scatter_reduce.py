"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete maxpool-offset scatter_add, affine/ReLU gate, groupnorm-backward tensor, and both sibling channel reductions directly from the pooled-gradient/offset domain, whereas Inductor currently materializes the [4096,256] scatter result and schedules the ReLU mask, dependent reductions, and groupnorm-backward epilogue as separate generic kernels; Inductor cannot do this today because scheduler/codegen does not model low-memory maxpool offsets as a structured scatter-reduce producer that can feed both the full tensor epilogue and the sibling reductions without materializing the scatter buffer; the fix is SCATTER_REDUCE: add a maxpool-backward scatter-reduce lowering that inverts the bounded pooling windows, keeps per-group reduction scalars live, and emits the complete three-output return tuple."""
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

BATCH = 64
CHANNELS = 64
GROUPS = 32
CHANNELS_PER_GROUP = 2
POOL_H = 8
POOL_W = 8
IN_H = 16
IN_W = 16
SPATIAL = IN_H * IN_W
GROUP_ELEMS = CHANNELS_PER_GROUP * SPATIAL
GROUP_REDUCTION_SCALE = 1.0 / GROUP_ELEMS


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _pool_scatter_groupnorm_kernel(
        where_ptr,
        getitem_ptr,
        offsets_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        gamma_ptr,
        beta_ptr,
        full_ptr,
        out_dx_ptr,
        partial_dgamma_ptr,
        partial_dbeta_ptr,
        BLOCK: tl.constexpr,
    ):
        n = tl.program_id(0)
        group = tl.program_id(1)
        elem = tl.arange(0, BLOCK)

        channel_in_group = elem // 256
        spatial = elem - channel_in_group * 256
        h = spatial // 16
        w = spatial - h * 16
        c = group * 2 + channel_in_group

        x_offset = n * 64 * 256 + c * 256 + spatial
        x = tl.load(x_ptr + x_offset).to(tl.float32)
        mean = tl.load(mean_ptr + n * 32 + group).to(tl.float32)
        invstd = tl.load(invstd_ptr + n * 32 + group).to(tl.float32)
        gamma = tl.load(gamma_ptr + c).to(tl.float32)
        beta = tl.load(beta_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        # Invert the 3x3 stride-2 pad-1 maxpool-offset mapping. Each input
        # pixel can be selected by at most four output windows.
        oh0 = h // 2
        ow0 = w // 2
        oh1 = (h + 1) // 2
        ow1 = (w + 1) // 2
        oh1_unique = oh1 != oh0
        ow1_unique = ow1 != ow0

        scatter = tl.zeros((BLOCK,), tl.float32)

        pool_offset = n * 64 * 8 * 8 + c * 8 * 8

        cand_mask = (oh0 < 8) & (ow0 < 8)
        cand_offset = pool_offset + oh0 * 8 + ow0
        offset_value = tl.load(offsets_ptr + cand_offset, mask=cand_mask, other=-1).to(tl.int64)
        kh = offset_value // 3
        kw = offset_value - kh * 3
        selected_h = oh0 * 2 - 1 + kh
        selected_w = ow0 * 2 - 1 + kw
        hit = cand_mask & (selected_h == h) & (selected_w == w)
        source = (
            tl.load(where_ptr + cand_offset, mask=hit, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + cand_offset, mask=hit, other=0.0).to(tl.float32)
        )
        scatter += tl.where(hit, source, 0.0)

        cand_mask = (oh0 < 8) & (ow1 < 8) & ow1_unique
        cand_offset = pool_offset + oh0 * 8 + ow1
        offset_value = tl.load(offsets_ptr + cand_offset, mask=cand_mask, other=-1).to(tl.int64)
        kh = offset_value // 3
        kw = offset_value - kh * 3
        selected_h = oh0 * 2 - 1 + kh
        selected_w = ow1 * 2 - 1 + kw
        hit = cand_mask & (selected_h == h) & (selected_w == w)
        source = (
            tl.load(where_ptr + cand_offset, mask=hit, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + cand_offset, mask=hit, other=0.0).to(tl.float32)
        )
        scatter += tl.where(hit, source, 0.0)

        cand_mask = (oh1 < 8) & (ow0 < 8) & oh1_unique
        cand_offset = pool_offset + oh1 * 8 + ow0
        offset_value = tl.load(offsets_ptr + cand_offset, mask=cand_mask, other=-1).to(tl.int64)
        kh = offset_value // 3
        kw = offset_value - kh * 3
        selected_h = oh1 * 2 - 1 + kh
        selected_w = ow0 * 2 - 1 + kw
        hit = cand_mask & (selected_h == h) & (selected_w == w)
        source = (
            tl.load(where_ptr + cand_offset, mask=hit, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + cand_offset, mask=hit, other=0.0).to(tl.float32)
        )
        scatter += tl.where(hit, source, 0.0)

        cand_mask = (oh1 < 8) & (ow1 < 8) & oh1_unique & ow1_unique
        cand_offset = pool_offset + oh1 * 8 + ow1
        offset_value = tl.load(offsets_ptr + cand_offset, mask=cand_mask, other=-1).to(tl.int64)
        kh = offset_value // 3
        kw = offset_value - kh * 3
        selected_h = oh1 * 2 - 1 + kh
        selected_w = ow1 * 2 - 1 + kw
        hit = cand_mask & (selected_h == h) & (selected_w == w)
        source = (
            tl.load(where_ptr + cand_offset, mask=hit, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + cand_offset, mask=hit, other=0.0).to(tl.float32)
        )
        scatter += tl.where(hit, source, 0.0)

        affine = (x - mean) * invstd * gamma + beta
        relu_pass = (affine > 0.0) | (affine != affine)
        grad = tl.where(relu_pass, scatter, full_value)

        ch0 = channel_in_group == 0
        ch1 = channel_in_group == 1
        sum_w0 = tl.sum(tl.where(ch0, grad, 0.0), axis=0)
        sum_w1 = tl.sum(tl.where(ch1, grad, 0.0), axis=0)
        sum_wx0 = tl.sum(tl.where(ch0, grad * x, 0.0), axis=0)
        sum_wx1 = tl.sum(tl.where(ch1, grad * x, 0.0), axis=0)

        gamma0 = tl.load(gamma_ptr + group * 2).to(tl.float32)
        gamma1 = tl.load(gamma_ptr + group * 2 + 1).to(tl.float32)
        sum_gamma_w = sum_w0 * gamma0 + sum_w1 * gamma1
        sum_gamma_wx = sum_wx0 * gamma0 + sum_wx1 * gamma1

        invstd2 = invstd * invstd
        reduction = 0.001953125
        group_term = (sum_gamma_w * mean - sum_gamma_wx) * invstd * invstd2 * reduction
        bias_term = -group_term * mean - sum_gamma_w * invstd * reduction
        dx = grad * invstd * gamma + x * group_term + bias_term
        tl.store(out_dx_ptr + x_offset, dx)

        partial_base = n * 64 + group * 2
        dgamma0 = (sum_wx0 - sum_w0 * mean) * invstd
        dgamma1 = (sum_wx1 - sum_w1 * mean) * invstd
        tl.store(partial_dgamma_ptr + partial_base, dgamma0)
        tl.store(partial_dgamma_ptr + partial_base + 1, dgamma1)
        tl.store(partial_dbeta_ptr + partial_base, sum_w0)
        tl.store(partial_dbeta_ptr + partial_base + 1, sum_w1)

    @triton.jit
    def _finalize_channel_vectors_kernel(
        partial_dgamma_ptr,
        partial_dbeta_ptr,
        out_dgamma_ptr,
        out_dbeta_ptr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        n = tl.arange(0, BLOCK_N)
        c = tl.arange(0, BLOCK_C)
        offsets = n[:, None] * 64 + c[None, :]
        dgamma = tl.load(partial_dgamma_ptr + offsets).to(tl.float32)
        dbeta = tl.load(partial_dbeta_ptr + offsets).to(tl.float32)
        tl.store(out_dgamma_ptr + c, tl.sum(dgamma, axis=0))
        tl.store(out_dbeta_ptr + c, tl.sum(dbeta, axis=0))


def _validate_inputs(inputs):
    if len(inputs) != 24:
        raise ValueError(f"expected 24 inputs, got {len(inputs)}")
    (
        where_14,
        getitem_54,
        arg47_1,
        arg43_1,
        arg44_1,
        arg45_1,
        arg2_1,
        arg3_1,
        full,
        *_shape_params,
    ) = inputs

    expected = {
        "where_14": ((BATCH, CHANNELS, POOL_H, POOL_W), torch.float32, where_14),
        "getitem_54": ((BATCH, CHANNELS, POOL_H, POOL_W), torch.float32, getitem_54),
        "arg47_1": ((BATCH, CHANNELS, POOL_H, POOL_W), torch.int8, arg47_1),
        "arg43_1": ((BATCH, CHANNELS, IN_H, IN_W), torch.float32, arg43_1),
        "arg44_1": ((BATCH, GROUPS, 1, 1), torch.float32, arg44_1),
        "arg45_1": ((BATCH, GROUPS, 1, 1), torch.float32, arg45_1),
        "arg2_1": ((CHANNELS,), torch.float32, arg2_1),
        "arg3_1": ((CHANNELS,), torch.float32, arg3_1),
        "full": ((), torch.float32, full),
    }
    for name, (shape, dtype, tensor) in expected.items():
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} != {dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return where_14, getitem_54, arg47_1, arg43_1, arg44_1, arg45_1, arg2_1, arg3_1, full


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
    (
        where_14,
        getitem_54,
        arg47_1,
        arg43_1,
        arg44_1,
        arg45_1,
        arg2_1,
        arg3_1,
        full,
    ) = _validate_inputs(inputs)

    if triton is None:
        raise RuntimeError("Triton is required for oracle_pool_scatter_reduce.py")
    if arg43_1.device.type != "cuda":
        raise RuntimeError("oracle_pool_scatter_reduce.py requires CUDA inputs")

    out_dx = torch.empty_like(arg43_1)
    partial_dgamma = torch.empty((BATCH, CHANNELS), device=arg43_1.device, dtype=torch.float32)
    partial_dbeta = torch.empty((BATCH, CHANNELS), device=arg43_1.device, dtype=torch.float32)
    out_dgamma = torch.empty_like(arg2_1)
    out_dbeta = torch.empty_like(arg2_1)

    _pool_scatter_groupnorm_kernel[(BATCH, GROUPS)](
        where_14,
        getitem_54,
        arg47_1,
        arg43_1,
        arg44_1,
        arg45_1,
        arg2_1,
        arg3_1,
        full,
        out_dx,
        partial_dgamma,
        partial_dbeta,
        BLOCK=GROUP_ELEMS,
        num_warps=8,
    )
    _finalize_channel_vectors_kernel[(1,)](
        partial_dgamma,
        partial_dbeta,
        out_dgamma,
        out_dbeta,
        BLOCK_N=BATCH,
        BLOCK_C=CHANNELS,
        num_warps=8,
    )
    return out_dx, out_dgamma, out_dbeta


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
