"""
Oracle for sum_sum_sum_8d668b8fcd9e

Gap diagnosis:
  Classification: SCATTER_REDUCE
  What oracle does differently: Treats the zero/select_scatter producer as a structured sparse scatter, computes the layer-norm-backward row reductions only for token 0, writes the required transposed side output, and accumulates all sibling channel reductions from that sparse producer.
  What Inductor change would fix: Recognize zero-filled select_scatter feeding row reductions, a materialized permute side output, and sibling channel reductions as one structured scatter-reduce producer instead of materializing dense intermediates.
"""
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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure (installed via pip install -e .)
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


BATCH = 128
TOKENS = 1370
CHANNELS = 768
ROWS = BATCH * TOKENS
ZERO_BLOCK = 4096
CHANNEL_BLOCK = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _zero_storage_and_vectors_kernel(
        side_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out4_ptr,
        total_elems: tl.constexpr,
        channels: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        zero = tl.zeros((BLOCK,), tl.float32)
        tl.store(side_ptr + offsets, zero, mask=offsets < total_elems)
        tl.store(out0_ptr + offsets, zero, mask=offsets < channels)
        tl.store(out1_ptr + offsets, zero, mask=offsets < channels)
        tl.store(out2_ptr + offsets, zero, mask=offsets < channels)
        tl.store(out4_ptr + offsets, zero, mask=offsets < channels)

    @triton.jit
    def _select_scatter_layernorm_reduce_kernel(
        source_ptr,
        gamma_ptr,
        xhat_ptr,
        scale_ptr,
        view_ptr,
        out_weight_ptr,
        out_mul_xhat_ptr,
        out_scatter_sum_ptr,
        out_weighted_view_sum_ptr,
        side_ptr,
        out_grad_sum_ptr,
        channels: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_offsets = tl.arange(0, BLOCK_C)
        c_mask = c_offsets < channels

        source_offsets = batch * channels + c_offsets
        token_offsets = batch * tokens * channels + c_offsets
        scale_offset = batch * tokens

        source = tl.load(source_ptr + source_offsets, mask=c_mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + token_offsets, mask=c_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + scale_offset).to(tl.float32)
        view_value = tl.load(view_ptr + token_offsets, mask=c_mask, other=0.0).to(tl.float32)
        out_weight = tl.load(out_weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        weighted = source * gamma
        row_sum = tl.sum(weighted, axis=0)
        row_dot = tl.sum(weighted * xhat, axis=0)
        grad = scale * (weighted * channels - row_sum - xhat * row_dot)
        side_value = grad * out_weight

        tl.store(side_ptr + token_offsets, side_value, mask=c_mask)
        tl.atomic_add(out_mul_xhat_ptr + c_offsets, source * xhat, sem="relaxed", mask=c_mask)
        tl.atomic_add(out_scatter_sum_ptr + c_offsets, source, sem="relaxed", mask=c_mask)
        tl.atomic_add(
            out_weighted_view_sum_ptr + c_offsets,
            grad * view_value,
            sem="relaxed",
            mask=c_mask,
        )
        tl.atomic_add(out_grad_sum_ptr + c_offsets, side_value, sem="relaxed", mask=c_mask)


def _validate_inputs(inputs):
    if len(inputs) != 10:
        raise ValueError(f"expected 10 inputs, got {len(inputs)}")
    (
        source,
        gamma,
        xhat,
        scale,
        view_flat,
        out_weight,
        _shape0,
        _shape1,
        _shape2,
        _shape3,
    ) = inputs
    del _shape0, _shape1, _shape2, _shape3

    if source.shape != (BATCH, CHANNELS):
        raise ValueError(f"unexpected source shape {tuple(source.shape)}")
    if gamma.shape != (CHANNELS,) or out_weight.shape != (CHANNELS,):
        raise ValueError("unexpected channel vector shape")
    if xhat.shape != (BATCH, TOKENS, CHANNELS):
        raise ValueError(f"unexpected xhat shape {tuple(xhat.shape)}")
    if scale.shape != (BATCH, TOKENS, 1):
        raise ValueError(f"unexpected scale shape {tuple(scale.shape)}")
    if view_flat.shape != (ROWS, CHANNELS):
        raise ValueError(f"unexpected flattened view shape {tuple(view_flat.shape)}")

    tensors = (source, gamma, xhat, scale, view_flat, out_weight)
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("oracle expects float32 tensor inputs")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("oracle expects contiguous captured tensor inputs")
    return source, gamma, xhat, scale, view_flat, out_weight


def _oracle_torch(inputs):
    source, gamma, xhat_all, scale_all, view_flat, out_weight = _validate_inputs(inputs)
    xhat = xhat_all[:, 0, :]
    scale = scale_all[:, 0, :]
    view_value = view_flat.view(BATCH, TOKENS, CHANNELS)[:, 0, :]
    weighted = source * gamma
    row_sum = weighted.sum(dim=1, keepdim=True)
    row_dot = (weighted * xhat).sum(dim=1, keepdim=True)
    grad = scale * (weighted * CHANNELS - row_sum - xhat * row_dot)
    side_rows = grad * out_weight

    side_storage = torch.zeros((ROWS, CHANNELS), device=source.device, dtype=source.dtype)
    side_storage[0::TOKENS, :] = side_rows
    return (
        (source * xhat).sum(dim=0),
        source.sum(dim=0),
        (grad * view_value).sum(dim=0),
        side_storage.t(),
        side_rows.sum(dim=0),
    )


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
    source, gamma, xhat, scale, view_flat, out_weight = _validate_inputs(inputs)
    if source.device.type != "cuda" or triton is None:
        return _oracle_torch(inputs)

    out0 = torch.empty((CHANNELS,), device=source.device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=source.device, dtype=torch.float32)
    out2 = torch.empty((CHANNELS,), device=source.device, dtype=torch.float32)
    out4 = torch.empty((CHANNELS,), device=source.device, dtype=torch.float32)
    side = torch.empty_strided(
        (CHANNELS, ROWS),
        (1, CHANNELS),
        device=source.device,
        dtype=torch.float32,
    )

    total_side_elems = ROWS * CHANNELS
    _zero_storage_and_vectors_kernel[(triton.cdiv(total_side_elems, ZERO_BLOCK),)](
        side,
        out0,
        out1,
        out2,
        out4,
        total_elems=total_side_elems,
        channels=CHANNELS,
        BLOCK=ZERO_BLOCK,
        num_warps=8,
    )
    _select_scatter_layernorm_reduce_kernel[(BATCH,)](
        source,
        gamma,
        xhat,
        scale,
        view_flat,
        out_weight,
        out0,
        out1,
        out2,
        side,
        out4,
        channels=CHANNELS,
        tokens=TOKENS,
        BLOCK_C=CHANNEL_BLOCK,
        num_warps=8,
    )
    return (out0, out1, out2, side, out4)


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
