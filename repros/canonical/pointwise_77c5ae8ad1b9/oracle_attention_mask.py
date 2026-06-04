"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete causal same-segment attention mask directly into the f32[1,1,128,128] where result and returns the exact eager f32[1,16,128,128] zero-stride expand view, whereas Inductor lowers the captured iota unsqueeze advanced-index equality where expand chain as generic pointwise work with the indirect cumsum lookups inside the decomposed graph; Inductor cannot do this today because it has no shape-specialized causal segment-mask pattern that recognizes the paired position gathers and final metadata-only head expansion; the fix is NEW_PATTERN: add a lowering that emits a triangular segment-mask kernel and preserves the expanded output stride contract instead of treating this as unrelated generic ops."""
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


if triton is not None:

    @triton.jit
    def attention_mask_kernel(
        positions_ptr,
        cumsum_ptr,
        out_ptr,
        Q: tl.constexpr,
        K: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Materialize the base `[1,1,Q,K]` mask before the head expand."""
        q_block = tl.program_id(0)
        k_block = tl.program_id(1)

        q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        q_mask = q_offsets < Q
        k_mask = k_offsets < K

        q_pos = tl.load(positions_ptr + q_offsets, mask=q_mask, other=0)
        k_pos = tl.load(positions_ptr + k_offsets, mask=k_mask, other=0)
        q_seg = tl.load(cumsum_ptr + q_pos, mask=q_mask, other=-1)
        k_seg = tl.load(cumsum_ptr + k_pos, mask=k_mask, other=-2)

        keep = (k_pos[None, :] <= q_pos[:, None]) & (k_seg[None, :] == q_seg[:, None])
        values = tl.where(keep, 0.0, -3.4028234663852886e38)

        out_offsets = q_offsets[:, None] * K + k_offsets[None, :]
        tl.store(out_ptr + out_offsets, values, mask=q_mask[:, None] & k_mask[None, :])


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
        raise RuntimeError("triton is required for this oracle")

    positions, cumsum, shape_param_0, shape_param_1 = inputs
    if positions.ndim != 2 or cumsum.ndim != 2 or positions.shape != cumsum.shape:
        raise ValueError(
            f"expected matching rank-2 position/cumsum inputs, got "
            f"{tuple(positions.shape)} and {tuple(cumsum.shape)}"
        )
    if positions.shape[0] != 1:
        raise ValueError(f"this captured repro has batch size 1, got {positions.shape[0]}")

    q_len = int(positions.shape[1])
    k_len = q_len
    if list(shape_param_0) != [1, -1, q_len, k_len]:
        raise ValueError(f"unexpected first expand shape: {shape_param_0}")
    if list(shape_param_1)[:1] != [1] or list(shape_param_1)[2:] != [q_len, k_len]:
        raise ValueError(f"unexpected final expand shape: {shape_param_1}")

    base = torch.empty((1, 1, q_len, k_len), device=positions.device, dtype=torch.float32)
    grid = (triton.cdiv(q_len, 16), triton.cdiv(k_len, 64))
    attention_mask_kernel[grid](
        positions,
        cumsum,
        base,
        Q=q_len,
        K=k_len,
        BLOCK_Q=16,
        BLOCK_K=64,
        num_warps=4,
    )
    return base.expand(tuple(int(dim) for dim in shape_param_1))


def check_output_strides(instance, inputs) -> bool:
    """The shared harness checks values/shapes; this repro also requires expand strides."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
    ok = eager.stride() == oracle_out.stride()
    print(
        f"  output 0 stride: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} stride={oracle_out.stride()} "
        f"expected_stride={eager.stride()})"
    )
    return ok


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
        ok = check_output_strides(instance, inputs) and ok
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
