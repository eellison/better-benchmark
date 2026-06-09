"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DistilBERT attention softmax/dropout return from Repro.forward, including the always-true iota/ge broadcast mask folded to zero, the [3072,128,128] to [256,12,128,128] view, stable last-dimension softmax with all-minus-inf row zeroing, exact Inductor RNG dropout from the provided seed tensor, expand/view, and final non-contiguous [3072,128,128] transpose view, whereas Inductor currently lowers the decomposed iota/expand/where/view/add/amax/sub/exp/sum/div/eq/any/where/inductor_random/dropout/expand/view/permute graph as generic reductions, pointwise RNG/dropout, and layout work over materialized intermediates; Inductor cannot do this today because its pattern matcher and scheduler do not canonicalize a removable structured attention mask, row-all-masked guard, stochastic dropout, and trailing layout-only transpose into one fused online-softmax template; the fix is NEW_PATTERN: add an Inductor lowering for attention softmax/dropout that recognizes removable structured masks, preserves the all-masked-row guard, and fuses dropout plus the output-layout epilogue into the row softmax kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401

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


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _softmax_any_dropout_kernel(
        bmm_ptr,
        seeds_ptr,
        out_base_ptr,
        K: tl.constexpr,
        N_TOTAL_ROWS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        DROPOUT_P_CONST: tl.constexpr,
        DROPOUT_SCALE_CONST: tl.constexpr,
        SEED_IDX: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_K)

        row_mask = rows < N_TOTAL_ROWS
        col_mask = cols < K
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * K + cols[None, :]

        scores = tl.load(
            bmm_ptr + offsets,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        row_max = tl.max(scores, axis=1)
        row_has_value = row_max != -float("inf")
        safe_max = tl.where(row_has_value, row_max, 0.0)
        numer = tl.exp(scores - safe_max[:, None])
        denom = tl.sum(numer, axis=1)
        safe_denom = tl.where(row_has_value, denom, 1.0)
        softmax = numer / safe_denom[:, None]
        value = tl.where(row_has_value[:, None], softmax, 0.0)

        seed = tl.load(seeds_ptr + SEED_IDX)
        random = tl.rand(seed, offsets.to(tl.uint32))
        keep = (random > DROPOUT_P_CONST).to(tl.float32)
        out = keep * value * DROPOUT_SCALE_CONST

        tl.store(out_base_ptr + offsets, out, mask=mask)


BATCH = 256
N_HEADS = 12
Q_LEN = 128
K_LEN = 128
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
SEED_INDEX = 1
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)


def _check_power_of_two(value: int, name: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def _check_shape_params(
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    _shape_param_3: object,
) -> None:
    assert list(_shape_param_0) == [BATCH, -1, Q_LEN, K_LEN]
    assert list(_shape_param_1) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_2) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_3) == [BH, Q_LEN, K_LEN]


def _launch_oracle(
    bmm: torch.Tensor,
    inductor_seeds: torch.Tensor,
    out_base: torch.Tensor,
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    assert bmm.is_cuda and inductor_seeds.is_cuda and out_base.is_cuda
    assert bmm.dtype == torch.float32 and bmm.shape == (BH, Q_LEN, K_LEN)
    assert inductor_seeds.dtype == torch.int64
    assert inductor_seeds.numel() > SEED_INDEX
    assert out_base.dtype == torch.float32 and out_base.shape == bmm.shape
    assert bmm.is_contiguous()
    assert out_base.is_contiguous()
    _check_power_of_two(block_m, "block_m")
    _check_power_of_two(block_k, "block_k")
    assert block_k >= K_LEN

    grid = (triton.cdiv(N_ROWS, block_m),)
    _softmax_any_dropout_kernel[grid](
        bmm,
        inductor_seeds,
        out_base,
        K=K_LEN,
        N_TOTAL_ROWS=N_ROWS,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        SEED_IDX=SEED_INDEX,
        num_warps=num_warps,
    )
    return out_base.permute(0, 2, 1)


def oracle_full_attention_softmax_dropout(
    bmm: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    _shape_param_3: object,
    *,
    out_base: torch.Tensor | None = None,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 4,
) -> torch.Tensor:
    _check_shape_params(
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )

    if out_base is None:
        out_base = torch.empty_like(bmm)
    return _launch_oracle(
        bmm,
        inductor_seeds,
        out_base,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )


@oracle_impl(hardware="H100", shapes="(T([3072, 128, 128], f32), T([13], i64), S([256, -1, 128, 128]), S([256, 12, 128, 128]), S([256, 12, 128, 128]), S([3072, 128, 128]))")
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
    return oracle_full_attention_softmax_dropout(*inputs)


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
