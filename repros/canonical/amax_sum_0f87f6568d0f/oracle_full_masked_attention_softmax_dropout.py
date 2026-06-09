"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DeBERTa masked attention softmax/dropout return from Repro.forward, including the [192,512,512] to [8,24,512,512] view, broadcast [8,1,512,512] bool mask with scalar fill, stable last-dimension softmax, seed-index-70 tl.rand dropout and scale, reshape, and final [192,512,512] transposed output view, whereas Inductor currently lowers the decomposed view/where/amax/sub/exp/sum/div/inductor_random/gt/mul/view/permute graph as separate generic mask, reduction, RNG/dropout, and layout kernels over materialized intermediates; Inductor cannot do this today because its pattern matcher and scheduler do not canonicalize scalar-fill masked attention softmax with stochastic Inductor RNG dropout and a trailing layout-only transpose into one persistent online-softmax template; the fix is NEW_PATTERN: add an Inductor lowering for scalar-fill masked attention softmax/dropout that fuses the mask fill, normalization, dropout scale, and output-layout epilogue into the row-softmax schedule. Exact stochastic eager equality is not established for prims.inductor_random, so this oracle is diagnostic/not_true_floor."""
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


BATCH = 8
N_HEADS = 24
Q_LEN = 512
K_LEN = 512
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
SEED_INDEX = 70

if triton is not None:

    @triton.jit
    def _masked_softmax_dropout_kernel(
        bmm_ptr,
        mask_ptr,
        fill_ptr,
        seeds_ptr,
        out_base_ptr,
        H: tl.constexpr,
        Q: tl.constexpr,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SEED_INDEX_CONST: tl.constexpr,
        DROPOUT_P_CONST: tl.constexpr,
        DROPOUT_SCALE_CONST: tl.constexpr,
    ):
        row = tl.program_id(0)
        bh = row // Q
        batch = bh // H
        q = row - bh * Q

        cols = tl.arange(0, BLOCK_K)
        col_mask = cols < K
        row_offsets = row * K + cols
        mask_offsets = (batch * Q + q) * K + cols

        bmm = tl.load(bmm_ptr + row_offsets, mask=col_mask, other=-float("inf")).to(tl.float32)
        mask = tl.load(mask_ptr + mask_offsets, mask=col_mask, other=0).to(tl.int1)
        fill = tl.load(fill_ptr).to(tl.float32)
        scores = tl.where(mask, fill, bmm)

        row_max = tl.max(scores, axis=0)
        numer = tl.exp(scores - row_max)
        denom = tl.sum(numer, axis=0)
        softmax = numer / denom

        seed = tl.load(seeds_ptr + SEED_INDEX_CONST)
        random = tl.rand(seed, row_offsets.to(tl.uint32))
        keep = (random > DROPOUT_P_CONST).to(tl.float32)
        out = softmax * keep * DROPOUT_SCALE_CONST

        tl.store(out_base_ptr + row_offsets, out, mask=col_mask)


def _check_shape_params(_shape_param_0: object, _shape_param_1: object) -> None:
    assert list(_shape_param_0) in (
        [-1, N_HEADS, Q_LEN, K_LEN],
        [BATCH, N_HEADS, Q_LEN, K_LEN],
    )
    assert list(_shape_param_1) in (
        [-1, Q_LEN, K_LEN],
        [BH, Q_LEN, K_LEN],
    )


def _launch_oracle(
    bmm_46: torch.Tensor,
    full_1: torch.Tensor,
    full_2: torch.Tensor,
    inductor_seeds: torch.Tensor,
    out_base: torch.Tensor,
    *,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    assert bmm_46.is_cuda and full_1.is_cuda and full_2.is_cuda and inductor_seeds.is_cuda
    assert out_base.is_cuda
    assert bmm_46.dtype == torch.float32 and bmm_46.shape == (BH, Q_LEN, K_LEN)
    assert full_1.dtype == torch.bool and full_1.shape == (BATCH, 1, Q_LEN, K_LEN)
    assert full_2.dtype == torch.float32 and full_2.shape == ()
    assert inductor_seeds.dtype == torch.int64 and inductor_seeds.shape == (73,)
    assert out_base.dtype == torch.float32 and out_base.shape == bmm_46.shape
    assert bmm_46.is_contiguous()
    assert full_1.is_contiguous()
    assert inductor_seeds.is_contiguous()
    assert out_base.is_contiguous()
    assert block_k >= K_LEN and (block_k & (block_k - 1)) == 0

    _masked_softmax_dropout_kernel[(N_ROWS,)](
        bmm_46,
        full_1,
        full_2,
        inductor_seeds,
        out_base,
        H=N_HEADS,
        Q=Q_LEN,
        K=K_LEN,
        BLOCK_K=block_k,
        SEED_INDEX_CONST=SEED_INDEX,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        num_warps=num_warps,
    )
    return out_base.permute(0, 2, 1)


def oracle_full_masked_attention_softmax_dropout(
    bmm_46: torch.Tensor,
    full_1: torch.Tensor,
    full_2: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    *,
    out_base: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 4,
) -> torch.Tensor:
    _check_shape_params(_shape_param_0, _shape_param_1)

    if out_base is None:
        out_base = torch.empty_like(bmm_46)
    return _launch_oracle(
        bmm_46,
        full_1,
        full_2,
        inductor_seeds,
        out_base,
        block_k=block_k,
        num_warps=num_warps,
    )


@oracle_impl(hardware="H100", shapes="(T([192, 512, 512], f32), T([8, 1, 512, 512], b8), T([], f32), T([73], i64), S([-1, 24, 512, 512]), S([-1, 512, 512]))")
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
    return oracle_full_masked_attention_softmax_dropout(*inputs)


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
