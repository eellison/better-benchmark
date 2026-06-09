"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete T5 additive-bias attention softmax/dropout Repro.forward scope, including the [64,1024,1024] to [8,8,1024,1024] score view, broadcast [8,1,1024,1024] bias add through the explicit zero full, stable last-dimension softmax, Inductor seed-index-29 dropout mask, dropout scale, expand/view, and final transposed [64,1024,1024] output stride in one Triton row kernel, whereas Inductor lowers the decomposed view/add/amax/sub/exp/sum/div/random/dropout/expand/view/permute graph through generic reduction, RNG/dropout, and layout scheduling over large intermediates; Inductor cannot do this today because its pattern matcher and scheduler do not canonicalize broadcast-bias attention softmax with seeded dropout and a trailing layout-only transpose into one persistent online-softmax template; the fix is NEW_PATTERN: add a guarded attention softmax/dropout lowering that folds the bias broadcast, normalization, RNG dropout, scale, and output-layout epilogue into the row-softmax schedule."""
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
N_HEADS = 8
Q_LEN = 1024
K_LEN = 1024
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
OUT_SHAPE = (BH, K_LEN, Q_LEN)
OUT_STRIDE = (Q_LEN * K_LEN, 1, K_LEN)
SEED_INDEX = 29
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)

if triton is not None:

    @triton.jit
    def _bias_softmax_dropout_transpose_kernel(
        bmm_ptr,
        where_ptr,
        seeds_ptr,
        out_ptr,
        H: tl.constexpr,
        Q: tl.constexpr,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SEED_IDX: tl.constexpr,
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
        bias_offsets = (batch * Q + q) * K + cols

        bmm = tl.load(bmm_ptr + row_offsets, mask=col_mask, other=-float("inf")).to(
            tl.float32
        )
        bias = tl.load(where_ptr + bias_offsets, mask=col_mask, other=0.0).to(
            tl.float32
        )
        scores = tl.where(col_mask, bmm + bias, -float("inf"))

        row_max = tl.max(scores, axis=0)
        numer = tl.exp(scores - row_max)
        numer = tl.where(col_mask, numer, 0.0)
        denom = tl.sum(numer, axis=0)
        probs = numer / denom

        seed = tl.load(seeds_ptr + SEED_IDX)
        random = tl.rand(seed, row_offsets.to(tl.uint32))
        keep = random > DROPOUT_P_CONST
        out = tl.where(keep, probs * DROPOUT_SCALE_CONST, 0.0)

        out_offsets = bh * (Q * K) + cols + q * K
        tl.store(out_ptr + out_offsets, out, mask=col_mask)


def _shape_tuple(value: object) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: object, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm_14: torch.Tensor,
    where: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
) -> None:
    if not (bmm_14.is_cuda and where.is_cuda and inductor_seeds.is_cuda):
        raise RuntimeError("CUDA tensor inputs are required")
    if bmm_14.dtype != torch.float32:
        raise TypeError(f"expected bmm_14 dtype torch.float32, got {bmm_14.dtype}")
    if where.dtype != torch.float32:
        raise TypeError(f"expected where dtype torch.float32, got {where.dtype}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(
            f"expected inductor_seeds dtype torch.int64, got {inductor_seeds.dtype}"
        )
    if tuple(bmm_14.shape) != (BH, Q_LEN, K_LEN):
        raise ValueError(
            f"expected bmm_14 shape {(BH, Q_LEN, K_LEN)}, got {tuple(bmm_14.shape)}"
        )
    if tuple(where.shape) != (BATCH, 1, Q_LEN, K_LEN):
        raise ValueError(
            f"expected where shape {(BATCH, 1, Q_LEN, K_LEN)}, got {tuple(where.shape)}"
        )
    if tuple(inductor_seeds.shape) != (BH,):
        raise ValueError(
            f"expected inductor_seeds shape {(BH,)}, got {tuple(inductor_seeds.shape)}"
        )
    if tuple(bmm_14.stride()) != (Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"expected contiguous bmm_14 stride, got {tuple(bmm_14.stride())}")
    if tuple(where.stride()) != (Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(
            f"expected broadcast-bias contiguous stride, got {tuple(where.stride())}"
        )

    _validate_shape_param("_shape_param_0", _shape_param_0, (BATCH, N_HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_1", _shape_param_1, (BATCH, N_HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_2", _shape_param_2, (BH, Q_LEN, K_LEN))


def oracle_full_t5_attention_softmax_dropout(
    bmm_14: torch.Tensor,
    where: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    *,
    out: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 8,
) -> torch.Tensor:
    """Run the full attention softmax/dropout/transpose scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    _validate_inputs(
        bmm_14,
        where,
        inductor_seeds,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")

    if out is None:
        out = torch.empty_strided(
            OUT_SHAPE,
            OUT_STRIDE,
            device=bmm_14.device,
            dtype=torch.float32,
        )
    else:
        if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
            raise ValueError(
                f"unexpected output layout: shape={tuple(out.shape)} stride={tuple(out.stride())}"
            )
        if out.dtype != torch.float32 or out.device != bmm_14.device:
            raise ValueError(f"unexpected output dtype/device: {out.dtype} {out.device}")

    _bias_softmax_dropout_transpose_kernel[(N_ROWS,)](
        bmm_14,
        where,
        inductor_seeds,
        out,
        H=N_HEADS,
        Q=Q_LEN,
        K=K_LEN,
        BLOCK_K=block_k,
        SEED_IDX=SEED_INDEX,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([64, 1024, 1024], f32), T([8, 1, 1024, 1024], f32), T([64], i64), S([8, 8, 1024, 1024]), S([8, 8, 1024, 1024]), S([64, 1024, 1024]))")
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
    return oracle_full_t5_attention_softmax_dropout(*inputs)


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
