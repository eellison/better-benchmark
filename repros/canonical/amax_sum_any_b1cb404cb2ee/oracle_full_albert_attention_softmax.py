"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Albert attention softmax return from Repro.forward, including the [96,512,512] to [8,12,512,512] view, the iota/ge/expand/where mask folded to its all-zero value, stable last-dimension softmax, the all-minus-inf row zeroing implied by eq/any/where, and the final contiguous [96,512,512] view, whereas the measured Inductor path is already within the same row-softmax read/exp/sum/write envelope; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new local pattern because the mandatory 512-wide softmax math and output traffic dominate after the removable mask is folded; the fix is BANDWIDTH_BOUND: record this row as at floor unless broader softmax-template or bandwidth work moves both implementations together."""
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

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
HEADS = 12
Q_LEN = 512
K_LEN = 512
FLAT_BH = BATCH * HEADS
N_ROWS = FLAT_BH * Q_LEN
OUT_SHAPE = (FLAT_BH, Q_LEN, K_LEN)
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _softmax_all_true_mask_kernel(
        bmm_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        k_len: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_K)
        row_mask = rows < total_rows
        col_mask = cols < k_len
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * k_len + cols[None, :]

        scores = tl.load(bmm_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(scores, axis=1)
        row_has_value = row_max != -float("inf")
        safe_max = tl.where(row_has_value, row_max, 0.0)
        numer = tl.exp2((scores - safe_max[:, None]) * 1.4426950408889634)
        denom = tl.sum(numer, axis=1)
        safe_denom = tl.where(row_has_value, denom, 1.0)
        softmax = numer / safe_denom[:, None]
        out = tl.where(row_has_value[:, None], softmax, 0.0)

        tl.store(out_ptr + offsets, out, mask=mask)


def _validate_shape_param(name: str, actual: object, expected: tuple[int, ...]) -> None:
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    bmm: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    _shape_param_3: object,
) -> None:
    if not bmm.is_cuda:
        raise RuntimeError("CUDA input is required")
    if bmm.dtype != torch.float32:
        raise TypeError(f"expected bmm float32, got {bmm.dtype}")
    if tuple(bmm.shape) != OUT_SHAPE:
        raise ValueError(f"unexpected bmm shape: {tuple(bmm.shape)}")
    if bmm.stride() != OUT_STRIDE:
        raise ValueError(f"unexpected bmm stride: {bmm.stride()}")
    _validate_shape_param("_shape_param_0", _shape_param_0, (BATCH, -1, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_1", _shape_param_1, (BATCH, HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_2", _shape_param_2, (BATCH, HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_3", _shape_param_3, OUT_SHAPE)


def oracle_full_albert_attention_softmax(
    bmm: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    _shape_param_3: object,
    *,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 4,
) -> torch.Tensor:
    _validate_inputs(
        bmm,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a positive power of two, got {block_m}")

    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=bmm.device, dtype=bmm.dtype)
    grid = (triton.cdiv(N_ROWS, block_m),)
    _softmax_all_true_mask_kernel[grid](
        bmm,
        out,
        total_rows=N_ROWS,
        k_len=K_LEN,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([96, 512, 512], f32), S([8, -1, 512, 512]), S([8, 12, 512, 512]), S([8, 12, 512, 512]), S([96, 512, 512]))")
def oracle_forward(inputs):
    """Run the oracle computation for the exact Repro()(*make_inputs()) scope."""
    return oracle_full_albert_attention_softmax(*inputs)


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
