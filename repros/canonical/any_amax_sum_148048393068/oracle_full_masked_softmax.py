"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Electra masked attention softmax returned by Repro.forward, including the [256,512,512] -> [64,4,512,512] view, the broadcast bool mask to fp32 0/-inf bias, the any(eq(-inf)) all-masked-row guard, stable last-dimension softmax, zero fill for all--inf rows, expand, and final contiguous [256,512,512] view in one Triton row kernel, whereas Inductor currently lowers the decomposed where/add/eq/logical_not/any/amax/sub/exp/sum/div/where chain as generic pointwise and reduction fragments over materialized intermediates; Inductor cannot do this today because its scheduler/codegen pattern library does not recognize this all--inf-safe broadcast masked-softmax idiom as one semantic row template; the fix is NEW_PATTERN: add an Inductor lowering that fuses the predicate, row reductions, all-masked-row handling, and layout-only epilogue into a single masked-softmax kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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

BATCH = 64
HEADS = 4
Q_LEN = 512
K_LEN = 512
N_ROWS = BATCH * HEADS * Q_LEN
BMM_SHAPE = (BATCH * HEADS, Q_LEN, K_LEN)
MASK_SHAPE = (BATCH, 1, Q_LEN, K_LEN)
VIEW_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
OUT_SHAPE = BMM_SHAPE
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _masked_softmax_kernel(
        bmm_ptr,
        mask_ptr,
        out_ptr,
        mask_s0: tl.constexpr,
        mask_s1: tl.constexpr,
        mask_s2: tl.constexpr,
        mask_s3: tl.constexpr,
        heads: tl.constexpr,
        q_len: tl.constexpr,
        k_len: tl.constexpr,
        n_rows: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_k)
        row_mask = rows < n_rows
        col_mask = cols < k_len
        tile_mask = row_mask[:, None] & col_mask[None, :]

        flat_bh = rows // q_len
        batch = flat_bh // heads
        q = rows - flat_bh * q_len

        bmm_offsets = rows[:, None] * k_len + cols[None, :]
        mask_offsets = (
            batch[:, None] * mask_s0
            + q[:, None] * mask_s2
            + cols[None, :] * mask_s3
        )

        keep = tl.load(mask_ptr + mask_offsets, mask=tile_mask, other=0).to(tl.int1)
        load_mask = tile_mask & keep
        scores = tl.load(bmm_ptr + bmm_offsets, mask=load_mask, other=-float("inf")).to(tl.float32)
        scores = tl.where(load_mask, scores, -float("inf"))

        live = load_mask & (scores != -float("inf"))
        has_any = tl.sum(tl.where(live, 1, 0), axis=1) > 0
        row_max = tl.max(scores, axis=1)
        stable_max = tl.where(has_any, row_max, 0.0)
        numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
        numer = tl.where(load_mask, numer, 0.0)
        denom = tl.sum(numer, axis=1)
        probs = tl.where(has_any[:, None], numer / denom[:, None], 0.0)

        tl.store(out_ptr + bmm_offsets, probs, mask=tile_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm_22: torch.Tensor,
    expand_4: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not bmm_22.is_cuda or not expand_4.is_cuda:
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if bmm_22.dtype != torch.float32 or expand_4.dtype != torch.bool:
        raise TypeError(
            f"expected f32 scores and bool mask, got {bmm_22.dtype} and {expand_4.dtype}"
        )
    if tuple(bmm_22.shape) != BMM_SHAPE:
        raise ValueError(f"expected bmm_22 shape {BMM_SHAPE}, got {tuple(bmm_22.shape)}")
    if tuple(bmm_22.stride()) != OUT_STRIDE:
        raise ValueError(
            f"expected contiguous bmm_22 stride {OUT_STRIDE}, got {tuple(bmm_22.stride())}"
        )
    if tuple(expand_4.shape) != MASK_SHAPE:
        raise ValueError(f"expected expand_4 shape {MASK_SHAPE}, got {tuple(expand_4.shape)}")

    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, VIEW_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def oracle_full_masked_softmax(
    bmm_22: torch.Tensor,
    expand_4: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    *,
    block_m: int = 2,
    block_k: int = K_LEN,
    num_warps: int = 4,
) -> torch.Tensor:
    _validate_inputs(bmm_22, expand_4, _shape_param_0, _shape_param_1, _shape_param_2)
    if block_k != K_LEN:
        raise ValueError(f"block_k must equal {K_LEN}, got {block_k}")
    if block_m <= 0 or N_ROWS % block_m != 0:
        raise ValueError(f"block_m must evenly divide {N_ROWS}, got {block_m}")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_22.device,
        dtype=torch.float32,
    )
    _masked_softmax_kernel[(N_ROWS // block_m,)](
        bmm_22,
        expand_4,
        out,
        mask_s0=expand_4.stride(0),
        mask_s1=expand_4.stride(1),
        mask_s2=expand_4.stride(2),
        mask_s3=expand_4.stride(3),
        heads=HEADS,
        q_len=Q_LEN,
        k_len=K_LEN,
        n_rows=N_ROWS,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([256, 512, 512], f32), T([64, 1, 512, 512], b8, stride=(0, 512, 1, 0)), S([64, 4, 512, 512]), S([64, 4, 512, 512]), S([256, 512, 512]))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation for the assigned input tuple."""
    return oracle_full_masked_softmax(*inputs)


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
