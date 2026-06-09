"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete MobileBERT attention softmax returned by Repro.forward, including the [1024,128,128] -> [256,4,128,128] view, generated arange(128) >= 0 mask, 0/-inf where bias, any(eq(-inf)) all-masked-row guard, stable last-dimension softmax, zero fill for all--inf rows, expand, and final contiguous [1024,128,128] view, by proving the generated mask always selects zero bias and folding the remaining work into one Triton row-softmax kernel; whereas Inductor lowers the decomposed view/iota/ge/expand/where/add/eq/any/amax/sub/exp/sum/div/where/expand/view graph as generic pointwise and reduction work; Inductor cannot do this today because shape-derived tautological predicates and identity mask-bias adds are not eliminated before reduction scheduling; the fix is ALGEBRAIC_ELIMINATION: fold provably constant generated masks and identity bias adds before scheduling the all--inf-safe row softmax."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

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


# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 256
N_HEADS = 4
Q_LEN = 128
K_LEN = 128
N_ROWS = BATCH * N_HEADS * Q_LEN
IN_SHAPE = (BATCH * N_HEADS, Q_LEN, K_LEN)
IN_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)
VIEW_SHAPE = (BATCH, N_HEADS, Q_LEN, K_LEN)
MASK_EXPAND_SHAPE = (BATCH, -1, Q_LEN, K_LEN)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _all_inf_safe_softmax_block_m_kernel(
    bmm_ptr,
    out_ptr,
    k_len: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_k)
    offsets = rows[:, None] * k_len + cols[None, :]

    # The repro's generated arange/ge/where mask contributes exactly zero.
    # Preserve the live semantic guard that maps all--inf rows to zeros.
    scores = tl.load(bmm_ptr + offsets).to(tl.float32)
    row_max = tl.max(scores, axis=1)
    has_any = row_max != -float("inf")
    stable_max = tl.where(has_any, row_max, 0.0)
    numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
    denom = tl.sum(numer, axis=1)
    probs = tl.where(has_any[:, None], numer / denom[:, None], 0.0)

    tl.store(out_ptr + offsets, probs)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
) -> None:
    if not bmm.is_cuda:
        raise RuntimeError("CUDA tensor input is required")
    if bmm.dtype != torch.float32:
        raise TypeError(f"expected bmm dtype torch.float32, got {bmm.dtype}")
    if tuple(bmm.shape) != IN_SHAPE:
        raise ValueError(f"expected bmm shape {IN_SHAPE}, got {tuple(bmm.shape)}")
    if tuple(bmm.stride()) != IN_STRIDE:
        raise ValueError(
            f"expected contiguous bmm stride {IN_STRIDE}, got {tuple(bmm.stride())}"
        )

    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, MASK_EXPAND_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, VIEW_SHAPE)
    _validate_shape_param("_shape_param_3", _shape_param_3, IN_SHAPE)


def oracle_full_attention_softmax(
    bmm: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
    *,
    out: torch.Tensor | None = None,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 1,
) -> torch.Tensor:
    _validate_inputs(
        bmm,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )
    if block_k != K_LEN:
        raise ValueError(f"block_k must equal {K_LEN} for this fixed-shape oracle, got {block_k}")
    if block_m <= 0 or N_ROWS % block_m != 0:
        raise ValueError(f"block_m must evenly divide {N_ROWS}, got {block_m}")

    if out is None:
        out = torch.empty_strided(
            IN_SHAPE,
            IN_STRIDE,
            device=bmm.device,
            dtype=torch.float32,
        )
    else:
        if tuple(out.shape) != IN_SHAPE or tuple(out.stride()) != IN_STRIDE:
            raise ValueError(
                f"unexpected output layout: shape={tuple(out.shape)} stride={tuple(out.stride())}"
            )
        if out.dtype != torch.float32 or out.device != bmm.device:
            raise ValueError(f"unexpected output dtype/device: {out.dtype} {out.device}")

    _all_inf_safe_softmax_block_m_kernel[(N_ROWS // block_m,)](
        bmm,
        out,
        k_len=K_LEN,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([1024, 128, 128], f32), S([256, 4, 128, 128]), S([256, -1, 128, 128]), S([256, 4, 128, 128]), S([1024, 128, 128]))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation for the assigned input tuple."""
    return oracle_full_attention_softmax(*inputs)


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
