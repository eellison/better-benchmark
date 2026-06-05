"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin relative-position-bias attention softmax returned by Repro.forward, including the [4096, 49, 49] to [128, 32, 49, 49] reshape, indexed [169, 32] bias-table gather through the [49, 49] index matrix, permute/contiguous/unsqueeze semantics, bias add, last-dimension softmax, expand, and final [4096, 49, 49] reshape by materializing the small [32, 49, 49] relative bias once and feeding it to a persistent small-row Triton softmax, whereas Inductor currently lowers the decomposed reshape/index/permute/clone/add/amax/sub/exp/sum/div/expand/reshape graph as a generic fused reduction kernel that replays the indexed relative-bias expression across the softmax loops instead of hoisting the window-invariant bias and using a dedicated small-row attention-softmax template; Inductor cannot do this today because its pattern library does not canonicalize indexed relative-position bias producers into the persistent row-softmax template or sink/hoist the gather-permute producer around the reduction schedule; the fix is NEW_PATTERN: add an Inductor lowering for Swin-style relative-position attention softmax that fuses or hoists bias-table lookup, score add, normalization, and output-layout epilogue into the row-kernel schedule."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "amax_sum_17ab35828f89"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

Q_LEN = 49
K_LEN = 49
REL_POS_ROWS = 169
DEFAULT_BLOCK_K = 64


@triton.jit
def _materialize_relpos_bias_kernel(
    index_ptr,
    table_ptr,
    bias_ptr,
    index_s0: tl.constexpr,
    index_s1: tl.constexpr,
    table_s0: tl.constexpr,
    table_s1: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    col = offsets % k_len
    tmp = offsets // k_len
    query = tmp % q_len
    head = tmp // q_len

    rel_index = tl.load(
        index_ptr + query * index_s0 + col * index_s1,
        mask=mask,
        other=0,
    ).to(tl.int32)
    rel_bias = tl.load(
        table_ptr + rel_index * table_s0 + head * table_s1,
        mask=mask,
        other=0.0,
    )
    tl.store(bias_ptr + offsets, rel_bias, mask=mask)


@triton.jit
def _swin_bias_softmax_kernel(
    bmm_ptr,
    bias_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    n_rows: tl.constexpr,
    n_heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    block_rows: tl.constexpr,
    block_k: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    cols = tl.arange(0, block_k)
    row_mask = row_offsets < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    query = row_offsets % q_len
    window_head = row_offsets // q_len
    head = window_head % n_heads

    score_offsets = (
        window_head[:, None] * bmm_s0
        + query[:, None] * bmm_s1
        + cols[None, :] * bmm_s2
    )
    bias_offsets = head[:, None] * q_len * k_len + query[:, None] * k_len + cols[None, :]

    bmm_vals = tl.load(bmm_ptr + score_offsets, mask=mask, other=0.0)
    rel_bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)
    scores = bmm_vals.to(tl.float32) + rel_bias.to(tl.float32)
    scores = tl.where(mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    numer = tl.exp2((scores - row_max[:, None]) * 1.4426950408889634)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    out_offsets = (
        window_head[:, None] * out_s0
        + query[:, None] * out_s1
        + cols[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, probs, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm_46: torch.Tensor,
    arg352_1: torch.Tensor,
    arg351_1: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
) -> tuple[int, int, int]:
    if not bmm_46.is_cuda or not arg352_1.is_cuda or not arg351_1.is_cuda:
        raise RuntimeError("CUDA tensor inputs are required")
    if bmm_46.dtype != torch.float32:
        raise TypeError(f"expected bmm_46 dtype torch.float32, got {bmm_46.dtype}")
    if arg352_1.dtype != torch.int64:
        raise TypeError(f"expected arg352_1 dtype torch.int64, got {arg352_1.dtype}")
    if arg351_1.dtype != torch.float32:
        raise TypeError(f"expected arg351_1 dtype torch.float32, got {arg351_1.dtype}")
    if bmm_46.ndim != 3 or tuple(bmm_46.shape[1:]) != (Q_LEN, K_LEN):
        raise ValueError(f"expected bmm_46 shape [*, {Q_LEN}, {K_LEN}], got {tuple(bmm_46.shape)}")
    if tuple(arg352_1.shape) != (Q_LEN, K_LEN):
        raise ValueError(f"expected arg352_1 shape {(Q_LEN, K_LEN)}, got {tuple(arg352_1.shape)}")
    if arg351_1.ndim != 2 or arg351_1.shape[0] != REL_POS_ROWS:
        raise ValueError(f"expected arg351_1 shape [{REL_POS_ROWS}, heads], got {tuple(arg351_1.shape)}")

    view_shape = _shape_tuple(_shape_param_0)
    if len(view_shape) != 4 or view_shape[2:] != (Q_LEN, K_LEN):
        raise ValueError(f"_shape_param_0 must be [windows, heads, {Q_LEN}, {K_LEN}], got {view_shape}")
    n_windows, n_heads = view_shape[0], view_shape[1]
    if n_windows * n_heads != bmm_46.shape[0]:
        raise ValueError(
            f"bmm_46 first dim must equal windows * heads, got {bmm_46.shape[0]} vs {n_windows} * {n_heads}"
        )
    if arg351_1.shape[1] != n_heads:
        raise ValueError(f"arg351_1 head dim mismatch: expected {n_heads}, got {arg351_1.shape[1]}")

    view_bias_shape = _shape_tuple(_shape_param_1)
    if len(view_bias_shape) != 3 or view_bias_shape[:2] != (Q_LEN, K_LEN):
        raise ValueError(f"_shape_param_1 must start with {(Q_LEN, K_LEN)}, got {view_bias_shape}")
    if view_bias_shape[2] not in (-1, n_heads):
        raise ValueError(f"_shape_param_1 final dim must be -1 or {n_heads}, got {view_bias_shape[2]}")

    _validate_shape_param("_shape_param_2", _shape_param_2, view_shape)
    _validate_shape_param("_shape_param_3", _shape_param_3, tuple(bmm_46.shape))
    return n_windows, n_heads, n_windows * n_heads * Q_LEN


def _contiguous_3d_stride(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return (shape[1] * shape[2], shape[2], 1)


def _launch_materialized_oracle(
    bmm_46: torch.Tensor,
    arg352_1: torch.Tensor,
    arg351_1: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    *,
    n_heads: int,
    n_rows: int,
    block_rows: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    if block_rows <= 0 or block_rows & (block_rows - 1):
        raise ValueError(f"block_rows must be a positive power of two, got {block_rows}")
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")
    if bias.shape != (n_heads, Q_LEN, K_LEN) or bias.dtype != torch.float32 or not bias.is_cuda:
        raise ValueError(f"bias must be CUDA fp32 with shape {(n_heads, Q_LEN, K_LEN)}")
    if not bias.is_contiguous():
        raise ValueError(f"bias must be contiguous, got stride {bias.stride()}")
    if out.shape != bmm_46.shape or out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError("preallocated output must be CUDA fp32 with bmm_46 shape")
    if out.stride() != _contiguous_3d_stride(tuple(bmm_46.shape)):
        raise ValueError(f"preallocated output must be contiguous, got stride {out.stride()}")

    bias_elems = n_heads * Q_LEN * K_LEN
    _materialize_relpos_bias_kernel[(triton.cdiv(bias_elems, 256),)](
        arg352_1,
        arg351_1,
        bias,
        index_s0=arg352_1.stride(0),
        index_s1=arg352_1.stride(1),
        table_s0=arg351_1.stride(0),
        table_s1=arg351_1.stride(1),
        q_len=Q_LEN,
        k_len=K_LEN,
        total=bias_elems,
        block_size=256,
        num_warps=4,
    )
    _swin_bias_softmax_kernel[(triton.cdiv(n_rows, block_rows),)](
        bmm_46,
        bias,
        out,
        bmm_s0=bmm_46.stride(0),
        bmm_s1=bmm_46.stride(1),
        bmm_s2=bmm_46.stride(2),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        n_rows=n_rows,
        n_heads=n_heads,
        q_len=Q_LEN,
        k_len=K_LEN,
        block_rows=block_rows,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


def oracle_full_swin_relative_position_softmax(
    bmm_46: torch.Tensor,
    arg352_1: torch.Tensor,
    arg351_1: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
    *,
    out: torch.Tensor | None = None,
    block_rows: int = 4,
    block_k: int | None = None,
    num_warps: int = 1,
) -> torch.Tensor:
    _n_windows, n_heads, n_rows = _validate_inputs(
        bmm_46,
        arg352_1,
        arg351_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )
    if out is None:
        out = torch.empty_strided(
            tuple(bmm_46.shape),
            _contiguous_3d_stride(tuple(bmm_46.shape)),
            device=bmm_46.device,
            dtype=torch.float32,
        )
    bias = torch.empty((n_heads, Q_LEN, K_LEN), device=bmm_46.device, dtype=torch.float32)
    actual_block_k = block_k if block_k is not None else DEFAULT_BLOCK_K
    return _launch_materialized_oracle(
        bmm_46,
        arg352_1,
        arg351_1,
        bias,
        out,
        n_heads=n_heads,
        n_rows=n_rows,
        block_rows=block_rows,
        block_k=actual_block_k,
        num_warps=num_warps,
    )


def oracle_forward(inputs):
    return oracle_full_swin_relative_position_softmax(*inputs)


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
