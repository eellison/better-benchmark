"""
Gap diagnosis (classification: NEW_PATTERN): this oracle covers the full
Reformer LSH equality-masked softmax materialization from repro.py, including
the rotated/current bucket concatenation, the broadcasted int64 equality mask,
the scalar replacement through aten.where, the stable logsumexp softmax over
the last dimension, and the final contiguous [6144, 64, 128] view. It differs
from Inductor by using one Triton program for several adjacent w rows of the
same (batch, head, q) group, so the 128-element comparison vector is loaded
once per tile instead of being materialized or redundantly reloaded around the
generic amax/sum lowering. Inductor cannot do this today because the mask is
expressed as view/slice/cat/unsqueeze/ne feeding a reduction pattern, and the
scheduler does not canonicalize that structured Reformer predicate into a
single masked-softmax row template with comparison-vector reuse. The fix class
is NEW_PATTERN: add a lowering for this Reformer LSH masked softmax that
recomputes the cheap predicate inside the online-softmax kernel and fuses the
epilogue view.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
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



REPRO_ID = "amax_sum_d112f48ea917"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
HEADS = 12
Q_LEN = 64
W_LEN = 64
C_LEN = 128
FLAT_BHQ = BATCH * HEADS * Q_LEN
OUT_SHAPE = (FLAT_BHQ, W_LEN, C_LEN)
OUT_STRIDE = (W_LEN * C_LEN, C_LEN, 1)

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
        "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
        "triton.multi_kernel=3",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


@triton.jit
def _reformer_masked_softmax_kernel(
    remainder_ptr,
    bmm_ptr,
    scalar_ptr,
    out_ptr,
    rem_s0: tl.constexpr,
    rem_s1: tl.constexpr,
    rem_s2: tl.constexpr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    w_len: tl.constexpr,
    c_len: tl.constexpr,
    block_w: tl.constexpr,
    block_c: tl.constexpr,
):
    group = tl.program_id(0)
    w_block = tl.program_id(1)

    batch = group // (heads * q_len)
    rem_group = group - batch * (heads * q_len)
    head = rem_group // q_len
    q = rem_group - head * q_len
    q_prev = tl.where(q == 0, q_len - 1, q - 1)

    w_offsets = w_block * block_w + tl.arange(0, block_w)
    c_offsets = tl.arange(0, block_c)
    w_mask = w_offsets < w_len
    c_mask = c_offsets < c_len

    rem_base = batch * rem_s0 + head * rem_s1
    left_vals = tl.load(
        remainder_ptr + rem_base + (q * w_len + w_offsets) * rem_s2,
        mask=w_mask,
        other=0,
    )

    right_q = tl.where(c_offsets < w_len, q_prev, q)
    right_w = tl.where(c_offsets < w_len, c_offsets, c_offsets - w_len)
    right_vals = tl.load(
        remainder_ptr + rem_base + (right_q * w_len + right_w) * rem_s2,
        mask=c_mask,
        other=0,
    )

    bmm_offsets = (
        group * bmm_s0
        + w_offsets[:, None] * bmm_s1
        + c_offsets[None, :] * bmm_s2
    )
    valid = w_mask[:, None] & c_mask[None, :]
    bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=valid, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    neq = left_vals[:, None] != right_vals[None, :]
    scores = tl.where(neq, bmm_vals, scalar)
    scores = tl.where(valid, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    stable_max = tl.where(tl.abs(row_max) == float("inf"), 0.0, row_max)
    exp_scores = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
    denom = tl.sum(exp_scores, axis=1)
    out_vals = exp_scores / denom[:, None]

    out_offsets = (
        group * out_s0
        + w_offsets[:, None] * out_s1
        + c_offsets[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, out_vals, mask=valid)


@triton.jit
def _reformer_masked_softmax_serial_w_kernel(
    remainder_ptr,
    bmm_ptr,
    scalar_ptr,
    out_ptr,
    rem_s0: tl.constexpr,
    rem_s1: tl.constexpr,
    rem_s2: tl.constexpr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    w_len: tl.constexpr,
    c_len: tl.constexpr,
    block_c: tl.constexpr,
):
    group = tl.program_id(0)

    batch = group // (heads * q_len)
    rem_group = group - batch * (heads * q_len)
    head = rem_group // q_len
    q = rem_group - head * q_len
    q_prev = tl.where(q == 0, q_len - 1, q - 1)

    c_offsets = tl.arange(0, block_c)
    c_mask = c_offsets < c_len

    rem_base = batch * rem_s0 + head * rem_s1
    right_q = tl.where(c_offsets < w_len, q_prev, q)
    right_w = tl.where(c_offsets < w_len, c_offsets, c_offsets - w_len)
    right_vals = tl.load(
        remainder_ptr + rem_base + (right_q * w_len + right_w) * rem_s2,
        mask=c_mask,
        other=0,
    )
    scalar = tl.load(scalar_ptr).to(tl.float32)

    for w in tl.range(0, w_len):
        left_val = tl.load(remainder_ptr + rem_base + (q * w_len + w) * rem_s2)
        bmm_offsets = group * bmm_s0 + w * bmm_s1 + c_offsets * bmm_s2
        bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=c_mask, other=0.0).to(tl.float32)

        scores = tl.where(left_val != right_vals, bmm_vals, scalar)
        scores = tl.where(c_mask, scores, -float("inf"))
        row_max = tl.max(scores, axis=0)
        stable_max = tl.where(tl.abs(row_max) == float("inf"), 0.0, row_max)
        exp_scores = tl.exp2((scores - stable_max) * 1.4426950408889634)
        denom = tl.sum(exp_scores, axis=0)
        out_vals = exp_scores / denom

        out_offsets = group * out_s0 + w * out_s1 + c_offsets * out_s2
        tl.store(out_ptr + out_offsets, out_vals, mask=c_mask)


def _validate_shape_param(name: str, actual, expected: tuple[int, ...]) -> None:
    if actual is None:
        return
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    remainder_1: torch.Tensor,
    bmm_1: torch.Tensor,
    arg5_1: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
    _shape_param_3=None,
) -> None:
    if not (remainder_1.is_cuda and bmm_1.is_cuda and arg5_1.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if remainder_1.dtype != torch.int64:
        raise TypeError(f"expected remainder_1 int64, got {remainder_1.dtype}")
    if bmm_1.dtype != torch.float32 or arg5_1.dtype != torch.float32:
        raise TypeError(f"expected fp32 score/scalar inputs, got {bmm_1.dtype} and {arg5_1.dtype}")
    if tuple(remainder_1.shape) != (BATCH, HEADS, Q_LEN * W_LEN):
        raise ValueError(f"unexpected remainder_1 shape: {tuple(remainder_1.shape)}")
    if tuple(bmm_1.shape) != OUT_SHAPE:
        raise ValueError(f"unexpected bmm_1 shape: {tuple(bmm_1.shape)}")
    if arg5_1.ndim != 0:
        raise ValueError(f"expected scalar arg5_1, got shape {tuple(arg5_1.shape)}")

    if _shape_param_0 is not None and tuple(_shape_param_0) != (BATCH, HEADS, -1, W_LEN):
        raise ValueError(f"_shape_param_0 mismatch: {tuple(_shape_param_0)}")
    _validate_shape_param("_shape_param_1", _shape_param_1, (BATCH, HEADS, Q_LEN, W_LEN, C_LEN))
    _validate_shape_param("_shape_param_2", _shape_param_2, (BATCH, HEADS, Q_LEN, W_LEN, C_LEN))
    _validate_shape_param("_shape_param_3", _shape_param_3, OUT_SHAPE)


def _launch_oracle(
    remainder_1: torch.Tensor,
    bmm_1: torch.Tensor,
    arg5_1: torch.Tensor,
    out: torch.Tensor,
    *,
    block_w: int,
    block_c: int,
    num_warps: int,
    serial_w: bool,
) -> torch.Tensor:
    if out.shape != OUT_SHAPE or out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError("preallocated output must be CUDA fp32 with shape [6144, 64, 128]")
    if out.stride() != OUT_STRIDE:
        raise ValueError(f"preallocated output stride must be {OUT_STRIDE}, got {out.stride()}")
    if block_c < C_LEN or block_c & (block_c - 1):
        raise ValueError(f"block_c must be a power of two >= {C_LEN}, got {block_c}")

    if serial_w:
        _reformer_masked_softmax_serial_w_kernel[(FLAT_BHQ,)](
            remainder_1,
            bmm_1,
            arg5_1,
            out,
            rem_s0=remainder_1.stride(0),
            rem_s1=remainder_1.stride(1),
            rem_s2=remainder_1.stride(2),
            bmm_s0=bmm_1.stride(0),
            bmm_s1=bmm_1.stride(1),
            bmm_s2=bmm_1.stride(2),
            out_s0=out.stride(0),
            out_s1=out.stride(1),
            out_s2=out.stride(2),
            heads=HEADS,
            q_len=Q_LEN,
            w_len=W_LEN,
            c_len=C_LEN,
            block_c=block_c,
            num_warps=num_warps,
        )
        return out

    if block_w <= 0 or W_LEN % block_w != 0:
        raise ValueError(f"block_w must divide {W_LEN}, got {block_w}")
    grid = (FLAT_BHQ, triton.cdiv(W_LEN, block_w))
    _reformer_masked_softmax_kernel[grid](
        remainder_1,
        bmm_1,
        arg5_1,
        out,
        rem_s0=remainder_1.stride(0),
        rem_s1=remainder_1.stride(1),
        rem_s2=remainder_1.stride(2),
        bmm_s0=bmm_1.stride(0),
        bmm_s1=bmm_1.stride(1),
        bmm_s2=bmm_1.stride(2),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        heads=HEADS,
        q_len=Q_LEN,
        w_len=W_LEN,
        c_len=C_LEN,
        block_w=block_w,
        block_c=block_c,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    remainder_1: torch.Tensor,
    bmm_1: torch.Tensor,
    arg5_1: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
    _shape_param_3=None,
    *,
    block_w: int = 64,
    block_c: int = 128,
    num_warps: int = 8,
    serial_w: bool = False,
) -> torch.Tensor:
    _validate_inputs(
        remainder_1,
        bmm_1,
        arg5_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_1.device,
        dtype=torch.float32,
    )
    return _launch_oracle(
        remainder_1,
        bmm_1,
        arg5_1,
        out,
        block_w=block_w,
        block_c=block_c,
        num_warps=num_warps,
        serial_w=serial_w,
    )


@oracle_impl(hardware="H100", shapes="(T([8, 12, 4096], i64), T([6144, 64, 128], f32), T([], f32), S([8, 12, -1, 64]), S([8, 12, 64, 64, 128]), S([8, 12, 64, 64, 128]), S([6144, 64, 128]))")
def oracle_forward(inputs):
    return oracle_online_softmax(*inputs)


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
