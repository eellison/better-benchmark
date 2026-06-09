"""
Gap diagnosis (classification: NEW_PATTERN): this oracle covers the full
Reformer logsumexp/softmax-style materialization from repro.py, including the
fp16 `[768,64,128]` input view as `[1,12,64,64,128]`, the fp32 max/exp/sum/log
over the last dimension, the `abs(max) == inf` replacement with zero, the fp16
logsumexp cast, the second fp16 subtract/exp, and the final contiguous
`[768,64,128]` output view. It differs from Inductor only by using a dedicated
multi-row Triton template that keeps each 128-wide row in registers
across the reduction and epilogue instead of relying on the generic generated
online-softmax schedule. Inductor cannot do this today because it does not
canonicalize this exact max/sum/log, fp16 logsumexp cast, and second fp16
exp epilogue into a multi-row persistent softmax template; the existing generic
single-kernel reduction schedule covers the scope but leaves small-row overhead
on the table. The fix class is NEW_PATTERN: add a specialized small-row
logsumexp-to-exp lowering that preserves the fp16 rounding points and final
view while processing multiple rows per Triton program.
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



REPRO_ID = "amax_sum_66dc4fac757d"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 1 * 12 * 64 * 64
COLS = 128
INPUT_SHAPE = (768, 64, 128)
VIEW_SHAPE = (1, 12, 64, 64, 128)
OUT_SHAPE = INPUT_SHAPE
OUT_STRIDE = (64 * 128, 128, 1)

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
def _logsumexp_exp_rows_kernel(
    x_ptr,
    out_ptr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    cols = tl.arange(0, block_cols)
    mask = (row_offsets[:, None] < 49152) & (cols[None, :] < 128)

    i0 = row_offsets // 64
    i1 = row_offsets - i0 * 64

    x_offsets = i0[:, None] * x_s0 + i1[:, None] * x_s1 + cols[None, :] * x_s2
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=-float("inf"))
    x_f32 = x_vals.to(tl.float32)
    x_f32 = tl.where(mask, x_f32, -float("inf"))

    row_max = tl.max(x_f32, axis=1)
    stable_max = tl.where(tl.abs(row_max) == float("inf"), 0.0, row_max)
    exp_shifted = tl.exp(x_f32 - stable_max[:, None])
    denom = tl.sum(exp_shifted, axis=1)
    logsumexp = tl.log(denom) + stable_max

    # repro.py casts logsumexp to fp16, then performs the final subtract/exp
    # on fp16 tensors. Round those two intermediate values the same way.
    logsumexp_h = logsumexp.to(tl.float16)
    shifted_h = (x_f32 - logsumexp_h[:, None].to(tl.float32)).to(tl.float16)
    out_vals = tl.exp(shifted_h.to(tl.float32))

    out_offsets = i0[:, None] * out_s0 + i1[:, None] * out_s1 + cols[None, :] * out_s2
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


def _validate_shape_param(name: str, actual, expected: tuple[int, ...]) -> None:
    if actual is None:
        return
    got = tuple(int(dim) for dim in actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm_10: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
) -> None:
    if not bmm_10.is_cuda:
        raise RuntimeError("CUDA tensor input is required")
    if bmm_10.dtype != torch.float16:
        raise TypeError(f"expected bmm_10 fp16, got {bmm_10.dtype}")
    if tuple(bmm_10.shape) != INPUT_SHAPE:
        raise ValueError(f"expected bmm_10 shape {INPUT_SHAPE}, got {tuple(bmm_10.shape)}")
    if bmm_10.stride() != OUT_STRIDE:
        raise ValueError(f"expected contiguous input stride {OUT_STRIDE}, got {bmm_10.stride()}")
    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, VIEW_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def _launch_oracle(
    bmm_10: torch.Tensor,
    out: torch.Tensor,
    *,
    block_rows: int,
    block_cols: int,
    num_warps: int,
) -> torch.Tensor:
    if block_rows <= 0 or block_rows & (block_rows - 1):
        raise ValueError(f"block_rows must be a positive power of two, got {block_rows}")
    if block_cols < COLS or block_cols & (block_cols - 1):
        raise ValueError(f"block_cols must be a power of two >= {COLS}, got {block_cols}")
    if out.shape != OUT_SHAPE or out.dtype != torch.float16 or not out.is_cuda:
        raise ValueError("preallocated output must be CUDA fp16 with repro output shape")
    if out.stride() != OUT_STRIDE:
        raise ValueError(f"preallocated output stride mismatch: expected {OUT_STRIDE}, got {out.stride()}")

    grid = (triton.cdiv(ROWS, block_rows),)
    _logsumexp_exp_rows_kernel[grid](
        bmm_10,
        out,
        x_s0=bmm_10.stride(0),
        x_s1=bmm_10.stride(1),
        x_s2=bmm_10.stride(2),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        block_rows=block_rows,
        block_cols=block_cols,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    bmm_10: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
    *,
    block_rows: int = 8,
    block_cols: int | None = None,
    num_warps: int = 4,
) -> torch.Tensor:
    _validate_inputs(bmm_10, _shape_param_0, _shape_param_1, _shape_param_2)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_10.device,
        dtype=torch.float16,
    )
    actual_block_cols = block_cols if block_cols is not None else triton.next_power_of_2(COLS)
    return _launch_oracle(
        bmm_10,
        out,
        block_rows=block_rows,
        block_cols=actual_block_cols,
        num_warps=num_warps,
    )


@oracle_impl(hardware="H100", shapes="(T([768, 64, 128], f16), S([1, 12, 64, 64, 128]), S([1, 12, 64, 64, 128]), S([768, 64, 128]))")
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
