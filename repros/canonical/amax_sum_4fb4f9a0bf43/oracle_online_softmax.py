"""
Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle covers the full T5 additive-score softmax materialization from repro.py, including the fp16 `[H,2048,2048] + [1,H,2048,2048]` add rounded through fp16, fp32 last-dimension softmax, fp16 cast, and contiguous `[H,2048,2048]` output view, using one Triton row program per query row instead of Inductor's generated softmax schedule. Inductor can already keep this default shape in a single compiled kernel under the required combo-looped configuration, so the remaining difference is a hand-written row template rather than a missing fusion across surrounding ops; Inductor cannot do materially better through the current scheduler-fusion classes because the required two fp16 reads, one fp16 write, and 2048-wide exp/reduction dominate this repro. The fix class is BANDWIDTH_BOUND: treat this artifact as diagnosis-only unless the measured Triton row kernel beats both required compile configs on the same full scope.
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



REPRO_ID = "amax_sum_4fb4f9a0bf43"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

DEFAULT_COLS = 2048

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
def _add_softmax_rows_kernel(
    bmm_ptr,
    bias_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    bias_s1: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    head = row // q_len
    query = row - head * q_len
    cols = tl.arange(0, block_k)
    mask = cols < k_len

    bmm_offsets = head * bmm_s0 + query * bmm_s1 + cols * bmm_s2
    bias_offsets = head * bias_s1 + query * bias_s2 + cols * bias_s3

    bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0)
    bias_vals = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)

    # repro.py performs aten.add on fp16 tensors before converting to fp32.
    scores = (bmm_vals + bias_vals).to(tl.float16).to(tl.float32)
    scores = tl.where(mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=0)
    numer = tl.exp2((scores - row_max) * 1.4426950408889634)
    denom = tl.sum(numer, axis=0)
    probs = numer / denom

    out_offsets = head * out_s0 + query * out_s1 + cols * out_s2
    tl.store(out_ptr + out_offsets, probs, mask=mask)


def _output_shape(
    bmm_34: torch.Tensor,
    _shape_param_2,
) -> tuple[int, int, int]:
    if _shape_param_2 is None:
        return tuple(bmm_34.shape)
    return tuple(int(dim) for dim in _shape_param_2)


def _validate_inputs(
    bmm_34: torch.Tensor,
    add_48: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[int, int, int]:
    if not bmm_34.is_cuda or not add_48.is_cuda:
        raise RuntimeError("CUDA tensors are required")
    if bmm_34.dtype != torch.float16 or add_48.dtype != torch.float16:
        raise TypeError(f"expected fp16 inputs, got {bmm_34.dtype} and {add_48.dtype}")
    if bmm_34.ndim != 3 or add_48.ndim != 4:
        raise ValueError(f"expected ranks 3 and 4, got {bmm_34.ndim} and {add_48.ndim}")

    heads, q_len, k_len = (int(dim) for dim in bmm_34.shape)
    expected_4d = (1, heads, q_len, k_len)
    if tuple(add_48.shape) != expected_4d:
        raise ValueError(f"expected add_48 shape {expected_4d}, got {tuple(add_48.shape)}")

    if _shape_param_0 is not None and tuple(_shape_param_0) != expected_4d:
        raise ValueError(f"_shape_param_0 mismatch: {tuple(_shape_param_0)} vs {expected_4d}")
    if _shape_param_1 is not None and tuple(_shape_param_1) != expected_4d:
        raise ValueError(f"_shape_param_1 mismatch: {tuple(_shape_param_1)} vs {expected_4d}")

    out_shape = _output_shape(bmm_34, _shape_param_2)
    if out_shape != (heads, q_len, k_len):
        raise ValueError(f"expected output view {(heads, q_len, k_len)}, got {out_shape}")
    return heads, q_len, k_len


def _contiguous_3d_stride(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return (shape[1] * shape[2], shape[2], 1)


def _launch_oracle(
    bmm_34: torch.Tensor,
    add_48: torch.Tensor,
    out: torch.Tensor,
    *,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    heads, q_len, k_len = (int(dim) for dim in bmm_34.shape)
    if block_k < k_len or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {k_len}, got {block_k}")
    if out.shape != bmm_34.shape or out.dtype != torch.float16 or not out.is_cuda:
        raise ValueError("preallocated output must be CUDA fp16 with bmm_34 shape")

    _add_softmax_rows_kernel[(heads * q_len,)](
        bmm_34,
        add_48,
        out,
        bmm_s0=bmm_34.stride(0),
        bmm_s1=bmm_34.stride(1),
        bmm_s2=bmm_34.stride(2),
        bias_s1=add_48.stride(1),
        bias_s2=add_48.stride(2),
        bias_s3=add_48.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        q_len=q_len,
        k_len=k_len,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    bmm_34: torch.Tensor,
    add_48: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
    *,
    block_k: int | None = None,
    num_warps: int = 8,
) -> torch.Tensor:
    heads, q_len, k_len = _validate_inputs(
        bmm_34,
        add_48,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )
    out_shape = (heads, q_len, k_len)
    out = torch.empty_strided(
        out_shape,
        _contiguous_3d_stride(out_shape),
        device=bmm_34.device,
        dtype=torch.float16,
    )
    actual_block_k = block_k if block_k is not None else triton.next_power_of_2(k_len)
    return _launch_oracle(
        bmm_34,
        add_48,
        out,
        block_k=actual_block_k,
        num_warps=num_warps,
    )


@oracle_impl(hardware="H100", shapes="(T([8, 2048, 2048], f16), T([1, 8, 2048, 2048], f16), S([1, 8, 2048, 2048]), S([1, 8, 2048, 2048]), S([8, 2048, 2048]))")
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
