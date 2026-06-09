"""Full-scope Triton diagnostic oracle for any_amax_sum_727a4ae37fb6.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle computes the
complete DistilBERT masked softmax region from repro.py, including the logical
view of bmm_10 to [256, 12, 128, 128], the broadcast bool mask to 0/-inf, the
any(eq(-inf)) all-masked-row guard, stable last-dimension softmax, zero fill
for all--inf rows, expand, and final contiguous [3072, 128, 128] view. It
differs from Inductor by using the original stride-zero mask semantics directly
inside one row-softmax Triton kernel and masking score loads for rows that the
broadcast mask turns into all-zero output. Inductor already recognizes this
decomposed graph as a prepare_softmax_online persistent reduction and emits one
full-scope fused kernel, so the standalone masked-load specialization does not
expose a true lower floor. There is no actionable Inductor fix for this repro
beyond preserving the existing online-softmax/persistent-reduction lowering and
tuning; the row is diagnosis-only when the historical-best gate is applied.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

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


REPRO_ID = "any_amax_sum_727a4ae37fb6"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
QUEUE_PATH = REPO_ROOT / "investigation_results" / "oracle_gap_closure_queue.csv"
HISTORICAL_BEST_COMPILE_US = 73.37599992752075

BATCH = 256
HEADS = 12
Q_LEN = 128
K_LEN = 128
BH = BATCH * HEADS
N_ROWS = BH * Q_LEN
BMM_SHAPE = (BH, Q_LEN, K_LEN)
MASK_SHAPE = (BATCH, 1, Q_LEN, K_LEN)
VIEW_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
OUT_SHAPE = BMM_SHAPE
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)

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
def _broadcast_masked_softmax_kernel(
    bmm_ptr,
    mask_ptr,
    out_ptr,
    mask_s0: tl.constexpr,
    mask_s1: tl.constexpr,
    mask_s2: tl.constexpr,
    mask_s3: tl.constexpr,
    n_rows: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    flat_bh = rows // q_len
    batch = flat_bh // heads
    q = rows - flat_bh * q_len

    cols = tl.arange(0, block_k)
    row_mask = rows < n_rows
    col_mask = cols < k_len

    mask_offsets = batch * mask_s0 + q * mask_s2
    keep_row = tl.load(mask_ptr + mask_offsets, mask=row_mask, other=0)
    load_mask = row_mask[:, None] & keep_row[:, None] & col_mask[None, :]

    offsets = rows[:, None] * k_len + cols[None, :]
    scores = tl.load(bmm_ptr + offsets, mask=load_mask, other=-float("inf")).to(tl.float32)
    scores = tl.where(load_mask, scores, -float("inf"))

    finite = (scores != -float("inf")) & load_mask
    has_any = tl.sum(tl.where(finite, 1, 0), axis=1) > 0

    row_max = tl.max(scores, axis=1)
    stable_max = tl.where(has_any, row_max, 0.0)
    numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
    numer = tl.where(load_mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    out_vals = tl.where(has_any[:, None], numer / denom[:, None], 0.0)

    store_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(out_ptr + offsets, out_vals, mask=store_mask)


def _load_repro_module():
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs() -> tuple[Any, ...]:
    module = _load_repro_module()
    return tuple(module.make_inputs())


def get_repro_instance() -> torch.nn.Module:
    module = _load_repro_module()
    return module.Repro()


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _validate_shape_param(name: str, actual, expected: tuple[int, ...]) -> None:
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    bmm_10: torch.Tensor,
    expand_2: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> None:
    if not (bmm_10.is_cuda and expand_2.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if bmm_10.dtype != torch.float32 or expand_2.dtype != torch.bool:
        raise TypeError(f"expected f32 scores and bool mask, got {bmm_10.dtype} and {expand_2.dtype}")
    if tuple(bmm_10.shape) != BMM_SHAPE:
        raise ValueError(f"unexpected bmm_10 shape: {tuple(bmm_10.shape)}")
    if tuple(expand_2.shape) != MASK_SHAPE:
        raise ValueError(f"unexpected expand_2 shape: {tuple(expand_2.shape)}")
    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, VIEW_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def _validate_launch(out: torch.Tensor, block_m: int, block_k: int) -> None:
    if out.shape != OUT_SHAPE or out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError(f"output must be CUDA fp32 with shape {OUT_SHAPE}")
    if out.stride() != OUT_STRIDE:
        raise ValueError(f"output stride must be {OUT_STRIDE}, got {tuple(out.stride())}")
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a positive power of two, got {block_m}")
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")


def _launch_oracle(
    bmm_10: torch.Tensor,
    expand_2: torch.Tensor,
    out: torch.Tensor,
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    _validate_launch(out, block_m, block_k)
    _broadcast_masked_softmax_kernel[(triton.cdiv(N_ROWS, block_m),)](
        bmm_10,
        expand_2,
        out,
        mask_s0=expand_2.stride(0),
        mask_s1=expand_2.stride(1),
        mask_s2=expand_2.stride(2),
        mask_s3=expand_2.stride(3),
        n_rows=N_ROWS,
        heads=HEADS,
        q_len=Q_LEN,
        k_len=K_LEN,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([3072, 128, 128], f32), T([256, 1, 128, 128], b8, stride=(0, 128, 1, 0)), S([256, 12, 128, 128]), S([256, 12, 128, 128]), S([3072, 128, 128]))")
def oracle_forward(
    inputs: tuple[Any, ...],
    *,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 4,
) -> torch.Tensor:
    """Compute exactly Repro()(*inputs): same inputs, one fp32 strided output."""
    bmm_10, expand_2, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    _validate_inputs(bmm_10, expand_2, _shape_param_0, _shape_param_1, _shape_param_2)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_10.device,
        dtype=torch.float32,
    )
    return _launch_oracle(
        bmm_10,
        expand_2,
        out,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )


def _mask_storage_view(mask: torch.Tensor) -> torch.Tensor:
    base = torch.empty((Q_LEN,), dtype=torch.bool, device=mask.device)
    base.copy_(mask[0, 0, :, 0])
    return base.as_strided(MASK_SHAPE, tuple(mask.stride()))


def _clone_with_edge_cases(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    bmm_10, expand_2, *shape_params = inputs
    bmm_clone = bmm_10.clone()
    mask_clone = _mask_storage_view(expand_2)

    mask_clone[0, 0, 0, 0] = False
    mask_clone[0, 0, 1, 0] = True
    mask_clone[0, 0, 2, 0] = True
    bmm_clone[0, 1, :] = -float("inf")
    bmm_clone[0, 2, 0:4] = -float("inf")
    return (bmm_clone, mask_clone, *shape_params)


def main() -> None:
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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
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
