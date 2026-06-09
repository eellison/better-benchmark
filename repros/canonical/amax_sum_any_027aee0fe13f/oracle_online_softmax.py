"""Full-scope Triton oracle for amax_sum_any_027aee0fe13f.

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the
complete BlenderBot masked attention-softmax region from repro.py, including
the bool mask select between the scalar f32 inputs full_1/full, the view of
the [512, 128, 128] scores to [16, 32, 128, 128], score plus mask-bias add,
the any(eq(-inf)) all-masked-row guard, stable last-dimension softmax, zero
fill for all--inf rows, expand, and final contiguous [512, 128, 128] view. It
differs from Inductor by consuming the score, mask, and scalar fill inputs
directly in one shape-specialized Triton row-softmax kernel that reuses the
broadcast mask across a block of heads, derives the all--inf predicate from
the row max, and reuses the numerator exponentials for both the denominator
and output. Inductor cannot do this today because its persistent softmax
lowering preserves the explicit any(eq(-inf)) reduction and emits separate
exp computations for sum and normalization even when the full reduction fits
in one tile. The fixing Inductor change is an algebraic/codegen simplification
for small-rnumel masked softmax: eliminate the redundant any reduction via the
max predicate and CSE/register-reuse the numerator exponential, preferably
with the existing fast-math exp2 path.
"""
from __future__ import annotations

import argparse
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


REPRO_ID = "amax_sum_any_027aee0fe13f"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
HISTORICAL_BEST_COMPILE_US = 25.567999109625816
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"

BATCH = 16
HEADS = 32
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
def _scalar_masked_softmax_heads_kernel(
    bmm_ptr,
    mask_ptr,
    keep_value_ptr,
    drop_value_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    mask_s0: tl.constexpr,
    mask_s2: tl.constexpr,
    mask_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    batch_q = tl.program_id(0)
    head_block = tl.program_id(1)
    batch = batch_q // q_len
    q = batch_q - batch * q_len

    h = head_block * block_h + tl.arange(0, block_h)
    cols = tl.arange(0, block_k)
    h_mask = h < heads
    col_mask = cols < k_len
    elem_mask = h_mask[:, None] & col_mask[None, :]

    keep_value = tl.load(keep_value_ptr).to(tl.float32)
    drop_value = tl.load(drop_value_ptr).to(tl.float32)
    mask_offsets = batch * mask_s0 + q * mask_s2 + cols * mask_s3
    keep_cols = tl.load(mask_ptr + mask_offsets, mask=col_mask, other=0)
    bias = tl.where(keep_cols[None, :], keep_value, drop_value)

    flat_bh = batch * heads + h
    bmm_offsets = (
        flat_bh[:, None] * bmm_s0
        + q * bmm_s1
        + cols[None, :] * bmm_s2
    )
    bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    scores = tl.where(elem_mask, bmm_vals + bias, -float("inf"))

    row_max = tl.max(scores, axis=1)
    has_any = row_max != -float("inf")
    stable_max = tl.where(has_any, row_max, 0.0)
    numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
    numer = tl.where(elem_mask & (scores != -float("inf")), numer, 0.0)
    denom = tl.sum(numer, axis=1)
    safe_denom = tl.where(has_any, denom, 1.0)
    out_vals = tl.where(has_any[:, None], numer / safe_denom[:, None], 0.0)

    out_offsets = (
        flat_bh[:, None] * out_s0
        + q * out_s1
        + cols[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, out_vals, mask=elem_mask)


@triton.jit
def _scalar_masked_softmax_rows_kernel(
    bmm_ptr,
    mask_ptr,
    keep_value_ptr,
    drop_value_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    mask_s0: tl.constexpr,
    mask_s2: tl.constexpr,
    mask_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
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
    elem_mask = row_mask[:, None] & col_mask[None, :]

    keep_value = tl.load(keep_value_ptr).to(tl.float32)
    drop_value = tl.load(drop_value_ptr).to(tl.float32)
    mask_offsets = batch[:, None] * mask_s0 + q[:, None] * mask_s2 + cols[None, :] * mask_s3
    keep = tl.load(mask_ptr + mask_offsets, mask=elem_mask, other=0)
    bias = tl.where(keep, keep_value, drop_value)

    bmm_offsets = (
        flat_bh[:, None] * bmm_s0
        + q[:, None] * bmm_s1
        + cols[None, :] * bmm_s2
    )
    bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    scores = tl.where(elem_mask, bmm_vals + bias, -float("inf"))

    row_max = tl.max(scores, axis=1)
    has_any = row_max != -float("inf")
    stable_max = tl.where(has_any, row_max, 0.0)
    numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
    numer = tl.where(elem_mask & (scores != -float("inf")), numer, 0.0)
    denom = tl.sum(numer, axis=1)
    safe_denom = tl.where(has_any, denom, 1.0)
    out_vals = tl.where(has_any[:, None], numer / safe_denom[:, None], 0.0)

    out_offsets = (
        flat_bh[:, None] * out_s0
        + q[:, None] * out_s1
        + cols[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, out_vals, mask=elem_mask)


def _load_repro_module() -> Any:
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


def _make_inputs(module: Any, seed: int) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return tuple(module.make_inputs())


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    actual_tuple = tuple(int(dim) for dim in actual)
    if actual_tuple != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual_tuple}")


def _validate_inputs(
    arg21_1: torch.Tensor,
    full_1: torch.Tensor,
    full: torch.Tensor,
    bmm: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
) -> None:
    if not (arg21_1.is_cuda and full_1.is_cuda and full.is_cuda and bmm.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if arg21_1.dtype != torch.bool or bmm.dtype != torch.float32:
        raise TypeError(f"expected bool mask and fp32 scores, got {arg21_1.dtype} and {bmm.dtype}")
    if full_1.dtype != torch.float32 or full.dtype != torch.float32:
        raise TypeError(f"expected fp32 scalar fills, got {full_1.dtype} and {full.dtype}")
    if tuple(arg21_1.shape) != MASK_SHAPE:
        raise ValueError(f"unexpected arg21_1 shape: {tuple(arg21_1.shape)}")
    if tuple(full_1.shape) != () or tuple(full.shape) != ():
        raise ValueError(f"expected scalar full_1/full, got {tuple(full_1.shape)} and {tuple(full.shape)}")
    if tuple(bmm.shape) != BMM_SHAPE:
        raise ValueError(f"unexpected bmm shape: {tuple(bmm.shape)}")
    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, VIEW_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def _validate_launch(
    out: torch.Tensor,
    *,
    block_h: int,
    block_m: int,
    block_k: int,
    tile_heads: bool,
) -> None:
    if out.shape != OUT_SHAPE or out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError(f"output must be CUDA fp32 with shape {OUT_SHAPE}")
    if out.stride() != OUT_STRIDE:
        raise ValueError(f"output stride must be {OUT_STRIDE}, got {tuple(out.stride())}")
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")
    if tile_heads:
        if block_h <= 0 or block_h & (block_h - 1):
            raise ValueError(f"block_h must be a positive power of two, got {block_h}")
    elif block_m <= 0 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a positive power of two, got {block_m}")


def _launch_oracle(
    arg21_1: torch.Tensor,
    full_1: torch.Tensor,
    full: torch.Tensor,
    bmm: torch.Tensor,
    out: torch.Tensor,
    *,
    block_h: int,
    block_m: int,
    block_k: int,
    num_warps: int,
    tile_heads: bool,
) -> torch.Tensor:
    _validate_launch(
        out,
        block_h=block_h,
        block_m=block_m,
        block_k=block_k,
        tile_heads=tile_heads,
    )
    if tile_heads:
        _scalar_masked_softmax_heads_kernel[(BATCH * Q_LEN, triton.cdiv(HEADS, block_h))](
            bmm,
            arg21_1,
            full_1,
            full,
            out,
            bmm_s0=bmm.stride(0),
            bmm_s1=bmm.stride(1),
            bmm_s2=bmm.stride(2),
            mask_s0=arg21_1.stride(0),
            mask_s2=arg21_1.stride(2),
            mask_s3=arg21_1.stride(3),
            out_s0=out.stride(0),
            out_s1=out.stride(1),
            out_s2=out.stride(2),
            heads=HEADS,
            q_len=Q_LEN,
            k_len=K_LEN,
            block_h=block_h,
            block_k=block_k,
            num_warps=num_warps,
        )
        return out

    _scalar_masked_softmax_rows_kernel[(triton.cdiv(N_ROWS, block_m),)](
        bmm,
        arg21_1,
        full_1,
        full,
        out,
        bmm_s0=bmm.stride(0),
        bmm_s1=bmm.stride(1),
        bmm_s2=bmm.stride(2),
        mask_s0=arg21_1.stride(0),
        mask_s2=arg21_1.stride(2),
        mask_s3=arg21_1.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        n_rows=N_ROWS,
        heads=HEADS,
        q_len=Q_LEN,
        k_len=K_LEN,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([16, 1, 128, 128], b8), T([], f32), T([], f32), T([512, 128, 128], f32), S([16, 32, 128, 128]), S([16, 32, 128, 128]), S([512, 128, 128]))")
def oracle_forward(
    inputs: tuple[Any, ...],
    *,
    block_h: int = 8,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 2,
    tile_heads: bool = True,
) -> torch.Tensor:
    """Compute exactly Repro()(*inputs): same inputs, one fp32 contiguous output."""
    arg21_1, full_1, full, bmm, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    _validate_inputs(arg21_1, full_1, full, bmm, _shape_param_0, _shape_param_1, _shape_param_2)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm.device,
        dtype=torch.float32,
    )
    return _launch_oracle(
        arg21_1,
        full_1,
        full,
        bmm,
        out,
        block_h=block_h,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        tile_heads=tile_heads,
    )


def _clone_with_edge_cases(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    arg21_1, full_1, full, bmm, *shape_params = inputs
    mask_clone = arg21_1.clone()
    keep_clone = torch.empty_like(full_1)
    drop_clone = torch.empty_like(full)
    bmm_clone = bmm.clone()

    keep_clone.fill_(0.0)
    drop_clone.fill_(-float("inf"))
    mask_clone[0, 0, 0, :] = False
    mask_clone[0, 0, 1, :] = True
    mask_clone[0, 0, 1, 0:4] = False
    mask_clone[0, 0, 2, :] = True
    bmm_clone[0, 2, :] = -float("inf")
    mask_clone[0, 0, 3, :] = True
    bmm_clone[0, 3, 0:4] = -float("inf")
    return (mask_clone, keep_clone, drop_clone, bmm_clone, *shape_params)


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
