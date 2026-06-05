"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `Repro.forward` Swin MLP/layer-norm-backward return tuple by doing the gamma-weighted row reductions, dropout-scaled gradient side output, both input column reductions, and final gradient column reduction from one row-tiled producer while preserving the returned non-contiguous `[1024, 6272]` stride, whereas Inductor currently schedules the row-local reductions, dropout/layout epilogue, transposed side-output store, and sibling `sum([0, 1])`/`sum([0])` column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that coordinates row-local reductions, a full-tensor transposed side output, broadcast dropout scaling, and multiple sibling column accumulators from the same producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across row tiles, fuse the dependent dropout/layout side store, and finalize the partial column accumulators in a dedicated epilogue."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
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


REPRO_ID = "sum_sum_sum_3814c418da03"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_690e88a9"

BATCH = 128
WINDOW_ELEMS = 49
M = BATCH * WINDOW_ELEMS
C = 1024
KEEP_PROB = 0.9043478220701218

TILE_M = 4
TILE_C = 1024
FINAL_TILE_C = 16


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda() if isinstance(value, torch.Tensor) else value
        for value in module.make_inputs()
    )


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_15,
        primals_342,
        mul_230,
        div_74,
        view_689,
        lt_42,
        *_shape_params,
    ) = inputs

    return (
        mm_15.reshape(M, C).contiguous(),
        primals_342.contiguous(),
        mul_230.reshape(M, C).contiguous(),
        div_74.reshape(M).contiguous(),
        view_689.reshape(M, C).contiguous(),
        lt_42.reshape(BATCH).contiguous(),
    )


@triton.jit
def _row_store_and_partial_reduce_kernel(
    x_ptr,
    gamma_ptr,
    rhs_ptr,
    row_scale_ptr,
    residual_ptr,
    keep_ptr,
    partial_sum_x_rhs_ptr,
    partial_sum_x_ptr,
    partial_sum_grad_ptr,
    out_transposed_ptr,
    M_: tl.constexpr,
    C_: tl.constexpr,
    WINDOW_ELEMS_: tl.constexpr,
    KEEP_PROB_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile_m = tl.program_id(0)
    m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_C)
    m_mask = m < M_
    c_mask = c < C_
    mask = m_mask[:, None] & c_mask[None, :]
    offsets = m[:, None] * C_ + c[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)

    weighted = x * gamma[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
    grad = row_scale[:, None] * (
        weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None]
    )

    batch = m // WINDOW_ELEMS_
    keep = tl.load(keep_ptr + batch, mask=m_mask, other=0).to(tl.float32)
    grad = (residual + grad) * (keep[:, None] / KEEP_PROB_)
    tl.store(out_transposed_ptr + offsets, grad, mask=mask)

    sum_x_rhs = tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    sum_grad = tl.sum(tl.where(mask, grad, 0.0), axis=0)
    partial_offsets = tile_m * C_ + c
    tl.store(partial_sum_x_rhs_ptr + partial_offsets, sum_x_rhs, mask=c_mask)
    tl.store(partial_sum_x_ptr + partial_offsets, sum_x, mask=c_mask)
    tl.store(partial_sum_grad_ptr + partial_offsets, sum_grad, mask=c_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_sum_x_rhs_ptr,
    partial_sum_x_ptr,
    partial_sum_grad_ptr,
    out_sum_x_rhs_ptr,
    out_sum_x_ptr,
    out_sum_grad_ptr,
    NUM_M_TILES: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tile = tl.arange(0, BLOCK_TILES)
    mask = (tile[:, None] < NUM_M_TILES) & (c[None, :] < C_)
    offsets = tile[:, None] * C_ + c[None, :]

    sum_x_rhs = tl.load(
        partial_sum_x_rhs_ptr + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    sum_x = tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_grad = tl.load(
        partial_sum_grad_ptr + offsets, mask=mask, other=0.0
    ).to(tl.float32)

    c_mask = c < C_
    tl.store(out_sum_x_rhs_ptr + c, tl.sum(sum_x_rhs, axis=0), mask=c_mask)
    tl.store(out_sum_x_ptr + c, tl.sum(sum_x, axis=0), mask=c_mask)
    tl.store(out_sum_grad_ptr + c, tl.sum(sum_grad, axis=0), mask=c_mask)


def oracle_triton_prepared(
    x_mc: torch.Tensor,
    gamma_c: torch.Tensor,
    rhs_mc: torch.Tensor,
    row_scale_m: torch.Tensor,
    residual_mc: torch.Tensor,
    keep_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_mc.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_mc.shape == (M, C)
    assert gamma_c.shape == (C,)
    assert rhs_mc.shape == (M, C)
    assert row_scale_m.shape == (M,)
    assert residual_mc.shape == (M, C)
    assert keep_b.shape == (BATCH,)
    assert x_mc.is_contiguous()
    assert gamma_c.is_contiguous()
    assert rhs_mc.is_contiguous()
    assert row_scale_m.is_contiguous()
    assert residual_mc.is_contiguous()
    assert keep_b.is_contiguous()

    device = x_mc.device
    num_m_tiles = triton.cdiv(M, TILE_M)
    partial_sum_x_rhs = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    partial_sum_x = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    partial_sum_grad = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    out_transposed = torch.empty_strided(
        (C, M),
        (1, C),
        device=device,
        dtype=torch.float32,
    )

    _row_store_and_partial_reduce_kernel[(num_m_tiles,)](
        x_mc,
        gamma_c,
        rhs_mc,
        row_scale_m,
        residual_mc,
        keep_b,
        partial_sum_x_rhs,
        partial_sum_x,
        partial_sum_grad,
        out_transposed,
        M_=M,
        C_=C,
        WINDOW_ELEMS_=WINDOW_ELEMS,
        KEEP_PROB_=KEEP_PROB,
        BLOCK_M=TILE_M,
        BLOCK_C=TILE_C,
        num_warps=8,
    )

    out_sum_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum_grad = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(C, FINAL_TILE_C),)](
        partial_sum_x_rhs,
        partial_sum_x,
        partial_sum_grad,
        out_sum_x_rhs,
        out_sum_x,
        out_sum_grad,
        NUM_M_TILES=num_m_tiles,
        C_=C,
        BLOCK_TILES=triton.next_power_of_2(num_m_tiles),
        BLOCK_C=FINAL_TILE_C,
        num_warps=8,
    )

    return out_sum_x_rhs, out_sum_x, out_transposed, out_sum_grad


def oracle_triton(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton(*inputs)


def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return oracle_full(*inputs)


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
