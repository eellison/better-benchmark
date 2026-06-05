"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `sum_sum_sum_56cdc50216ea` Swin layer-norm-backward return tuple by reducing each `[1024]` row for the input-gradient epilogue, writing the returned non-contiguous `[1024, 6272]` transposed side output, and cooperatively accumulating both returned `[1024]` column reductions from the same row-tiled producer, whereas Inductor currently schedules the row reductions, transposed side-output store, and two sibling `sum([0, 1, 2])` column reductions as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reductions, a materialized transposed side output, and sibling column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across row tiles, finalize their partials, and fuse the dependent transposed side store."""
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


REPRO_ID = "sum_sum_sum_56cdc50216ea"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_f98bd35c"

N = 128
H = 7
W = 7
C = 1024
M = N * H * W
INV_C = 1.0 / C

TILE_M = 4
TILE_C = 1024


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
        mm_19,
        primals_334,
        mm_2,
        getitem_163,
        rsqrt_48,
        view_696,
        *_shape_params,
    ) = inputs

    return (
        mm_19.contiguous(),
        primals_334.contiguous(),
        mm_2.contiguous(),
        getitem_163.reshape(M).contiguous(),
        rsqrt_48.reshape(M).contiguous(),
        view_696.reshape(M, C).contiguous(),
    )


@triton.jit
def _row_store_and_partial_reduce_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    residual_ptr,
    partial_sum_x_rhs_ptr,
    partial_sum_x_ptr,
    out_transposed_ptr,
    M_: tl.constexpr,
    C_: tl.constexpr,
    INV_C_: tl.constexpr,
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
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + m, mask=m_mask, other=0.0).to(tl.float32)

    rhs = (y - mean[:, None]) * rstd[:, None]
    weighted = x * weight[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
    grad = residual + rstd[:, None] * INV_C_ * (
        weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None]
    )
    tl.store(out_transposed_ptr + offsets, grad, mask=mask)

    sum_x_rhs = tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    partial_offsets = tile_m * C_ + c
    tl.store(partial_sum_x_rhs_ptr + partial_offsets, sum_x_rhs, mask=c_mask)
    tl.store(partial_sum_x_ptr + partial_offsets, sum_x, mask=c_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_sum_x_rhs_ptr,
    partial_sum_x_ptr,
    out_sum_x_rhs_ptr,
    out_sum_x_ptr,
    NUM_M_TILES: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tile = tl.arange(0, BLOCK_TILES)
    mask = (tile[:, None] < NUM_M_TILES) & (c[None, :] < C_)
    offsets = tile[:, None] * C_ + c[None, :]

    sum_x_rhs = tl.load(partial_sum_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c_mask = c < C_
    tl.store(out_sum_x_rhs_ptr + c, tl.sum(sum_x_rhs, axis=0), mask=c_mask)
    tl.store(out_sum_x_ptr + c, tl.sum(sum_x, axis=0), mask=c_mask)


def oracle_triton_prepared(
    x_mc: torch.Tensor,
    weight_c: torch.Tensor,
    y_mc: torch.Tensor,
    mean_m: torch.Tensor,
    rstd_m: torch.Tensor,
    residual_mc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_mc.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_mc.shape == (M, C)
    assert weight_c.shape == (C,)
    assert y_mc.shape == (M, C)
    assert mean_m.shape == (M,)
    assert rstd_m.shape == (M,)
    assert residual_mc.shape == (M, C)
    assert x_mc.is_contiguous()
    assert weight_c.is_contiguous()
    assert y_mc.is_contiguous()
    assert mean_m.is_contiguous()
    assert rstd_m.is_contiguous()
    assert residual_mc.is_contiguous()

    device = x_mc.device
    num_m_tiles = triton.cdiv(M, TILE_M)
    partial_sum_x_rhs = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    partial_sum_x = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    out_transposed = torch.empty_strided(
        (C, M),
        (1, C),
        device=device,
        dtype=torch.float32,
    )

    _row_store_and_partial_reduce_kernel[(num_m_tiles,)](
        x_mc,
        weight_c,
        y_mc,
        mean_m,
        rstd_m,
        residual_mc,
        partial_sum_x_rhs,
        partial_sum_x,
        out_transposed,
        M_=M,
        C_=C,
        INV_C_=INV_C,
        BLOCK_M=TILE_M,
        BLOCK_C=TILE_C,
        num_warps=8,
    )

    out_sum_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum_x = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(C, 16),)](
        partial_sum_x_rhs,
        partial_sum_x,
        out_sum_x_rhs,
        out_sum_x,
        NUM_M_TILES=num_m_tiles,
        C_=C,
        BLOCK_TILES=triton.next_power_of_2(num_m_tiles),
        BLOCK_C=16,
        num_warps=8,
    )

    return out_sum_x_rhs, out_sum_x, out_transposed


def oracle_triton(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
