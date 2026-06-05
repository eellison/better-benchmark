"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `sum_sum_sum_f2c6042682bc` Longformer GELU/layer-norm-backward return tuple by row-tiling the `[8192, 768]` producer, recomputing the GELU forward and derivative, preserving the returned `[768, 8192]` transpose-view stride, and cooperatively accumulating all three returned `[768]` column reductions from the same row tiles, whereas Inductor currently schedules the GELU pointwise chain, row-local layer-norm reductions, transposed side-output materialization, and sibling `sum([0, 1])`/`sum([0])` reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates row-local reductions, expensive fused pointwise epilogues, a required transposed side output, and multiple sibling column accumulators in one producer; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer mode that fuses GELU/layer-norm-backward epilogues, writes required side outputs with target layout, emits compatible column partials, and finalizes them in one epilogue."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps CPU-only syntax checks usable.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_f2c6042682bc"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_allenailongformerbase_train_004_e38cb0c3"

BATCH = 8
SEQ = 1024
M = BATCH * SEQ
D = 768

ROW_SPLIT = 8
XBLOCK = 1
BLOCK_D = 1024
FINAL_BLOCK_D = 16
FINAL_BLOCK_TILES = 1024
ROW_NUM_WARPS = 8
FINAL_NUM_WARPS = 8

RSQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def make_inputs(device: torch.device) -> tuple[object, ...]:
    module = _load_repro_module()
    moved: list[object] = []
    for value in module.make_inputs():
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_torch(
    mm: torch.Tensor,
    arg1_1: torch.Tensor,
    arg5_1: torch.Tensor,
    arg6_1: torch.Tensor,
    arg7_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = mm.view(_shape_param_0)
    weighted = x * arg1_1
    weighted_row_sum = weighted.sum(dim=2, keepdim=True)

    gelu_x = arg5_1.view(_shape_param_1)
    erf_plus_one = torch.erf(gelu_x * RSQRT2) + 1.0
    gelu = gelu_x * 0.5 * erf_plus_one
    normalized_gelu = (gelu - arg6_1) * arg7_1

    weighted_norm_row_sum = (weighted * normalized_gelu).sum(dim=2, keepdim=True)
    ln_grad = (arg7_1 / D) * (
        weighted * D - weighted_row_sum - normalized_gelu * weighted_norm_row_sum
    )
    gelu_grad = 0.5 * erf_plus_one + gelu_x * torch.exp(gelu_x * gelu_x * -0.5) * NORMAL_PDF_SCALE
    grad = ln_grad * gelu_grad

    out_x_norm = (x * normalized_gelu).sum(dim=(0, 1))
    out_x = x.sum(dim=(0, 1))
    grad_md = grad.view(_shape_param_2)
    out_grad = grad_md.sum(dim=0, keepdim=True).view(_shape_param_3)
    return out_x_norm, out_x, grad_md.t(), out_grad


if triton is not None:

    @triton.jit
    def _row_tile_kernel(
        x_ptr,
        gamma_ptr,
        gelu_x_ptr,
        mean_ptr,
        scale_ptr,
        partial_x_norm_ptr,
        partial_x_ptr,
        partial_grad_ptr,
        out_transposed_ptr,
        M_: tl.constexpr,
        D_: tl.constexpr,
        ROW_SPLIT_: tl.constexpr,
        XBLOCK_: tl.constexpr,
        BLOCK_D_: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        d = tl.arange(0, BLOCK_D_)
        d_mask = d < D_

        gamma = tl.load(gamma_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        acc_x_norm = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        acc_grad = tl.zeros((BLOCK_D_,), dtype=tl.float32)

        for start in tl.range(0, ROW_SPLIT_, XBLOCK_):
            m = row_block * ROW_SPLIT_ + start + tl.arange(0, XBLOCK_)
            m_mask = m < M_
            mask = m_mask[:, None] & d_mask[None, :]
            offsets = m[:, None] * D_ + d[None, :]

            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            gelu_x = tl.load(gelu_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            mean = tl.load(mean_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)

            erf_plus_one = tl.math.erf(gelu_x * 0.7071067811865476) + 1.0
            gelu = gelu_x * 0.5 * erf_plus_one
            normalized_gelu = (gelu - mean[:, None]) * scale[:, None]

            weighted = x * gamma[None, :]
            row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
            row_dot = tl.sum(tl.where(mask, weighted * normalized_gelu, 0.0), axis=1)
            ln_grad = (scale[:, None] / D_) * (
                weighted * D_ - row_sum[:, None] - normalized_gelu * row_dot[:, None]
            )
            gelu_grad = (
                0.5 * erf_plus_one
                + gelu_x * tl.exp(gelu_x * gelu_x * -0.5) * 0.3989422804014327
            )
            grad = ln_grad * gelu_grad

            tl.store(out_transposed_ptr + offsets, grad, mask=mask)
            acc_x_norm += tl.sum(tl.where(mask, x * normalized_gelu, 0.0), axis=0)
            acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
            acc_grad += tl.sum(tl.where(mask, grad, 0.0), axis=0)

        partial_offsets = row_block * D_ + d
        tl.store(partial_x_norm_ptr + partial_offsets, acc_x_norm, mask=d_mask)
        tl.store(partial_x_ptr + partial_offsets, acc_x, mask=d_mask)
        tl.store(partial_grad_ptr + partial_offsets, acc_grad, mask=d_mask)


    @triton.jit
    def _finalize_column_sums_kernel(
        partial_x_norm_ptr,
        partial_x_ptr,
        partial_grad_ptr,
        out_x_norm_ptr,
        out_x_ptr,
        out_grad_ptr,
        NUM_M_TILES: tl.constexpr,
        D_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
        BLOCK_D_: tl.constexpr,
    ):
        d = tl.program_id(0) * BLOCK_D_ + tl.arange(0, BLOCK_D_)
        d_mask = d < D_
        tiles = tl.arange(0, BLOCK_TILES_)

        acc_x_norm = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        acc_grad = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        for tile_base in range(0, NUM_M_TILES, BLOCK_TILES_):
            tile_ids = tile_base + tiles
            mask = (tile_ids[:, None] < NUM_M_TILES) & d_mask[None, :]
            offsets = tile_ids[:, None] * D_ + d[None, :]
            acc_x_norm += tl.sum(
                tl.load(partial_x_norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
                axis=0,
            )
            acc_x += tl.sum(
                tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
                axis=0,
            )
            acc_grad += tl.sum(
                tl.load(partial_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
                axis=0,
            )

        tl.store(out_x_norm_ptr + d, acc_x_norm, mask=d_mask)
        tl.store(out_x_ptr + d, acc_x, mask=d_mask)
        tl.store(out_grad_ptr + d, acc_grad, mask=d_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm,
        arg1_1,
        arg5_1,
        arg6_1,
        arg7_1,
        *_shape_params,
    ) = inputs

    return (
        mm.reshape(M, D).contiguous(),
        arg1_1.contiguous(),
        arg5_1.reshape(M, D).contiguous(),
        arg6_1.reshape(M).contiguous(),
        arg7_1.reshape(M).contiguous(),
    )


def oracle_triton_prepared(
    x_md: torch.Tensor,
    gamma_d: torch.Tensor,
    gelu_x_md: torch.Tensor,
    mean_m: torch.Tensor,
    scale_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if x_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_md.shape == (M, D)
    assert gamma_d.shape == (D,)
    assert gelu_x_md.shape == (M, D)
    assert mean_m.shape == (M,)
    assert scale_m.shape == (M,)
    assert x_md.is_contiguous()
    assert gamma_d.is_contiguous()
    assert gelu_x_md.is_contiguous()
    assert mean_m.is_contiguous()
    assert scale_m.is_contiguous()

    device = x_md.device
    num_m_tiles = triton.cdiv(M, ROW_SPLIT)
    partial_x_norm = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_grad = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    out_transposed = torch.empty_strided((D, M), (1, D), device=device, dtype=torch.float32)

    _row_tile_kernel[(num_m_tiles,)](
        x_md,
        gamma_d,
        gelu_x_md,
        mean_m,
        scale_m,
        partial_x_norm,
        partial_x,
        partial_grad,
        out_transposed,
        M_=M,
        D_=D,
        ROW_SPLIT_=ROW_SPLIT,
        XBLOCK_=XBLOCK,
        BLOCK_D_=BLOCK_D,
        num_warps=ROW_NUM_WARPS,
    )

    out_x_norm = torch.empty((D,), device=device, dtype=torch.float32)
    out_x = torch.empty((D,), device=device, dtype=torch.float32)
    out_grad = torch.empty((D,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partial_x_norm,
        partial_x,
        partial_grad,
        out_x_norm,
        out_x,
        out_grad,
        NUM_M_TILES=num_m_tiles,
        D_=D,
        BLOCK_TILES_=FINAL_BLOCK_TILES,
        BLOCK_D_=FINAL_BLOCK_D,
        num_warps=FINAL_NUM_WARPS,
    )

    return out_x_norm, out_x, out_transposed, out_grad


def oracle_triton(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def oracle_full(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    if impl == "auto":
        impl = "triton" if first_tensor.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


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
