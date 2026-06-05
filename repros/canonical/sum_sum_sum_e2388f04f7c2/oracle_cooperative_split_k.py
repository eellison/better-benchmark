"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Google FNet layer-norm/dropout-backward return tuple by row-tiling the selected-real `[32,512,768]` producer, preserving the returned `[768,16384]` transpose-view stride, and cooperatively accumulating the two pre-mask `[768]` column sums plus the dependent masked-gradient `[768]` column sum from the same row tiles, whereas Inductor currently schedules the select/add producer, row-local layer-norm reductions, mask conversion/multiply, transpose side output, and sibling `sum([0, 1])`/`sum([0])` reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that combines a selected-view producer, row-local reduction scalars, a required transposed side-output store, and multiple column accumulators in one coordinated producer/finalizer; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer mode that reads selected-view inputs directly, writes required side outputs while emitting column partials, and finalizes compatible layer-norm-backward reductions together."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps CPU-only syntax checks usable.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_e2388f04f7c2"

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
SHAPE_LABEL = "hf_GoogleFnet_train_001_22bbd240"

BATCH = 32
SEQ = 512
M = BATCH * SEQ
D = 768

ROW_SPLIT = 29
XBLOCK = 1
BLOCK_D = 1024
FINAL_BLOCK_D = 16
FINAL_BLOCK_TILES = 1024
ROW_NUM_WARPS = 8
FINAL_NUM_WARPS = 8


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


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda() if isinstance(value, torch.Tensor) and torch.cuda.is_available() else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    model = module.Repro().to(first_tensor.device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_torch(
    view_as_real_10: torch.Tensor,
    mul_306: torch.Tensor,
    arg9_1: torch.Tensor,
    arg65_1: torch.Tensor,
    arg163_1: torch.Tensor,
    arg64_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    add = mul_306 + view_as_real_10.select(3, 0)
    weighted = add * arg9_1
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * arg65_1).sum(dim=2, keepdim=True)
    grad = arg163_1 * (weighted * D - row_sum - arg65_1 * row_dot)
    grad = grad * arg64_1.to(torch.float32)

    out_add_rhs = (add * arg65_1).sum(dim=(0, 1))
    out_add = add.sum(dim=(0, 1))
    grad_md = grad.view(_shape_param_0)
    out_grad_sum = grad_md.sum(dim=0, keepdim=True).view(_shape_param_1)
    return out_add_rhs, out_add, grad_md.permute(1, 0), out_grad_sum


if triton is not None:

    @triton.jit
    def _row_tile_kernel(
        real_pair_ptr,
        residual_ptr,
        gamma_ptr,
        rhs_ptr,
        row_scale_ptr,
        keep_ptr,
        partial_add_rhs_ptr,
        partial_add_ptr,
        partial_grad_ptr,
        grad_md_ptr,
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
        acc_add_rhs = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        acc_add = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        acc_grad = tl.zeros((BLOCK_D_,), dtype=tl.float32)

        for start in tl.range(0, ROW_SPLIT_, XBLOCK_):
            m = row_block * ROW_SPLIT_ + start + tl.arange(0, XBLOCK_)
            m_mask = m < M_
            mask = m_mask[:, None] & d_mask[None, :]
            offsets = m[:, None] * D_ + d[None, :]
            real_offsets = offsets * 2

            selected_real = tl.load(
                real_pair_ptr + real_offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            residual = tl.load(
                residual_ptr + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            row_scale = tl.load(
                row_scale_ptr + m,
                mask=m_mask,
                other=0.0,
            ).to(tl.float32)
            keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)

            add = residual + selected_real
            weighted = add * gamma[None, :]
            row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
            row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
            grad = row_scale[:, None] * (
                weighted * D_ - row_sum[:, None] - rhs * row_dot[:, None]
            )
            grad = grad * keep

            tl.store(grad_md_ptr + offsets, grad, mask=mask)
            acc_add_rhs += tl.sum(tl.where(mask, add * rhs, 0.0), axis=0)
            acc_add += tl.sum(tl.where(mask, add, 0.0), axis=0)
            acc_grad += tl.sum(tl.where(mask, grad, 0.0), axis=0)

        partial_offsets = row_block * D_ + d
        tl.store(partial_add_rhs_ptr + partial_offsets, acc_add_rhs, mask=d_mask)
        tl.store(partial_add_ptr + partial_offsets, acc_add, mask=d_mask)
        tl.store(partial_grad_ptr + partial_offsets, acc_grad, mask=d_mask)


    @triton.jit
    def _finalize_column_sums_kernel(
        partial_add_rhs_ptr,
        partial_add_ptr,
        partial_grad_ptr,
        out_add_rhs_ptr,
        out_add_ptr,
        out_grad_ptr,
        NUM_M_TILES: tl.constexpr,
        D_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
        BLOCK_D_: tl.constexpr,
    ):
        d = tl.program_id(0) * BLOCK_D_ + tl.arange(0, BLOCK_D_)
        d_mask = d < D_
        tiles = tl.arange(0, BLOCK_TILES_)

        acc_add_rhs = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        acc_add = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        acc_grad = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        for tile_base in range(0, NUM_M_TILES, BLOCK_TILES_):
            tile_ids = tile_base + tiles
            mask = (tile_ids[:, None] < NUM_M_TILES) & d_mask[None, :]
            offsets = tile_ids[:, None] * D_ + d[None, :]
            acc_add_rhs += tl.sum(
                tl.load(partial_add_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
                axis=0,
            )
            acc_add += tl.sum(
                tl.load(partial_add_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
                axis=0,
            )
            acc_grad += tl.sum(
                tl.load(partial_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
                axis=0,
            )

        tl.store(out_add_rhs_ptr + d, acc_add_rhs, mask=d_mask)
        tl.store(out_add_ptr + d, acc_add, mask=d_mask)
        tl.store(out_grad_ptr + d, acc_grad, mask=d_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        view_as_real_10,
        mul_306,
        arg9_1,
        arg65_1,
        arg163_1,
        arg64_1,
        *_shape_params,
    ) = inputs

    return (
        view_as_real_10.contiguous(),
        mul_306.reshape(M, D).contiguous(),
        arg9_1.contiguous(),
        arg65_1.reshape(M, D).contiguous(),
        arg163_1.reshape(M).contiguous(),
        arg64_1.reshape(M, D).contiguous(),
    )


def oracle_triton_prepared(
    real_pair_mdk: torch.Tensor,
    residual_md: torch.Tensor,
    gamma_d: torch.Tensor,
    rhs_md: torch.Tensor,
    row_scale_m: torch.Tensor,
    keep_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if real_pair_mdk.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert real_pair_mdk.shape == (BATCH, SEQ, D, 2)
    assert residual_md.shape == (M, D)
    assert gamma_d.shape == (D,)
    assert rhs_md.shape == (M, D)
    assert row_scale_m.shape == (M,)
    assert keep_md.shape == (M, D)
    assert real_pair_mdk.is_contiguous()
    assert residual_md.is_contiguous()
    assert gamma_d.is_contiguous()
    assert rhs_md.is_contiguous()
    assert row_scale_m.is_contiguous()
    assert keep_md.is_contiguous()

    device = real_pair_mdk.device
    num_m_tiles = triton.cdiv(M, ROW_SPLIT)
    partial_add_rhs = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_add = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_grad = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    grad_md = torch.empty((M, D), device=device, dtype=torch.float32)

    _row_tile_kernel[(num_m_tiles,)](
        real_pair_mdk,
        residual_md,
        gamma_d,
        rhs_md,
        row_scale_m,
        keep_md,
        partial_add_rhs,
        partial_add,
        partial_grad,
        grad_md,
        M_=M,
        D_=D,
        ROW_SPLIT_=ROW_SPLIT,
        XBLOCK_=XBLOCK,
        BLOCK_D_=BLOCK_D,
        num_warps=ROW_NUM_WARPS,
    )

    out_add_rhs = torch.empty((D,), device=device, dtype=torch.float32)
    out_add = torch.empty((D,), device=device, dtype=torch.float32)
    out_grad = torch.empty((D,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partial_add_rhs,
        partial_add,
        partial_grad,
        out_add_rhs,
        out_add,
        out_grad,
        NUM_M_TILES=num_m_tiles,
        D_=D,
        BLOCK_TILES_=FINAL_BLOCK_TILES,
        BLOCK_D_=FINAL_BLOCK_D,
        num_warps=FINAL_NUM_WARPS,
    )

    return out_add_rhs, out_add, grad_md.t(), out_grad


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
