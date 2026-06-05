"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `Repro.forward` tuple by using one row-tiled producer to share the ViT layer-norm-backward row reductions, write the returned non-contiguous `[768, 32768]` gradient transpose, and emit split-K partials for the three returned `[768]` column sums, whereas Inductor currently schedules the row reductions, gradient materialization/permute, and sibling column reductions as separate generic pointwise and reduction kernels over the same `[32768, 768]` data; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K multi-output reduction template that can combine row-local reductions, materialized side-output stores, and several small-output column accumulators in one coordinated producer/finalizer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across row tiles, finalize shared partials, and fuse the dependent transposed side store with those accumulators."""
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


REPRO_ID = "sum_sum_sum_a431ef798d5d"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_vit_base_patch16_siglip_256_train"

B = 128
T = 256
C = 768
ROWS = B * T

ROW_SPLIT = 32
XBLOCK = 2
BLOCK_C = 1024
FINAL_BLOCK_C = 32
ROW_NUM_WARPS = 1
FINAL_NUM_WARPS = 8
CLAIMED_COMPILE_US = 78.688
HISTORICAL_BAD_ORACLE_US = 95.040


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


@triton.jit
def _row_store_and_partial_reduce_kernel(
    x_ptr,
    weight_ptr,
    xhat_ptr,
    scale_ptr,
    partial_sum_x_xhat_ptr,
    partial_sum_x_ptr,
    partial_sum_grad_ptr,
    out_transposed_storage_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    ROW_SPLIT_: tl.constexpr,
    XBLOCK_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C_)
    c_mask = c < C_
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    acc_sum_x_xhat = tl.zeros([BLOCK_C_], dtype=tl.float32)
    acc_sum_x = tl.zeros([BLOCK_C_], dtype=tl.float32)
    acc_sum_grad = tl.zeros([BLOCK_C_], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT_, XBLOCK_):
        row = pid * ROW_SPLIT_ + start + tl.arange(0, XBLOCK_)
        row_mask = row < ROWS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * C_ + c[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

        weighted = x * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * xhat, 0.0), axis=1)
        grad = scale[:, None] * (
            weighted * C_ - row_sum[:, None] - xhat * row_dot[:, None]
        )

        tl.store(out_transposed_storage_ptr + offsets, grad, mask=mask)
        acc_sum_x_xhat += tl.sum(tl.where(mask, x * xhat, 0.0), axis=0)
        acc_sum_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_sum_grad += tl.sum(tl.where(mask, grad, 0.0), axis=0)

    partial_offsets = pid * C_ + c
    tl.store(partial_sum_x_xhat_ptr + partial_offsets, acc_sum_x_xhat, mask=c_mask)
    tl.store(partial_sum_x_ptr + partial_offsets, acc_sum_x, mask=c_mask)
    tl.store(partial_sum_grad_ptr + partial_offsets, acc_sum_grad, mask=c_mask)


@triton.jit
def _finalize_column_partials_kernel(
    partial_sum_x_xhat_ptr,
    partial_sum_x_ptr,
    partial_sum_grad_ptr,
    out_sum_x_xhat_ptr,
    out_sum_x_ptr,
    out_sum_grad_ptr,
    NUM_ROW_BLOCKS_: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_ROW_BLOCKS_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    block = tl.arange(0, BLOCK_ROW_BLOCKS_)
    mask = (block[:, None] < NUM_ROW_BLOCKS_) & (c[None, :] < C_)
    offsets = block[:, None] * C_ + c[None, :]

    sum_x_xhat = tl.load(
        partial_sum_x_xhat_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sum_x = tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_grad = tl.load(partial_sum_grad_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    c_mask = c < C_
    tl.store(out_sum_x_xhat_ptr + c, tl.sum(sum_x_xhat, axis=0), mask=c_mask)
    tl.store(out_sum_x_ptr + c, tl.sum(sum_x, axis=0), mask=c_mask)
    tl.store(out_sum_grad_ptr + c, tl.sum(sum_grad, axis=0), mask=c_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_7,
        primals_149,
        mul_84,
        div_1,
        *_shape_params,
    ) = inputs

    return (
        mm_7.contiguous(),
        primals_149.contiguous(),
        mul_84.reshape(ROWS, C).contiguous(),
        div_1.reshape(ROWS).contiguous(),
    )


def oracle_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    xhat: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    assert x.shape == (ROWS, C)
    assert weight.shape == (C,)
    assert xhat.shape == (ROWS, C)
    assert scale.shape == (ROWS,)
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert xhat.is_contiguous()
    assert scale.is_contiguous()

    device = x.device
    num_row_blocks = triton.cdiv(ROWS, ROW_SPLIT)
    partial_sum_x_xhat = torch.empty(
        (num_row_blocks, C),
        device=device,
        dtype=torch.float32,
    )
    partial_sum_x = torch.empty((num_row_blocks, C), device=device, dtype=torch.float32)
    partial_sum_grad = torch.empty(
        (num_row_blocks, C),
        device=device,
        dtype=torch.float32,
    )
    out_transposed = torch.empty_strided(
        (C, ROWS),
        (1, C),
        device=device,
        dtype=torch.float32,
    )

    _row_store_and_partial_reduce_kernel[(num_row_blocks,)](
        x,
        weight,
        xhat,
        scale,
        partial_sum_x_xhat,
        partial_sum_x,
        partial_sum_grad,
        out_transposed,
        ROWS_=ROWS,
        C_=C,
        ROW_SPLIT_=ROW_SPLIT,
        XBLOCK_=XBLOCK,
        BLOCK_C_=BLOCK_C,
        num_warps=ROW_NUM_WARPS,
    )

    out_sum_x_xhat = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum_grad = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_partials_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_sum_x_xhat,
        partial_sum_x,
        partial_sum_grad,
        out_sum_x_xhat,
        out_sum_x,
        out_sum_grad,
        NUM_ROW_BLOCKS_=num_row_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS_=triton.next_power_of_2(num_row_blocks),
        BLOCK_C_=FINAL_BLOCK_C,
        num_warps=FINAL_NUM_WARPS,
    )

    return out_sum_x_xhat, out_sum_x, out_transposed, out_sum_grad


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_fused(*prepare_oracle_inputs(*inputs))


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
