"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GPTNeo layer-norm-backward return tuple by streaming the three-matmul sum once, doing the two row-local 2048-wide reductions, writing the returned non-contiguous `[2048, 4096]` transpose view, and cooperatively accumulating all three returned `[2048]` column reductions from the same producer, whereas Inductor currently schedules the sibling column reductions, row reductions, dense epilogue, transpose view, and final column reduction as separate generic reduction/pointwise regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local summaries, a dependent full-tensor side output, and multiple compatible column accumulators tied to a shared multi-input producer in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across token-row tiles, fuse the dense side-output store, and finalize all sibling column partials together."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps CPU-only checks usable.
    triton = None
    tl = None

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
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 4096
CHANNELS = 2048
ROW_SPLIT = 4
XBLOCK = 1
BLOCK_CHANNELS = 2048
FINAL_BLOCK_CHANNELS = 16


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def oracle_torch(
    mm_273: torch.Tensor,
    mm_275: torch.Tensor,
    mm_277: torch.Tensor,
    arg11_1: torch.Tensor,
    arg231_1: torch.Tensor,
    arg537_1: torch.Tensor,
    add_181: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = (
        mm_273.view(_shape_param_0)
        + mm_275.view(_shape_param_1)
        + mm_277.view(_shape_param_2)
    )
    weighted = x * arg11_1
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * arg231_1).sum(dim=2, keepdim=True)
    grad = arg537_1 * (weighted * CHANNELS - row_sum - arg231_1 * row_dot)

    out0 = (x * arg231_1).sum(dim=(0, 1))
    out1 = x.sum(dim=(0, 1))
    out_base = add_181 + grad
    out2 = out_base.view(_shape_param_3).permute(1, 0)
    out3 = out_base.view(_shape_param_3).sum(dim=0, keepdim=True).view(_shape_param_4)
    return out0, out1, out2, out3


if triton is not None:

    @triton.jit
    def _row_store_and_partial_reduce_kernel(
        mm0_ptr,
        mm1_ptr,
        mm2_ptr,
        gamma_ptr,
        rhs_ptr,
        row_scale_ptr,
        residual_ptr,
        out_base_ptr,
        partial_x_rhs_ptr,
        partial_x_ptr,
        partial_out_ptr,
        ROWS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        ROW_SPLIT_: tl.constexpr,
        XBLOCK_: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        pid = tl.program_id(0)
        c = tl.arange(0, BLOCK_CHANNELS_)
        c_mask = c < CHANNELS_
        gamma = tl.load(gamma_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

        acc_x_rhs = tl.zeros((BLOCK_CHANNELS_,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_CHANNELS_,), dtype=tl.float32)
        acc_out = tl.zeros((BLOCK_CHANNELS_,), dtype=tl.float32)

        for start in tl.range(0, ROW_SPLIT_, XBLOCK_):
            row = pid * ROW_SPLIT_ + start + tl.arange(0, XBLOCK_)
            row_mask = row < ROWS_
            mask = row_mask[:, None] & c_mask[None, :]
            offsets = row[:, None] * CHANNELS_ + c[None, :]

            x = (
                tl.load(mm0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                + tl.load(mm1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            )
            x = x + tl.load(mm2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            row_scale = tl.load(row_scale_ptr + row, mask=row_mask, other=0.0).to(
                tl.float32
            )

            weighted = x * gamma[None, :]
            row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
            row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
            grad = row_scale[:, None] * (
                weighted * CHANNELS_ - row_sum[:, None] - rhs * row_dot[:, None]
            )
            out = residual + grad
            tl.store(out_base_ptr + offsets, out, mask=mask)

            acc_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
            acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
            acc_out += tl.sum(tl.where(mask, out, 0.0), axis=0)

        partial_offsets = pid * CHANNELS_ + c
        tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=c_mask)
        tl.store(partial_x_ptr + partial_offsets, acc_x, mask=c_mask)
        tl.store(partial_out_ptr + partial_offsets, acc_out, mask=c_mask)


    @triton.jit
    def _finalize_column_partials_kernel(
        partial_x_rhs_ptr,
        partial_x_ptr,
        partial_out_ptr,
        out_x_rhs_ptr,
        out_x_ptr,
        out_sum_out_ptr,
        NUM_ROW_BLOCKS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_ROW_BLOCKS_: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
        row_block = tl.arange(0, BLOCK_ROW_BLOCKS_)
        c_mask = c < CHANNELS_
        mask = (row_block[:, None] < NUM_ROW_BLOCKS_) & c_mask[None, :]
        offsets = row_block[:, None] * CHANNELS_ + c[None, :]

        sum_x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        sum_x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_out = tl.load(partial_out_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        tl.store(out_x_rhs_ptr + c, tl.sum(sum_x_rhs, axis=0), mask=c_mask)
        tl.store(out_x_ptr + c, tl.sum(sum_x, axis=0), mask=c_mask)
        tl.store(out_sum_out_ptr + c, tl.sum(sum_out, axis=0), mask=c_mask)


def _prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_273,
        mm_275,
        mm_277,
        arg11_1,
        arg231_1,
        arg537_1,
        add_181,
        *_shape_params,
    ) = inputs

    return (
        mm_273.reshape(ROWS, CHANNELS).contiguous(),
        mm_275.reshape(ROWS, CHANNELS).contiguous(),
        mm_277.reshape(ROWS, CHANNELS).contiguous(),
        arg11_1.contiguous(),
        arg231_1.reshape(ROWS, CHANNELS).contiguous(),
        arg537_1.reshape(ROWS).contiguous(),
        add_181.reshape(ROWS, CHANNELS).contiguous(),
    )


def oracle_triton_prepared(
    mm0: torch.Tensor,
    mm1: torch.Tensor,
    mm2: torch.Tensor,
    gamma: torch.Tensor,
    rhs: torch.Tensor,
    row_scale: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm0.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm0.shape == (ROWS, CHANNELS)
    assert mm1.shape == (ROWS, CHANNELS)
    assert mm2.shape == (ROWS, CHANNELS)
    assert gamma.shape == (CHANNELS,)
    assert rhs.shape == (ROWS, CHANNELS)
    assert row_scale.shape == (ROWS,)
    assert residual.shape == (ROWS, CHANNELS)
    assert mm0.is_contiguous()
    assert mm1.is_contiguous()
    assert mm2.is_contiguous()
    assert gamma.is_contiguous()
    assert rhs.is_contiguous()
    assert row_scale.is_contiguous()
    assert residual.is_contiguous()

    device = mm0.device
    num_row_blocks = triton.cdiv(ROWS, ROW_SPLIT)
    partial_x_rhs = torch.empty(
        (num_row_blocks, CHANNELS), device=device, dtype=torch.float32
    )
    partial_x = torch.empty(
        (num_row_blocks, CHANNELS), device=device, dtype=torch.float32
    )
    partial_out = torch.empty(
        (num_row_blocks, CHANNELS), device=device, dtype=torch.float32
    )
    out_base = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)

    _row_store_and_partial_reduce_kernel[(num_row_blocks,)](
        mm0,
        mm1,
        mm2,
        gamma,
        rhs,
        row_scale,
        residual,
        out_base,
        partial_x_rhs,
        partial_x,
        partial_out,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        ROW_SPLIT_=ROW_SPLIT,
        XBLOCK_=XBLOCK,
        BLOCK_CHANNELS_=BLOCK_CHANNELS,
        num_warps=8,
    )

    out_x_rhs = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    _finalize_column_partials_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partial_x_rhs,
        partial_x,
        partial_out,
        out_x_rhs,
        out_x,
        out_sum_out,
        NUM_ROW_BLOCKS_=num_row_blocks,
        CHANNELS_=CHANNELS,
        BLOCK_ROW_BLOCKS_=triton.next_power_of_2(num_row_blocks),
        BLOCK_CHANNELS_=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return out_x_rhs, out_x, out_base.permute(1, 0), out_sum_out


def oracle_triton(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*_prepare_oracle_inputs(*inputs))


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
    """Run the oracle computation."""
    return oracle_full(*inputs)


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

    inputs = get_inputs()
    instance = get_repro_instance()

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
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
