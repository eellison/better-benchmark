"""
Full-scope oracle for sum_sum_sum_ed8809aa30cf (Blenderbot backward tail).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): The timed oracle consumes
the same original inputs as repro.py and returns the same two `[2560]`
reductions plus the full `[32, 128, 2560]` epilogue tensor. It streams the
shared `mm_6 + mm_8 + mm_10` producer once through a Triton split-tile reduction
that computes the row-local scalar accumulators needed by the dependent
epilogue and the sibling column reductions needed for the two vector outputs,
then finalizes those partials and recomputes the cheap pointwise producer for
the returned tensor. Inductor cannot do this today because the graph mixes
reductions over the hidden dimension with reductions over batch/sequence and a
dependent pointwise consumer, so the scheduler emits separate reductions and an
epilogue instead of one coordinated multi-output reduction plan. The fix is
COOPERATIVE_SPLIT_K: add codegen support for cross-axis multi-output reductions
with row/column partials and recompute in the dependent epilogue.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_ed8809aa30cf"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ
CHANNELS = 2560
INV_CHANNELS = 1.0 / CHANNELS



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
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


@triton.jit
def _reduction_partials_kernel(
    mm6_ptr,
    mm8_ptr,
    mm10_ptr,
    weight_ptr,
    source_ptr,
    row_shift_ptr,
    row_scale_ptr,
    row_partial0_ptr,
    row_partial1_ptr,
    col_partial0_ptr,
    col_partial1_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    NUM_C_BLOCKS_: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS_
    col_mask = cols < CHANNELS_
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * CHANNELS_ + cols[None, :]

    mm6 = tl.load(mm6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm8 = tl.load(mm8_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm10 = tl.load(mm10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = mm6 + mm8 + mm10
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    m2 = (source - row_shift[:, None]) * row_scale[:, None]
    scaled = x * weight[None, :]

    row_sum0 = tl.sum(tl.where(mask, scaled, 0.0), axis=1)
    row_sum1 = tl.sum(tl.where(mask, scaled * m2, 0.0), axis=1)
    row_partial_offsets = rows * NUM_C_BLOCKS_ + col_block
    tl.store(row_partial0_ptr + row_partial_offsets, row_sum0, mask=row_mask)
    tl.store(row_partial1_ptr + row_partial_offsets, row_sum1, mask=row_mask)

    col_sum0 = tl.sum(tl.where(mask, x * m2, 0.0), axis=0)
    col_sum1 = tl.sum(tl.where(mask, x, 0.0), axis=0)
    col_partial_offsets = row_block * CHANNELS_ + cols
    tl.store(col_partial0_ptr + col_partial_offsets, col_sum0, mask=col_mask)
    tl.store(col_partial1_ptr + col_partial_offsets, col_sum1, mask=col_mask)


@triton.jit
def _finalize_row_sums_kernel(
    row_partial0_ptr,
    row_partial1_ptr,
    row_sum0_ptr,
    row_sum1_ptr,
    ROWS_: tl.constexpr,
    NUM_C_BLOCKS_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    tiles = tl.arange(0, BLOCK_TILES)
    row_mask = rows < ROWS_
    tile_mask = tiles < NUM_C_BLOCKS_
    mask = row_mask[:, None] & tile_mask[None, :]
    offsets = rows[:, None] * NUM_C_BLOCKS_ + tiles[None, :]

    partial0 = tl.load(row_partial0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial1 = tl.load(row_partial1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(row_sum0_ptr + rows, tl.sum(partial0, axis=1), mask=row_mask)
    tl.store(row_sum1_ptr + rows, tl.sum(partial1, axis=1), mask=row_mask)


@triton.jit
def _finalize_col_sums_kernel(
    col_partial0_ptr,
    col_partial1_ptr,
    out0_ptr,
    out1_ptr,
    NUM_ROW_BLOCKS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    col_mask = cols < CHANNELS_
    block_mask = blocks < NUM_ROW_BLOCKS_
    mask = block_mask[:, None] & col_mask[None, :]
    offsets = blocks[:, None] * CHANNELS_ + cols[None, :]

    partial0 = tl.load(col_partial0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial1 = tl.load(col_partial1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out0_ptr + cols, tl.sum(partial0, axis=0), mask=col_mask)
    tl.store(out1_ptr + cols, tl.sum(partial1, axis=0), mask=col_mask)


@triton.jit
def _epilogue_kernel(
    mm6_ptr,
    mm8_ptr,
    mm10_ptr,
    weight_ptr,
    source_ptr,
    row_shift_ptr,
    row_scale_ptr,
    residual_ptr,
    row_sum0_ptr,
    row_sum1_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    INV_CHANNELS_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = offsets < NUMEL_
    cols = offsets % CHANNELS_
    rows = offsets // CHANNELS_

    mm6 = tl.load(mm6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm8 = tl.load(mm8_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm10 = tl.load(mm10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = mm6 + mm8 + mm10
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift = tl.load(row_shift_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    row_scale = tl.load(row_scale_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum0 = tl.load(row_sum0_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    row_sum1 = tl.load(row_sum1_ptr + rows, mask=mask, other=0.0).to(tl.float32)

    scaled = x * weight
    m2 = (source - row_shift) * row_scale
    ln_tail = row_scale * INV_CHANNELS_ * (scaled * CHANNELS_ - row_sum0 - m2 * row_sum1)
    tl.store(out_ptr + offsets, residual + ln_tail, mask=mask)

def oracle_fused(
    mm_6: torch.Tensor,
    mm_8: torch.Tensor,
    mm_10: torch.Tensor,
    arg0_1: torch.Tensor,
    arg1_1: torch.Tensor,
    arg9_1: torch.Tensor,
    arg10_1: torch.Tensor,
    add_3: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_6.shape == (ROWS, CHANNELS)
    assert mm_8.shape == (ROWS, CHANNELS)
    assert mm_10.shape == (ROWS, CHANNELS)
    assert arg0_1.shape == (CHANNELS,)
    assert arg1_1.shape == (BATCH, SEQ, CHANNELS)
    assert arg9_1.shape == (BATCH, SEQ, 1)
    assert arg10_1.shape == (BATCH, SEQ, 1)
    assert add_3.shape == (BATCH, SEQ, CHANNELS)
    assert mm_6.is_contiguous()
    assert mm_8.is_contiguous()
    assert mm_10.is_contiguous()
    assert arg0_1.is_contiguous()
    assert arg1_1.is_contiguous()
    assert arg9_1.is_contiguous()
    assert arg10_1.is_contiguous()
    assert add_3.is_contiguous()

    device = mm_6.device
    block_r = 32
    block_c = 64
    num_row_blocks = triton.cdiv(ROWS, block_r)
    num_c_blocks = triton.cdiv(CHANNELS, block_c)

    row_partials = torch.empty((2, ROWS, num_c_blocks), device=device, dtype=torch.float32)
    col_partials = torch.empty((2, num_row_blocks, CHANNELS), device=device, dtype=torch.float32)
    row_sums = torch.empty((2, ROWS), device=device, dtype=torch.float32)
    vector_outputs = torch.empty((2, CHANNELS), device=device, dtype=torch.float32)
    out_tensor = torch.empty((BATCH, SEQ, CHANNELS), device=device, dtype=torch.float32)

    _reduction_partials_kernel[(num_row_blocks, num_c_blocks)](
        mm_6,
        mm_8,
        mm_10,
        arg0_1,
        arg1_1,
        arg9_1,
        arg10_1,
        row_partials[0],
        row_partials[1],
        col_partials[0],
        col_partials[1],
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        NUM_C_BLOCKS_=num_c_blocks,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        num_warps=4,
    )

    row_block_rows = 8
    _finalize_row_sums_kernel[(triton.cdiv(ROWS, row_block_rows),)](
        row_partials[0],
        row_partials[1],
        row_sums[0],
        row_sums[1],
        ROWS_=ROWS,
        NUM_C_BLOCKS_=num_c_blocks,
        BLOCK_ROWS=row_block_rows,
        BLOCK_TILES=triton.next_power_of_2(num_c_blocks),
        num_warps=4,
    )

    finalize_block_c = 16
    _finalize_col_sums_kernel[(triton.cdiv(CHANNELS, finalize_block_c),)](
        col_partials[0],
        col_partials[1],
        vector_outputs[0],
        vector_outputs[1],
        NUM_ROW_BLOCKS_=num_row_blocks,
        CHANNELS_=CHANNELS,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=finalize_block_c,
        num_warps=8,
    )

    numel = ROWS * CHANNELS
    block_elems = 1024
    _epilogue_kernel[(triton.cdiv(numel, block_elems),)](
        mm_6,
        mm_8,
        mm_10,
        arg0_1,
        arg1_1,
        arg9_1,
        arg10_1,
        add_3,
        row_sums[0],
        row_sums[1],
        out_tensor,
        NUMEL_=numel,
        CHANNELS_=CHANNELS,
        INV_CHANNELS_=INV_CHANNELS,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return vector_outputs[0], vector_outputs[1], out_tensor


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual = oracle_fused(*inputs)
        torch.cuda.synchronize()

    ok = True
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        output_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == expected.stride()
        ok = ok and output_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={output_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(model: torch.nn.Module, inputs: tuple[object, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(rep: int, warmup: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        oracle_fused(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_fused(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"oracle_fused full-scope Blenderbot multi-output reduction: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    compile_configs = [
        ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3",
            {
                "combo_kernels": True,
                "combo_kernel_per_subkernel_blocks": True,
                "coordinate_descent_tuning": True,
                "benchmark_combo_kernel": True,
                "triton.multi_kernel": 3,
            },
        ),
    ]
    for label, config in compile_configs:
        model = module.Repro().cuda()
        with torch.no_grad():
            compiled = _compile_with_config(model, inputs, config)
            compiled_us = triton.testing.do_bench(
                lambda: compiled(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            ) * 1000.0
        print(f"torch.compile {label}: {compiled_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
