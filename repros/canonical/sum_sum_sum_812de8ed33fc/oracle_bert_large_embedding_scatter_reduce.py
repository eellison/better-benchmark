"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full BERT-large embedding/layernorm-backward return tuple in a row-wise scatter-reduce pass plus initialization/finalization, whereas Inductor currently materializes the rowwise gradient and lowers the two hidden reductions and three index_put(accumulate=True) embedding-gradient outputs as separate scheduled kernels; Inductor cannot do this today because its scheduler/codegen has no structured scatter-reduce template that shares a row-reduction producer across column reductions and multiple indexed accumulator destinations with different output sizes; the fix is SCATTER_REDUCE: add scheduler/codegen support for fused embedding-backward scatter-reduce kernels that keep the row layernorm algebra in registers, emit sibling hidden reductions, and atomically accumulate position, segment, and vocabulary gradients directly."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_812de8ed33fc"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
SEQ = 512
HIDDEN = 1024
ROWS = BATCH * SEQ
POSITION_ROWS = 512
SEGMENT_ROWS = 2
VOCAB_ROWS = 30522
ROW_BLOCK = 16
ROW_BLOCKS = triton.cdiv(ROWS, ROW_BLOCK)
HIDDEN_BLOCK = 1024

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _init_outputs_kernel(
    mm1_ptr,
    out_sum_vx_ptr,
    out_sum_v_ptr,
    out_position_ptr,
    out_segment_ptr,
    out_vocab_ptr,
    TOTAL_VOCAB: tl.constexpr,
    TOTAL_POSITION: tl.constexpr,
    TOTAL_SEGMENT: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    vocab_mask = offset < TOTAL_VOCAB
    vocab_value = tl.load(mm1_ptr + offset, mask=vocab_mask, other=0.0).to(tl.float32)
    tl.store(out_vocab_ptr + offset, vocab_value, mask=vocab_mask)

    position_mask = offset < TOTAL_POSITION
    tl.store(out_position_ptr + offset, tl.zeros([BLOCK], tl.float32), mask=position_mask)

    segment_mask = offset < TOTAL_SEGMENT
    tl.store(out_segment_ptr + offset, tl.zeros([BLOCK], tl.float32), mask=segment_mask)

    hidden_mask = offset < HIDDEN_
    tl.store(out_sum_vx_ptr + offset, tl.zeros([BLOCK], tl.float32), mask=hidden_mask)
    tl.store(out_sum_v_ptr + offset, tl.zeros([BLOCK], tl.float32), mask=hidden_mask)


@triton.jit
def _row_scatter_reduce_kernel(
    mm286_ptr,
    mul681_ptr,
    mm288_ptr,
    mm290_ptr,
    dropout_mask_ptr,
    weight_ptr,
    saved_x_ptr,
    row_scale_ptr,
    position_index_ptr,
    full_ptr,
    segment_index_ptr,
    token_index_ptr,
    out_sum_vx_ptr,
    out_sum_v_ptr,
    out_position_ptr,
    out_segment_ptr,
    out_vocab_ptr,
    HIDDEN_: tl.constexpr,
    SEQ_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    SEGMENT_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    hidden_mask = h < HIDDEN_
    base = row * HIDDEN_ + h
    seq = row % SEQ_

    added = (
        tl.load(mul681_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        + tl.load(mm286_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        + tl.load(mm288_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        + tl.load(mm290_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
    )
    keep = tl.load(dropout_mask_ptr + base, mask=hidden_mask, other=0).to(tl.float32)
    v = added * keep * 1.1111111111111112
    weight = tl.load(weight_ptr + h, mask=hidden_mask, other=0.0).to(tl.float32)
    saved_x = tl.load(saved_x_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
    weighted_v = v * weight
    row_sum = tl.sum(tl.where(hidden_mask, weighted_v, 0.0), axis=0)
    row_dot = tl.sum(tl.where(hidden_mask, weighted_v * saved_x, 0.0), axis=0)
    scale = tl.load(row_scale_ptr + row).to(tl.float32)
    grad = scale * (weighted_v * 1024.0 - row_sum - saved_x * row_dot)

    tl.atomic_add(out_sum_vx_ptr + h, v * saved_x, sem="relaxed", mask=hidden_mask)
    tl.atomic_add(out_sum_v_ptr + h, v, sem="relaxed", mask=hidden_mask)

    full_value = tl.load(full_ptr).to(tl.float32)

    position_raw = tl.load(position_index_ptr + seq).to(tl.int64)
    position_wrapped = tl.where(position_raw < 0, position_raw + POSITION_ROWS_, position_raw)
    position_value = tl.where(position_raw == -1, full_value, grad)
    tl.atomic_add(
        out_position_ptr + position_wrapped * HIDDEN_ + h,
        position_value,
        sem="relaxed",
        mask=hidden_mask & (position_wrapped >= 0) & (position_wrapped < POSITION_ROWS_),
    )

    segment_raw = tl.load(segment_index_ptr + seq).to(tl.int64)
    segment_wrapped = tl.where(segment_raw < 0, segment_raw + SEGMENT_ROWS_, segment_raw)
    segment_value = tl.where(segment_raw == -1, full_value, grad)
    tl.atomic_add(
        out_segment_ptr + segment_wrapped * HIDDEN_ + h,
        segment_value,
        sem="relaxed",
        mask=hidden_mask & (segment_wrapped >= 0) & (segment_wrapped < SEGMENT_ROWS_),
    )

    token_raw = tl.load(token_index_ptr + row).to(tl.int64)
    token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_ROWS_, token_raw)
    token_value = tl.where(token_raw == 0, full_value, grad)
    tl.atomic_add(
        out_vocab_ptr + token_wrapped * HIDDEN_ + h,
        token_value,
        sem="relaxed",
        mask=hidden_mask & (token_wrapped >= 0) & (token_wrapped < VOCAB_ROWS_),
    )


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
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def oracle_bert_large_embedding_scatter_reduce(
    mm_286: torch.Tensor,
    mul_681: torch.Tensor,
    mm_288: torch.Tensor,
    mm_290: torch.Tensor,
    arg201_1: torch.Tensor,
    arg3_1: torch.Tensor,
    arg200_1: torch.Tensor,
    arg623_1: torch.Tensor,
    arg1_1: torch.Tensor,
    full_1: torch.Tensor,
    arg199_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if list(_shape_param_0) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 0: {_shape_param_0}")
    if list(_shape_param_1) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 1: {_shape_param_1}")
    if list(_shape_param_2) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 2: {_shape_param_2}")
    if list(_shape_param_3) != [BATCH, SEQ]:
        raise ValueError(f"unexpected shape parameter 3: {_shape_param_3}")

    mm_286 = mm_286.contiguous()
    mul_681 = mul_681.contiguous()
    mm_288 = mm_288.contiguous()
    mm_290 = mm_290.contiguous()
    arg201_1 = arg201_1.contiguous()
    arg3_1 = arg3_1.contiguous()
    arg200_1 = arg200_1.contiguous()
    arg623_1 = arg623_1.contiguous()
    arg1_1 = arg1_1.contiguous()
    full_1 = full_1.contiguous()
    arg199_1 = arg199_1.contiguous()
    arg0_1 = arg0_1.contiguous()
    mm_1 = mm_1.contiguous()

    assert mm_286.shape == (ROWS, HIDDEN)
    assert mul_681.shape == (BATCH, SEQ, HIDDEN)
    assert mm_288.shape == (ROWS, HIDDEN)
    assert mm_290.shape == (ROWS, HIDDEN)
    assert arg201_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg3_1.shape == (HIDDEN,)
    assert arg200_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg623_1.shape == (BATCH, SEQ, 1)
    assert arg1_1.shape == (1, SEQ)
    assert arg199_1.shape == (1, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm_1.shape == (VOCAB_ROWS, HIDDEN)

    device = mm_286.device
    out_sum_vx = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_v = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_position = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_segment = torch.empty((SEGMENT_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    block = 1024
    total_vocab = VOCAB_ROWS * HIDDEN
    total_position = POSITION_ROWS * HIDDEN
    total_segment = SEGMENT_ROWS * HIDDEN
    _init_outputs_kernel[(triton.cdiv(total_vocab, block),)](
        mm_1,
        out_sum_vx,
        out_sum_v,
        out_position,
        out_segment,
        out_vocab,
        TOTAL_VOCAB=total_vocab,
        TOTAL_POSITION=total_position,
        TOTAL_SEGMENT=total_segment,
        HIDDEN_=HIDDEN,
        BLOCK=block,
        num_warps=8,
    )
    _row_scatter_reduce_kernel[(ROWS,)](
        mm_286,
        mul_681,
        mm_288,
        mm_290,
        arg201_1,
        arg3_1,
        arg200_1,
        arg623_1,
        arg1_1,
        full_1,
        arg199_1,
        arg0_1,
        out_sum_vx,
        out_sum_v,
        out_position,
        out_segment,
        out_vocab,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        POSITION_ROWS_=POSITION_ROWS,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        BLOCK_H=HIDDEN_BLOCK,
        num_warps=8,
    )
    return out_sum_vx, out_sum_v, out_position, out_segment, out_vocab


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = oracle_bert_large_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"output_count: got={len(actual)} expected={len(expected)} match=False")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        ok = ok and value_ok and shape_ok and dtype_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={value_ok} shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok}"
        )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(rep: int, warmup: int) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        oracle_bert_large_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_bert_large_embedding_scatter_reduce(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"oracle_bert_large_embedding_scatter_reduce full-scope: {oracle_us:.3f} us")
    return oracle_us


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare every oracle output with repro.py")
    parser.add_argument("--bench", action="store_true", help="benchmark the full-scope oracle")
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup)


if __name__ == "__main__":
    with torch.no_grad():
        main()
