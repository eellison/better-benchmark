"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full DistilBert layernorm-backward return tuple by fusing the shared dropout/add row producer, hidden-dimension layernorm reductions, two returned hidden-column reductions, and both duplicate-index embedding scatter-add outputs, whereas Inductor currently materializes the `[8, 512, 768]` layernorm-gradient producer and lowers the `sum` and `index_put(accumulate=True)` consumers as separate generic reduction and scatter kernels; Inductor cannot do this today because scheduler/codegen has no structured scatter-reduce template that shares one rowwise reduction producer across sibling column reductions and multiple indexed accumulation destinations; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that emits the rowwise layernorm math once, accumulates the column reductions, and writes the position and word embedding scatter-add outputs directly."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_94e3624cdd05"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
SEQ = 512
HIDDEN = 768
ROWS = BATCH * SEQ
VOCAB = 30522
DROPOUT_SCALE = 1.1111111111111112



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


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_70,
        mul_167,
        mm_72,
        mm_74,
        arg59_1,
        arg3_1,
        arg55_1,
        arg56_1,
        arg57_1,
        arg58_1,
        arg2_1,
        full_1,
        arg0_1,
        mm_1,
        *_shape_params,
    ) = inputs
    return (
        mm_70.contiguous(),
        mul_167.contiguous(),
        mm_72.contiguous(),
        mm_74.contiguous(),
        arg59_1.contiguous(),
        arg3_1.contiguous(),
        arg55_1.contiguous(),
        arg56_1.contiguous(),
        arg57_1.contiguous(),
        arg58_1.contiguous(),
        arg2_1.contiguous(),
        full_1.contiguous(),
        arg0_1.contiguous(),
        mm_1.contiguous(),
    )


@triton.jit
def _seq_layernorm_scatter_kernel(
    mm70_ptr,
    mul167_ptr,
    mm72_ptr,
    mm74_ptr,
    dropout_mask_ptr,
    weight_ptr,
    arg55_ptr,
    arg56_ptr,
    arg57_ptr,
    arg58_ptr,
    position_idx_ptr,
    full_ptr,
    token_idx_ptr,
    partial_sum_weighted_xhat_ptr,
    partial_sum_weighted_ptr,
    position_out_ptr,
    vocab_out_ptr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BATCH_: tl.constexpr,
    VOCAB_: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
):
    seq = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    h_mask = h < HIDDEN_

    weight = tl.load(weight_ptr + h, mask=h_mask, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    seq_hidden_offsets = seq * HIDDEN_ + h
    position_base = tl.load(arg56_ptr + seq_hidden_offsets, mask=h_mask, other=0.0).to(tl.float32)

    acc_sum_weighted_xhat = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_sum_weighted = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_position_grad = tl.zeros([BLOCK_H], dtype=tl.float32)

    for b in tl.static_range(0, BATCH_):
        row = b * SEQ_ + seq
        offsets = row * HIDDEN_ + h
        elem_mask = h_mask

        source = (
            tl.load(mul167_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
            + tl.load(mm70_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
            + tl.load(mm72_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
            + tl.load(mm74_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        )
        keep = tl.load(dropout_mask_ptr + offsets, mask=elem_mask, other=0).to(tl.float32)
        weighted_pre = source * keep * DROPOUT_SCALE_
        weighted = weighted_pre * weight

        center = tl.load(arg57_ptr + row, mask=True, other=0.0).to(tl.float32)
        invstd = tl.load(arg58_ptr + row, mask=True, other=0.0).to(tl.float32)
        xhat = (
            tl.load(arg55_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
            + position_base
            - center
        ) * invstd

        row_sum = tl.sum(tl.where(h_mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(h_mask, weighted * xhat, 0.0), axis=0)
        grad = (invstd * (1.0 / HIDDEN_)) * (weighted * HIDDEN_ - row_sum - xhat * row_dot)

        acc_sum_weighted_xhat += tl.where(h_mask, weighted_pre * xhat, 0.0)
        acc_sum_weighted += tl.where(h_mask, weighted_pre, 0.0)
        acc_position_grad += tl.where(h_mask, grad, 0.0)

        token_raw = tl.load(token_idx_ptr + row).to(tl.int64)
        token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_, token_raw)
        token_valid = h_mask & (token_wrapped >= 0) & (token_wrapped < VOCAB_)
        token_value = tl.where(token_raw == 0, full_value, grad)
        tl.atomic_add(
            vocab_out_ptr + token_wrapped * HIDDEN_ + h,
            token_value,
            sem="relaxed",
            mask=token_valid,
        )

    position_raw = tl.load(position_idx_ptr + seq).to(tl.int64)
    position_wrapped = tl.where(position_raw < 0, position_raw + SEQ_, position_raw)
    position_valid = h_mask & (position_wrapped >= 0) & (position_wrapped < SEQ_)
    position_value = tl.where(position_raw == -1, full_value, acc_position_grad)
    tl.atomic_add(
        position_out_ptr + position_wrapped * HIDDEN_ + h,
        position_value,
        sem="relaxed",
        mask=position_valid,
    )

    partial_offsets = seq * HIDDEN_ + h
    tl.store(partial_sum_weighted_xhat_ptr + partial_offsets, acc_sum_weighted_xhat, mask=h_mask)
    tl.store(partial_sum_weighted_ptr + partial_offsets, acc_sum_weighted, mask=h_mask)


@triton.jit
def _finalize_hidden_sums_kernel(
    partial_sum_weighted_xhat_ptr,
    partial_sum_weighted_ptr,
    out_sum_weighted_xhat_ptr,
    out_sum_weighted_ptr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    h = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    n = tl.arange(0, BLOCK_N)
    mask = (n[:, None] < SEQ_) & (h[None, :] < HIDDEN_)
    offsets = n[:, None] * HIDDEN_ + h[None, :]

    sum_weighted_xhat = tl.load(partial_sum_weighted_xhat_ptr + offsets, mask=mask, other=0.0)
    sum_weighted = tl.load(partial_sum_weighted_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_sum_weighted_xhat_ptr + h, tl.sum(sum_weighted_xhat, axis=0), mask=h < HIDDEN_)
    tl.store(out_sum_weighted_ptr + h, tl.sum(sum_weighted, axis=0), mask=h < HIDDEN_)


def oracle_distilbert_embedding_scatter_reduce(
    mm_70: torch.Tensor,
    mul_167: torch.Tensor,
    mm_72: torch.Tensor,
    mm_74: torch.Tensor,
    arg59_1: torch.Tensor,
    arg3_1: torch.Tensor,
    arg55_1: torch.Tensor,
    arg56_1: torch.Tensor,
    arg57_1: torch.Tensor,
    arg58_1: torch.Tensor,
    arg2_1: torch.Tensor,
    full_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_70.shape == (ROWS, HIDDEN)
    assert mul_167.shape == (BATCH, SEQ, HIDDEN)
    assert mm_72.shape == (ROWS, HIDDEN)
    assert mm_74.shape == (ROWS, HIDDEN)
    assert arg59_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg3_1.shape == (HIDDEN,)
    assert arg55_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg56_1.shape == (1, SEQ, HIDDEN)
    assert arg57_1.shape == (BATCH, SEQ, 1)
    assert arg58_1.shape == (BATCH, SEQ, 1)
    assert arg2_1.shape == (1, SEQ)
    assert full_1.shape == ()
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm_1.shape == (VOCAB, HIDDEN)

    device = mm_70.device
    partial_sum_weighted_xhat = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    partial_sum_weighted = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    out_sum_weighted_xhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_weighted = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    position_out = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = mm_1.clone()
    position_out.zero_()

    _seq_layernorm_scatter_kernel[(SEQ,)](
        mm_70,
        mul_167,
        mm_72,
        mm_74,
        arg59_1,
        arg3_1,
        arg55_1,
        arg56_1,
        arg57_1,
        arg58_1,
        arg2_1,
        full_1,
        arg0_1,
        partial_sum_weighted_xhat,
        partial_sum_weighted,
        position_out,
        vocab_out,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BATCH_=BATCH,
        VOCAB_=VOCAB,
        BLOCK_H=triton.next_power_of_2(HIDDEN),
        DROPOUT_SCALE_=DROPOUT_SCALE,
        num_warps=8,
    )

    block_h = 16
    _finalize_hidden_sums_kernel[(triton.cdiv(HIDDEN, block_h),)](
        partial_sum_weighted_xhat,
        partial_sum_weighted,
        out_sum_weighted_xhat,
        out_sum_weighted,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BLOCK_N=triton.next_power_of_2(SEQ),
        BLOCK_H=block_h,
        num_warps=8,
    )

    return out_sum_weighted_xhat, out_sum_weighted, position_out, vocab_out


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
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = oracle_distilbert_embedding_scatter_reduce(*oracle_inputs)
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)}")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        ok = ok and value_ok and shape_ok and dtype_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} shape_match={shape_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(rep: int, warmup: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = prepare_oracle_inputs(*make_inputs())
    with torch.no_grad():
        oracle_distilbert_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_distilbert_embedding_scatter_reduce(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"oracle_distilbert_embedding_scatter_reduce full-scope: {oracle_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("pass --check and/or --bench")

    ok = True
    if args.check:
        ok = run_check(args.rtol, args.atol)
    if args.bench:
        run_bench(args.rep, args.warmup)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
