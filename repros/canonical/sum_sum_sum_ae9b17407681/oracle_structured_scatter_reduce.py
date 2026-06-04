"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full GPT2 layernorm-backward and embedding-gradient tuple by fusing the hidden reductions, dropout/mask epilogue, and both duplicate-index scatter-add outputs into Triton row and scatter kernels, whereas Inductor materializes the `[4,512,768]` gradient producer and lowers the two `index_put(accumulate=True)` embedding updates plus sibling reductions as generic scheduled kernels; Inductor cannot do this today because scheduler/codegen has no structured scatter-reduce template that shares one rowwise layernorm-backward producer across multiple indexed accumulation outputs and column reductions; the fix is SCATTER_REDUCE: add an Inductor lowering for embedding-backward scatter-reduce fed by rowwise reductions that emits shared reduction/scatter epilogues without materializing the intermediate."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_ae9b17407681"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
SEQ = 512
HIDDEN = 768
ROWS = BATCH * SEQ
POSITION_ROWS = 1024
VOCAB_ROWS = 50257
PARTIAL_ROWS = SEQ
DROPOUT_SCALE = 1.1111111111111112



def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@triton.jit
def _seq_layernorm_scatter_kernel(
    mm96_ptr,
    weight_ptr,
    xhat_ptr,
    row_scale_ptr,
    add71_ptr,
    dropout_mask_ptr,
    position_idx_ptr,
    token_idx_ptr,
    partial_sum_xxhat_ptr,
    partial_sum_x_ptr,
    position_out_ptr,
    vocab_out_ptr,
    BATCH_: tl.constexpr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
):
    seq = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    h_mask = h < HIDDEN_
    weight = tl.load(weight_ptr + h, mask=h_mask, other=0.0).to(tl.float32)

    acc_sum_xxhat = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_sum_x = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_position_grad = tl.zeros([BLOCK_H], dtype=tl.float32)

    for b in tl.static_range(0, BATCH_):
        row = b * SEQ_ + seq
        offsets = row * HIDDEN_ + h

        x = tl.load(mm96_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
        weighted = x * weight
        row_sum = tl.sum(tl.where(h_mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(h_mask, weighted * xhat, 0.0), axis=0)
        scale = tl.load(row_scale_ptr + row).to(tl.float32)
        ln_grad = scale * (weighted * HIDDEN_ - row_sum - xhat * row_dot)

        add_value = tl.load(add71_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
        keep = tl.load(dropout_mask_ptr + offsets, mask=h_mask, other=0).to(tl.float32)
        grad = (add_value + ln_grad) * keep * DROPOUT_SCALE_

        acc_sum_xxhat += tl.where(h_mask, x * xhat, 0.0)
        acc_sum_x += tl.where(h_mask, x, 0.0)
        acc_position_grad += tl.where(h_mask, grad, 0.0)

        token_raw = tl.load(token_idx_ptr + row).to(tl.int64)
        token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_ROWS_, token_raw)
        token_valid = (
            h_mask
            & (token_raw != -1)
            & (token_wrapped >= 0)
            & (token_wrapped < VOCAB_ROWS_)
        )
        tl.atomic_add(
            vocab_out_ptr + token_wrapped * HIDDEN_ + h,
            grad,
            sem="relaxed",
            mask=token_valid,
        )

    position_raw = tl.load(position_idx_ptr + seq).to(tl.int64)
    position_wrapped = tl.where(position_raw < 0, position_raw + POSITION_ROWS_, position_raw)
    position_valid = (
        h_mask
        & (position_raw != -1)
        & (position_wrapped >= 0)
        & (position_wrapped < POSITION_ROWS_)
    )
    tl.atomic_add(
        position_out_ptr + position_wrapped * HIDDEN_ + h,
        acc_position_grad,
        sem="relaxed",
        mask=position_valid,
    )

    partial_offsets = seq * HIDDEN_ + h
    tl.store(partial_sum_xxhat_ptr + partial_offsets, acc_sum_xxhat, mask=h_mask)
    tl.store(partial_sum_x_ptr + partial_offsets, acc_sum_x, mask=h_mask)


@triton.jit
def _finalize_hidden_sums_kernel(
    partial_sum_xxhat_ptr,
    partial_sum_x_ptr,
    out_sum_xxhat_ptr,
    out_sum_x_ptr,
    ROW_BLOCKS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    h = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    n = tl.arange(0, BLOCK_N)
    mask = (n[:, None] < ROW_BLOCKS_) & (h[None, :] < HIDDEN_)
    offsets = n[:, None] * HIDDEN_ + h[None, :]

    sum_xxhat = tl.load(partial_sum_xxhat_ptr + offsets, mask=mask, other=0.0)
    sum_x = tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_sum_xxhat_ptr + h, tl.sum(sum_xxhat, axis=0), mask=h < HIDDEN_)
    tl.store(out_sum_x_ptr + h, tl.sum(sum_x, axis=0), mask=h < HIDDEN_)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_96,
        arg2_1,
        arg77_1,
        arg284_1,
        add_71,
        arg76_1,
        arg75_1,
        arg0_1,
        mm,
        _shape_param_0,
    ) = inputs
    del _shape_param_0
    return (
        mm_96.contiguous(),
        arg2_1.contiguous(),
        arg77_1.contiguous(),
        arg284_1.contiguous(),
        add_71.contiguous(),
        arg76_1.contiguous(),
        arg75_1.contiguous(),
        arg0_1.contiguous(),
        mm.contiguous(),
    )


def oracle_structured_scatter_reduce(
    mm_96: torch.Tensor,
    arg2_1: torch.Tensor,
    arg77_1: torch.Tensor,
    arg284_1: torch.Tensor,
    add_71: torch.Tensor,
    arg76_1: torch.Tensor,
    arg75_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_96.shape == (ROWS, HIDDEN)
    assert arg2_1.shape == (HIDDEN,)
    assert arg77_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg284_1.shape == (BATCH, SEQ, 1)
    assert add_71.shape == (BATCH, SEQ, HIDDEN)
    assert arg76_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg75_1.shape == (1, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm.shape == (VOCAB_ROWS, HIDDEN)

    device = mm_96.device
    partial_sum_xxhat = torch.empty((PARTIAL_ROWS, HIDDEN), device=device, dtype=torch.float32)
    partial_sum_x = torch.empty((PARTIAL_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_sum_xxhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    position_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = mm.clone()
    position_out.zero_()

    _seq_layernorm_scatter_kernel[(SEQ,)](
        mm_96,
        arg2_1,
        arg77_1,
        arg284_1,
        add_71,
        arg76_1,
        arg75_1,
        arg0_1,
        partial_sum_xxhat,
        partial_sum_x,
        position_out,
        vocab_out,
        BATCH_=BATCH,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        BLOCK_H=triton.next_power_of_2(HIDDEN),
        DROPOUT_SCALE_=DROPOUT_SCALE,
        num_warps=8,
    )

    finalize_block_h = 16
    _finalize_hidden_sums_kernel[(triton.cdiv(HIDDEN, finalize_block_h),)](
        partial_sum_xxhat,
        partial_sum_x,
        out_sum_xxhat,
        out_sum_x,
        ROW_BLOCKS_=PARTIAL_ROWS,
        HIDDEN_=HIDDEN,
        BLOCK_N=triton.next_power_of_2(PARTIAL_ROWS),
        BLOCK_H=finalize_block_h,
        num_warps=8,
    )

    return out_sum_xxhat, out_sum_x, position_out, vocab_out


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
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = oracle_structured_scatter_reduce(*prepare_oracle_inputs(*inputs))
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
            f"allclose={value_ok} shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(rep: int, warmup: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)
    with torch.no_grad():
        oracle_structured_scatter_reduce(*oracle_inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_structured_scatter_reduce(*oracle_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"oracle_structured_scatter_reduce full-scope: {oracle_us:.3f} us")


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
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup)


if __name__ == "__main__":
    with torch.no_grad():
        main()
