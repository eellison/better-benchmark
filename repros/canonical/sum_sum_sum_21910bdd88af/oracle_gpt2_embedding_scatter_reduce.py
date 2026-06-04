"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full GPT2 layernorm-backward/dropout return tuple by fusing the shared rowwise hidden reductions with both duplicate-index position and vocabulary scatter-add outputs, whereas Inductor currently materializes the `[8,1024,768]` producer and schedules the sibling hidden reductions plus two `index_put(accumulate=True)` outputs as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured scatter-reduce template that keeps row-local layernorm reduction scalars live while directly feeding multiple indexed accumulation destinations and column reductions; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that computes each token-row producer once and emits the hidden reductions and indexed accumulation epilogues without materializing the intermediate."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile/import usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_21910bdd88af"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
SEQ = 1024
HIDDEN = 768
ROWS = BATCH * SEQ
POSITION_ROWS = 1024
VOCAB_ROWS = 50257
DROPOUT_SCALE = 1.1111111111111112
INIT_BLOCK = 1024
HIDDEN_BLOCK = 1024
FINAL_BLOCK_H = 16



if triton is not None:

    @triton.jit
    def _init_outputs_kernel(
        out_sum_x_centered_ptr,
        out_sum_x_ptr,
        position_out_ptr,
        vocab_out_ptr,
        TOTAL_POSITION_: tl.constexpr,
        TOTAL_VOCAB_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        zeros = tl.zeros([BLOCK_], dtype=tl.float32)

        hidden_mask = offsets < HIDDEN_
        position_mask = offsets < TOTAL_POSITION_
        vocab_mask = offsets < TOTAL_VOCAB_
        tl.store(out_sum_x_centered_ptr + offsets, zeros, mask=hidden_mask)
        tl.store(out_sum_x_ptr + offsets, zeros, mask=hidden_mask)
        tl.store(position_out_ptr + offsets, zeros, mask=position_mask)
        tl.store(vocab_out_ptr + offsets, zeros, mask=vocab_mask)


    @triton.jit
    def _seq_layernorm_scatter_kernel(
        mm96_ptr,
        weight_ptr,
        arg75_ptr,
        arg77_ptr,
        dropout_mask_ptr,
        mean_ptr,
        invstd_ptr,
        add71_ptr,
        position_idx_ptr,
        full_ptr,
        token_idx_ptr,
        partial_sum_x_centered_ptr,
        partial_sum_x_ptr,
        position_out_ptr,
        vocab_out_ptr,
        BATCH_: tl.constexpr,
        SEQ_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        VOCAB_ROWS_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        seq = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        h_mask = h < HIDDEN_
        weight = tl.load(weight_ptr + h, mask=h_mask, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        acc_sum_x_centered = tl.zeros([BLOCK_H_], dtype=tl.float32)
        acc_sum_x = tl.zeros([BLOCK_H_], dtype=tl.float32)
        acc_position_grad = tl.zeros([BLOCK_H_], dtype=tl.float32)

        seq_hidden_offsets = seq * HIDDEN_ + h
        broadcast_bias = tl.load(arg77_ptr + seq_hidden_offsets, mask=h_mask, other=0.0).to(
            tl.float32
        )

        for batch in tl.static_range(0, BATCH_):
            row = batch * SEQ_ + seq
            offsets = row * HIDDEN_ + h

            x = tl.load(mm96_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            weighted = x * weight
            row_sum = tl.sum(tl.where(h_mask, weighted, 0.0), axis=0)

            add_saved = tl.load(arg75_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            keep = tl.load(dropout_mask_ptr + offsets, mask=h_mask, other=0).to(tl.float32)
            dropped_saved = keep * (add_saved + broadcast_bias) * DROPOUT_SCALE_
            mean = tl.load(mean_ptr + row).to(tl.float32)
            invstd = tl.load(invstd_ptr + row).to(tl.float32)
            centered = (dropped_saved - mean) * invstd

            row_dot = tl.sum(tl.where(h_mask, weighted * centered, 0.0), axis=0)
            ln_grad = (invstd * (1.0 / HIDDEN_)) * (
                weighted * HIDDEN_ - row_sum - centered * row_dot
            )

            residual = tl.load(add71_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            producer = (residual + ln_grad) * keep * DROPOUT_SCALE_

            acc_sum_x_centered += tl.where(h_mask, x * centered, 0.0)
            acc_sum_x += tl.where(h_mask, x, 0.0)
            acc_position_grad += tl.where(h_mask, producer, 0.0)

            token_raw = tl.load(token_idx_ptr + row).to(tl.int64)
            token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_ROWS_, token_raw)
            token_value = tl.where(token_raw == -1, full_value, producer)
            token_valid = h_mask & (token_wrapped >= 0) & (token_wrapped < VOCAB_ROWS_)
            tl.atomic_add(
                vocab_out_ptr + token_wrapped * HIDDEN_ + h,
                token_value,
                sem="relaxed",
                mask=token_valid,
            )

        position_raw = tl.load(position_idx_ptr + seq).to(tl.int64)
        position_wrapped = tl.where(
            position_raw < 0,
            position_raw + POSITION_ROWS_,
            position_raw,
        )
        position_value = tl.where(position_raw == -1, full_value, acc_position_grad)
        position_valid = h_mask & (position_wrapped >= 0) & (position_wrapped < POSITION_ROWS_)
        tl.atomic_add(
            position_out_ptr + position_wrapped * HIDDEN_ + h,
            position_value,
            sem="relaxed",
            mask=position_valid,
        )

        partial_offsets = seq * HIDDEN_ + h
        tl.store(partial_sum_x_centered_ptr + partial_offsets, acc_sum_x_centered, mask=h_mask)
        tl.store(partial_sum_x_ptr + partial_offsets, acc_sum_x, mask=h_mask)


    @triton.jit
    def _finalize_hidden_sums_kernel(
        partial_sum_x_centered_ptr,
        partial_sum_x_ptr,
        out_sum_x_centered_ptr,
        out_sum_x_ptr,
        SEQ_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        h = tl.program_id(0) * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        n = tl.arange(0, BLOCK_N_)
        mask = (n[:, None] < SEQ_) & (h[None, :] < HIDDEN_)
        offsets = n[:, None] * HIDDEN_ + h[None, :]

        sum_x_centered = tl.load(partial_sum_x_centered_ptr + offsets, mask=mask, other=0.0)
        sum_x = tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_sum_x_centered_ptr + h, tl.sum(sum_x_centered, axis=0), mask=h < HIDDEN_)
        tl.store(out_sum_x_ptr + h, tl.sum(sum_x, axis=0), mask=h < HIDDEN_)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(device: torch.device) -> tuple[object, ...]:
    module = _load_repro_module()
    inputs = module.make_inputs()
    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )


def oracle_gpt2_embedding_scatter_reduce(
    mm_96: torch.Tensor,
    arg1_1: torch.Tensor,
    arg75_1: torch.Tensor,
    arg77_1: torch.Tensor,
    arg79_1: torch.Tensor,
    arg80_1: torch.Tensor,
    arg81_1: torch.Tensor,
    add_71: torch.Tensor,
    arg76_1: torch.Tensor,
    full: torch.Tensor,
    arg0_1: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if mm_96.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if list(_shape_param_0) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter: {_shape_param_0}")

    mm_96 = mm_96.contiguous()
    arg1_1 = arg1_1.contiguous()
    arg75_1 = arg75_1.contiguous()
    arg77_1 = arg77_1.contiguous()
    arg79_1 = arg79_1.contiguous()
    arg80_1 = arg80_1.contiguous()
    arg81_1 = arg81_1.contiguous()
    add_71 = add_71.contiguous()
    arg76_1 = arg76_1.contiguous()
    full = full.contiguous()
    arg0_1 = arg0_1.contiguous()

    assert mm_96.shape == (ROWS, HIDDEN)
    assert arg1_1.shape == (HIDDEN,)
    assert arg75_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg77_1.shape == (1, SEQ, HIDDEN)
    assert arg79_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg80_1.shape == (BATCH, SEQ, 1)
    assert arg81_1.shape == (BATCH, SEQ, 1)
    assert add_71.shape == (BATCH, SEQ, HIDDEN)
    assert arg76_1.shape == (1, SEQ)
    assert full.shape == ()
    assert arg0_1.shape == (BATCH, SEQ)

    device = mm_96.device
    partial_sum_x_centered = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    partial_sum_x = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    out_sum_x_centered = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    position_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    total_position = POSITION_ROWS * HIDDEN
    total_vocab = VOCAB_ROWS * HIDDEN
    _init_outputs_kernel[(triton.cdiv(total_vocab, INIT_BLOCK),)](
        out_sum_x_centered,
        out_sum_x,
        position_out,
        vocab_out,
        TOTAL_POSITION_=total_position,
        TOTAL_VOCAB_=total_vocab,
        HIDDEN_=HIDDEN,
        BLOCK_=INIT_BLOCK,
        num_warps=8,
    )

    _seq_layernorm_scatter_kernel[(SEQ,)](
        mm_96,
        arg1_1,
        arg75_1,
        arg77_1,
        arg79_1,
        arg80_1,
        arg81_1,
        add_71,
        arg76_1,
        full,
        arg0_1,
        partial_sum_x_centered,
        partial_sum_x,
        position_out,
        vocab_out,
        BATCH_=BATCH,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=HIDDEN_BLOCK,
        num_warps=8,
    )

    _finalize_hidden_sums_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_H),)](
        partial_sum_x_centered,
        partial_sum_x,
        out_sum_x_centered,
        out_sum_x,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BLOCK_N_=triton.next_power_of_2(SEQ),
        BLOCK_H_=FINAL_BLOCK_H,
        num_warps=8,
    )
    return out_sum_x_centered, out_sum_x, position_out, vocab_out


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs(torch.device("cuda"))
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = oracle_gpt2_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: got={len(actual)} expected={len(expected)}")

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
    return bool(ok)


def run_bench(rep: int, warmup: int) -> float:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs(torch.device("cuda"))
    with torch.no_grad():
        oracle_gpt2_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_gpt2_embedding_scatter_reduce(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"oracle_gpt2_embedding_scatter_reduce full-scope: {oracle_us:.3f} us")
    return float(oracle_us)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare every oracle output with repro.py")
    parser.add_argument("--bench", action="store_true", help="benchmark the full-scope oracle")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup)


if __name__ == "__main__":
    with torch.no_grad():
        main()
