"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete DistillGPT2 layernorm-backward/dropout return tuple, including the two hidden-column reductions, position-embedding scatter-add, and token-embedding scatter-add added to `mm[:50257]`, whereas Inductor currently materializes the rowwise layernorm/dropout producer and schedules the sibling reductions and duplicate-index `index_put(accumulate=True)` outputs as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured embedding scatter-reduce template that keeps row-local layernorm reduction scalars live while feeding multiple column reductions and indexed accumulator destinations; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that emits the rowwise layernorm math once, accumulates the hidden-column reductions, and atomically scatters position and token gradients directly into their destination rows."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_c01ffea2a781"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_distillgpt2_train_003_b6cfacd8"

BATCH = 32
SEQ = 512
HIDDEN = 768
ROWS = BATCH * SEQ
POSITION_ROWS = 1024
VOCAB = 50257
BLOCK_SUM_H = 16



if triton is not None:

    @triton.jit
    def _seq_layernorm_scatter_kernel(
        mm48_ptr,
        gamma_ptr,
        saved_ptr,
        position_base_ptr,
        dropout_mask_ptr,
        mean_ptr,
        invstd_ptr,
        add_ptr,
        position_idx_ptr,
        token_idx_ptr,
        partial_sum_xhat_ptr,
        partial_sum_plain_ptr,
        position_out_ptr,
        vocab_out_ptr,
        HIDDEN_: tl.constexpr,
        SEQ_: tl.constexpr,
        BATCH_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        VOCAB_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        seq = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        h_mask = h < HIDDEN_
        seq_offsets = seq * HIDDEN_ + h

        gamma = tl.load(gamma_ptr + h, mask=h_mask, other=0.0).to(tl.float32)
        position_base = tl.load(position_base_ptr + seq_offsets, mask=h_mask, other=0.0).to(tl.float32)

        acc_sum_xhat = tl.zeros([BLOCK_H_], dtype=tl.float32)
        acc_sum_plain = tl.zeros([BLOCK_H_], dtype=tl.float32)
        acc_position = tl.zeros([BLOCK_H_], dtype=tl.float32)

        for b in tl.static_range(0, BATCH_):
            row = b * SEQ_ + seq
            offsets = row * HIDDEN_ + h

            upstream = tl.load(mm48_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            weighted = upstream * gamma
            keep = tl.load(dropout_mask_ptr + offsets, mask=h_mask, other=0).to(tl.float32)
            saved_value = tl.load(saved_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            mean = tl.load(mean_ptr + row).to(tl.float32)
            invstd = tl.load(invstd_ptr + row).to(tl.float32)
            normalized = (keep * (saved_value + position_base) - mean) * invstd

            row_sum = tl.sum(tl.where(h_mask, weighted, 0.0), axis=0)
            row_dot = tl.sum(tl.where(h_mask, weighted * normalized, 0.0), axis=0)
            layernorm_grad = invstd * (weighted - (row_sum + normalized * row_dot) / HIDDEN_)

            residual_grad = tl.load(add_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            embed_grad = (residual_grad + layernorm_grad) * keep

            acc_sum_xhat += tl.where(h_mask, upstream * normalized, 0.0)
            acc_sum_plain += tl.where(h_mask, upstream, 0.0)
            acc_position += tl.where(h_mask, embed_grad, 0.0)

            token_raw = tl.load(token_idx_ptr + row).to(tl.int64)
            token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_, token_raw)
            token_value = tl.where(token_raw == -1, 0.0, embed_grad)
            token_valid = h_mask & (token_wrapped >= 0) & (token_wrapped < VOCAB_)
            tl.atomic_add(
                vocab_out_ptr + token_wrapped * HIDDEN_ + h,
                token_value,
                sem="relaxed",
                mask=token_valid,
            )

        position_raw = tl.load(position_idx_ptr + seq).to(tl.int64)
        position_wrapped = tl.where(position_raw < 0, position_raw + POSITION_ROWS_, position_raw)
        position_value = tl.where(position_raw == -1, 0.0, acc_position)
        position_valid = h_mask & (position_wrapped >= 0) & (position_wrapped < POSITION_ROWS_)
        tl.atomic_add(
            position_out_ptr + position_wrapped * HIDDEN_ + h,
            position_value,
            sem="relaxed",
            mask=position_valid,
        )

        partial_offsets = seq * HIDDEN_ + h
        tl.store(partial_sum_xhat_ptr + partial_offsets, acc_sum_xhat, mask=h_mask)
        tl.store(partial_sum_plain_ptr + partial_offsets, acc_sum_plain, mask=h_mask)

    @triton.jit
    def _finalize_hidden_sums_kernel(
        partial_sum_xhat_ptr,
        partial_sum_plain_ptr,
        out_sum_xhat_ptr,
        out_sum_plain_ptr,
        SEQ_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        h = tl.program_id(0) * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        n = tl.arange(0, BLOCK_N_)
        mask = (n[:, None] < SEQ_) & (h[None, :] < HIDDEN_)
        offsets = n[:, None] * HIDDEN_ + h[None, :]

        sum_xhat = tl.load(partial_sum_xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_plain = tl.load(partial_sum_plain_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out_mask = h < HIDDEN_
        tl.store(out_sum_xhat_ptr + h, tl.sum(sum_xhat, axis=0), mask=out_mask)
        tl.store(out_sum_plain_ptr + h, tl.sum(sum_plain, axis=0), mask=out_mask)


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
    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def oracle_distillgpt2_embedding_scatter_reduce(
    mm: torch.Tensor,
    mm_48: torch.Tensor,
    arg2_1: torch.Tensor,
    arg39_1: torch.Tensor,
    arg41_1: torch.Tensor,
    arg42_1: torch.Tensor,
    arg43_1: torch.Tensor,
    arg44_1: torch.Tensor,
    add_35: torch.Tensor,
    arg40_1: torch.Tensor,
    arg0_1: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if mm.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if list(_shape_param_0) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter: {_shape_param_0}")

    mm = mm.contiguous()
    mm_48 = mm_48.contiguous()
    arg2_1 = arg2_1.contiguous()
    arg39_1 = arg39_1.contiguous()
    arg41_1 = arg41_1.contiguous()
    arg42_1 = arg42_1.contiguous()
    arg43_1 = arg43_1.contiguous()
    arg44_1 = arg44_1.contiguous()
    add_35 = add_35.contiguous()
    arg40_1 = arg40_1.contiguous()
    arg0_1 = arg0_1.contiguous()

    assert mm.shape == (VOCAB + 3, HIDDEN)
    assert mm_48.shape == (ROWS, HIDDEN)
    assert arg2_1.shape == (HIDDEN,)
    assert arg39_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg41_1.shape == (1, SEQ, HIDDEN)
    assert arg42_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg43_1.shape == (BATCH, SEQ, 1)
    assert arg44_1.shape == (BATCH, SEQ, 1)
    assert add_35.shape == (BATCH, SEQ, HIDDEN)
    assert arg40_1.shape == (1, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)

    device = mm.device
    partial_sum_xhat = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    partial_sum_plain = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    out_sum_xhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_plain = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    position_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = mm[:VOCAB].clone()
    position_out.zero_()

    _seq_layernorm_scatter_kernel[(SEQ,)](
        mm_48,
        arg2_1,
        arg39_1,
        arg41_1,
        arg42_1,
        arg43_1,
        arg44_1,
        add_35,
        arg40_1,
        arg0_1,
        partial_sum_xhat,
        partial_sum_plain,
        position_out,
        vocab_out,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        BATCH_=BATCH,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_=VOCAB,
        BLOCK_H_=triton.next_power_of_2(HIDDEN),
        num_warps=8,
    )
    _finalize_hidden_sums_kernel[(triton.cdiv(HIDDEN, BLOCK_SUM_H),)](
        partial_sum_xhat,
        partial_sum_plain,
        out_sum_xhat,
        out_sum_plain,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BLOCK_N_=triton.next_power_of_2(SEQ),
        BLOCK_H_=BLOCK_SUM_H,
        num_warps=8,
    )

    return out_sum_xhat, out_sum_plain, position_out, vocab_out


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
    inputs = make_inputs(torch.device("cuda"))
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = oracle_distillgpt2_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)}")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        ok = ok and shape_ok and dtype_ok and stride_ok and value_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
            f"expected_stride={ref.stride()} max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={value_ok} shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(warmup: int, rep: int) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")
    if triton is None:
        raise RuntimeError("triton is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs(torch.device("cuda"))
    with torch.no_grad():
        oracle_distillgpt2_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_distillgpt2_embedding_scatter_reduce(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    dense_mb = VOCAB * HIDDEN * 4 / 1e6
    producer_mb = ROWS * HIDDEN * 4 / 1e6
    print(
        f"oracle_distillgpt2_embedding_scatter_reduce: {oracle_us:.3f} us shape={SHAPE_LABEL} "
        f"dense_output={dense_mb:.1f} MB row_producer={producer_mb:.1f} MB "
        f"warmup={warmup} rep={rep}"
    )
    return oracle_us


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare every oracle output against Repro.forward")
    parser.add_argument("--bench", action="store_true", help="time the Triton oracle")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("pass --check and/or --bench")

    ok = True
    if args.check:
        ok = run_check(rtol=args.rtol, atol=args.atol)
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
