"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete GPT-2 large layernorm-backward plus dropout, position-embedding scatter-add, token-embedding scatter-add, and sibling hidden-column sums from one rowwise producer, whereas Inductor currently materializes the layernorm-gradient and dropout tensors then schedules the two sums and two index_put(accumulate=True) embedding-gradient consumers as separate generic reduction/scatter kernels; Inductor cannot do this today because scheduler/codegen has no structured embedding scatter-reduce template that shares rowwise layernorm reductions across sibling column reductions and multiple duplicate-index accumulation outputs with a dense add epilogue; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that computes each token row producer once, accumulates compatible hidden-column reductions, and atomically scatters position and token gradients directly into their destination rows."""
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
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_280231331614"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_hf_gpt2_large_train_002_634af270"

BATCH = 4
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 1280
POSITION_ROWS = 1024
VOCAB = 50257
DROPOUT_SCALE = 1.1111111111111112
BLOCK_SUM_H = 16

sys.path.insert(0, str(REPO_ROOT))


if triton is not None:

    @triton.jit
    def _seq_layernorm_scatter_kernel(
        mm_ptr,
        gamma_ptr,
        xhat_ptr,
        inv_factor_ptr,
        add_ptr,
        dropout_mask_ptr,
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
        BLOCK_H: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
    ):
        seq = tl.program_id(0)
        h = tl.arange(0, BLOCK_H)
        h_mask = h < HIDDEN_

        gamma = tl.load(gamma_ptr + h, mask=h_mask, other=0.0).to(tl.float32)

        acc_sum_xhat = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc_sum_plain = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc_position = tl.zeros([BLOCK_H], dtype=tl.float32)

        for b in tl.static_range(0, BATCH_):
            row = b * SEQ_ + seq
            offsets = row * HIDDEN_ + h
            mm = tl.load(mm_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            saved = tl.load(xhat_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            weighted = mm * gamma

            row_sum = tl.sum(tl.where(h_mask, weighted, 0.0), axis=0)
            row_dot = tl.sum(tl.where(h_mask, weighted * saved, 0.0), axis=0)
            inv_factor = tl.load(inv_factor_ptr + row).to(tl.float32)
            add = tl.load(add_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            keep = tl.load(dropout_mask_ptr + offsets, mask=h_mask, other=0).to(tl.float32)
            grad = add + inv_factor * (weighted * HIDDEN_ - row_sum - saved * row_dot)
            grad = grad * keep * DROPOUT_SCALE_

            acc_sum_xhat += tl.where(h_mask, mm * saved, 0.0)
            acc_sum_plain += tl.where(h_mask, mm, 0.0)
            acc_position += tl.where(h_mask, grad, 0.0)

            token_raw = tl.load(token_idx_ptr + row).to(tl.int64)
            token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_, token_raw)
            token_value = tl.where(token_raw == -1, 0.0, grad)
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
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        h = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
        n = tl.arange(0, BLOCK_N)
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


def oracle_embedding_scatter_reduce(
    mm_288: torch.Tensor,
    arg2_1: torch.Tensor,
    arg221_1: torch.Tensor,
    arg836_1: torch.Tensor,
    add_215: torch.Tensor,
    arg220_1: torch.Tensor,
    arg219_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if mm_288.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    assert mm_288.shape == (ROWS, HIDDEN)
    assert arg2_1.shape == (HIDDEN,)
    assert arg221_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg836_1.shape == (BATCH, SEQ, 1)
    assert add_215.shape == (BATCH, SEQ, HIDDEN)
    assert arg220_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg219_1.shape == (1, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm.shape == (VOCAB, HIDDEN)

    device = mm_288.device
    partial_sum_xhat = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    partial_sum_plain = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    out_sum_xhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_plain = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    position_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = mm.clone()
    position_out.zero_()

    _seq_layernorm_scatter_kernel[(SEQ,)](
        mm_288,
        arg2_1,
        arg221_1,
        arg836_1,
        add_215,
        arg220_1,
        arg219_1,
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
        BLOCK_H=triton.next_power_of_2(HIDDEN),
        DROPOUT_SCALE_=DROPOUT_SCALE,
        num_warps=8,
    )
    _finalize_hidden_sums_kernel[(triton.cdiv(HIDDEN, BLOCK_SUM_H),)](
        partial_sum_xhat,
        partial_sum_plain,
        out_sum_xhat,
        out_sum_plain,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BLOCK_N=triton.next_power_of_2(SEQ),
        BLOCK_H=BLOCK_SUM_H,
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
        actual = oracle_embedding_scatter_reduce(*inputs)
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


def run_bench(warmup: int, rep: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs(torch.device("cuda"))
    with torch.no_grad():
        oracle_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_embedding_scatter_reduce(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    dense_mb = VOCAB * HIDDEN * 4 / 1e6
    producer_mb = ROWS * HIDDEN * 4 / 1e6
    print(
        f"oracle_embedding_scatter_reduce: {oracle_us:.3f} us shape={SHAPE_LABEL} "
        f"dense_output={dense_mb:.1f} MB row_producer={producer_mb:.1f} MB "
        f"warmup={warmup} rep={rep}"
    )


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
