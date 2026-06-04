"""Full-scope oracle for sum_sum_sum_cad02fe18a69.

Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full
BERT embedding/layernorm-backward tuple in one row-wise Triton pass after a
Triton initialization pass: it forms the summed activation, computes the two
per-token layernorm reductions, emits the two per-hidden outer reductions, and
accumulates the same three indexed side outputs returned by Repro.forward.
Inductor currently materializes the layernorm-backward intermediate and lowers
the three `index_put(accumulate=True)` consumers plus sibling reductions as
generic scheduled work; it cannot represent this as one structured
gather/scatter-reduce producer with several accumulator destinations. The fix
is SCATTER_REDUCE: teach scheduler/codegen to recognize embedding/backward
row-index scatter-reduces fed by a shared rowwise reduction producer, then fuse
the per-hidden reductions and indexed accumulation epilogues without
materializing the full `[4,512,768]` intermediate.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_cad02fe18a69"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
SEQ = 512
HIDDEN = 768
ROWS = BATCH * SEQ
VOCAB = 30522

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]



@triton.jit
def _init_outputs_kernel(
    mm1_ptr,
    out0_ptr,
    out1_ptr,
    out_index512_ptr,
    out_index2_ptr,
    out_vocab_ptr,
    TOTAL_VOCAB: tl.constexpr,
    TOTAL_512: tl.constexpr,
    TOTAL_2: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    vocab_mask = offsets < TOTAL_VOCAB
    values = tl.load(mm1_ptr + offsets, mask=vocab_mask, other=0.0)
    tl.store(out_vocab_ptr + offsets, values, mask=vocab_mask)

    small_mask = offsets < TOTAL_512
    tl.store(out_index512_ptr + offsets, tl.zeros([BLOCK], tl.float32), mask=small_mask)

    two_mask = offsets < TOTAL_2
    tl.store(out_index2_ptr + offsets, tl.zeros([BLOCK], tl.float32), mask=two_mask)

    hidden_mask = offsets < HIDDEN_
    tl.store(out0_ptr + offsets, tl.zeros([BLOCK], tl.float32), mask=hidden_mask)
    tl.store(out1_ptr + offsets, tl.zeros([BLOCK], tl.float32), mask=hidden_mask)


@triton.jit
def _row_scatter_reduce_kernel(
    mm142_ptr,
    mul345_ptr,
    mm144_ptr,
    mm146_ptr,
    dropout_mask_ptr,
    weight_ptr,
    saved_x_ptr,
    row_scale_ptr,
    index512_ptr,
    full_ptr,
    index2_ptr,
    token_index_ptr,
    out0_ptr,
    out1_ptr,
    out_index512_ptr,
    out_index2_ptr,
    out_vocab_ptr,
    HIDDEN_: tl.constexpr,
    SEQ_: tl.constexpr,
    VOCAB_: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    mask = c < HIDDEN_
    base = row * HIDDEN_ + c
    seq = row % SEQ_

    x = (
        tl.load(mul345_ptr + base, mask=mask, other=0.0).to(tl.float32)
        + tl.load(mm142_ptr + base, mask=mask, other=0.0).to(tl.float32)
        + tl.load(mm144_ptr + base, mask=mask, other=0.0).to(tl.float32)
        + tl.load(mm146_ptr + base, mask=mask, other=0.0).to(tl.float32)
    )
    keep = tl.load(dropout_mask_ptr + base, mask=mask, other=0).to(tl.float32)
    mul1 = x * keep * 1.1111111111111112
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    saved_x = tl.load(saved_x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    mul2 = mul1 * weight
    weighted_saved = mul2 * saved_x
    row_sum = tl.sum(tl.where(mask, mul2, 0.0), axis=0)
    row_weighted_sum = tl.sum(tl.where(mask, weighted_saved, 0.0), axis=0)
    scale = tl.load(row_scale_ptr + row).to(tl.float32)
    grad = scale * (mul2 * 768.0 - row_sum - saved_x * row_weighted_sum)

    tl.atomic_add(out0_ptr + c, mul1 * saved_x, sem="relaxed", mask=mask)
    tl.atomic_add(out1_ptr + c, mul1, sem="relaxed", mask=mask)

    idx512 = tl.load(index512_ptr + seq).to(tl.int64)
    idx512_valid = idx512 >= 0
    tl.atomic_add(
        out_index512_ptr + idx512 * HIDDEN_ + c,
        grad,
        sem="relaxed",
        mask=mask & idx512_valid & (idx512 < SEQ_),
    )

    idx2 = tl.load(index2_ptr + seq).to(tl.int64)
    idx2_valid = idx2 >= 0
    tl.atomic_add(
        out_index2_ptr + idx2 * HIDDEN_ + c,
        grad,
        sem="relaxed",
        mask=mask & idx2_valid & (idx2 < 2),
    )

    token = tl.load(token_index_ptr + row).to(tl.int64)
    full_value = tl.load(full_ptr).to(tl.float32)
    vocab_value = tl.where(token == 0, full_value, grad)
    tl.atomic_add(
        out_vocab_ptr + token * HIDDEN_ + c,
        vocab_value,
        sem="relaxed",
        mask=mask & (token >= 0) & (token < VOCAB_),
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
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_bert_embedding_scatter_reduce(
    mm_142: torch.Tensor,
    mul_345: torch.Tensor,
    mm_144: torch.Tensor,
    mm_146: torch.Tensor,
    arg105_1: torch.Tensor,
    arg3_1: torch.Tensor,
    arg104_1: torch.Tensor,
    arg323_1: torch.Tensor,
    arg1_1: torch.Tensor,
    full_1: torch.Tensor,
    arg103_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3
    assert mm_142.shape == (ROWS, HIDDEN)
    assert mul_345.shape == (BATCH, SEQ, HIDDEN)
    assert mm_144.shape == (ROWS, HIDDEN)
    assert mm_146.shape == (ROWS, HIDDEN)
    assert arg105_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg3_1.shape == (HIDDEN,)
    assert arg104_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg323_1.shape == (BATCH, SEQ, 1)
    assert arg1_1.shape == (1, SEQ)
    assert arg103_1.shape == (1, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm_1.shape == (VOCAB, HIDDEN)

    out0 = torch.empty((HIDDEN,), device=mm_142.device, dtype=torch.float32)
    out1 = torch.empty((HIDDEN,), device=mm_142.device, dtype=torch.float32)
    out_index512 = torch.empty((SEQ, HIDDEN), device=mm_142.device, dtype=torch.float32)
    out_index2 = torch.empty((2, HIDDEN), device=mm_142.device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB, HIDDEN), device=mm_142.device, dtype=torch.float32)

    block = 1024
    total_vocab = VOCAB * HIDDEN
    total_512 = SEQ * HIDDEN
    total_2 = 2 * HIDDEN
    grid = (triton.cdiv(total_vocab, block),)
    _init_outputs_kernel[grid](
        mm_1,
        out0,
        out1,
        out_index512,
        out_index2,
        out_vocab,
        TOTAL_VOCAB=total_vocab,
        TOTAL_512=total_512,
        TOTAL_2=total_2,
        HIDDEN_=HIDDEN,
        BLOCK=block,
        num_warps=8,
    )
    _row_scatter_reduce_kernel[(ROWS,)](
        mm_142,
        mul_345,
        mm_144,
        mm_146,
        arg105_1,
        arg3_1,
        arg104_1,
        arg323_1,
        arg1_1,
        full_1,
        arg103_1,
        arg0_1,
        out0,
        out1,
        out_index512,
        out_index2,
        out_vocab,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        VOCAB_=VOCAB,
        BLOCK_C=triton.next_power_of_2(HIDDEN),
        num_warps=8,
    )
    return out0, out1, out_index512, out_index2, out_vocab


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
        actual = oracle_bert_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == ref.stride()
        dtype_ok = got.dtype == ref.dtype
        ok = ok and value_ok and stride_ok and dtype_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} "
            f"mean_abs={mean_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} stride_match={stride_ok} dtype_match={dtype_ok}"
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
        oracle_bert_embedding_scatter_reduce(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_bert_embedding_scatter_reduce(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"oracle_bert_embedding_scatter_reduce full-scope: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    for label, config in COMPILE_CONFIGS:
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
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
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
