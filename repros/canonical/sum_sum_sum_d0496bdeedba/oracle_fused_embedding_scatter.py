"""
Oracle for sum_sum_sum_d0496bdeedba (GPT-Neo layer-norm backward plus embedding scatters).

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the layer-norm-backward row reductions, the two column reductions, and both returned embedding-gradient scatter-add outputs in one Triton producer pass, so it never writes the intermediate `sub_2` tensor that Inductor currently materializes and rereads to build the position-embedding scatter; Inductor can fuse the word-embedding scatter with the row-reduction producer, but it splits the position path because that consumer is a second reduction/scatter over the batch dimension with a different output layout, so the scheduler has no representation for one dependent multi-output reduction with two side-effect scatter stores; the Inductor fix is scheduler/codegen support for this fused multi-output reduction/scatter template, allowing a row-reduction producer to feed sibling column accumulators and multiple final scatter-add side outputs without forcing an intermediate tensor.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_d0496bdeedba"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
SEQ = 128
HIDDEN = 2048
ROWS = BATCH * SEQ
POSITION_ROWS = 2048
VOCAB_ROWS = 50257
ROW_BLOCK = 16
ROW_BLOCKS = triton.cdiv(ROWS, ROW_BLOCK)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@triton.jit
def _fused_ln_embedding_scatter_kernel(
    mm0_ptr,
    mm1_ptr,
    mm2_ptr,
    gamma_ptr,
    arg219_ptr,
    arg222_ptr,
    arg223_ptr,
    arg224_ptr,
    add189_ptr,
    pos_idx_ptr,
    full_ptr,
    word_idx_ptr,
    partial_sum_ax_ptr,
    partial_sum_a_ptr,
    pos_out_ptr,
    word_out_ptr,
    ROWS_: tl.constexpr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BATCH_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    ROW_BLOCK_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    block = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    h_mask = h < HIDDEN_
    gamma = tl.load(gamma_ptr + h, mask=h_mask, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    acc_sum_ax = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_sum_a = tl.zeros([BLOCK_H], dtype=tl.float32)

    for i in tl.static_range(0, ROW_BLOCK_):
        row = block * ROW_BLOCK_ + i
        row_mask = row < ROWS_
        token = row % SEQ_
        offsets = row * HIDDEN_ + h
        token_offsets = token * HIDDEN_ + h
        elem_mask = row_mask & h_mask

        mm0 = tl.load(mm0_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        mm1 = tl.load(mm1_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        mm2 = tl.load(mm2_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        a = mm0 + mm1 + mm2

        y = a * gamma
        row_sum = tl.sum(y, axis=0)

        aux = tl.load(arg219_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        pos_aux = tl.load(arg222_ptr + token_offsets, mask=h_mask, other=0.0).to(tl.float32)
        center = tl.load(arg223_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        scale = tl.load(arg224_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        xhat = (aux + pos_aux - center) * scale

        row_dot = tl.sum(y * xhat, axis=0)
        pre_scaled_grad = y * HIDDEN_ - row_sum - xhat * row_dot
        grad = tl.load(add189_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        grad += (scale * (1.0 / HIDDEN_)) * pre_scaled_grad

        word_raw = tl.load(word_idx_ptr + row, mask=row_mask, other=0).to(tl.int64)
        word_wrapped = tl.where(word_raw < 0, word_raw + VOCAB_ROWS_, word_raw)
        word_valid = row_mask & (word_wrapped >= 0) & (word_wrapped < VOCAB_ROWS_) & h_mask
        word_value = tl.where(word_raw == -1, full_value, grad)
        tl.atomic_add(
            word_out_ptr + word_wrapped * HIDDEN_ + h,
            word_value,
            sem="relaxed",
            mask=word_valid,
        )

        pos_raw = tl.load(pos_idx_ptr + token, mask=row_mask, other=0).to(tl.int64)
        pos_wrapped = tl.where(pos_raw < 0, pos_raw + POSITION_ROWS_, pos_raw)
        pos_valid = row_mask & (pos_wrapped >= 0) & (pos_wrapped < POSITION_ROWS_) & h_mask
        pos_value = tl.where(pos_raw == -1, full_value / BATCH_, grad)
        tl.atomic_add(
            pos_out_ptr + pos_wrapped * HIDDEN_ + h,
            pos_value,
            sem="relaxed",
            mask=pos_valid,
        )

        acc_sum_ax += tl.where(elem_mask, a * xhat, 0.0)
        acc_sum_a += tl.where(elem_mask, a, 0.0)

    partial_offsets = block * HIDDEN_ + h
    tl.store(partial_sum_ax_ptr + partial_offsets, acc_sum_ax, mask=h_mask)
    tl.store(partial_sum_a_ptr + partial_offsets, acc_sum_a, mask=h_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_sum_ax_ptr,
    partial_sum_a_ptr,
    out_sum_ax_ptr,
    out_sum_a_ptr,
    ROW_BLOCKS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    h = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    n = tl.arange(0, BLOCK_N)
    mask = (n[:, None] < ROW_BLOCKS_) & (h[None, :] < HIDDEN_)
    offsets = n[:, None] * HIDDEN_ + h[None, :]

    sum_ax = tl.load(partial_sum_ax_ptr + offsets, mask=mask, other=0.0)
    sum_a = tl.load(partial_sum_a_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_sum_ax_ptr + h, tl.sum(sum_ax, axis=0), mask=h < HIDDEN_)
    tl.store(out_sum_a_ptr + h, tl.sum(sum_a, axis=0), mask=h < HIDDEN_)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_285,
        mm_287,
        mm_289,
        arg1_1,
        arg219_1,
        arg222_1,
        arg223_1,
        arg224_1,
        add_189,
        arg220_1,
        full,
        arg0_1,
        *_shape_params,
    ) = inputs

    return (
        mm_285.contiguous(),
        mm_287.contiguous(),
        mm_289.contiguous(),
        arg1_1.contiguous(),
        arg219_1.contiguous(),
        arg222_1.contiguous(),
        arg223_1.contiguous(),
        arg224_1.contiguous(),
        add_189.contiguous(),
        arg220_1.contiguous(),
        full.contiguous(),
        arg0_1.contiguous(),
    )


def oracle_fused(
    mm_285: torch.Tensor,
    mm_287: torch.Tensor,
    mm_289: torch.Tensor,
    arg1_1: torch.Tensor,
    arg219_1: torch.Tensor,
    arg222_1: torch.Tensor,
    arg223_1: torch.Tensor,
    arg224_1: torch.Tensor,
    add_189: torch.Tensor,
    arg220_1: torch.Tensor,
    full: torch.Tensor,
    arg0_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_285.shape == (ROWS, HIDDEN)
    assert mm_287.shape == (ROWS, HIDDEN)
    assert mm_289.shape == (ROWS, HIDDEN)
    assert arg1_1.shape == (HIDDEN,)
    assert arg219_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg222_1.shape == (1, SEQ, HIDDEN)
    assert arg223_1.shape == (BATCH, SEQ, 1)
    assert arg224_1.shape == (BATCH, SEQ, 1)
    assert add_189.shape == (BATCH, SEQ, HIDDEN)
    assert arg220_1.shape == (1, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)

    device = mm_285.device
    partial_sum_ax = torch.empty((ROW_BLOCKS, HIDDEN), device=device, dtype=torch.float32)
    partial_sum_a = torch.empty((ROW_BLOCKS, HIDDEN), device=device, dtype=torch.float32)
    pos_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    word_out = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)
    pos_out.zero_()
    word_out.zero_()

    _fused_ln_embedding_scatter_kernel[(ROW_BLOCKS,)](
        mm_285,
        mm_287,
        mm_289,
        arg1_1,
        arg219_1,
        arg222_1,
        arg223_1,
        arg224_1,
        add_189,
        arg220_1,
        full,
        arg0_1,
        partial_sum_ax,
        partial_sum_a,
        pos_out,
        word_out,
        ROWS_=ROWS,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BATCH_=BATCH,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        ROW_BLOCK_=ROW_BLOCK,
        BLOCK_H=HIDDEN,
        num_warps=8,
    )

    sum_ax = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_a = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    block_h = 16
    _finalize_column_sums_kernel[(triton.cdiv(HIDDEN, block_h),)](
        partial_sum_ax,
        partial_sum_a,
        sum_ax,
        sum_a,
        ROW_BLOCKS_=ROW_BLOCKS,
        HIDDEN_=HIDDEN,
        BLOCK_N=ROW_BLOCKS,
        BLOCK_H=block_h,
        num_warps=8,
    )

    return sum_ax, sum_a, pos_out, word_out


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


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
        actual = oracle_fused(*prepare_oracle_inputs(*inputs))
        torch.cuda.synchronize()

    ok = True
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        output_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={output_ok}"
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

    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)
    with torch.no_grad():
        oracle_fused(*oracle_inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_fused(*oracle_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    print(f"oracle_fused embedding scatter: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    compile_configs = [
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
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark oracle")
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
