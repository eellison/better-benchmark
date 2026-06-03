"""
Canonical oracle for sum_sum_sum_6770dc6efbe8 (Swin patch-expand backward).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle differs from Inductor by making the final patch-expand/dropout producer also cooperatively reduce the returned `[401408, 128]` buffer into the sibling `[128]` output, while separate kernels compute the two shared `[512]` column reductions and the per-row dual reduction needed by the producer; Inductor cannot do this today because the final `sum(dim=0)` has a tiny output and very large reduction extent but is attached to a required materialized transpose/reshape side output, so the scheduler keeps the side output and reduction as ordinary producer/consumer work instead of emitting one split-K producer with atomic/partial coordination; the fix is COOPERATIVE_SPLIT_K support for reductions with materialized side outputs, so a pointwise producer can write the returned buffer and accumulate the sibling reduction across row tiles without rereading the buffer.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_6770dc6efbe8"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
H = 28
W = 28
C = 512
K = 128
ROWS = N * H * W
OUT_H = H * 2
OUT_W = W * 2
OUT_ROWS_PER_N = OUT_H * OUT_W
OUT_ROWS = N * OUT_ROWS_PER_N
DROPOUT_SCALE = 1.0 / 0.9956521736457944


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@triton.jit
def _summary_reduce_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    row_sum_ptr,
    row_dot_ptr,
    partial_sum_x_rhs_ptr,
    partial_sum_x_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    ROW_SPLIT: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C_
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    acc_sum_x_rhs = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_sum_x = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * C_ + c[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        xw = x * weight[None, :]

        tl.store(row_sum_ptr + row, tl.sum(tl.where(mask, xw, 0.0), axis=1), mask=row_mask)
        tl.store(row_dot_ptr + row, tl.sum(tl.where(mask, xw * rhs, 0.0), axis=1), mask=row_mask)

        acc_sum_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        acc_sum_x += tl.sum(tl.where(mask, x, 0.0), axis=0)

    partial_offsets = pid * C_ + c
    tl.store(partial_sum_x_rhs_ptr + partial_offsets, acc_sum_x_rhs, mask=c_mask)
    tl.store(partial_sum_x_ptr + partial_offsets, acc_sum_x, mask=c_mask)


@triton.jit
def _finalize_column_partials_kernel(
    partial_sum_x_rhs_ptr,
    partial_sum_x_ptr,
    out_sum_x_rhs_ptr,
    out_sum_x_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block = tl.arange(0, BLOCK_ROW_BLOCKS)
    mask = (block[:, None] < NUM_ROW_BLOCKS) & (c[None, :] < C_)
    offsets = block[:, None] * C_ + c[None, :]

    sum_x_rhs = tl.load(partial_sum_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_sum_x_rhs_ptr + c, tl.sum(sum_x_rhs, axis=0), mask=c < C_)
    tl.store(out_sum_x_ptr + c, tl.sum(sum_x, axis=0), mask=c < C_)


@triton.jit
def _patch_expand_store_and_reduce_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    gate_ptr,
    keep_ptr,
    row_sum_ptr,
    row_dot_ptr,
    out_base_ptr,
    partial_out_sum_ptr,
    OUT_ROWS_: tl.constexpr,
    OUT_ROWS_PER_N_: tl.constexpr,
    OUT_W_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    C_: tl.constexpr,
    K_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    BLOCK_OUT_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    out_row = tl.program_id(0) * BLOCK_OUT_ROWS + tl.arange(0, BLOCK_OUT_ROWS)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)

    row_mask = out_row < OUT_ROWS_
    k_mask = k < K_
    mask = row_mask[:, None] & k_mask[None, :]

    n = out_row // OUT_ROWS_PER_N_
    inner = out_row - n * OUT_ROWS_PER_N_
    out_h = inner // OUT_W_
    out_w = inner - out_h * OUT_W_
    src_h = out_h // 2
    src_w = out_w // 2
    split_b = out_h - src_h * 2
    split_a = out_w - src_w * 2

    src_row = n * (H_ * W_) + src_h * W_ + src_w
    c = split_a[:, None] * 256 + split_b[:, None] * 128 + k[None, :]
    src_offsets = src_row[:, None] * C_ + c

    x = tl.load(x_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + src_row, mask=row_mask, other=0.0).to(tl.float32)
    keep = tl.load(keep_ptr + n, mask=row_mask, other=0).to(tl.float32) * DROPOUT_SCALE_
    row_sum = tl.load(row_sum_ptr + src_row, mask=row_mask, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + src_row, mask=row_mask, other=0.0).to(tl.float32)

    value = gate[:, None] * (x * weight * 512.0 - row_sum[:, None] - rhs * row_dot[:, None])
    value = value * keep[:, None]

    out_offsets = out_row[:, None] * K_ + k[None, :]
    tl.store(out_base_ptr + out_offsets, value, mask=mask)

    partial = tl.sum(tl.where(mask, value, 0.0), axis=0)
    partial_offsets = tl.program_id(0) * K_ + k
    tl.store(partial_out_sum_ptr + partial_offsets, partial, mask=k_mask)


@triton.jit
def _finalize_out_sum_kernel(
    partial_out_sum_ptr,
    out_sum_ptr,
    NUM_OUT_ROW_BLOCKS: tl.constexpr,
    K_: tl.constexpr,
    BLOCK_OUT_ROW_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    block = tl.arange(0, BLOCK_OUT_ROW_BLOCKS)
    mask = (block[:, None] < NUM_OUT_ROW_BLOCKS) & (k[None, :] < K_)
    offsets = block[:, None] * K_ + k[None, :]

    partial = tl.load(partial_out_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_sum_ptr + k, tl.sum(partial, axis=0), mask=k < K_)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_183,
        arg18_1,
        arg203_1,
        arg549_1,
        arg202_1,
        *_shape_params,
    ) = inputs

    return (
        mm_183.contiguous(),
        arg18_1.contiguous(),
        arg203_1.contiguous(),
        arg549_1.reshape(ROWS).contiguous(),
        arg202_1.reshape(N).contiguous(),
    )


def oracle_fused(
    mm_183: torch.Tensor,
    arg18_1: torch.Tensor,
    arg203_1: torch.Tensor,
    arg549_flat: torch.Tensor,
    arg202_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_183.shape == (ROWS, C)
    assert arg18_1.shape == (C,)
    assert arg203_1.shape == (N, H, W, C)
    assert arg549_flat.shape == (ROWS,)
    assert arg202_flat.shape == (N,)

    rhs = arg203_1.reshape(ROWS, C)
    device = mm_183.device

    row_sum = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_split = 128
    xblock = 2
    num_summary_blocks = triton.cdiv(ROWS, row_split)
    partial_sum_x_rhs = torch.empty((num_summary_blocks, C), device=device, dtype=torch.float32)
    partial_sum_x = torch.empty((num_summary_blocks, C), device=device, dtype=torch.float32)
    _summary_reduce_kernel[(num_summary_blocks,)](
        mm_183,
        arg18_1,
        rhs,
        row_sum,
        row_dot,
        partial_sum_x_rhs,
        partial_sum_x,
        ROWS_=ROWS,
        C_=C,
        ROW_SPLIT=row_split,
        XBLOCK=xblock,
        BLOCK_C=512,
        num_warps=1,
    )

    sum_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    sum_x = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_partials_kernel[(triton.cdiv(C, 16),)](
        partial_sum_x_rhs,
        partial_sum_x,
        sum_x_rhs,
        sum_x,
        NUM_ROW_BLOCKS=num_summary_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS=1024,
        BLOCK_C=16,
        num_warps=8,
    )

    out_base = torch.empty((OUT_ROWS, K), device=device, dtype=torch.float32)
    block_out_rows = 64
    block_k = 64
    num_out_row_blocks = triton.cdiv(OUT_ROWS, block_out_rows)
    partial_out_sum = torch.empty((num_out_row_blocks, K), device=device, dtype=torch.float32)
    _patch_expand_store_and_reduce_kernel[
        (num_out_row_blocks, triton.cdiv(K, block_k))
    ](
        mm_183,
        arg18_1,
        rhs,
        arg549_flat,
        arg202_flat,
        row_sum,
        row_dot,
        out_base,
        partial_out_sum,
        OUT_ROWS_=OUT_ROWS,
        OUT_ROWS_PER_N_=OUT_ROWS_PER_N,
        OUT_W_=OUT_W,
        H_=H,
        W_=W,
        C_=C,
        K_=K,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_OUT_ROWS=block_out_rows,
        BLOCK_K=block_k,
        num_warps=4,
    )
    out_sum = torch.empty((K,), device=device, dtype=torch.float32)
    _finalize_out_sum_kernel[(triton.cdiv(K, 8),)](
        partial_out_sum,
        out_sum,
        NUM_OUT_ROW_BLOCKS=num_out_row_blocks,
        K_=K,
        BLOCK_OUT_ROW_BLOCKS=8192,
        BLOCK_K=8,
        num_warps=8,
    )

    return sum_x_rhs, sum_x, out_base.permute(1, 0), out_sum


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


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
        oracle_inputs = prepare_oracle_inputs(*inputs)
        actual = oracle_fused(*oracle_inputs)
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

    inputs = make_inputs()
    with torch.no_grad():
        oracle_inputs = prepare_oracle_inputs(*inputs)
        oracle_fused(*oracle_inputs)
        torch.cuda.synchronize()

        oracle_us = triton.testing.do_bench(
            lambda: oracle_fused(*oracle_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    print(f"oracle_fused cooperative patch-expand reduction: {oracle_us:.3f} us")

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
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-2)
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
