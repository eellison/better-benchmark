"""
Full-scope oracle for sum_sum_sum_4de1559f24a7 (MegatronBert LN/dropout backward).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): the timed oracle covers
the same scope as `repro.py`, including the three matmul-gradient reshapes, the
shared add, affine multiply, per-row layernorm-backward reductions, dropout mask
scaling, the returned `[1024,8192]` permuted materialized buffer, and all three
returned `[1024]` column reductions. It differs from Inductor by streaming each
row tile once through a Triton producer that computes the dependent row scalars,
writes the side output, and records partials for the sibling column sums instead
of materializing/re-reading the row-local intermediates across separate template
regions. Inductor cannot express this today because the scheduler has to mix a
row reduction, downstream column reductions, and a required materialized side
output; the fix is COOPERATIVE_SPLIT_K support for dependent multi-output
reductions with partial buffers or equivalent atomic coordination.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_4de1559f24a7"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 16
SEQ = 512
C = 1024
ROWS = BATCH * SEQ
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
def _full_scope_partial_kernel(
    mm0_ptr,
    mm1_ptr,
    mm2_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    residual_ptr,
    keep_ptr,
    out_base_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_out_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    ROW_SPLIT: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C_
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_x = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_out = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * C_ + c[None, :]

        x0 = tl.load(mm0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(mm1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(mm2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = x0 + x1 + x2
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

        weighted = x * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
        out = residual + row_scale[:, None] * (
            weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None]
        )
        out = out * keep * DROPOUT_SCALE_

        tl.store(out_base_ptr + offsets, out, mask=mask)
        acc_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_out += tl.sum(tl.where(mask, out, 0.0), axis=0)

    partial_offsets = pid * C_ + c
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=c_mask)
    tl.store(partial_out_ptr + partial_offsets, acc_out, mask=c_mask)


@triton.jit
def _finalize_partials_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_out_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_sum_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block = tl.arange(0, BLOCK_ROW_BLOCKS)
    mask = (block[:, None] < NUM_ROW_BLOCKS) & (c[None, :] < C_)
    offsets = block[:, None] * C_ + c[None, :]

    x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.load(partial_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    c_mask = c < C_
    tl.store(out_x_rhs_ptr + c, tl.sum(x_rhs, axis=0), mask=c_mask)
    tl.store(out_x_ptr + c, tl.sum(x, axis=0), mask=c_mask)
    tl.store(out_sum_ptr + c, tl.sum(out, axis=0), mask=c_mask)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    mm_274: torch.Tensor,
    mm_276: torch.Tensor,
    mm_278: torch.Tensor,
    arg11_1: torch.Tensor,
    arg216_1: torch.Tensor,
    arg641_1: torch.Tensor,
    add_137: torch.Tensor,
    arg215_1: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_274.shape == (ROWS, C)
    assert mm_276.shape == (ROWS, C)
    assert mm_278.shape == (ROWS, C)
    assert arg11_1.shape == (C,)
    assert arg216_1.shape == (BATCH, SEQ, C)
    assert arg641_1.shape == (BATCH, SEQ, 1)
    assert add_137.shape == (BATCH, SEQ, C)
    assert arg215_1.shape == (BATCH, SEQ, C)

    x0 = mm_274.contiguous()
    x1 = mm_276.contiguous()
    x2 = mm_278.contiguous()
    weight = arg11_1.contiguous()
    rhs = arg216_1.contiguous().reshape(ROWS, C)
    row_scale = arg641_1.contiguous().reshape(ROWS)
    residual = add_137.contiguous().reshape(ROWS, C)
    keep = arg215_1.contiguous().reshape(ROWS, C)

    row_split = 16
    xblock = 1
    block_c = 1024
    num_row_blocks = triton.cdiv(ROWS, row_split)

    partials = torch.empty((3, num_row_blocks, C), device=x0.device, dtype=torch.float32)
    partial_x_rhs = partials[0]
    partial_x = partials[1]
    partial_out = partials[2]
    out_base = torch.empty((ROWS, C), device=x0.device, dtype=torch.float32)

    _full_scope_partial_kernel[(num_row_blocks,)](
        x0,
        x1,
        x2,
        weight,
        rhs,
        row_scale,
        residual,
        keep,
        out_base,
        partial_x_rhs,
        partial_x,
        partial_out,
        ROWS_=ROWS,
        C_=C,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        ROW_SPLIT=row_split,
        XBLOCK=xblock,
        BLOCK_C=block_c,
        num_warps=8,
    )

    vector_outputs = torch.empty((3, C), device=x0.device, dtype=torch.float32)
    out_x_rhs = vector_outputs[0]
    out_x = vector_outputs[1]
    out_sum = vector_outputs[2]
    finalize_block_c = 16
    _finalize_partials_kernel[(triton.cdiv(C, finalize_block_c),)](
        partial_x_rhs,
        partial_x,
        partial_out,
        out_x_rhs,
        out_x,
        out_sum,
        NUM_ROW_BLOCKS=num_row_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=finalize_block_c,
        num_warps=8,
    )

    return out_x_rhs, out_x, out_base.t(), out_sum


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
    print(f"oracle_fused full-scope MegatronBert LN/dropout reductions: {oracle_us:.3f} us")

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
