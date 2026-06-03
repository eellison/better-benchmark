"""
Full-scope oracle for sum_sum_sum_59e022113eeb (DistilBERT LN backward tail).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle covers the
same scope as `repro.py`, including the `mm_66` view, residual add with
`mul_152`, affine multiply by `arg8_1`, the two row-local reductions over the
last dimension, the `arg164_1` scaling, the two sibling `[768]` reductions over
the residual input, the `[768]` reduction of the scaled gradient, and the
returned `[768, 32768]` transpose view. It differs from Inductor by computing
each row tile's scalar reductions, transpose backing store, and three compatible
column partials in one Triton pass, then finalizing the column vectors in a
small split-K reduction, instead of scheduling the row reductions, pointwise
epilogue, column reductions, and transpose-producing view as separate template
work. Inductor cannot do this today because the scheduler does not represent a
dependent multi-output reduction with row-wise scalar intermediates plus a
materialized transpose side output as one cooperatively split reduction; the fix
is COOPERATIVE_SPLIT_K support for dependent multi-output reductions that
accumulate compatible column partials while writing the side output.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_59e022113eeb"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 256
SEQ = 128
ROWS = BATCH * SEQ
C = 768

sys.path.insert(0, str(REPO_ROOT))


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
    mm_ptr,
    residual_ptr,
    weight_ptr,
    rhs_ptr,
    gate_ptr,
    out_base_ptr,
    partial_sum_add_rhs_ptr,
    partial_sum_add_ptr,
    partial_sum_out_ptr,
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

    acc_add_rhs = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_add = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_out = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * C_ + c[None, :]

        add_value = (
            tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weighted = add_value * weight[None, :]

        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
        gate = tl.load(gate_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        out_value = gate[:, None] * (
            weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None]
        )

        tl.store(out_base_ptr + offsets, out_value, mask=mask)
        acc_add_rhs += tl.sum(tl.where(mask, add_value * rhs, 0.0), axis=0)
        acc_add += tl.sum(tl.where(mask, add_value, 0.0), axis=0)
        acc_out += tl.sum(tl.where(mask, out_value, 0.0), axis=0)

    partial_offsets = pid * C_ + c
    tl.store(partial_sum_add_rhs_ptr + partial_offsets, acc_add_rhs, mask=c_mask)
    tl.store(partial_sum_add_ptr + partial_offsets, acc_add, mask=c_mask)
    tl.store(partial_sum_out_ptr + partial_offsets, acc_out, mask=c_mask)


@triton.jit
def _finalize_column_partials_kernel(
    partial_sum_add_rhs_ptr,
    partial_sum_add_ptr,
    partial_sum_out_ptr,
    out_sum_add_rhs_ptr,
    out_sum_add_ptr,
    out_sum_out_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block = tl.arange(0, BLOCK_ROW_BLOCKS)
    mask = (block[:, None] < NUM_ROW_BLOCKS) & (c[None, :] < C_)
    offsets = block[:, None] * C_ + c[None, :]

    sum_add_rhs = tl.load(
        partial_sum_add_rhs_ptr + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    sum_add = tl.load(partial_sum_add_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_out = tl.load(partial_sum_out_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    c_mask = c < C_
    tl.store(out_sum_add_rhs_ptr + c, tl.sum(sum_add_rhs, axis=0), mask=c_mask)
    tl.store(out_sum_add_ptr + c, tl.sum(sum_add, axis=0), mask=c_mask)
    tl.store(out_sum_out_ptr + c, tl.sum(sum_out, axis=0), mask=c_mask)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    mm_66: torch.Tensor,
    mul_152: torch.Tensor,
    arg8_1: torch.Tensor,
    arg68_1: torch.Tensor,
    arg164_1: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_66.shape == (ROWS, C)
    assert mul_152.shape == (BATCH, SEQ, C)
    assert arg8_1.shape == (C,)
    assert arg68_1.shape == (BATCH, SEQ, C)
    assert arg164_1.shape == (BATCH, SEQ, 1)
    assert mm_66.stride() == (C, 1)
    assert mul_152.stride() == (SEQ * C, C, 1)
    assert arg68_1.stride() == (SEQ * C, C, 1)

    mm = mm_66.contiguous()
    residual = mul_152.contiguous().reshape(ROWS, C)
    weight = arg8_1.contiguous()
    rhs = arg68_1.contiguous().reshape(ROWS, C)
    gate = arg164_1.contiguous().reshape(ROWS)
    device = mm.device

    row_split = 16
    xblock = 1
    block_c = 1024
    num_row_blocks = triton.cdiv(ROWS, row_split)

    out_base = torch.empty((ROWS, C), device=device, dtype=torch.float32)
    partial_sum_add_rhs = torch.empty(
        (num_row_blocks, C), device=device, dtype=torch.float32
    )
    partial_sum_add = torch.empty((num_row_blocks, C), device=device, dtype=torch.float32)
    partial_sum_out = torch.empty((num_row_blocks, C), device=device, dtype=torch.float32)

    _full_scope_partial_kernel[(num_row_blocks,)](
        mm,
        residual,
        weight,
        rhs,
        gate,
        out_base,
        partial_sum_add_rhs,
        partial_sum_add,
        partial_sum_out,
        ROWS_=ROWS,
        C_=C,
        ROW_SPLIT=row_split,
        XBLOCK=xblock,
        BLOCK_C=block_c,
        num_warps=8,
    )

    out_sum_add_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum_add = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_partials_kernel[(triton.cdiv(C, 16),)](
        partial_sum_add_rhs,
        partial_sum_add,
        partial_sum_out,
        out_sum_add_rhs,
        out_sum_add,
        out_sum_out,
        NUM_ROW_BLOCKS=num_row_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=16,
        num_warps=8,
    )

    return out_sum_add_rhs, out_sum_add, out_base.permute(1, 0), out_sum_out


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
    print(f"oracle_fused full-scope DistilBERT reductions+transpose: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    compile_configs = [
        ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3",
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
