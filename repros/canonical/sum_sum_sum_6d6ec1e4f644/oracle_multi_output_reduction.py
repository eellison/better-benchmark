"""
Full-scope oracle for sum_sum_sum_6d6ec1e4f644 (GPT-2 LN/dropout backward).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the
same full repro scope as `repro.py`, including the `mm_1` reshape, weight
multiply by `arg73_1`, per-row layernorm reductions over C=768, dependent
pointwise expression with `arg233_1`, dropout mask scaling, and all three
returned `[768]` column reductions. It differs from Inductor by doing the
row-local reductions and sibling column reductions inside one split-row partial
kernel, so `mm_1`, `arg226_1`, and the dropout mask are streamed once per row
tile instead of materializing row-sum intermediates and scheduling the final
column sum as a separate producer/consumer chain. Inductor cannot express this
today because the final reduction depends on row-wise reductions over a
different axis, and the scheduler does not generate one cooperative split-K
multi-output reduction that coordinates those row-local scalars with the three
column accumulators; the fix is COOPERATIVE_SPLIT_K support for dependent
multi-output reductions with row-tile partial buffers or equivalent atomics.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_6d6ec1e4f644"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 8 * 1024
C = 768
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
    x_ptr,
    weight_ptr,
    rhs_ptr,
    gate_ptr,
    keep_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_grad_ptr,
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
    acc_grad = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * C_ + c[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        gate = tl.load(gate_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

        weighted = x * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)

        acc_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)

        grad = gate[:, None] * (weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None])
        grad = grad * keep * DROPOUT_SCALE_
        acc_grad += tl.sum(tl.where(mask, grad, 0.0), axis=0)

    partial_offsets = pid * C_ + c
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=c_mask)
    tl.store(partial_grad_ptr + partial_offsets, acc_grad, mask=c_mask)


@triton.jit
def _finalize_partials_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_grad_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_grad_ptr,
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
    grad = tl.load(partial_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    c_mask = c < C_
    tl.store(out_x_rhs_ptr + c, tl.sum(x_rhs, axis=0), mask=c_mask)
    tl.store(out_x_ptr + c, tl.sum(x, axis=0), mask=c_mask)
    tl.store(out_grad_ptr + c, tl.sum(grad, axis=0), mask=c_mask)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    mm_1: torch.Tensor,
    arg73_1: torch.Tensor,
    arg226_1: torch.Tensor,
    arg233_1: torch.Tensor,
    arg225_1: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_1.shape == (ROWS, C)
    assert arg73_1.shape == (C,)
    assert arg226_1.shape == (8, 1024, C)
    assert arg233_1.shape == (8, 1024, 1)
    assert arg225_1.shape == (8, 1024, C)

    x = mm_1.contiguous()
    rhs = arg226_1.contiguous().reshape(ROWS, C)
    keep = arg225_1.contiguous().reshape(ROWS, C)
    gate = arg233_1.contiguous().reshape(ROWS)

    row_split = 16
    xblock = 1
    block_c = 1024
    num_row_blocks = triton.cdiv(ROWS, row_split)
    partial_x_rhs = torch.empty((num_row_blocks, C), device=mm_1.device, dtype=torch.float32)
    partial_x = torch.empty((num_row_blocks, C), device=mm_1.device, dtype=torch.float32)
    partial_grad = torch.empty((num_row_blocks, C), device=mm_1.device, dtype=torch.float32)

    _full_scope_partial_kernel[(num_row_blocks,)](
        x,
        arg73_1.contiguous(),
        rhs,
        gate,
        keep,
        partial_x_rhs,
        partial_x,
        partial_grad,
        ROWS_=ROWS,
        C_=C,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        ROW_SPLIT=row_split,
        XBLOCK=xblock,
        BLOCK_C=block_c,
        num_warps=8,
    )

    out_x_rhs = torch.empty((C,), device=mm_1.device, dtype=torch.float32)
    out_x = torch.empty((C,), device=mm_1.device, dtype=torch.float32)
    out_grad = torch.empty((C,), device=mm_1.device, dtype=torch.float32)
    _finalize_partials_kernel[(triton.cdiv(C, 16),)](
        partial_x_rhs,
        partial_x,
        partial_grad,
        out_x_rhs,
        out_x,
        out_grad,
        NUM_ROW_BLOCKS=num_row_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=16,
        num_warps=8,
    )
    return out_x_rhs, out_x, out_grad


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
    print(f"oracle_fused full-scope LN/dropout dependent reductions: {oracle_us:.3f} us")

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
