"""
Full-scope oracle for sum_sum_sum_fb9996974d8f (GhostNet BN backward).

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the
captured add, the value-equivalent channels-last copy/contiguous clone paths,
and both BN-backward-style dual reductions for C=24 and the sliced C=12 tail in
one explicit multi-output schedule; it avoids materializing and rereading the
intermediate add/copy/clone tensors while still returning the same four outputs
and output layouts as Repro.forward. Inductor cannot do this today because its
scheduler treats the memory-format copy/clone/slice path and the two dependent
sum reductions as separate producer/consumer regions, so it cannot represent a
single fused template that shares the add producer across sibling reductions and
materialized outputs with different layouts. The fix is scheduler/codegen
support for this memory-format-aware multi-output reduction pattern, using
masked power-of-two channel blocks for non-power-of-two C.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_fb9996974d8f"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C24 = 24
C12 = 12
H = 56
W = 56
HW = H * W
ROWS = N * HW
INV_ROWS = 1.0 / ROWS


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@triton.jit
def _reduce_c24_kernel(
    clone_ptr,
    getitem_ptr,
    arg230_ptr,
    mean24_ptr,
    partial_sum_x_ptr,
    partial_sum_x_rhs_ptr,
    C24_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    row = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c_mask = c < C24_
    row_mask = row < ROWS_

    n = row // HW_
    hw = row - n * HW_
    offsets = n[None, :] * (C24_ * HW_) + c[:, None] * HW_ + hw[None, :]
    mask = c_mask[:, None] & row_mask[None, :]

    x = (
        tl.load(clone_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    centered = (
        tl.load(arg230_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        - tl.load(mean24_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    )

    partial_offsets = pid * C24_ + c
    tl.store(
        partial_sum_x_ptr + partial_offsets,
        tl.sum(tl.where(mask, x, 0.0), axis=1),
        mask=c_mask,
    )
    tl.store(
        partial_sum_x_rhs_ptr + partial_offsets,
        tl.sum(tl.where(mask, x * centered, 0.0), axis=1),
        mask=c_mask,
    )


@triton.jit
def _reduce_c12_kernel(
    clone_ptr,
    getitem_ptr,
    arg225_ptr,
    mean12_ptr,
    partial_sum_x_rhs_ptr,
    C24_: tl.constexpr,
    C12_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    row = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c_mask = c < C12_
    row_mask = row < ROWS_

    n = row // HW_
    hw = row - n * HW_
    x_offsets = n[None, :] * (C24_ * HW_) + (c[:, None] + 12) * HW_ + hw[None, :]
    rhs_offsets = n[None, :] * (C12_ * HW_) + c[:, None] * HW_ + hw[None, :]
    mask = c_mask[:, None] & row_mask[None, :]

    x = (
        tl.load(clone_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    )
    centered = (
        tl.load(arg225_ptr + rhs_offsets, mask=mask, other=0.0).to(tl.float32)
        - tl.load(mean12_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    )

    partial_offsets = pid * C12_ + c
    tl.store(
        partial_sum_x_rhs_ptr + partial_offsets,
        tl.sum(tl.where(mask, x * centered, 0.0), axis=1),
        mask=c_mask,
    )


@triton.jit
def _finalize_c24_kernel(
    partial_sum_x_ptr,
    partial_sum_x_rhs_ptr,
    scale24_ptr,
    sum_x_ptr,
    sum_x_rhs_ptr,
    out_vec_ptr,
    C24_: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C24_
    acc_x = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_x_rhs = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, NUM_BLOCKS, BLOCK_BLOCKS):
        block = start + tl.arange(0, BLOCK_BLOCKS)
        mask = (block[:, None] < NUM_BLOCKS) & c_mask[None, :]
        offsets = block[:, None] * C24_ + c[None, :]
        acc_x += tl.sum(
            tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_x_rhs += tl.sum(
            tl.load(partial_sum_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )

    scale = tl.load(scale24_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(sum_x_ptr + c, acc_x, mask=c_mask)
    tl.store(sum_x_rhs_ptr + c, acc_x_rhs, mask=c_mask)
    tl.store(out_vec_ptr + c, acc_x_rhs * scale, mask=c_mask)


@triton.jit
def _finalize_c12_kernel(
    partial_sum_x_rhs_ptr,
    scale12_ptr,
    sum_x_rhs_ptr,
    out_vec_ptr,
    C12_: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C12_
    acc_x_rhs = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, NUM_BLOCKS, BLOCK_BLOCKS):
        block = start + tl.arange(0, BLOCK_BLOCKS)
        mask = (block[:, None] < NUM_BLOCKS) & c_mask[None, :]
        offsets = block[:, None] * C12_ + c[None, :]
        acc_x_rhs += tl.sum(
            tl.load(partial_sum_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )

    scale = tl.load(scale12_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(sum_x_rhs_ptr + c, acc_x_rhs, mask=c_mask)
    tl.store(out_vec_ptr + c, acc_x_rhs * scale, mask=c_mask)


@triton.jit
def _store_out24_kernel(
    clone_ptr,
    getitem_ptr,
    arg230_ptr,
    mean24_ptr,
    scale24_ptr,
    grad24_ptr,
    sum_x_ptr,
    sum_x_rhs_ptr,
    out_ptr,
    C24_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    INV_ROWS_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    row = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c_mask = c < C24_
    row_mask = row < ROWS_

    n = row // HW_
    hw = row - n * HW_
    offsets = n[None, :] * (C24_ * HW_) + c[:, None] * HW_ + hw[None, :]
    mask = c_mask[:, None] & row_mask[None, :]

    x = (
        tl.load(clone_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    centered = (
        tl.load(arg230_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        - tl.load(mean24_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    )
    scale = tl.load(scale24_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    grad = tl.load(grad24_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    sum_x = tl.load(sum_x_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    sum_x_rhs = tl.load(sum_x_rhs_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    correction = sum_x_rhs[:, None] * INV_ROWS_ * scale[:, None] * scale[:, None]
    mean_grad = sum_x[:, None] * INV_ROWS_
    out = (x - centered * correction - mean_grad) * (scale * grad)[:, None]
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _store_out12_kernel(
    clone_ptr,
    getitem_ptr,
    arg225_ptr,
    mean12_ptr,
    scale12_ptr,
    grad12_ptr,
    sum_x24_ptr,
    sum_x_rhs12_ptr,
    out_ptr,
    C24_: tl.constexpr,
    C12_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    INV_ROWS_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    row = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c_mask = c < C12_
    row_mask = row < ROWS_

    n = row // HW_
    hw = row - n * HW_
    x_offsets = n[None, :] * (C24_ * HW_) + (c[:, None] + 12) * HW_ + hw[None, :]
    rhs_offsets = n[None, :] * (C12_ * HW_) + c[:, None] * HW_ + hw[None, :]
    out_offsets = n[None, :] * (C12_ * HW_) + hw[None, :] * C12_ + c[:, None]
    mask = c_mask[:, None] & row_mask[None, :]

    x = (
        tl.load(clone_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    )
    centered = (
        tl.load(arg225_ptr + rhs_offsets, mask=mask, other=0.0).to(tl.float32)
        - tl.load(mean12_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    )
    scale = tl.load(scale12_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    grad = tl.load(grad12_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    sum_x = tl.load(sum_x24_ptr + c + 12, mask=c_mask, other=0.0).to(tl.float32)
    sum_x_rhs = tl.load(sum_x_rhs12_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    correction = sum_x_rhs[:, None] * INV_ROWS_ * scale[:, None] * scale[:, None]
    mean_grad = sum_x[:, None] * INV_ROWS_
    out = (x - centered * correction - mean_grad) * (scale * grad)[:, None]
    tl.store(out_ptr + out_offsets, out, mask=mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        clone_13,
        getitem_246,
        arg230_1,
        arg530_1,
        arg231_1,
        arg26_1,
        arg225_1,
        arg532_1,
        arg226_1,
        arg22_1,
    ) = inputs

    return (
        clone_13.contiguous(),
        getitem_246.contiguous(),
        arg230_1.contiguous(),
        arg530_1.contiguous(),
        arg231_1.contiguous(),
        arg26_1.contiguous(),
        arg225_1.contiguous(),
        arg532_1.contiguous(),
        arg226_1.contiguous(),
        arg22_1.contiguous(),
    )


def oracle_fused(
    clone_13: torch.Tensor,
    getitem_246: torch.Tensor,
    arg230_1: torch.Tensor,
    arg530_1: torch.Tensor,
    arg231_1: torch.Tensor,
    arg26_1: torch.Tensor,
    arg225_1: torch.Tensor,
    arg532_1: torch.Tensor,
    arg226_1: torch.Tensor,
    arg22_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert clone_13.shape == (N, C24, H, W)
    assert getitem_246.shape == (N, C24, H, W)
    assert arg230_1.shape == (N, C24, H, W)
    assert arg530_1.shape == (1, C24, 1, 1)
    assert arg231_1.shape == (C24,)
    assert arg26_1.shape == (C24,)
    assert arg225_1.shape == (N, C12, H, W)
    assert arg532_1.shape == (1, C12, 1, 1)
    assert arg226_1.shape == (C12,)
    assert arg22_1.shape == (C12,)

    device = clone_13.device
    block_rows_reduce = 256
    block_rows_store = 256
    block_c24 = 32
    block_c12 = 16
    num_row_blocks = triton.cdiv(ROWS, block_rows_reduce)

    partial_sum_x24 = torch.empty((num_row_blocks, C24), device=device, dtype=torch.float32)
    partial_sum_x_rhs24 = torch.empty((num_row_blocks, C24), device=device, dtype=torch.float32)
    partial_sum_x_rhs12 = torch.empty((num_row_blocks, C12), device=device, dtype=torch.float32)

    _reduce_c24_kernel[(num_row_blocks,)](
        clone_13,
        getitem_246,
        arg230_1,
        arg530_1,
        partial_sum_x24,
        partial_sum_x_rhs24,
        C24_=C24,
        ROWS_=ROWS,
        HW_=HW,
        BLOCK_ROWS=block_rows_reduce,
        BLOCK_C=block_c24,
        num_warps=8,
    )
    _reduce_c12_kernel[(num_row_blocks,)](
        clone_13,
        getitem_246,
        arg225_1,
        arg532_1,
        partial_sum_x_rhs12,
        C24_=C24,
        C12_=C12,
        ROWS_=ROWS,
        HW_=HW,
        BLOCK_ROWS=block_rows_reduce,
        BLOCK_C=block_c12,
        num_warps=4,
    )

    sum_x24 = torch.empty((C24,), device=device, dtype=torch.float32)
    sum_x_rhs24 = torch.empty((C24,), device=device, dtype=torch.float32)
    out8 = torch.empty((C24,), device=device, dtype=torch.float32)
    sum_x_rhs12 = torch.empty((C12,), device=device, dtype=torch.float32)
    out17 = torch.empty((C12,), device=device, dtype=torch.float32)

    _finalize_c24_kernel[(1,)](
        partial_sum_x24,
        partial_sum_x_rhs24,
        arg231_1,
        sum_x24,
        sum_x_rhs24,
        out8,
        C24_=C24,
        NUM_BLOCKS=num_row_blocks,
        BLOCK_BLOCKS=1024,
        BLOCK_C=block_c24,
        num_warps=8,
    )
    _finalize_c12_kernel[(1,)](
        partial_sum_x_rhs12,
        arg226_1,
        sum_x_rhs12,
        out17,
        C12_=C12,
        NUM_BLOCKS=num_row_blocks,
        BLOCK_BLOCKS=1024,
        BLOCK_C=block_c12,
        num_warps=4,
    )

    out7 = torch.empty((N, C24, H, W), device=device, dtype=torch.float32)
    out16 = torch.empty_strided(
        (N, C12, H, W),
        (C12 * HW, 1, W * C12, C12),
        device=device,
        dtype=torch.float32,
    )
    num_store_blocks = triton.cdiv(ROWS, block_rows_store)
    _store_out24_kernel[(num_store_blocks,)](
        clone_13,
        getitem_246,
        arg230_1,
        arg530_1,
        arg231_1,
        arg26_1,
        sum_x24,
        sum_x_rhs24,
        out7,
        C24_=C24,
        ROWS_=ROWS,
        HW_=HW,
        INV_ROWS_=INV_ROWS,
        BLOCK_ROWS=block_rows_store,
        BLOCK_C=block_c24,
        num_warps=8,
    )
    _store_out12_kernel[(num_store_blocks,)](
        clone_13,
        getitem_246,
        arg225_1,
        arg532_1,
        arg226_1,
        arg22_1,
        sum_x24,
        sum_x_rhs12,
        out16,
        C24_=C24,
        C12_=C12,
        ROWS_=ROWS,
        HW_=HW,
        INV_ROWS_=INV_ROWS,
        BLOCK_ROWS=block_rows_store,
        BLOCK_C=block_c12,
        num_warps=4,
    )

    return out7, out8, out16, out17


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

    print(f"oracle_fused full-scope BN-backward reductions: {oracle_us:.3f} us")

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
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
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
