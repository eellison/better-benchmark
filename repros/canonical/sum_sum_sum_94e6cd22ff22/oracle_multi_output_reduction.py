"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `Repro.forward` return tuple by cooperatively reducing each token over the batch dimension while sharing the row-local layernorm-backward sums with the sibling global channel reductions and the class/patch token reductions, whereas Inductor currently emits separate row reductions, dependent pointwise layernorm-backward work, and downstream `sum([0, 1])`, `sum([0])`, and patch-token reductions around materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K multi-output reduction template that can combine row-local scalar reductions with several compatible batch/token reductions and view-equivalent epilogues in one full-scope kernel; the fix is COOPERATIVE_SPLIT_K: add an Inductor template that splits the batch/token K domain for compatible small-output reductions, shares dependent row-reduction scalars, and emits the full tuple epilogue without materializing the layernorm-backward tensor."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_94e6cd22ff22"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

B = 128
T = 197
C = 192
ROWS = B * T



@triton.jit
def _zero_outputs_kernel(
    out_x_norm_ptr,
    out_x_ptr,
    out_token_ptr,
    out_patch_ptr,
    TOKEN_NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token_mask = offsets < TOKEN_NUMEL_
    c_mask = offsets < C_

    zeros = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(out_token_ptr + offsets, zeros, mask=token_mask)
    tl.store(out_x_norm_ptr + offsets, zeros, mask=c_mask)
    tl.store(out_x_ptr + offsets, zeros, mask=c_mask)
    tl.store(out_patch_ptr + offsets, zeros, mask=c_mask)


@triton.jit
def _token_full_scope_kernel(
    x_ptr,
    weight_ptr,
    cat_ptr,
    pos_ptr,
    mean_ptr,
    rsqrt_ptr,
    add_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    out_token_ptr,
    out_patch_ptr,
    B_: tl.constexpr,
    T_: tl.constexpr,
    C_: tl.constexpr,
    INV_C_: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    b_block = tl.program_id(1)
    b = b_block * BLOCK_B + tl.arange(0, BLOCK_B)
    c = tl.arange(0, BLOCK_C)
    b_mask = b < B_
    c_mask = c < C_
    mask = b_mask[:, None] & c_mask[None, :]

    row_offsets = (b[:, None] * T_ + token) * C_ + c[None, :]
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    cat = tl.load(cat_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    add_base = tl.load(add_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)

    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptr + token * C_ + c, mask=c_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + b * T_ + token, mask=b_mask, other=0.0).to(tl.float32)
    inv = tl.load(rsqrt_ptr + b * T_ + token, mask=b_mask, other=0.0).to(tl.float32)

    weighted_x = x * weight[None, :]
    normed = (cat + pos[None, :] - mean[:, None]) * inv[:, None]
    row_weighted_sum = tl.sum(tl.where(mask, weighted_x, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted_x * normed, 0.0), axis=1)

    grad = inv[:, None] * INV_C_ * (
        weighted_x * C_ - row_weighted_sum[:, None] - normed * row_dot[:, None]
    )
    add_value = add_base + grad

    token_x_norm = tl.sum(tl.where(mask, x * normed, 0.0), axis=0)
    token_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    token_add = tl.sum(tl.where(mask, add_value, 0.0), axis=0)

    tl.atomic_add(out_token_ptr + token * C_ + c, token_add, mask=c_mask, sem="relaxed")
    tl.atomic_add(out_x_norm_ptr + c, token_x_norm, mask=c_mask, sem="relaxed")
    tl.atomic_add(out_x_ptr + c, token_x, mask=c_mask, sem="relaxed")
    tl.atomic_add(out_patch_ptr + c, token_add, mask=c_mask & (token != 0), sem="relaxed")


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


def oracle_fused(
    mm_96: torch.Tensor,
    primals_6: torch.Tensor,
    cat: torch.Tensor,
    primals_5: torch.Tensor,
    getitem_1: torch.Tensor,
    rsqrt: torch.Tensor,
    add_133: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_96.shape == (ROWS, C)
    assert primals_6.shape == (C,)
    assert cat.shape == (B, T, C)
    assert primals_5.shape == (1, T, C)
    assert getitem_1.shape == (B, T, 1)
    assert rsqrt.shape == (B, T, 1)
    assert add_133.shape == (B, T, C)

    x = mm_96.contiguous()
    weight = primals_6.contiguous()
    cat_contig = cat.contiguous()
    pos = primals_5.contiguous()
    mean = getitem_1.contiguous()
    inv = rsqrt.contiguous()
    add_base = add_133.contiguous()

    block_b = 32
    block_c = 256
    num_b_blocks = triton.cdiv(B, block_b)
    out_x_norm = torch.empty((C,), device=x.device, dtype=torch.float32)
    out_x = torch.empty((C,), device=x.device, dtype=torch.float32)
    out_token = torch.empty((1, T, C), device=x.device, dtype=torch.float32)
    out_patch = torch.empty((C,), device=x.device, dtype=torch.float32)

    zero_block = 1024
    _zero_outputs_kernel[(triton.cdiv(T * C, zero_block),)](
        out_x_norm,
        out_x,
        out_token,
        out_patch,
        TOKEN_NUMEL_=T * C,
        C_=C,
        BLOCK=zero_block,
        num_warps=4,
    )

    _token_full_scope_kernel[(T, num_b_blocks)](
        x,
        weight,
        cat_contig,
        pos,
        mean,
        inv,
        add_base,
        out_x_norm,
        out_x,
        out_token,
        out_patch,
        B_=B,
        T_=T,
        C_=C,
        INV_C_=1.0 / C,
        BLOCK_B=block_b,
        BLOCK_C=block_c,
        num_warps=8,
    )
    out_cls = torch.as_strided(out_token, (1, 1, C), (C, C, 1))
    return out_x_norm, out_x, out_token, out_cls, out_patch


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
    print(f"oracle_fused full-scope ViT reductions: {oracle_us:.3f} us")

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
