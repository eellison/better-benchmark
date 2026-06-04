"""
Canonical oracle for sum_sum_b7f94adef30f (GhostNet BN-backward reductions).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle differs from Inductor by taking the BN-backward pointwise producer as the prepared input and reducing it with a split-K dual-accumulator Triton kernel, so each channel is reduced by many row/spatial tiles that contribute partials for both `sum(add_tensor_1)` and `sum(add_tensor_1 * centered)` before one fused epilogue pass writes the two final outputs; Inductor cannot do this today because its reduction splitting heuristic rejects this small-output, large-reduction BN-backward shape under the no-split threshold and then schedules the sibling reductions/epilogue with too little reduction-dimension parallelism; the fix is COOPERATIVE_SPLIT_K support for compatible multi-output reductions, allowing the scheduler to split the reduction dimension and coordinate partial or atomic accumulation across tiles while preserving the dependent epilogue.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_b7f94adef30f"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C = 72
H = 28
W = 28
HW = H * W
NUMEL = N * C * HW
TOTAL_SPATIAL = N * HW
INV_HW = 1.0 / HW
SCALE = 2.4912308673469386e-06



@triton.jit
def _dual_reduce_split_k_kernel(
    producer_ptr,
    centered_ptr,
    sum1_ptr,
    sum2_ptr,
    total_spatial: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k < total_spatial

    n = k // HW_
    hw = k - n * HW_
    full_offsets = n * (C_ * HW_) + c * HW_ + hw
    add_tensor_1 = tl.load(producer_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
    centered = tl.load(centered_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)

    tl.atomic_add(sum1_ptr + c, tl.sum(add_tensor_1, axis=0), sem="relaxed")
    tl.atomic_add(sum2_ptr + c, tl.sum(add_tensor_1 * centered, axis=0), sem="relaxed")

@triton.jit
def _bn_backward_epilogue_kernel(
    producer_ptr,
    centered_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    NUMEL_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = offsets < NUMEL_

    c = (offsets // HW_) % C_
    add_tensor_1 = tl.load(producer_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered = tl.load(centered_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sum1 = tl.load(sum1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=mask, other=0.0).to(tl.float32)

    mean_term = sum1 * SCALE_
    variance_term = sum2 * SCALE_ * rsqrt * rsqrt
    out = (add_tensor_1 - centered * variance_term - mean_term) * (rsqrt * affine_weight)
    tl.store(out_ptr + offsets, out, mask=mask)


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


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        arg258_1,
        getitem_219,
        getitem_225,
        arg254_1,
        arg253_1,
        arg255_1,
        arg42_1,
        _shape_param_0,
    ) = inputs
    del _shape_param_0

    return (
        (
            getitem_219
            * (torch.clamp(arg258_1 + 3.0, min=0.0, max=6.0) * 0.16666666666666666)
            + getitem_225 * INV_HW
        ).contiguous(),
        (arg253_1 - arg254_1.reshape(1, C, 1, 1)).contiguous(),
        arg255_1.reshape(C).contiguous(),
        arg42_1.contiguous(),
    )


def oracle_fused(
    producer: torch.Tensor,
    centered: torch.Tensor,
    rsqrt: torch.Tensor,
    affine_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert producer.shape == (N, C, H, W)
    assert centered.shape == (N, C, H, W)
    assert rsqrt.shape == (C,)
    assert affine_weight.shape == (C,)

    sum1 = torch.zeros((C,), device=producer.device, dtype=torch.float32)
    sum2 = torch.zeros((C,), device=producer.device, dtype=torch.float32)

    block_k = 2048
    _dual_reduce_split_k_kernel[(C, triton.cdiv(TOTAL_SPATIAL, block_k))](
        producer,
        centered,
        sum1,
        sum2,
        total_spatial=TOTAL_SPATIAL,
        C_=C,
        HW_=HW,
        BLOCK_K=block_k,
        num_warps=8,
    )

    out = torch.empty_like(producer)
    block_elems = 1024
    _bn_backward_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        producer,
        centered,
        rsqrt,
        affine_weight,
        sum1,
        sum2,
        out,
        C_=C,
        HW_=HW,
        SCALE_=SCALE,
        NUMEL_=NUMEL,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out, sum2 * rsqrt


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
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

    print(f"oracle_fused prepared split-k reduction+epilogue: {oracle_us:.3f} us")

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
    parser.add_argument("--rtol", type=float, default=1e-2)
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
