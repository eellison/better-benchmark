"""
Oracle kernel for sum_sum_ee85624361a0.

Pattern: MobileNetV3 batch-norm backward style sibling reductions:

    sum1[c] = sum_{n,h,w} where_self[n,c,h,w]
    sum2[c] = sum_{n,h,w} where_self[n,c,h,w] * (arg165_1[n,c,h,w] - mean[c])

The captured repro then uses both channel reductions in a pointwise epilogue.
This oracle keeps the repro's setup ops in PyTorch, then measures the intended
Inductor optimization target: one Triton pass feeding both accumulators, followed
by one fused pointwise epilogue.

Gap diagnosis: this oracle reads where_self and centered once in a single-pass
dual-accumulator Triton reduction, computing sum(where_self) and
sum(where_self * centered) together before feeding both results into the fused
post-reduction batch-norm epilogue; Inductor emits separate sibling reductions
because its scheduler does not currently group reductions with the same
iteration domain, reduction domain, and shared inputs when their per-element
reduction expressions differ. The fix class is SCHEDULER_FUSION: teach the
scheduler/codegen path to recognize compatible sibling reductions and lower
them as one multi-output reduction template with multiple accumulators, then
preserve fusion into the dependent epilogue.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_ee85624361a0"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C = 72
H = 28
W = 28
HW = H * W
SCALE = 2.4912308673469386e-06

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _dual_reduce_nchw_kernel(
    where_ptr,
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
    offsets = n * (C_ * HW_) + c * HW_ + hw

    where_vals = tl.load(where_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered_vals = tl.load(centered_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.atomic_add(sum1_ptr + c, tl.sum(where_vals, axis=0))
    tl.atomic_add(sum2_ptr + c, tl.sum(where_vals * centered_vals, axis=0))


@triton.jit
def _post_reduce_pointwise_nchw_kernel(
    where_ptr,
    centered_ptr,
    sum1_ptr,
    sum2_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    out_ptr,
    numel: tl.constexpr,
    scale: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = offsets < numel
    c = (offsets // HW_) % C_

    where_vals = tl.load(where_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered_vals = tl.load(centered_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=mask, other=0.0).to(tl.float32)

    mean_term = sum1 * scale
    variance_term = sum2 * scale * rsqrt * rsqrt
    out = (where_vals - centered_vals * variance_term - mean_term) * (rsqrt * affine_weight)
    tl.store(out_ptr + offsets, out, mask=mask)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple:
    module = _load_repro_module()
    inputs = module.make_inputs()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)


def prepare_oracle_inputs(
    arg170_1: torch.Tensor,
    getitem_147: torch.Tensor,
    getitem_153: torch.Tensor,
    relu_2: torch.Tensor,
    full: torch.Tensor,
    arg166_1: torch.Tensor,
    arg165_1: torch.Tensor,
    arg167_1: torch.Tensor,
    arg23_1: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0

    gate = torch.clamp(arg170_1 + 3.0, min=0.0, max=6.0) / 6.0
    add_tensor = getitem_147 * gate + getitem_153 / float(HW)
    where_self = torch.where(relu_2 <= 0, full, add_tensor).contiguous()

    mean = arg166_1.squeeze(0).squeeze(-1).squeeze(-1)
    centered = (arg165_1 - mean[None, :, None, None]).contiguous()
    rsqrt = arg167_1.squeeze(0).squeeze(-1).squeeze(-1).contiguous()
    affine_weight = arg23_1.contiguous()
    return where_self, centered, rsqrt, affine_weight


def oracle_fused(
    where_self: torch.Tensor,
    centered: torch.Tensor,
    rsqrt: torch.Tensor,
    affine_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert where_self.shape == (N, C, H, W)
    assert centered.shape == (N, C, H, W)
    assert where_self.is_contiguous()
    assert centered.is_contiguous()

    sum1 = torch.zeros((C,), device=where_self.device, dtype=torch.float32)
    sum2 = torch.zeros((C,), device=where_self.device, dtype=torch.float32)

    block_k = 1024
    total_spatial = N * HW
    grid_reduce = (C, triton.cdiv(total_spatial, block_k))
    _dual_reduce_nchw_kernel[grid_reduce](
        where_self,
        centered,
        sum1,
        sum2,
        total_spatial=total_spatial,
        C_=C,
        HW_=HW,
        BLOCK_K=block_k,
    )

    out = torch.empty_like(where_self)
    block_elems = 256
    numel = where_self.numel()
    _post_reduce_pointwise_nchw_kernel[(triton.cdiv(numel, block_elems),)](
        where_self,
        centered,
        sum1,
        sum2,
        rsqrt,
        affine_weight,
        out,
        numel=numel,
        scale=SCALE,
        C_=C,
        HW_=HW,
        BLOCK_ELEMS=block_elems,
    )

    return out, sum2 * rsqrt


def reference_outputs(inputs: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check() -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()

    with torch.no_grad():
        ref = reference_outputs(inputs)
        oracle_inputs = prepare_oracle_inputs(*inputs)
        actual = oracle_fused(*oracle_inputs)

    ok = True
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        output_ok = torch.allclose(got.float(), expected.float(), rtol=1e-2, atol=1e-2)
        ok = ok and output_ok
        print(
            f"output[{idx}]: max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={output_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(model: torch.nn.Module, inputs: tuple, config: dict[str, object]):
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

    print(f"oracle_fused reduction+epilogue: {oracle_us:.3f} us")
    print("note: the oracle times the shared reductions and BN-backward epilogue only")

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
            us = triton.testing.do_bench(
                lambda: compiled(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            ) * 1000.0
        print(f"torch.compile {label}: {us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check():
        sys.exit(1)
    if args.bench:
        run_bench(args.rep, args.warmup, args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
