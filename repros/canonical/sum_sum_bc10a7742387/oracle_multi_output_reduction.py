"""Full-scope oracle for sum_sum_bc10a7742387 (VoVNet BN-backward tail).

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle consumes the same
eight original inputs as repro.py and returns the same contiguous
`[32, 1024, 7, 7]` tensor plus `[1024]` vector. It differs from Inductor by
computing the ReLU-backward `where(mm / 49)` producer, centered activation,
two sibling channel reductions, vector side output, and dependent BN-backward
full-tensor epilogue inside one per-channel Triton program, so the large
activation and gradient producer are not materialized and reread across generic
pointwise/reduction kernels. Inductor cannot do this today because its scheduler
does not form a full-scope multi-output reduction template that shares a masked
pointwise producer across sibling sums and then sinks the finalized sums into a
dependent full-tensor epilogue. The fix class is SCHEDULER_FUSION.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_bc10a7742387"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 32
C = 1024
H = 7
W = 7
HW = H * W
TOTAL_REDUCE = N * HW
NUMEL = N * C * HW
AVGPOOL_SCALE = 1.0 / 49.0
BN_SCALE = 1.0 / float(TOTAL_REDUCE)

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3"
        ),
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
def _full_channel_bn_backward_kernel(
    mm_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    vector_out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_REDUCE_: tl.constexpr,
    AVGPOOL_SCALE_: tl.constexpr,
    BN_SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    active = k < TOTAL_REDUCE_

    n = k // HW_
    hw = k - n * HW_
    offsets = n * (C_ * HW_) + c * HW_ + hw
    mm_offsets = n * C_ + c

    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    beta = tl.load(beta_ptr + c).to(tl.float32)

    source = tl.load(activation_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    grad = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32) * AVGPOOL_SCALE_
    centered = source - mean
    relu_input = centered * invstd * gamma + beta
    where_self = tl.where(relu_input <= 0.0, 0.0, grad)
    where_self = tl.where(active, where_self, 0.0)

    sum1 = tl.sum(where_self, axis=0)
    sum2 = tl.sum(where_self * centered, axis=0)

    variance_term = sum2 * BN_SCALE_ * invstd * invstd
    mean_term = sum1 * BN_SCALE_
    affine_term = invstd * gamma
    result = (where_self - centered * variance_term - mean_term) * affine_term

    tl.store(out_ptr + offsets, result, mask=active)
    tl.store(vector_out_ptr + c, sum2 * invstd)


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
    return tuple(x.cuda() if isinstance(x, torch.Tensor) and x.device.type != "cuda" else x for x in module.make_inputs())


def oracle_fused(
    mm: torch.Tensor,
    arg216_1: torch.Tensor,
    arg217_1: torch.Tensor,
    arg218_1: torch.Tensor,
    arg87_1: torch.Tensor,
    arg88_1: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mm.shape == (N, C)
    assert arg216_1.shape == (N, C, H, W)
    assert arg217_1.shape == (1, C, 1, 1)
    assert arg218_1.shape == (1, C, 1, 1)
    assert arg87_1.shape == (C,)
    assert arg88_1.shape == (C,)
    assert mm.is_contiguous()
    assert arg216_1.is_contiguous()
    assert arg217_1.is_contiguous()
    assert arg218_1.is_contiguous()
    assert arg87_1.is_contiguous()
    assert arg88_1.is_contiguous()

    out = torch.empty((N, C, H, W), device=mm.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=mm.device, dtype=torch.float32)
    block_k = triton.next_power_of_2(TOTAL_REDUCE)
    _full_channel_bn_backward_kernel[(C,)](
        mm,
        arg216_1,
        arg217_1,
        arg218_1,
        arg87_1,
        arg88_1,
        out,
        vector_out,
        C_=C,
        HW_=HW,
        TOTAL_REDUCE_=TOTAL_REDUCE,
        AVGPOOL_SCALE_=AVGPOOL_SCALE,
        BN_SCALE_=BN_SCALE,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out, vector_out


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
        actual = oracle_fused(*inputs)
        torch.cuda.synchronize()

    ok = True
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        output_ok = (
            got.shape == expected.shape
            and got.dtype == expected.dtype
            and got.stride() == expected.stride()
            and torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        )
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} dtype={got.dtype} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={output_ok}"
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
    print(f"oracle_fused full-scope masked BN dual reduction + epilogue: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    compile_times: list[tuple[str, float]] = []
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
        compile_times.append((label, compiled_us))
        print(f"torch.compile {label}: {compiled_us:.3f} us")

    if all(oracle_us < compiled_us for _, compiled_us in compile_times):
        print("valid_floor: oracle is faster than both required compile configs")
    else:
        print("diagnosis_only: oracle is not a true floor because a required compile config is faster")


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
