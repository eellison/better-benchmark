"""
Full-scope oracle for sum_27f8a4b9ab09 (NFNet GELU-backward channel reduce).

Gap diagnosis (classification: BANDWIDTH_BOUND): The oracle consumes the same
`mm`, activation tensor, and shape parameters as repro.py, fuses the view,
expand/divide, GELU-derivative pointwise chain, and `[0, 2, 3]` reduction into
one Triton channel-reduction kernel, and directly indexes `mm[n, c]` instead of
materializing the broadcasted `[128, 3072, 6, 6]` pooled-gradient tensor.
Inductor's coordinate-descent compile is slightly faster than this full-scope
oracle on the same inputs, which means Inductor already reaches an equivalent
fused reduction schedule for this contiguous layout; the apparent SOL gap is not
explained by a missing fusion in this graph. The fix classification is
BANDWIDTH_BOUND: no Inductor change is justified by this oracle, and the
artifact should be kept as diagnosis-only rather than a true floor.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_27f8a4b9ab09"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 3072
H = 6
W = 6
HW = H * W
GAMMA_OVER_HW = 1.7015043497085571 / HW
RSQRT2 = 0.7071067811865476
RSQRT2PI = 0.3989422804014327



@triton.jit
def _gelu_grad_flat_channel_kernel(
    mm_ptr,
    x_ptr,
    out_ptr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    C_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    N_HW_: tl.constexpr,
    GAMMA_OVER_HW_: tl.constexpr,
    RSQRT2_: tl.constexpr,
    RSQRT2PI_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    active = k < N_HW_
    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_

    x_offsets = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
    mm_offsets = n * C_ + c
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    mm_values = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32)

    erf_plus_one = tl.math.erf(x * RSQRT2_) + 1.0
    cdf = 0.5 * erf_plus_one
    pdf_term = x * tl.exp(-0.5 * x * x) * RSQRT2PI_
    value = mm_values * GAMMA_OVER_HW_ * (cdf + pdf_term)
    tl.store(out_ptr + c, tl.sum(tl.where(active, value, 0.0), axis=0))


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
    mm: torch.Tensor,
    arg413_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    assert mm.shape == (N, C)
    assert arg413_1.shape == (N, C, H, W)
    assert mm.is_contiguous()

    out = torch.empty((C,), device=mm.device, dtype=torch.float32)
    _gelu_grad_flat_channel_kernel[(C,)](
        mm,
        arg413_1,
        out,
        x_stride_n=arg413_1.stride(0),
        x_stride_c=arg413_1.stride(1),
        x_stride_h=arg413_1.stride(2),
        x_stride_w=arg413_1.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        N_HW_=N * HW,
        GAMMA_OVER_HW_=GAMMA_OVER_HW,
        RSQRT2_=RSQRT2,
        RSQRT2PI_=RSQRT2PI,
        BLOCK_K=triton.next_power_of_2(N * HW),
        num_warps=4,
    )
    return out


def reference_outputs(inputs: tuple[object, ...]) -> torch.Tensor:
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

    max_abs, max_rel = _max_diff(actual, ref)
    output_ok = torch.allclose(actual.float(), ref.float(), rtol=rtol, atol=atol)
    stride_ok = actual.stride() == ref.stride()
    ok = output_ok and stride_ok
    print(
        f"output: shape={list(actual.shape)} stride={actual.stride()} "
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
    print(f"oracle_fused full-scope flat GELU reduce: {oracle_us:.3f} us")

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
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
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
