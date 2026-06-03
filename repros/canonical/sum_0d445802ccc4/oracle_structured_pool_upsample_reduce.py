"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full NFNet adaptive-average-pool backward, GELU derivative, and returned channel sum directly from the original `[128, 3072]` pooled gradient and strided `[128, 3072, 6, 6]` activation without materializing the zero-fill `as_strided_scatter -> as_strided -> expand -> div` pool-gradient tensor, whereas Inductor currently lowers that structured scatter/expand producer, GELU pointwise chain, and `sum([0, 2, 3])` as generic scheduled tensor work; Inductor cannot do this today because scheduler/codegen does not recognize zero-fill view/as_strided scatter followed by broadcasted average-pool backward as a structured scatter-reduce feeding a channel reduction; the fix is SCATTER_REDUCE: add a structured average-pool-backward scatter/expand lowering that maps the pooled-gradient source directly into the GELU-gradient channel-reduction template."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_0d445802ccc4"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_dm_nfnet_f0_train_70c9d47c"

N = 128
C = 3072
H = 6
W = 6
HW = H * W
N_HW = N * HW
INV_HW = 1.0 / HW
GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476
RSQRT2PI = 0.3989422804014327

BLOCK_K = 32
BLOCK_C = 64
K_TILES = triton.cdiv(N_HW, BLOCK_K)

COMPILE_CONFIGS = [
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

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _avgpool_gelu_atomic_kernel(
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
    INV_HW_: tl.constexpr,
    GAMMA_: tl.constexpr,
    RSQRT2_: tl.constexpr,
    RSQRT2PI_: tl.constexpr,
    BLOCK_K_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    channel_block = tl.program_id(0)
    k_block = tl.program_id(1)
    offsets_k = k_block * BLOCK_K_ + tl.arange(0, BLOCK_K_)
    offsets_c = channel_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)

    n = offsets_k // HW_
    hw = offsets_k - n * HW_
    h = hw // W_
    w = hw - h * W_

    base = (
        n[:, None] * x_stride_n
        + h[:, None] * x_stride_h
        + w[:, None] * x_stride_w
    )
    x_offsets = base + offsets_c[None, :] * x_stride_c
    mm_offsets = n[:, None] * C_ + offsets_c[None, :]
    mask = (offsets_k[:, None] < N_HW_) & (offsets_c[None, :] < C_)

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)

    cdf = 0.5 * (tl.erf(x * RSQRT2_) + 1.0)
    pdf_term = x * tl.exp(-0.5 * x * x) * RSQRT2PI_
    value = mm * (INV_HW_ * GAMMA_) * (cdf + pdf_term)
    partial = tl.sum(tl.where(mask, value, 0.0), axis=0)

    tl.atomic_add(out_ptr + offsets_c, partial, sem="relaxed", mask=offsets_c < C_)


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


def oracle_structured_pool_upsample_reduce(
    mm: torch.Tensor,
    convolution_80: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    """Full-scope source-space average-pool-backward + GELU-gradient channel sum."""
    if mm.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if mm.shape != (N, C):
        raise ValueError(f"unexpected mm shape: {tuple(mm.shape)}")
    if convolution_80.shape != (N, C, H, W):
        raise ValueError(f"unexpected activation shape: {tuple(convolution_80.shape)}")
    if not mm.is_contiguous():
        raise ValueError("expected contiguous mm input")

    out = torch.zeros((C,), device=mm.device, dtype=torch.float32)

    _avgpool_gelu_atomic_kernel[(triton.cdiv(C, BLOCK_C), K_TILES)](
        mm,
        convolution_80,
        out,
        x_stride_n=convolution_80.stride(0),
        x_stride_c=convolution_80.stride(1),
        x_stride_h=convolution_80.stride(2),
        x_stride_w=convolution_80.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        N_HW_=N_HW,
        INV_HW_=INV_HW,
        GAMMA_=GAMMA,
        RSQRT2_=RSQRT2,
        RSQRT2PI_=RSQRT2PI,
        BLOCK_K_=BLOCK_K,
        BLOCK_C_=BLOCK_C,
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
        actual = oracle_structured_pool_upsample_reduce(*inputs)
        torch.cuda.synchronize()

    max_abs, max_rel = _max_diff(actual, ref)
    ok = torch.allclose(actual.float(), ref.float(), rtol=rtol, atol=atol)
    stride_ok = actual.stride() == ref.stride()
    print(
        f"output: shape={list(actual.shape)} stride={actual.stride()} "
        f"expected_stride={ref.stride()} max_abs={max_abs:.6e} "
        f"max_rel={max_rel:.6e} allclose={ok} stride_match={stride_ok}"
    )
    print(f"Correctness: {'PASS' if ok and stride_ok else 'FAIL'}")
    return ok and stride_ok


def _compile_with_config(model: torch.nn.Module, inputs: tuple[object, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        with torch.no_grad():
            compiled(*inputs)
            torch.cuda.synchronize()
    return compiled


def run_bench(rep: int, warmup: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    activation = inputs[1]
    assert isinstance(activation, torch.Tensor)

    print(
        f"oracle shape: mm=f32[{N}, {C}], activation=f32[{N}, {C}, {H}, {W}] "
        f"stride={activation.stride()} block_k={BLOCK_K} block_c={BLOCK_C}"
    )
    with torch.no_grad():
        oracle_structured_pool_upsample_reduce(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_structured_pool_upsample_reduce(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"oracle_structured_pool_upsample_reduce: {oracle_us:.3f} us shape={SHAPE_LABEL}")

    if no_compile:
        return

    module = _load_repro_module()
    print("torch.compile full repro timings:")
    for label, config in COMPILE_CONFIGS:
        try:
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
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")


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
        parser.error("select at least one mode: --check and/or --bench")

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
