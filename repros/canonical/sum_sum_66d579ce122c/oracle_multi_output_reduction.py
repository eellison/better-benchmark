"""
Full-scope Triton oracle for sum_sum_66d579ce122c.

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle consumes the
same seven inputs as repro.py and returns the same contiguous float32[32]
channel-gradient vector, but rewrites the dependent reduction chain
`sum_n(sigmoid'(arg175[n,c]) * sum_hw(pointwise[n,c,h,w]))` into one weighted
sum over `(n,h,w)` using Triton-only zeroing plus one atomic add per `(n,c)`
spatial tile. Inductor currently emits a spatial-reduction kernel that
materializes the `[128,32,1,1]` intermediate and a second reduction kernel for
the batch sum because it does not have a reduction-chain algebraic rewrite that
can distribute a per-`(n,c)` multiplier through the inner spatial sum. The fix
is ALGEBRAIC_ELIMINATION: canonicalize this linear dependent-reduction pattern
to a single weighted channel reduction before scheduling/codegen.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_66d579ce122c"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 32
H = 112
W = 112
HW = H * W
HISTORICAL_BEST_COMPILE_US = 79.71200346946716

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
            "triton.multi_kernel=3"
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

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _zero_output_kernel(out_ptr, C_: tl.constexpr, BLOCK_C: tl.constexpr):
    c = tl.arange(0, BLOCK_C)
    tl.store(out_ptr + c, tl.zeros([BLOCK_C], dtype=tl.float32), mask=c < C_)


@triton.jit
def _spatial_weighted_atomic_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    grad_ptr,
    gate_ptr,
    out_ptr,
    HW_: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.arange(0, BLOCK_HW)
    mask = hw < HW_
    offsets = n * C_ * HW_ + c * HW_ + hw

    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c).to(tl.float32)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    act = ((x - mean) * invstd) * affine_weight + affine_bias
    swish = act / (tl.exp(-act) + 1.0)
    spatial_sum = tl.sum(tl.where(mask, grad * swish, 0.0), axis=0)

    gate = tl.load(gate_ptr + n * C_ + c).to(tl.float32)
    sigmoid = 1.0 / (tl.exp(-gate) + 1.0)
    gate_grad = sigmoid * (1.0 - sigmoid)
    tl.atomic_add(out_ptr + c, spatial_sum * gate_grad, sem="relaxed")


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def get_repro_instance() -> torch.nn.Module:
    module = _load_repro_module()
    return module.Repro().cuda()


def _validate_inputs(inputs: tuple[object, ...]) -> None:
    (
        arg169_1,
        arg170_1,
        arg171_1,
        arg4_1,
        arg5_1,
        getitem_228,
        arg175_1,
    ) = inputs
    expected = (
        ((N, C, H, W), "arg169_1"),
        ((1, C, 1, 1), "arg170_1"),
        ((1, C, 1, 1), "arg171_1"),
        ((C,), "arg4_1"),
        ((C,), "arg5_1"),
        ((N, C, H, W), "getitem_228"),
        ((N, C, 1, 1), "arg175_1"),
    )
    for tensor, (shape, name) in zip(inputs, expected):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.shape != shape:
            raise ValueError(f"unexpected {name} shape: {tuple(tensor.shape)}")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def oracle_fused(
    arg169_1: torch.Tensor,
    arg170_1: torch.Tensor,
    arg171_1: torch.Tensor,
    arg4_1: torch.Tensor,
    arg5_1: torch.Tensor,
    getitem_228: torch.Tensor,
    arg175_1: torch.Tensor,
) -> torch.Tensor:
    inputs = (arg169_1, arg170_1, arg171_1, arg4_1, arg5_1, getitem_228, arg175_1)
    _validate_inputs(inputs)

    out = torch.empty((C,), device=arg169_1.device, dtype=torch.float32)
    _zero_output_kernel[(1,)](
        out,
        C_=C,
        BLOCK_C=triton.next_power_of_2(C),
        num_warps=1,
    )
    _spatial_weighted_atomic_kernel[(N, C)](
        arg169_1,
        arg170_1,
        arg171_1,
        arg4_1,
        arg5_1,
        getitem_228,
        arg175_1,
        out,
        HW_=HW,
        C_=C,
        BLOCK_HW=triton.next_power_of_2(HW),
        num_warps=8,
    )
    return out


def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Run the full-scope oracle with the same input tuple as Repro.forward."""
    return oracle_fused(*inputs)


def reference_output(inputs: tuple[object, ...]) -> torch.Tensor:
    model = get_repro_instance()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return float(diff.max().item()), float(rel.max().item())


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = get_inputs()
    with torch.no_grad():
        ref = reference_output(inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    shape_ok = actual.shape == ref.shape
    dtype_ok = actual.dtype == ref.dtype
    stride_ok = actual.stride() == ref.stride()
    max_abs, max_rel = _max_diff(actual, ref)
    close_ok = torch.allclose(actual.float(), ref.float(), rtol=rtol, atol=atol)
    ok = shape_ok and dtype_ok and stride_ok and close_ok
    print(
        f"output[0]: shape={list(actual.shape)} dtype={actual.dtype} "
        f"stride={actual.stride()} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
        f"allclose={close_ok} shape_match={shape_ok} stride_match={stride_ok} "
        f"dtype_match={dtype_ok}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    config: dict[str, object],
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(5):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def _bench_compile_config(
    label: str,
    config: dict[str, object],
    inputs: tuple[object, ...],
    warmup: int,
    rep: int,
) -> float:
    module = _load_repro_module()
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
    return float(compiled_us)


def run_bench(rep: int, warmup: int, no_compile: bool) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = get_inputs()
    with torch.no_grad():
        oracle_forward(inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_forward(inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    oracle_us = float(oracle_us)
    print(f"oracle_fused full-scope weighted channel reduction: {oracle_us:.3f} us")

    compile_results: dict[str, float] = {}
    if not no_compile:
        for label, config in COMPILE_CONFIGS:
            compile_results[label] = _bench_compile_config(label, config, inputs, warmup, rep)

    best_required_compile_us = min(compile_results.values()) if compile_results else None
    gate_values = [HISTORICAL_BEST_COMPILE_US]
    if best_required_compile_us is not None:
        gate_values.append(best_required_compile_us)
    best_gate_us = min(gate_values)
    true_floor = oracle_us < best_gate_us
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": oracle_us,
        "best_required_compile_us": best_required_compile_us,
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "best_gate_us": best_gate_us,
        "ratio_vs_best_required_compile": (
            best_required_compile_us / oracle_us if best_required_compile_us is not None else None
        ),
        "classification": "ALGEBRAIC_ELIMINATION",
        "true_floor": true_floor,
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
        "compile_configs": compile_results,
    }
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"Valid floor: {'yes' if true_floor else 'no (diagnosis-only)'}")
    print(json.dumps(result, sort_keys=True))
    if not true_floor:
        print("note: oracle is diagnosis-only because it does not beat all required compile baselines")
    return result


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
