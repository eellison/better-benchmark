"""
Full-scope Triton diagnostic artifact for sum_sum_cc11724ac17a.

Gap diagnosis (classification: BANDWIDTH_BOUND): this artifact consumes the
same five tensor inputs plus two shape parameters as repro.py, computes the
`(mul_296 + view(mm_126)) * arg9_1` producer, accumulates both hidden-dimension
row sums, applies the dependent layer-norm-backward epilogue, and writes the
returned `float32[4096, 4096]` transpose-view output with stride `(1, 4096)` in
one Triton row kernel. It differs from Inductor only by using a hand-written
one-row, 4096-column Triton schedule; Inductor already emits a single fused
`add/mul/sub/sum/view` reduction kernel for this full scope, and the historical
best compiled timing is faster than this artifact. There is no scheduler-fusion
or algebraic Inductor change to claim here; any remaining improvement would be a
low-level tiling/autotuning constant-factor win for an already bandwidth-heavy
fused kernel.
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


REPRO_ID = "sum_sum_cc11724ac17a"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 8 * 512
HIDDEN = 4096
HISTORICAL_BEST_COMPILE_US = 53.05600166320801

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
def _layernorm_backward_transpose_kernel(
    mm_ptr,
    mul_ptr,
    weight_ptr,
    rhs_ptr,
    gate_ptr,
    out_ptr,
    HIDDEN_: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    offsets = row * HIDDEN_ + cols

    mm = tl.load(mm_ptr + offsets).to(tl.float32)
    mul = tl.load(mul_ptr + offsets).to(tl.float32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets).to(tl.float32)

    producer = (mul + mm) * weight
    sum_producer = tl.sum(producer, axis=0)
    sum_producer_rhs = tl.sum(producer * rhs, axis=0)
    gate = tl.load(gate_ptr + row).to(tl.float32)
    out = gate * (producer * HIDDEN_ - sum_producer - rhs * sum_producer_rhs)

    # The returned eager tensor is a permute view with stride (1, 4096).  Writing
    # row-major backing storage here makes logical output[col, row] equal out.
    tl.store(out_ptr + offsets, out)


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
        mm_126,
        mul_296,
        arg9_1,
        arg30_1,
        arg175_1,
        shape0,
        shape1,
    ) = inputs
    if tuple(shape0) != (8, 512, HIDDEN):
        raise ValueError(f"unexpected first view shape parameter: {shape0}")
    if tuple(shape1) != (ROWS, HIDDEN):
        raise ValueError(f"unexpected second view shape parameter: {shape1}")
    if mm_126.shape != (ROWS, HIDDEN):
        raise ValueError(f"unexpected mm_126 shape: {tuple(mm_126.shape)}")
    if mul_296.shape != (8, 512, HIDDEN):
        raise ValueError(f"unexpected mul_296 shape: {tuple(mul_296.shape)}")
    if arg9_1.shape != (HIDDEN,):
        raise ValueError(f"unexpected arg9_1 shape: {tuple(arg9_1.shape)}")
    if arg30_1.shape != (8, 512, HIDDEN):
        raise ValueError(f"unexpected arg30_1 shape: {tuple(arg30_1.shape)}")
    if arg175_1.shape != (8, 512, 1):
        raise ValueError(f"unexpected arg175_1 shape: {tuple(arg175_1.shape)}")

    tensors = (mm_126, mul_296, arg9_1, arg30_1, arg175_1)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("oracle expects all tensor inputs to be float32")
    if not mm_126.is_contiguous():
        raise ValueError("oracle expects contiguous mm_126")
    if not mul_296.is_contiguous():
        raise ValueError("oracle expects contiguous mul_296")
    if not arg9_1.is_contiguous():
        raise ValueError("oracle expects contiguous arg9_1")
    if not arg30_1.is_contiguous():
        raise ValueError("oracle expects contiguous arg30_1")
    if not arg175_1.is_contiguous():
        raise ValueError("oracle expects contiguous arg175_1")


def oracle_fused(
    mm_126: torch.Tensor,
    mul_296: torch.Tensor,
    arg9_1: torch.Tensor,
    arg30_1: torch.Tensor,
    arg175_1: torch.Tensor,
    shape0,
    shape1,
) -> torch.Tensor:
    inputs = (mm_126, mul_296, arg9_1, arg30_1, arg175_1, shape0, shape1)
    _validate_inputs(inputs)

    out = torch.empty_strided(
        (HIDDEN, ROWS),
        (1, HIDDEN),
        device=mm_126.device,
        dtype=torch.float32,
    )
    _layernorm_backward_transpose_kernel[(ROWS,)](
        mm_126,
        mul_296,
        arg9_1,
        arg30_1,
        arg175_1,
        out,
        HIDDEN_=HIDDEN,
        BLOCK_N=HIDDEN,
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
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(rep: int, warmup: int, no_compile: bool) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = get_inputs()
    timings: dict[str, float] = {}
    with torch.no_grad():
        oracle_forward(inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_forward(inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    timings["oracle_fused"] = oracle_us
    print(f"oracle_fused full-scope dual row reduction + transpose epilogue: {oracle_us:.3f} us")

    if not no_compile:
        module = _load_repro_module()
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
            timings[label] = compiled_us
            print(f"torch.compile {label}: {compiled_us:.3f} us")

    best_required_compile = min(
        (timings[label] for label, _ in COMPILE_CONFIGS if label in timings),
        default=float("inf"),
    )
    best_reference = min(best_required_compile, HISTORICAL_BEST_COMPILE_US)
    valid_floor = oracle_us < best_reference
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"Valid floor: {'yes' if valid_floor else 'no (diagnosis-only)'}")
    print(
        json.dumps(
            {
                "repro_id": REPRO_ID,
                "oracle_us": round(oracle_us, 3),
                "best_required_compile_us": (
                    round(best_required_compile, 3)
                    if best_required_compile != float("inf")
                    else None
                ),
                "historical_best_compile_us": round(HISTORICAL_BEST_COMPILE_US, 3),
                "valid_floor": valid_floor,
                "classification": "BANDWIDTH_BOUND",
            }
        )
    )
    return timings


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
