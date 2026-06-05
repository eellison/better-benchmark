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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
