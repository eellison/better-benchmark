"""
Oracle for sum_21c4018e2b7b

Gap diagnosis:
  Classification: BANDWIDTH_BOUND
  What oracle does differently: computes the full sigmoid-backward-like
    pointwise expression and channelwise reduction from repro.py in Triton,
    keeping the pointwise work inside the reduction tile and writing only the
    final f32[16] result.
  Why Inductor cannot do it today: Inductor already fuses the surrounding
    pointwise expression into its reduction, so the remaining gap is dominated
    by reading two 102.8 MB fp32 inputs plus one exp/sigmoid per element rather
    than by a missing cross-kernel fusion opportunity.
  What Inductor change would fix it: no scheduler-only change is expected to
    create a lower true floor for this exact full-scope graph; any improvement
    would need a narrowly gated sigmoid/reduction template that beats the
    historical best on the same shape.
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


REPRO_ID = "sum_21c4018e2b7b"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

HISTORICAL_BEST_COMPILE_US = 50.87999999523163
DEFAULT_VARIANT = "partial"
DEFAULT_GROUP_N = 4
DEFAULT_BLOCK_R = 1024
DEFAULT_BLOCK_HW = 1024
DEFAULT_NUM_WARPS = 4
DEFAULT_SIGMOID_IMPL = "exp"

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
        "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
        "triton.multi_kernel=3",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_load_repro_module().make_inputs())


def get_repro_instance():
    """Create the eager repro instance used for reference comparison."""
    return _load_repro_module().Repro()


@triton.jit
def _sigmoid_reduce_grouped_partial_kernel(
    arg_ptr,
    grad_ptr,
    partial_ptr,
    N: tl.constexpr,
    HW: tl.constexpr,
    C: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_TL_SIGMOID: tl.constexpr,
):
    c = tl.program_id(0)
    group = tl.program_id(1)
    rbase = tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)

    for roffset in tl.range(0, GROUP_N * HW, BLOCK_R):
        r = roffset + rbase
        n_in_group = r // HW
        hw_offsets = r - n_in_group * HW
        n = group * GROUP_N + n_in_group
        mask = n < N
        offsets = n * C * HW + c * HW + hw_offsets

        x = tl.load(arg_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")

        if USE_TL_SIGMOID:
            sig = tl.sigmoid(x)
        else:
            sig = 1.0 / (1.0 + tl.exp(-x))

        value = grad * sig * (x * (1.0 - sig) + 1.0)
        acc += tl.where(mask, value, 0.0)

    tl.store(partial_ptr + c * N_GROUPS + group, tl.sum(acc, axis=0))


@triton.jit
def _sigmoid_reduce_partial_kernel(
    arg_ptr,
    grad_ptr,
    partial_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    HW_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    USE_TL_SIGMOID: tl.constexpr,
):
    c = tl.program_id(0)
    n = tl.program_id(1)
    hw_tile = tl.program_id(2)

    hw_offsets = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW
    offsets = n * C * HW + c * HW + hw_offsets

    x = tl.load(arg_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)

    if USE_TL_SIGMOID:
        sig = tl.sigmoid(x)
    else:
        sig = 1.0 / (1.0 + tl.exp(-x))

    value = grad * sig * (x * (1.0 - sig) + 1.0)
    value = tl.where(mask, value, 0.0)
    partial = tl.sum(value, axis=0)

    partial_idx = (c * tl.num_programs(1) + n) * HW_TILES + hw_tile
    tl.store(partial_ptr + partial_idx, partial)


@triton.jit
def _sigmoid_reduce_finalize_kernel(
    partial_ptr,
    out_ptr,
    PARTIALS_PER_C: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
):
    c = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_PARTIALS)
    mask = offsets < PARTIALS_PER_C
    values = tl.load(
        partial_ptr + c * PARTIALS_PER_C + offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(out_ptr + c, tl.sum(values, axis=0))


@triton.jit
def _zero_output_kernel(out_ptr, C: tl.constexpr, BLOCK_C: tl.constexpr):
    offsets = tl.arange(0, BLOCK_C)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK_C,), dtype=tl.float32), mask=offsets < C)


@triton.jit
def _sigmoid_reduce_atomic_kernel(
    arg_ptr,
    grad_ptr,
    out_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    USE_TL_SIGMOID: tl.constexpr,
):
    c = tl.program_id(0)
    n = tl.program_id(1)
    hw_tile = tl.program_id(2)

    hw_offsets = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW
    offsets = n * C * HW + c * HW + hw_offsets

    x = tl.load(arg_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)

    if USE_TL_SIGMOID:
        sig = tl.sigmoid(x)
    else:
        sig = 1.0 / (1.0 + tl.exp(-x))

    value = grad * sig * (x * (1.0 - sig) + 1.0)
    value = tl.where(mask, value, 0.0)
    tl.atomic_add(out_ptr + c, tl.sum(value, axis=0), sem="relaxed")


def _validate_inputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    if len(inputs) != 2:
        raise RuntimeError(f"expected two inputs, got {len(inputs)}")
    arg, grad = inputs
    if not isinstance(arg, torch.Tensor) or not isinstance(grad, torch.Tensor):
        raise RuntimeError("both inputs must be tensors")
    if not arg.is_cuda or not grad.is_cuda:
        raise RuntimeError("CUDA tensors are required")
    if arg.dtype != torch.float32 or grad.dtype != torch.float32:
        raise RuntimeError(f"expected float32 inputs, got {arg.dtype} and {grad.dtype}")
    if arg.shape != grad.shape or arg.ndim != 4:
        raise RuntimeError(f"expected matching 4D inputs, got {arg.shape} and {grad.shape}")
    if not arg.is_contiguous() or not grad.is_contiguous():
        raise RuntimeError("this canonical oracle expects contiguous NCHW inputs")
    n, c, h, w = (int(dim) for dim in arg.shape)
    return arg, grad, n, c, h, w


def _use_tl_sigmoid(sigmoid_impl: str) -> bool:
    if sigmoid_impl == "exp":
        return False
    if sigmoid_impl == "tl-sigmoid":
        return True
    raise ValueError(f"unknown sigmoid implementation: {sigmoid_impl}")


def _launch_partial_oracle(
    arg: torch.Tensor,
    grad: torch.Tensor,
    out: torch.Tensor,
    partials: torch.Tensor,
    *,
    block_hw: int,
    num_warps: int,
    sigmoid_impl: str,
) -> None:
    n, c, h, w = (int(dim) for dim in arg.shape)
    hw = h * w
    hw_tiles = triton.cdiv(hw, block_hw)
    partials_per_c = n * hw_tiles
    block_partials = triton.next_power_of_2(partials_per_c)
    use_tl_sigmoid = _use_tl_sigmoid(sigmoid_impl)

    _sigmoid_reduce_partial_kernel[(c, n, hw_tiles)](
        arg,
        grad,
        partials,
        HW=hw,
        C=c,
        HW_TILES=hw_tiles,
        BLOCK_HW=block_hw,
        USE_TL_SIGMOID=use_tl_sigmoid,
        num_warps=num_warps,
    )
    _sigmoid_reduce_finalize_kernel[(c,)](
        partials,
        out,
        PARTIALS_PER_C=partials_per_c,
        BLOCK_PARTIALS=block_partials,
        num_warps=1,
    )


def _launch_grouped_oracle(
    arg: torch.Tensor,
    grad: torch.Tensor,
    out: torch.Tensor,
    partials: torch.Tensor,
    *,
    group_n: int,
    block_r: int,
    num_warps: int,
    sigmoid_impl: str,
) -> None:
    n, c, h, w = (int(dim) for dim in arg.shape)
    hw = h * w
    n_groups = triton.cdiv(n, group_n)
    block_partials = triton.next_power_of_2(n_groups)
    use_tl_sigmoid = _use_tl_sigmoid(sigmoid_impl)

    _sigmoid_reduce_grouped_partial_kernel[(c, n_groups)](
        arg,
        grad,
        partials,
        N=n,
        HW=hw,
        C=c,
        N_GROUPS=n_groups,
        GROUP_N=group_n,
        BLOCK_R=block_r,
        USE_TL_SIGMOID=use_tl_sigmoid,
        num_warps=num_warps,
    )
    _sigmoid_reduce_finalize_kernel[(c,)](
        partials,
        out,
        PARTIALS_PER_C=n_groups,
        BLOCK_PARTIALS=block_partials,
        num_warps=1,
    )


def _launch_atomic_oracle(
    arg: torch.Tensor,
    grad: torch.Tensor,
    out: torch.Tensor,
    *,
    block_hw: int,
    num_warps: int,
    sigmoid_impl: str,
) -> None:
    n, c, h, w = (int(dim) for dim in arg.shape)
    hw = h * w
    hw_tiles = triton.cdiv(hw, block_hw)
    use_tl_sigmoid = _use_tl_sigmoid(sigmoid_impl)

    _zero_output_kernel[(1,)](
        out,
        C=c,
        BLOCK_C=triton.next_power_of_2(c),
        num_warps=1,
    )
    _sigmoid_reduce_atomic_kernel[(c, n, hw_tiles)](
        arg,
        grad,
        out,
        HW=hw,
        C=c,
        BLOCK_HW=block_hw,
        USE_TL_SIGMOID=use_tl_sigmoid,
        num_warps=num_warps,
    )


def oracle_forward_with_options(
    inputs: tuple[object, ...],
    *,
    variant: str = DEFAULT_VARIANT,
    group_n: int = DEFAULT_GROUP_N,
    block_r: int = DEFAULT_BLOCK_R,
    block_hw: int = DEFAULT_BLOCK_HW,
    num_warps: int = DEFAULT_NUM_WARPS,
    sigmoid_impl: str = DEFAULT_SIGMOID_IMPL,
) -> torch.Tensor:
    """Run the full repro.py computation using Triton kernels only."""
    arg, grad, n, c, h, w = _validate_inputs(tuple(inputs))
    del n, h, w

    out = torch.empty_strided((c,), (1,), device=arg.device, dtype=torch.float32)
    if variant == "grouped":
        n_groups = triton.cdiv(arg.shape[0], group_n)
        partials = torch.empty((c, n_groups), device=arg.device, dtype=torch.float32)
        _launch_grouped_oracle(
            arg,
            grad,
            out,
            partials,
            group_n=group_n,
            block_r=block_r,
            num_warps=num_warps,
            sigmoid_impl=sigmoid_impl,
        )
    elif variant == "partial":
        hw_tiles = triton.cdiv(arg.shape[2] * arg.shape[3], block_hw)
        partials = torch.empty((c, arg.shape[0] * hw_tiles), device=arg.device, dtype=torch.float32)
        _launch_partial_oracle(
            arg,
            grad,
            out,
            partials,
            block_hw=block_hw,
            num_warps=num_warps,
            sigmoid_impl=sigmoid_impl,
        )
    elif variant == "atomic":
        _launch_atomic_oracle(
            arg,
            grad,
            out,
            block_hw=block_hw,
            num_warps=num_warps,
            sigmoid_impl=sigmoid_impl,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")
    return out


def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Scope invariant: same inputs and same single f32[16] output as Repro.forward."""
    return oracle_forward_with_options(inputs)


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
