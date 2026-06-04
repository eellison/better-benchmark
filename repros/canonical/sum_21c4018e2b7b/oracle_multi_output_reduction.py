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
import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


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


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    max_abs = torch.nan_to_num(diff, nan=0.0).max().item()
    rel = diff / expected_f32.abs().clamp_min(1e-8)
    max_rel = torch.nan_to_num(rel, nan=0.0).max().item()
    return max_abs, max_rel


def _bench_cuda_graph(fn, *, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return min(times)


def _compile_with_config(module, inputs: tuple[object, ...], config: dict[str, object], warmup: int):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = module.Repro().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_check(
    *,
    variant: str,
    group_n: int,
    block_r: int,
    block_hw: int,
    num_warps: int,
    sigmoid_impl: str,
    rtol: float,
    atol: float,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    module = _load_repro_module()
    inputs = tuple(module.make_inputs())
    model = module.Repro().cuda()

    with torch.no_grad():
        ref = model(*inputs)
        got = oracle_forward_with_options(
            inputs,
            variant=variant,
            group_n=group_n,
            block_r=block_r,
            block_hw=block_hw,
            num_warps=num_warps,
            sigmoid_impl=sigmoid_impl,
        )
        torch.cuda.synchronize()

    max_abs, max_rel = _max_diff(got, ref)
    shape_match = tuple(got.shape) == tuple(ref.shape)
    dtype_match = got.dtype == ref.dtype
    stride_match = got.stride() == ref.stride()
    value_match = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
    ok = shape_match and dtype_match and stride_match and value_match

    print(
        "check full-scope sigmoid reduction: "
        f"shape={tuple(got.shape)} dtype={got.dtype} stride={got.stride()} "
        f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
        f"shape_match={shape_match} dtype_match={dtype_match} "
        f"stride_match={stride_match} allclose={value_match}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_oracle_variant(
    inputs: tuple[object, ...],
    *,
    variant: str,
    group_n: int,
    block_r: int,
    block_hw: int,
    num_warps: int,
    sigmoid_impl: str,
    warmup: int,
    rep: int,
) -> float:
    arg, grad, _n, c, h, w = _validate_inputs(inputs)
    out = torch.empty_strided((c,), (1,), device=arg.device, dtype=torch.float32)

    if variant == "grouped":
        n_groups = triton.cdiv(arg.shape[0], group_n)
        partials = torch.empty((c, n_groups), device=arg.device, dtype=torch.float32)
        fn = lambda: _launch_grouped_oracle(
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
        hw_tiles = triton.cdiv(h * w, block_hw)
        partials = torch.empty((c, arg.shape[0] * hw_tiles), device=arg.device, dtype=torch.float32)
        fn = lambda: _launch_partial_oracle(
            arg,
            grad,
            out,
            partials,
            block_hw=block_hw,
            num_warps=num_warps,
            sigmoid_impl=sigmoid_impl,
        )
    elif variant == "atomic":
        fn = lambda: _launch_atomic_oracle(
            arg,
            grad,
            out,
            block_hw=block_hw,
            num_warps=num_warps,
            sigmoid_impl=sigmoid_impl,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    return _bench_cuda_graph(fn, warmup=warmup, rep=rep)


def run_bench(
    *,
    variant: str,
    group_n: int,
    block_r: int,
    block_hw: int,
    num_warps: int,
    sigmoid_impl: str,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    module = _load_repro_module()
    inputs = tuple(module.make_inputs())
    arg, _grad, n, c, h, w = _validate_inputs(inputs)
    logical_bytes = arg.numel() * 2 * arg.element_size() + c * 4

    print(
        "oracle shape: "
        f"arg=f32[{n},{c},{h},{w}] grad=f32[{n},{c},{h},{w}] "
        f"out=f32[{c}] variant={variant} group_n={group_n} "
        f"block_r={block_r} block_hw={block_hw} sigmoid={sigmoid_impl}"
    )
    print(f"full-scope logical traffic: {logical_bytes / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_us = _bench_oracle_variant(
            inputs,
            variant=variant,
            group_n=group_n,
            block_r=block_r,
            block_hw=block_hw,
            num_warps=num_warps,
            sigmoid_impl=sigmoid_impl,
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = logical_bytes / (oracle_us * 1e-6) / 1e12
    print(f"oracle full-scope Triton: {oracle_us:.3f} us ({oracle_bw:.3f} TB/s logical bytes)")
    print(f"oracle_us={oracle_us:.3f}")

    compile_times: list[tuple[str, float]] = []
    if not no_compile:
        print("CUDA graph replay timings cover the same repro.py inputs, output, dtypes, shapes, and strides")
        for label, config in COMPILE_CONFIGS:
            try:
                compiled = _compile_with_config(module, inputs, config, warmup)
                us = _bench_cuda_graph(lambda: compiled(*inputs), warmup=warmup, rep=rep)
                compile_times.append((label, us))
                print(f"torch.compile {label}: {us:.3f} us")
            except Exception as exc:
                print(f"torch.compile {label}: FAILED ({exc})")

    best_required = min((us for _, us in compile_times), default=math.nan)
    gate = min(best_required, HISTORICAL_BEST_COMPILE_US) if compile_times else HISTORICAL_BEST_COMPILE_US
    true_floor = oracle_us < gate
    ratio = best_required / oracle_us if compile_times and oracle_us > 0 else math.nan
    status = "GOOD" if true_floor else "DIAGNOSIS_ONLY"

    print(f"historical_best_compile_us={HISTORICAL_BEST_COMPILE_US:.6f}")
    if compile_times:
        print(f"best_required_compile_us={best_required:.3f}")
    print(f"true_floor={true_floor}")
    if not true_floor:
        print("diagnosis_only: oracle is not a true floor because a required or historical compile timing is faster")

    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(best_required, 3) if compile_times else None,
        "ratio": round(ratio, 3) if ratio == ratio else None,
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "true_floor": true_floor,
        "classification": "BANDWIDTH_BOUND",
        "status": status,
    }
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Oracle for {REPRO_ID}")
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--variant", choices=("grouped", "partial", "atomic"), default=DEFAULT_VARIANT)
    parser.add_argument("--group-n", type=int, default=DEFAULT_GROUP_N)
    parser.add_argument("--block-r", type=int, default=DEFAULT_BLOCK_R)
    parser.add_argument("--block-hw", type=int, default=DEFAULT_BLOCK_HW)
    parser.add_argument("--num-warps", type=int, default=DEFAULT_NUM_WARPS)
    parser.add_argument("--sigmoid-impl", choices=("exp", "tl-sigmoid"), default=DEFAULT_SIGMOID_IMPL)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile baselines")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check:
        ok = run_check(
            variant=args.variant,
            group_n=args.group_n,
            block_r=args.block_r,
            block_hw=args.block_hw,
            num_warps=args.num_warps,
            sigmoid_impl=args.sigmoid_impl,
            rtol=args.rtol,
            atol=args.atol,
        )
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(
            variant=args.variant,
            group_n=args.group_n,
            block_r=args.block_r,
            block_hw=args.block_hw,
            num_warps=args.num_warps,
            sigmoid_impl=args.sigmoid_impl,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
