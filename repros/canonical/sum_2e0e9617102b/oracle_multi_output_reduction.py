"""
Full-scope Triton oracle for sum_2e0e9617102b.

The compiled repro computes:

    out[c] = sum_{n,h,w}(permute_46[n,c,h,w] + getitem_123[n,c,h,w])

for two original contiguous f32[128, 80, 56, 56] inputs and returns one
contiguous f32[80] tensor. This oracle covers that full add+channel-reduction
scope; it does not time a reduction-only subset.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle differs from the
eager expression by never materializing the full add tensor, instead reading the
two original inputs once and accumulating per-channel partials in Triton. That
is also the optimization Inductor can already express for this simple add+sum
region, so any remaining gap is expected to be a memory-bandwidth/tuning issue
rather than missing scheduler fusion, scatter-reduce handling, split-K,
algebraic elimination, or recomputation fusion. If this full-scope Triton
timing does not beat both required compile configurations, this artifact is
diagnosis-only rather than a true floor.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_2e0e9617102b"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 80
H = 56
W = 56
HW = H * W
BLOCK_HW = triton.next_power_of_2(HW)
DEFAULT_GROUP_N = 2



@triton.jit
def _add_sum_partials_kernel(
    x_ptr,
    y_ptr,
    partials_ptr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    y_stride_n: tl.constexpr,
    y_stride_c: tl.constexpr,
    y_stride_h: tl.constexpr,
    y_stride_w: tl.constexpr,
    N_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_HW_: tl.constexpr,
):
    c = tl.program_id(0)
    group = tl.program_id(1)
    hw = tl.arange(0, BLOCK_HW_)
    h = hw // W_
    w = hw - h * W_
    hw_mask = hw < HW_

    acc = tl.zeros([BLOCK_HW_], dtype=tl.float32)
    for i in range(GROUP_N):
        n = group * GROUP_N + i
        valid = (n < N_) & hw_mask
        x_offsets = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
        y_offsets = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
        x_vals = tl.load(x_ptr + x_offsets, mask=valid, other=0.0).to(tl.float32)
        y_vals = tl.load(y_ptr + y_offsets, mask=valid, other=0.0).to(tl.float32)
        acc += x_vals + y_vals

    partial = tl.sum(acc, axis=0)
    tl.store(partials_ptr + group * C_ + c, partial, mask=group < NUM_GROUPS)


@triton.jit
def _finish_partials_kernel(
    partials_ptr,
    out_ptr,
    C_: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    c = tl.program_id(0)
    group_offsets = tl.arange(0, BLOCK_GROUPS)
    mask = group_offsets < NUM_GROUPS
    values = tl.load(partials_ptr + group_offsets * C_ + c, mask=mask, other=0.0)
    tl.store(out_ptr + c, tl.sum(values, axis=0))


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


def _num_groups(group_n: int) -> int:
    return triton.cdiv(N, group_n)


def _make_workspace(device: torch.device, group_n: int) -> tuple[torch.Tensor, torch.Tensor]:
    partials = torch.empty((_num_groups(group_n), C), device=device, dtype=torch.float32)
    out = torch.empty((C,), device=device, dtype=torch.float32)
    return partials, out


def _oracle_into(
    permute_46: torch.Tensor,
    getitem_123: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    group_n: int,
) -> torch.Tensor:
    assert permute_46.shape == (N, C, H, W)
    assert getitem_123.shape == (N, C, H, W)
    assert permute_46.dtype is torch.float32
    assert getitem_123.dtype is torch.float32
    assert out.shape == (C,)
    assert out.dtype is torch.float32

    num_groups = _num_groups(group_n)
    _add_sum_partials_kernel[(C, num_groups)](
        permute_46,
        getitem_123,
        partials,
        x_stride_n=permute_46.stride(0),
        x_stride_c=permute_46.stride(1),
        x_stride_h=permute_46.stride(2),
        x_stride_w=permute_46.stride(3),
        y_stride_n=getitem_123.stride(0),
        y_stride_c=getitem_123.stride(1),
        y_stride_h=getitem_123.stride(2),
        y_stride_w=getitem_123.stride(3),
        N_=N,
        C_=C,
        HW_=HW,
        W_=W,
        NUM_GROUPS=num_groups,
        GROUP_N=group_n,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
    )
    _finish_partials_kernel[(C,)](
        partials,
        out,
        C_=C,
        NUM_GROUPS=num_groups,
        BLOCK_GROUPS=triton.next_power_of_2(num_groups),
        num_warps=1,
    )
    return out


def oracle_fused(
    permute_46: torch.Tensor,
    getitem_123: torch.Tensor,
    group_n: int = DEFAULT_GROUP_N,
) -> torch.Tensor:
    partials, out = _make_workspace(permute_46.device, group_n)
    return _oracle_into(permute_46, getitem_123, partials, out, group_n)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        out = model(*inputs)
    return (out,) if isinstance(out, torch.Tensor) else tuple(out)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float, group_n: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual_raw = oracle_fused(*inputs, group_n=group_n)
        actual = (actual_raw,) if isinstance(actual_raw, torch.Tensor) else tuple(actual_raw)
        torch.cuda.synchronize()

    if len(actual) != len(ref):
        print(f"output_count_match=False actual={len(actual)} expected={len(ref)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (actual_tensor, ref_tensor) in enumerate(zip(actual, ref, strict=True)):
        max_abs, max_rel = _max_diff(actual_tensor, ref_tensor)
        output_ok = torch.allclose(actual_tensor.float(), ref_tensor.float(), rtol=rtol, atol=atol)
        stride_ok = actual_tensor.stride() == ref_tensor.stride()
        shape_ok = actual_tensor.shape == ref_tensor.shape
        dtype_ok = actual_tensor.dtype == ref_tensor.dtype
        tensor_ok = output_ok and stride_ok and shape_ok and dtype_ok
        ok = ok and tensor_ok
        print(
            f"output[{idx}]: shape={list(actual_tensor.shape)} stride={actual_tensor.stride()} "
            f"dtype={actual_tensor.dtype} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={output_ok} shape_match={shape_ok} stride_match={stride_ok} "
            f"dtype_match={dtype_ok}"
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


def run_bench(rep: int, warmup: int, no_compile: bool, group_n: int) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    permute_46, getitem_123 = inputs
    partials, out = _make_workspace(permute_46.device, group_n)

    timings: dict[str, float] = {}
    with torch.no_grad():
        _oracle_into(permute_46, getitem_123, partials, out, group_n)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: _oracle_into(permute_46, getitem_123, partials, out, group_n),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    timings["oracle_fused"] = oracle_us
    print(f"oracle_fused full-scope add+sum: {oracle_us:.3f} us (group_n={group_n})")

    if no_compile:
        return timings

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
        timings[label] = compiled_us
        print(f"torch.compile {label}: {compiled_us:.3f} us")

    if oracle_us < min(timings["coordinate_descent_tuning"], timings["combo_looped_cd"]):
        print("Valid floor: yes")
    else:
        print("Valid floor: no (diagnosis-only; oracle does not beat both compile configs)")
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--group-n", type=int, default=DEFAULT_GROUP_N)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if args.group_n <= 0:
        raise ValueError("--group-n must be positive")
    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol, group_n=args.group_n):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile, group_n=args.group_n)


if __name__ == "__main__":
    with torch.no_grad():
        main()
