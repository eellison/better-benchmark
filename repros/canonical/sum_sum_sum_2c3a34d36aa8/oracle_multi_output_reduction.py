"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet gating backward return tuple from Repro.forward by streaming the shared [128,1536,12,12] pointwise producer once into a per-(N,C) spatial summary and finalizing both the scalar loss-gradient sum and [1536] sigmoid-gradient channel sum, whereas Inductor currently schedules the multiply/add/sigmoid producer, the global sum, the spatial sum, the sigmoid-derivative epilogue, and the channel sum as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler cannot fuse sibling reductions with different output ranks when one output needs a per-(N,C) reduced epilogue before the final channel reduction; the fix is SCHEDULER_FUSION: teach Inductor to form one multi-output reduction schedule that shares the pointwise producer, emits per-(N,C) partial summaries, and lowers the scalar and channel finalizers together."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_2c3a34d36aa8"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 1536
H = 12
W = 12
HW = H * W
SCALE0 = 0.9622504486493761
SCALE1 = 1.7015043497085571
POINTWISE_SCALE = 0.2

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@triton.jit
def _spatial_summary_kernel(
    getitem_ptr,
    arg450_ptr,
    add52_ptr,
    arg272_ptr,
    arg269_ptr,
    arg68_ptr,
    scalar_nc_ptr,
    channel_nc_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    SCALE0_: tl.constexpr,
    SCALE1_: tl.constexpr,
    POINTWISE_SCALE_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.arange(0, BLOCK_HW)
    c_mask = c < C_
    hw_mask = hw < HW_
    mask = c_mask[:, None] & hw_mask[None, :]

    offsets = n * C_ * HW_ + c[:, None] * HW_ + hw[None, :]
    x = tl.load(getitem_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(arg450_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    base = tl.load(add52_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(arg269_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    mul0 = x * SCALE0_
    mul1 = mul0 * SCALE1_
    pointwise = (base + mul1 * a) * POINTWISE_SCALE_
    sum_pointwise_y = tl.sum(tl.where(mask, pointwise * y, 0.0), axis=1)

    sigmoid = tl.sigmoid(tl.load(arg272_ptr + n * C_ + c, mask=c_mask, other=0.0).to(tl.float32))
    sigmoid_grad = sigmoid * (1.0 - sigmoid)
    scalar_scale = tl.load(arg68_ptr).to(tl.float32)

    out_offsets = n * C_ + c
    tl.store(scalar_nc_ptr + out_offsets, sum_pointwise_y * sigmoid * 2.0, mask=c_mask)
    tl.store(channel_nc_ptr + out_offsets, sum_pointwise_y * scalar_scale * 2.0 * sigmoid_grad, mask=c_mask)


@triton.jit
def _finalize_channel_kernel(
    scalar_nc_ptr,
    channel_nc_ptr,
    scalar_block_ptr,
    out_channel_ptr,
    N_: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    block = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)
    c = block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (n[:, None] < N_) & (c[None, :] < C_)
    offsets = n[:, None] * C_ + c[None, :]

    scalar_vals = tl.load(scalar_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    channel_vals = tl.load(channel_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    channel_sum = tl.sum(channel_vals, axis=0)
    scalar_per_c = tl.sum(scalar_vals, axis=0)

    tl.store(out_channel_ptr + c, channel_sum, mask=c < C_)
    tl.store(scalar_block_ptr + block, tl.sum(scalar_per_c, axis=0))


@triton.jit
def _finalize_scalar_kernel(
    scalar_block_ptr,
    out_scalar_ptr,
    NUM_C_BLOCKS: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    offsets = tl.arange(0, BLOCKS)
    mask = offsets < NUM_C_BLOCKS
    vals = tl.load(scalar_block_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_scalar_ptr, tl.sum(vals, axis=0))


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    getitem_129: torch.Tensor,
    arg450_1: torch.Tensor,
    add_52: torch.Tensor,
    arg272_1: torch.Tensor,
    arg269_1: torch.Tensor,
    arg68_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert getitem_129.shape == (N, C, H, W)
    assert arg450_1.shape == (N, C, H, W)
    assert add_52.shape == (N, C, H, W)
    assert arg272_1.shape == (N, C, 1, 1)
    assert arg269_1.shape == (N, C, H, W)
    assert arg68_1.shape == ()

    getitem = getitem_129.contiguous()
    arg450 = arg450_1.contiguous()
    add52 = add_52.contiguous()
    arg272 = arg272_1.contiguous()
    arg269 = arg269_1.contiguous()
    arg68 = arg68_1.contiguous()
    device = getitem.device

    block_c = 16
    num_c_blocks = triton.cdiv(C, block_c)
    scalar_nc = torch.empty((N, C), device=device, dtype=torch.float32)
    channel_nc = torch.empty((N, C), device=device, dtype=torch.float32)

    _spatial_summary_kernel[(N, num_c_blocks)](
        getitem,
        arg450,
        add52,
        arg272,
        arg269,
        arg68,
        scalar_nc,
        channel_nc,
        C_=C,
        HW_=HW,
        SCALE0_=SCALE0,
        SCALE1_=SCALE1,
        POINTWISE_SCALE_=POINTWISE_SCALE,
        BLOCK_C=block_c,
        BLOCK_HW=triton.next_power_of_2(HW),
        num_warps=8,
    )

    out_scalar = torch.empty((), device=device, dtype=torch.float32)
    out_channel = torch.empty((C,), device=device, dtype=torch.float32)
    scalar_blocks = torch.empty((num_c_blocks,), device=device, dtype=torch.float32)

    _finalize_channel_kernel[(num_c_blocks,)](
        scalar_nc,
        channel_nc,
        scalar_blocks,
        out_channel,
        N_=N,
        C_=C,
        BLOCK_N=triton.next_power_of_2(N),
        BLOCK_C=block_c,
        num_warps=8,
    )
    _finalize_scalar_kernel[(1,)](
        scalar_blocks,
        out_scalar,
        NUM_C_BLOCKS=num_c_blocks,
        BLOCKS=triton.next_power_of_2(num_c_blocks),
        num_warps=4,
    )

    return out_scalar, out_channel


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
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

    ok = len(actual) == len(ref)
    if not ok:
        print(f"output_count: got={len(actual)} expected={len(ref)}")

    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        output_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == expected.stride()
        ok = ok and output_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
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
    print(f"oracle_fused full-scope NFNet multi-output reduction: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    compile_configs = [
        ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
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
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-1)
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
