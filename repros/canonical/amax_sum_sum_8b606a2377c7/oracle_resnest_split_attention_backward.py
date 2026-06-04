"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ResNeSt split-attention backward tail returned by Repro.forward, including the avg_pool2d_backward producer, BN-affine/ReLU gate, radix-2 softmax, softmax-backward fma, layout views/permutes, spatial reductions, and final [256] channel reduction using Triton kernels, whereas Inductor currently lowers the decomposed avg_pool2d_backward/expand/BN-ReLU/view/mul/softmax/sum/fma/permute/view/sum graph as generic producer, pointwise, softmax, and reduction kernels over materialized [32, 2, 128, 56, 56] intermediates; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize ResNeSt split-attention backward into a fused structured pool-backward plus radix-softmax-gradient reduction template; the fix is NEW_PATTERN: add an Inductor lowering for split-attention backward that computes pool-gradient/gated spatial summaries and the radix softmax-gradient channel reduction directly."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "amax_sum_sum_8b606a2377c7"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_timm_resnest_train_001_c9d1356e"

BATCH = 32
RADIX = 2
GROUP_CHANNELS = 128
CHANNELS = RADIX * GROUP_CHANNELS
H = 56
W = 56
HW = H * W
POOLED_H = 28
POOLED_W = 28
KERNEL_AREA = 9.0
BLOCK_HW = 256
NUM_HW_TILES = 13
BLOCK_TILES = 16
BLOCK_BATCH = 32



if triton is not None:

    @triton.jit
    def _partial_spatial_sums_kernel(
        grad_pool_ptr,
        bn_input_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        partial_ptr,
        H_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        POOLED_H_: tl.constexpr,
        POOLED_W_: tl.constexpr,
        BATCH_: tl.constexpr,
        RADIX_: tl.constexpr,
        GROUP_CHANNELS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        KERNEL_AREA_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        tile = tl.program_id(0)
        batch = tl.program_id(1)
        c = tl.program_id(2)

        offs = tile * BLOCK_HW_ + tl.arange(0, BLOCK_HW_)
        mask = offs < HW_
        h = offs // W_
        w = offs - h * W_

        h_base = h // 2
        w_base = w // 2
        pool = tl.zeros([BLOCK_HW_], dtype=tl.float32)
        for dh in tl.static_range(0, 2):
            oh = h_base + dh
            valid_h = (oh < POOLED_H_) & (h >= (oh * 2 - 1)) & (h <= (oh * 2 + 1))
            for dw in tl.static_range(0, 2):
                ow = w_base + dw
                valid = mask & valid_h & (ow < POOLED_W_) & (w >= (ow * 2 - 1)) & (w <= (ow * 2 + 1))
                grad = tl.load(
                    grad_pool_ptr + ((batch * GROUP_CHANNELS_ + c) * POOLED_H_ + oh) * POOLED_W_ + ow,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                pool += grad * (1.0 / KERNEL_AREA_)

        j0 = c
        j1 = c + GROUP_CHANNELS_
        spatial = h * W_ + w
        x0 = tl.load(
            bn_input_ptr + (batch * CHANNELS_ + j0) * HW_ + spatial,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            bn_input_ptr + (batch * CHANNELS_ + j1) * HW_ + spatial,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        mean0 = tl.load(mean_ptr + j0).to(tl.float32)
        mean1 = tl.load(mean_ptr + j1).to(tl.float32)
        inv0 = tl.load(invstd_ptr + j0).to(tl.float32)
        inv1 = tl.load(invstd_ptr + j1).to(tl.float32)
        weight0 = tl.load(weight_ptr + j0).to(tl.float32)
        weight1 = tl.load(weight_ptr + j1).to(tl.float32)
        bias0 = tl.load(bias_ptr + j0).to(tl.float32)
        bias1 = tl.load(bias_ptr + j1).to(tl.float32)

        relu0 = tl.maximum((x0 - mean0) * inv0 * weight0 + bias0, 0.0)
        relu1 = tl.maximum((x1 - mean1) * inv1 * weight1 + bias1, 0.0)
        sum0 = tl.sum(tl.where(mask, pool * relu0, 0.0), axis=0)
        sum1 = tl.sum(tl.where(mask, pool * relu1, 0.0), axis=0)

        base = ((tile * BATCH_ + batch) * RADIX_) * GROUP_CHANNELS_ + c
        tl.store(partial_ptr + base, sum0)
        tl.store(partial_ptr + base + GROUP_CHANNELS_, sum1)

    @triton.jit
    def _final_softmax_backward_reduce_kernel(
        partial_ptr,
        logits_ptr,
        out_ptr,
        BATCH_: tl.constexpr,
        RADIX_: tl.constexpr,
        GROUP_CHANNELS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        NUM_HW_TILES_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
        BLOCK_BATCH_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES_)
        batch = tl.arange(0, BLOCK_BATCH_)
        mask = (tiles[:, None] < NUM_HW_TILES_) & (batch[None, :] < BATCH_)

        offsets0 = ((tiles[:, None] * BATCH_ + batch[None, :]) * RADIX_) * GROUP_CHANNELS_ + c
        offsets1 = offsets0 + GROUP_CHANNELS_
        partial0 = tl.load(partial_ptr + offsets0, mask=mask, other=0.0).to(tl.float32)
        partial1 = tl.load(partial_ptr + offsets1, mask=mask, other=0.0).to(tl.float32)
        spatial0 = tl.sum(partial0, axis=0)
        spatial1 = tl.sum(partial1, axis=0)

        logits0 = tl.load(logits_ptr + batch * CHANNELS_ + c, mask=batch < BATCH_, other=0.0).to(tl.float32)
        logits1 = tl.load(
            logits_ptr + batch * CHANNELS_ + GROUP_CHANNELS_ + c,
            mask=batch < BATCH_,
            other=0.0,
        ).to(tl.float32)
        row_max = tl.maximum(logits0, logits1)
        exp0 = tl.exp(logits0 - row_max)
        exp1 = tl.exp(logits1 - row_max)
        denom = exp0 + exp1
        prob0 = exp0 / denom
        prob1 = exp1 / denom

        weighted_sum = prob0 * spatial0 + prob1 * spatial1
        grad0 = prob0 * (spatial0 - weighted_sum)
        grad1 = prob1 * (spatial1 - weighted_sum)

        batch_mask = batch < BATCH_
        tl.store(out_ptr + c, tl.sum(tl.where(batch_mask, grad0, 0.0), axis=0))
        tl.store(out_ptr + GROUP_CHANNELS_ + c, tl.sum(tl.where(batch_mask, grad1, 0.0), axis=0))


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(seed: int) -> tuple[Any, ...]:
    module = _load_repro_module()
    torch.manual_seed(seed)
    inputs = module.make_inputs()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)


def _require_cuda() -> None:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _validate_shape_params(
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
) -> None:
    expected = (
        [BATCH, RADIX, GROUP_CHANNELS, H, W],
        [BATCH, RADIX, GROUP_CHANNELS, H, W],
        [BATCH, 1, RADIX, -1],
        [BATCH, CHANNELS, 1, 1],
        [BATCH, CHANNELS],
        [BATCH, RADIX, 1, GROUP_CHANNELS],
        [BATCH, CHANNELS, 1, 1],
    )
    got = (
        list(_shape_param_0),
        list(_shape_param_1),
        list(_shape_param_2),
        list(_shape_param_3),
        list(_shape_param_4),
        list(_shape_param_5),
        list(_shape_param_6),
    )
    if got != expected:
        raise ValueError(f"unexpected shape params: {got}")


def _launch_oracle(
    getitem_39: torch.Tensor,
    arg90_1: torch.Tensor,
    arg91_1: torch.Tensor,
    arg92_1: torch.Tensor,
    arg23_1: torch.Tensor,
    arg24_1: torch.Tensor,
    arg97_1: torch.Tensor,
    partial: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _partial_spatial_sums_kernel[(NUM_HW_TILES, BATCH, GROUP_CHANNELS)](
        getitem_39,
        arg90_1,
        arg91_1,
        arg92_1,
        arg23_1,
        arg24_1,
        partial,
        H_=H,
        W_=W,
        HW_=HW,
        POOLED_H_=POOLED_H,
        POOLED_W_=POOLED_W,
        BATCH_=BATCH,
        RADIX_=RADIX,
        GROUP_CHANNELS_=GROUP_CHANNELS,
        CHANNELS_=CHANNELS,
        KERNEL_AREA_=KERNEL_AREA,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
    )
    _final_softmax_backward_reduce_kernel[(GROUP_CHANNELS,)](
        partial,
        arg97_1,
        out,
        BATCH_=BATCH,
        RADIX_=RADIX,
        GROUP_CHANNELS_=GROUP_CHANNELS,
        CHANNELS_=CHANNELS,
        NUM_HW_TILES_=NUM_HW_TILES,
        BLOCK_TILES_=triton.next_power_of_2(NUM_HW_TILES),
        BLOCK_BATCH_=triton.next_power_of_2(BATCH),
        num_warps=8,
    )
    return out


def oracle_resnest_split_attention_backward(
    getitem_39: torch.Tensor,
    arg98_1: torch.Tensor,
    arg90_1: torch.Tensor,
    arg91_1: torch.Tensor,
    arg92_1: torch.Tensor,
    arg23_1: torch.Tensor,
    arg24_1: torch.Tensor,
    arg97_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
) -> torch.Tensor:
    del arg98_1
    _validate_shape_params(
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    )
    if getitem_39.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if getitem_39.shape != (BATCH, GROUP_CHANNELS, POOLED_H, POOLED_W):
        raise ValueError(f"unexpected getitem_39 shape: {tuple(getitem_39.shape)}")
    if arg90_1.shape != (BATCH, CHANNELS, H, W):
        raise ValueError(f"unexpected arg90_1 shape: {tuple(arg90_1.shape)}")
    if arg97_1.shape != (BATCH, CHANNELS, 1, 1):
        raise ValueError(f"unexpected arg97_1 shape: {tuple(arg97_1.shape)}")

    partial = torch.empty(
        (NUM_HW_TILES, BATCH, RADIX, GROUP_CHANNELS),
        device=getitem_39.device,
        dtype=torch.float32,
    )
    out = torch.empty((CHANNELS,), device=getitem_39.device, dtype=torch.float32)
    return _launch_oracle(
        getitem_39,
        arg90_1,
        arg91_1,
        arg92_1,
        arg23_1,
        arg24_1,
        arg97_1,
        partial,
        out,
    )


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / expected.float().abs().clamp_min(1.0e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def run_check(args: argparse.Namespace) -> bool:
    _require_cuda()
    inputs = make_inputs(seed=args.seed)
    model = _load_repro_module().Repro().cuda()
    with torch.no_grad():
        expected = _as_tuple(model(*inputs))
        actual = _as_tuple(oracle_resnest_split_attention_backward(*inputs))
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output arity mismatch: oracle={len(actual)} expected={len(expected)}")
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        values_ok = torch.allclose(got.float(), ref.float(), rtol=args.rtol, atol=args.atol)
        item_ok = shape_ok and dtype_ok and stride_ok and values_ok
        ok = ok and item_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok} "
            f"allclose={values_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_cuda(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def run_bench(args: argparse.Namespace) -> None:
    _require_cuda()
    inputs = make_inputs(seed=args.seed)
    (
        getitem_39,
        _arg98_1,
        arg90_1,
        arg91_1,
        arg92_1,
        arg23_1,
        arg24_1,
        arg97_1,
        *_shape_params,
    ) = inputs
    partial = torch.empty(
        (NUM_HW_TILES, BATCH, RADIX, GROUP_CHANNELS),
        device=getitem_39.device,
        dtype=torch.float32,
    )
    out = torch.empty((CHANNELS,), device=getitem_39.device, dtype=torch.float32)

    logical_bytes = (
        BATCH * GROUP_CHANNELS * POOLED_H * POOLED_W * 4
        + BATCH * CHANNELS * H * W * 4
        + CHANNELS * 4 * 4
        + BATCH * CHANNELS * 4
        + NUM_HW_TILES * BATCH * CHANNELS * 4
        + CHANNELS * 4
    )
    print(
        f"bench full-scope ResNeSt split-attention backward: "
        f"pool_grad=f32[{BATCH}, {GROUP_CHANNELS}, {POOLED_H}, {POOLED_W}] "
        f"bn_input=f32[{BATCH}, {CHANNELS}, {H}, {W}]"
    )
    print(f"logical oracle traffic={logical_bytes / 1e6:.3f} MB")

    with torch.no_grad():
        oracle_us = _bench_cuda(
            lambda: _launch_oracle(
                getitem_39,
                arg90_1,
                arg91_1,
                arg92_1,
                arg23_1,
                arg24_1,
                arg97_1,
                partial,
                out,
            ),
            warmup=args.warmup,
            rep=args.rep,
        )
    bandwidth = logical_bytes / (oracle_us * 1e-6) / 1e12
    print(
        f"oracle_resnest_split_attention_backward: {oracle_us:.3f} us "
        f"({bandwidth:.3f} TB/s logical) shape={SHAPE_LABEL}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle output against Repro.forward")
    parser.add_argument("--bench", action="store_true", help="benchmark the Triton oracle")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("choose --check and/or --bench")

    if args.check and not run_check(args):
        sys.exit(1)
    if args.bench:
        run_bench(args)


if __name__ == "__main__":
    with torch.no_grad():
        main()
