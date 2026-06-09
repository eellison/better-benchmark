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



from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

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


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / expected.float().abs().clamp_min(1.0e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


@oracle_impl(hardware="H100", shapes="(T([32, 128, 28, 28], f32), T([32, 128, 56, 56], f32), T([32, 256, 56, 56], f32), T([1, 256, 1, 1], f32), T([1, 256, 1, 1], f32), T([256], f32), T([256], f32), T([32, 256, 1, 1], f32), S([32, 2, 128, 56, 56]), S([32, 2, 128, 56, 56]), S([32, 1, 2, -1]), S([32, 256, 1, 1]), S([32, 256]), S([32, 2, 1, 128]), S([32, 256, 1, 1]))")
def oracle_forward(inputs):
    return oracle_resnest_split_attention_backward(*inputs)


def main():
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
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
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
