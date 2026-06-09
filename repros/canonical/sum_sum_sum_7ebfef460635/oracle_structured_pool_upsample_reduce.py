"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full twelve-output BEiT relative-position scatter-reduce return tuple by reducing each `[128, 12, 197, 197]` source over batch and atomically accumulating the resulting `[197, 197, 12]` values into duplicate `[732, 12]` relative-position buckets, whereas Inductor currently lowers each `sum(dim=0) -> full/slice_scatter/constant_pad_nd -> squeeze/permute/reshape -> index_put(accumulate=True)` branch as separate generic reduction, layout materialization, and scatter kernels; Inductor cannot do this today because scheduler/codegen does not recognize the zero-fill slice_scatter plus negative-pad crop as a structured no-op feeding a duplicate-index scatter-reduce and cannot fuse the batch reduction with indexed accumulation across the repeated return branches; the fix is SCATTER_REDUCE: add a structured scatter-reduce lowering that folds the slice_scatter/pad view chain, reduces source tiles over batch, and accumulates duplicate relative-position indices directly into output buckets for every branch in the full return tuple."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

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

REPRO_ID = "sum_sum_sum_7ebfef460635"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_beit_base_patch16_224_train_001_485c0134"

N = 128
C = 12
H = 197
W = 197
SPATIAL = H * W
BUCKETS = 732
N_BRANCHES = 12
N_TENSOR_ARGS = N_BRANCHES * 2
N_REPRO_ARGS = N_TENSOR_ARGS + N_BRANCHES

BLOCK_N = 4
BLOCK_S = 256



if triton is not None:

    @triton.jit
    def _batch_sum_relative_position_scatter_one(
        x_ptr,
        index_ptr,
        out_ptr,
        channel,
        offsets_s,
        h,
        w,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        index_stride_h: tl.constexpr,
        index_stride_w: tl.constexpr,
        n_items: tl.constexpr,
        channels: tl.constexpr,
        spatial: tl.constexpr,
        buckets: tl.constexpr,
        block_n: tl.constexpr,
        block_s: tl.constexpr,
    ):
        partial = tl.zeros((block_s,), tl.float32)
        for n_base in tl.range(0, n_items, block_n):
            n_offsets = n_base + tl.arange(0, block_n)
            x_offsets = (
                n_offsets[:, None] * x_stride_n
                + channel * x_stride_c
                + h[None, :] * x_stride_h
                + w[None, :] * x_stride_w
            )
            load_mask = (n_offsets[:, None] < n_items) & (offsets_s[None, :] < spatial)
            values = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0).to(tl.float32)
            partial += tl.sum(values, axis=0)

        bucket = tl.load(
            index_ptr + h * index_stride_h + w * index_stride_w,
            mask=offsets_s < spatial,
            other=0,
        ).to(tl.int64)
        store_mask = (offsets_s < spatial) & (bucket >= 0) & (bucket < buckets)
        tl.atomic_add(
            out_ptr + bucket * channels + channel,
            partial,
            sem="relaxed",
            mask=store_mask,
        )

    @triton.jit
    def _batch_sum_relative_position_scatter_all12_kernel(
        x0,
        index0,
        x1,
        index1,
        x2,
        index2,
        x3,
        index3,
        x4,
        index4,
        x5,
        index5,
        x6,
        index6,
        x7,
        index7,
        x8,
        index8,
        x9,
        index9,
        x10,
        index10,
        x11,
        index11,
        out_ptr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        index_stride_h: tl.constexpr,
        index_stride_w: tl.constexpr,
        n_items: tl.constexpr,
        channels: tl.constexpr,
        width: tl.constexpr,
        spatial: tl.constexpr,
        buckets: tl.constexpr,
        block_n: tl.constexpr,
        block_s: tl.constexpr,
    ):
        channel = tl.program_id(0)
        spatial_block = tl.program_id(1)

        offsets_s = spatial_block * block_s + tl.arange(0, block_s)
        h = offsets_s // width
        w = offsets_s - h * width
        output_stride = buckets * channels

        _batch_sum_relative_position_scatter_one(
            x0,
            index0,
            out_ptr,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )
        _batch_sum_relative_position_scatter_one(
            x1,
            index1,
            out_ptr + output_stride,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )
        _batch_sum_relative_position_scatter_one(
            x2,
            index2,
            out_ptr + output_stride * 2,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )
        _batch_sum_relative_position_scatter_one(
            x3,
            index3,
            out_ptr + output_stride * 3,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )
        _batch_sum_relative_position_scatter_one(
            x4,
            index4,
            out_ptr + output_stride * 4,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )
        _batch_sum_relative_position_scatter_one(
            x5,
            index5,
            out_ptr + output_stride * 5,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )
        _batch_sum_relative_position_scatter_one(
            x6,
            index6,
            out_ptr + output_stride * 6,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )
        _batch_sum_relative_position_scatter_one(
            x7,
            index7,
            out_ptr + output_stride * 7,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )
        _batch_sum_relative_position_scatter_one(
            x8,
            index8,
            out_ptr + output_stride * 8,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )
        _batch_sum_relative_position_scatter_one(
            x9,
            index9,
            out_ptr + output_stride * 9,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )
        _batch_sum_relative_position_scatter_one(
            x10,
            index10,
            out_ptr + output_stride * 10,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )
        _batch_sum_relative_position_scatter_one(
            x11,
            index11,
            out_ptr + output_stride * 11,
            channel,
            offsets_s,
            h,
            w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            index_stride_h,
            index_stride_w,
            n_items,
            channels,
            spatial,
            buckets,
            block_n,
            block_s,
        )

    @triton.jit
    def _batch_sum_relative_position_scatter_kernel(
        x_ptr,
        index_ptr,
        out_ptr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        index_stride_h: tl.constexpr,
        index_stride_w: tl.constexpr,
        n_items: tl.constexpr,
        channels: tl.constexpr,
        width: tl.constexpr,
        spatial: tl.constexpr,
        buckets: tl.constexpr,
        block_n: tl.constexpr,
        block_s: tl.constexpr,
    ):
        channel = tl.program_id(0)
        spatial_block = tl.program_id(1)

        s_offsets = spatial_block * block_s + tl.arange(0, block_s)
        h = s_offsets // width
        w = s_offsets - h * width

        partial = tl.zeros((block_s,), tl.float32)
        for n_base in tl.range(0, n_items, block_n):
            n_offsets = n_base + tl.arange(0, block_n)
            x_offsets = (
                n_offsets[:, None] * x_stride_n
                + channel * x_stride_c
                + h[None, :] * x_stride_h
                + w[None, :] * x_stride_w
            )
            load_mask = (n_offsets[:, None] < n_items) & (s_offsets[None, :] < spatial)
            values = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0).to(tl.float32)
            partial += tl.sum(values, axis=0)

        bucket = tl.load(
            index_ptr + h * index_stride_h + w * index_stride_w,
            mask=s_offsets < spatial,
            other=0,
        ).to(tl.int64)
        store_mask = (s_offsets < spatial) & (bucket >= 0) & (bucket < buckets)
        tl.atomic_add(
            out_ptr + bucket * channels + channel,
            partial,
            sem="relaxed",
            mask=store_mask,
        )


def make_inputs(device: torch.device) -> tuple[object, ...]:
    module = _load_repro_module()
    inputs = module.make_inputs()
    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def _check_branch_inputs(x: torch.Tensor, index: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    if x.device.type != "cuda" or index.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if tuple(x.shape) != (N, C, H, W):
        raise ValueError(f"unexpected gradient shape: {tuple(x.shape)}")
    if tuple(index.shape) != (H, W):
        raise ValueError(f"unexpected index shape: {tuple(index.shape)}")
    if x.dtype != torch.float32:
        raise ValueError(f"unexpected gradient dtype: {x.dtype}")
    if index.dtype != torch.int64:
        raise ValueError(f"unexpected index dtype: {index.dtype}")


def _scatter_reduce_one(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    _check_branch_inputs(x, index)
    out = torch.zeros((BUCKETS, C), device=x.device, dtype=torch.float32)
    grid = (C, triton.cdiv(SPATIAL, BLOCK_S))
    _batch_sum_relative_position_scatter_kernel[grid](
        x,
        index,
        out,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        index_stride_h=index.stride(0),
        index_stride_w=index.stride(1),
        n_items=N,
        channels=C,
        width=W,
        spatial=SPATIAL,
        buckets=BUCKETS,
        block_n=BLOCK_N,
        block_s=BLOCK_S,
        num_warps=4,
    )
    return out


def _scatter_reduce_all(grads: list[torch.Tensor], indices: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    first_grad = grads[0]
    first_index = indices[0]
    for grad in grads[1:]:
        if grad.stride() != first_grad.stride():
            raise ValueError("expected all gradient tensors to have the same stride")
    for index in indices[1:]:
        if index.stride() != first_index.stride():
            raise ValueError("expected all index tensors to have the same stride")

    out = torch.zeros((N_BRANCHES, BUCKETS, C), device=first_grad.device, dtype=torch.float32)
    grid = (C, triton.cdiv(SPATIAL, BLOCK_S))
    _batch_sum_relative_position_scatter_all12_kernel[grid](
        grads[0],
        indices[0],
        grads[1],
        indices[1],
        grads[2],
        indices[2],
        grads[3],
        indices[3],
        grads[4],
        indices[4],
        grads[5],
        indices[5],
        grads[6],
        indices[6],
        grads[7],
        indices[7],
        grads[8],
        indices[8],
        grads[9],
        indices[9],
        grads[10],
        indices[10],
        grads[11],
        indices[11],
        out,
        x_stride_n=first_grad.stride(0),
        x_stride_c=first_grad.stride(1),
        x_stride_h=first_grad.stride(2),
        x_stride_w=first_grad.stride(3),
        index_stride_h=first_index.stride(0),
        index_stride_w=first_index.stride(1),
        n_items=N,
        channels=C,
        width=W,
        spatial=SPATIAL,
        buckets=BUCKETS,
        block_n=BLOCK_N,
        block_s=BLOCK_S,
        num_warps=4,
    )
    return tuple(out[i] for i in range(N_BRANCHES))


def oracle_structured_pool_upsample_reduce(*inputs: object) -> tuple[torch.Tensor, ...]:
    """Compute the complete Repro.forward return tuple with source-space scatter reductions."""
    if len(inputs) != N_REPRO_ARGS:
        raise ValueError(f"expected {N_REPRO_ARGS} repro inputs, got {len(inputs)}")

    tensor_args = inputs[:N_TENSOR_ARGS]
    shape_args = inputs[N_TENSOR_ARGS:]
    for shape in shape_args:
        if list(shape) != [SPATIAL, C]:
            raise ValueError(f"unexpected view shape parameter: {shape}")

    grads: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for grad, index in zip(tensor_args[0::2], tensor_args[1::2]):
        if not isinstance(grad, torch.Tensor) or not isinstance(index, torch.Tensor):
            raise TypeError("expected alternating gradient/index tensor repro arguments")
        _check_branch_inputs(grad, index)
        grads.append(grad)
        indices.append(index)
    return _scatter_reduce_all(grads, indices)


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().to(device)
    out = model(*inputs)
    if not isinstance(out, tuple):
        raise TypeError(f"expected tuple output from repro, got {type(out)!r}")
    return out


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    if triton is not None and device.type == "cuda":
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

    for _ in range(warmup):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


@oracle_impl(hardware="H100", shapes="(T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), T([128, 12, 197, 197], f32, stride=(491712, 40976, 208, 1)), T([197, 197], i64, gen=Index(732)), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]), S([38809, 12]))")
def oracle_forward(inputs):
    return oracle_structured_pool_upsample_reduce(*inputs)


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
