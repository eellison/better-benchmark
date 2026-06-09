"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete four-output MT5 attention-backward tuple by fusing each rowwise softmax-backward producer with the returned dense dscore side output and the duplicate-index relative-position bucket accumulation, whereas Inductor currently materializes the dense dscore branches, separately reduces residual-plus-dscore tensors over batch, permutes/clones the `[128, 128, 6]` bucket values, and lowers both `index_put(accumulate=True)` operations as generic high-contention scatter kernels; Inductor cannot do this today because scheduler/codegen does not model a rowwise softmax-backward producer feeding both a full side-output store and a structured duplicate-index scatter-reduce epilogue; the fix is SCATTER_REDUCE: add a relative-position scatter-reduce lowering that keeps the softmax-backward row tile in registers, emits the dense side output, and accumulates bucket gradients directly from the same producer."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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

REPRO_ID = "sum_sum_sum_b8d91626261d"
SHAPE_LABEL = "hf_mt5forconditionalgeneration_train_001_16a0a2a4"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
HEADS = 6
QUERY = 128
KEY = 128
BUCKETS = 32
ROWS = BATCH * HEADS * QUERY
SCALE = 1.1111111111111112
NEG_INF_F32 = -3.4028234663852886e38
BLOCK_B = 32
BLOCK_K = 128



if triton is not None:

    @triton.jit
    def _stored_dscore_kernel(
        bmm_ptr,
        keep_mask_ptr,
        softmax_ptr,
        dscore_out_ptr,
        BATCH_: tl.constexpr,
        HEADS_: tl.constexpr,
        QUERY_: tl.constexpr,
        KEY_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK_B_: tl.constexpr,
        BLOCK_K_: tl.constexpr,
    ):
        h = tl.program_id(0)
        q = tl.program_id(1)
        b_block = tl.program_id(2)
        b_offsets = b_block * BLOCK_B_ + tl.arange(0, BLOCK_B_)
        cols = tl.arange(0, BLOCK_K_)
        row_offsets = (b_offsets * HEADS_ + h) * QUERY_ + q
        offsets = row_offsets[:, None] * KEY_ + cols[None, :]
        active = (b_offsets[:, None] < BATCH_) & (cols[None, :] < KEY_)

        bmm = tl.load(bmm_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        keep = tl.load(keep_mask_ptr + offsets, mask=active, other=0).to(tl.float32)
        probs = tl.load(softmax_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        product = bmm * keep * SCALE_ * probs
        row_dot = tl.sum(product, axis=1)
        dscore = product - probs * row_dot[:, None]
        tl.store(dscore_out_ptr + offsets, dscore, mask=active)

    @triton.jit
    def _recomputed_dscore_kernel(
        bmm_ptr,
        keep_mask_ptr,
        causal_mask_ptr,
        logits_ptr,
        position_bias_ptr,
        row_max_ptr,
        row_sum_ptr,
        fill_ptr,
        dscore_out_ptr,
        BATCH_: tl.constexpr,
        HEADS_: tl.constexpr,
        QUERY_: tl.constexpr,
        KEY_: tl.constexpr,
        SCALE_: tl.constexpr,
        NEG_INF_: tl.constexpr,
        BLOCK_B_: tl.constexpr,
        BLOCK_K_: tl.constexpr,
    ):
        h = tl.program_id(0)
        q = tl.program_id(1)
        b_block = tl.program_id(2)
        b_offsets = b_block * BLOCK_B_ + tl.arange(0, BLOCK_B_)
        cols = tl.arange(0, BLOCK_K_)
        row_offsets = (b_offsets * HEADS_ + h) * QUERY_ + q
        offsets = row_offsets[:, None] * KEY_ + cols[None, :]
        active = (b_offsets[:, None] < BATCH_) & (cols[None, :] < KEY_)

        fill = tl.load(fill_ptr).to(tl.float32)
        causal_keep = tl.load(causal_mask_ptr + q).to(tl.int1)
        mask_bias = tl.where(causal_keep, fill, NEG_INF_)
        logits = tl.load(logits_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        bias = tl.load(
            position_bias_ptr + (q * KEY_ + cols) * HEADS_ + h,
            mask=cols < KEY_,
            other=0.0,
        ).to(tl.float32)
        row_max = tl.load(row_max_ptr + row_offsets, mask=b_offsets < BATCH_, other=0.0).to(tl.float32)
        denom = tl.load(row_sum_ptr + row_offsets, mask=b_offsets < BATCH_, other=1.0).to(tl.float32)
        probs = tl.exp(logits + bias[None, :] + mask_bias - row_max[:, None]) / denom[:, None]

        bmm = tl.load(bmm_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        keep = tl.load(keep_mask_ptr + offsets, mask=active, other=0).to(tl.float32)
        product = bmm * keep * SCALE_ * probs
        row_dot = tl.sum(product, axis=1)
        dscore = product - probs * row_dot[:, None]
        tl.store(dscore_out_ptr + offsets, dscore, mask=active)

    @triton.jit
    def _bucket_partials_from_dscore_kernel(
        residual0_ptr,
        residual1_ptr,
        residual2_ptr,
        residual3_ptr,
        residual4_ptr,
        residual5_ptr,
        residual6_ptr,
        dscore_ptr,
        bucket_ptr,
        fill_ptr,
        partial_bucket_ptr,
        BATCH_: tl.constexpr,
        HEADS_: tl.constexpr,
        QUERY_: tl.constexpr,
        KEY_: tl.constexpr,
        BUCKETS_: tl.constexpr,
        BLOCK_B_: tl.constexpr,
        BLOCK_K_: tl.constexpr,
    ):
        h = tl.program_id(0)
        q = tl.program_id(1)
        b_block = tl.program_id(2)
        b_offsets = b_block * BLOCK_B_ + tl.arange(0, BLOCK_B_)
        cols = tl.arange(0, BLOCK_K_)
        row_offsets = (b_offsets * HEADS_ + h) * QUERY_ + q
        offsets = row_offsets[:, None] * KEY_ + cols[None, :]
        active = (b_offsets[:, None] < BATCH_) & (cols[None, :] < KEY_)

        values = (
            tl.load(residual0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            + tl.load(residual1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            + tl.load(residual2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            + tl.load(residual3_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            + tl.load(residual4_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            + tl.load(residual5_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            + tl.load(residual6_ptr + offsets, mask=active, other=0.0).to(tl.float32)
            + tl.load(dscore_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        )
        bucket = tl.load(bucket_ptr + q * KEY_ + cols, mask=cols < KEY_, other=0).to(tl.int64)
        dest = tl.where(bucket < 0, bucket + BUCKETS_, bucket)
        fill = tl.load(fill_ptr).to(tl.float32)
        values = tl.where(
            bucket[None, :] == -1,
            tl.where(b_offsets[:, None] == 0, fill, 0.0),
            values,
        )
        partial = tl.sum(tl.where(active, values, 0.0), axis=0)
        partial_base = (h * QUERY_ + q) * BUCKETS_
        for bucket_id in tl.static_range(0, 32):
            bucket_sum = tl.sum(
                tl.where((cols < KEY_) & (dest == bucket_id), partial, 0.0),
                axis=0,
            )
            tl.store(partial_bucket_ptr + partial_base + bucket_id, bucket_sum)

    @triton.jit
    def _finalize_bucket_partials_kernel(
        partial_bucket_ptr,
        bucket_out_ptr,
        HEADS_: tl.constexpr,
        QUERY_: tl.constexpr,
        BUCKETS_: tl.constexpr,
        BLOCK_Q_: tl.constexpr,
    ):
        h = tl.program_id(0)
        bucket = tl.program_id(1)
        q_offsets = tl.arange(0, BLOCK_Q_)
        partial_offsets = (h * QUERY_ + q_offsets) * BUCKETS_ + bucket
        values = tl.load(
            partial_bucket_ptr + partial_offsets,
            mask=q_offsets < QUERY_,
            other=0.0,
        ).to(tl.float32)
        total = tl.sum(values, axis=0)
        tl.store(bucket_out_ptr + bucket * HEADS_ + h, total)

def _config_on_device(config: dict, device: torch.device) -> dict:
    return {
        "inputs": [
            {**spec, "device": str(device)}
            if isinstance(spec, dict) and spec.get("kind") == "tensor"
            else spec
            for spec in config["inputs"]
        ]
    }


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        inputs = make_inputs_from_config(
            _config_on_device(next(iter(configs.values())), device)
        )
    else:
        inputs = _load_repro_module().make_inputs()

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def _check_cuda_tensor(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tensor.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous for this captured-shape oracle")


def oracle_mt5_attention_scatter_reduce(
    view_44: torch.Tensor,
    view_95: torch.Tensor,
    view_146: torch.Tensor,
    view_197: torch.Tensor,
    view_248: torch.Tensor,
    view_299: torch.Tensor,
    view_350: torch.Tensor,
    bmm_61: torch.Tensor,
    arg324_1: torch.Tensor,
    arg323_1: torch.Tensor,
    arg322_1: torch.Tensor,
    full_1: torch.Tensor,
    view_432: torch.Tensor,
    view_462: torch.Tensor,
    view_492: torch.Tensor,
    view_522: torch.Tensor,
    view_552: torch.Tensor,
    view_582: torch.Tensor,
    view_612: torch.Tensor,
    bmm_93: torch.Tensor,
    arg199_1: torch.Tensor,
    arg190_1: torch.Tensor,
    arg194_1: torch.Tensor,
    arg196_1: torch.Tensor,
    arg197_1: torch.Tensor,
    arg198_1: torch.Tensor,
    arg195_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
    _shape_param_8,
    _shape_param_9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
    )
    if triton is None:
        raise RuntimeError("triton is not available")

    attention_shape = (BATCH, HEADS, QUERY, KEY)
    bmm_shape = (BATCH * HEADS, QUERY, KEY)
    bucket_shape = (QUERY, KEY)
    row_shape = (BATCH, HEADS, QUERY, 1)
    for name, tensor in (
        ("view_44", view_44),
        ("view_95", view_95),
        ("view_146", view_146),
        ("view_197", view_197),
        ("view_248", view_248),
        ("view_299", view_299),
        ("view_350", view_350),
        ("arg324_1", arg324_1),
        ("arg323_1", arg323_1),
        ("view_432", view_432),
        ("view_462", view_462),
        ("view_492", view_492),
        ("view_522", view_522),
        ("view_552", view_552),
        ("view_582", view_582),
        ("view_612", view_612),
        ("arg199_1", arg199_1),
    ):
        _check_cuda_tensor(name, tensor, attention_shape)
    for name, tensor in (("bmm_61", bmm_61), ("bmm_93", bmm_93), ("arg194_1", arg194_1)):
        _check_cuda_tensor(name, tensor, bmm_shape)
    for name, tensor in (("arg322_1", arg322_1), ("arg195_1", arg195_1)):
        _check_cuda_tensor(name, tensor, bucket_shape)
    _check_cuda_tensor("arg190_1", arg190_1, (1, 1, QUERY, 1))
    _check_cuda_tensor("arg196_1", arg196_1, (QUERY, KEY, HEADS))
    _check_cuda_tensor("arg197_1", arg197_1, row_shape)
    _check_cuda_tensor("arg198_1", arg198_1, row_shape)

    out0 = torch.empty(bmm_shape, device=bmm_61.device, dtype=bmm_61.dtype)
    partial0 = torch.empty(
        (HEADS, QUERY, BUCKETS),
        device=bmm_61.device,
        dtype=torch.float32,
    )
    bucket0 = torch.empty((BUCKETS, HEADS), device=bmm_61.device, dtype=bmm_61.dtype)

    grid = (HEADS, QUERY, triton.cdiv(BATCH, BLOCK_B))
    _stored_dscore_kernel[grid](
        bmm_61,
        arg324_1,
        arg323_1,
        out0,
        BATCH_=BATCH,
        HEADS_=HEADS,
        QUERY_=QUERY,
        KEY_=KEY,
        SCALE_=SCALE,
        BLOCK_B_=BLOCK_B,
        BLOCK_K_=BLOCK_K,
        num_warps=4,
    )
    _bucket_partials_from_dscore_kernel[grid](
        view_44,
        view_95,
        view_146,
        view_197,
        view_248,
        view_299,
        view_350,
        out0,
        arg322_1,
        full_1,
        partial0,
        BATCH_=BATCH,
        HEADS_=HEADS,
        QUERY_=QUERY,
        KEY_=KEY,
        BUCKETS_=BUCKETS,
        BLOCK_B_=BLOCK_B,
        BLOCK_K_=BLOCK_K,
        num_warps=1,
    )
    _finalize_bucket_partials_kernel[(HEADS, BUCKETS)](
        partial0,
        bucket0,
        HEADS_=HEADS,
        QUERY_=QUERY,
        BUCKETS_=BUCKETS,
        BLOCK_Q_=QUERY,
        num_warps=1,
    )

    out2 = torch.empty(bmm_shape, device=bmm_93.device, dtype=bmm_93.dtype)
    partial1 = torch.empty(
        (HEADS, QUERY, BUCKETS),
        device=bmm_93.device,
        dtype=torch.float32,
    )
    bucket1 = torch.empty((BUCKETS, HEADS), device=bmm_93.device, dtype=bmm_93.dtype)
    _recomputed_dscore_kernel[grid](
        bmm_93,
        arg199_1,
        arg190_1,
        arg194_1,
        arg196_1,
        arg197_1,
        arg198_1,
        full_1,
        out2,
        BATCH_=BATCH,
        HEADS_=HEADS,
        QUERY_=QUERY,
        KEY_=KEY,
        SCALE_=SCALE,
        NEG_INF_=NEG_INF_F32,
        BLOCK_B_=BLOCK_B,
        BLOCK_K_=BLOCK_K,
        num_warps=1,
    )
    _bucket_partials_from_dscore_kernel[grid](
        view_432,
        view_462,
        view_492,
        view_522,
        view_552,
        view_582,
        view_612,
        out2,
        arg195_1,
        full_1,
        partial1,
        BATCH_=BATCH,
        HEADS_=HEADS,
        QUERY_=QUERY,
        KEY_=KEY,
        BUCKETS_=BUCKETS,
        BLOCK_B_=BLOCK_B,
        BLOCK_K_=BLOCK_K,
        num_warps=4,
    )
    _finalize_bucket_partials_kernel[(HEADS, BUCKETS)](
        partial1,
        bucket1,
        HEADS_=HEADS,
        QUERY_=QUERY,
        BUCKETS_=BUCKETS,
        BLOCK_Q_=QUERY,
        num_warps=4,
    )
    return out0, bucket0, out2, bucket1


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    if device.type != "cuda":
        raise RuntimeError("reference repro uses captured CUDA device literals; run checks on CUDA")
    module = _load_repro_module()
    return module.Repro().to(device)(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(max(0, warmup)):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


@oracle_impl(hardware="H100", shapes="(T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([192, 128, 128], f32), T([32, 6, 128, 128], b8), T([32, 6, 128, 128], f32), T([128, 128], i64, gen=Index(32)), T([], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([32, 6, 128, 128], f32), T([192, 128, 128], f32), T([32, 6, 128, 128], b8), T([1, 1, 128, 1], b8), T([192, 128, 128], f32), T([128, 128, 6], f32), T([32, 6, 128, 1], f32), T([32, 6, 128, 1], f32), T([128, 128], i64, gen=Index(32)), S([32, 6, 128, 128]), S([192, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]), S([32, 6, 128, 128]), S([32, -1, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]))")
def oracle_forward(inputs):
    return oracle_mt5_attention_scatter_reduce(*inputs)


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
