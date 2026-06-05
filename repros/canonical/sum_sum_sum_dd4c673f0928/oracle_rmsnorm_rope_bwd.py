"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Qwen grouped-RoPE plus two-branch RMSNorm-backward return tuple with fused CUDA kernels for the grouped-head rotary producer, row RMSNorm dot products, transposed input-gradient side outputs, and sibling `[128]` weight-gradient reductions, whereas Inductor currently materializes the grouped-head sum, rotary `slice_scatter` reconstructions, permutes, RMSNorm row gradients, clones/views, and weight-gradient sums as separate generic pointwise, reduction, and layout kernels; Inductor cannot do this today because its scheduler/codegen cannot represent a row-persistent fusion whose producer includes grouped-head reduction and half-rotation scatter algebra while also emitting layout-changing side outputs and cross-row reductions; the fix is SCHEDULER_FUSION: add a fused grouped-RoPE/RMSNorm-backward template that keeps each row dot product in registers, writes the contiguous bases for transposed outputs directly, and coordinates the two weight-gradient reductions."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful on CPU-only hosts.
    triton = None
    tl = None



from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_sum_sum_dd4c673f0928"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_Qwen_Qwen3-0.6B_001_6a9f0787"
CLAIMED_COMPILE_US = 65.4

BATCH = 4
SEQ = 512
TOKENS = BATCH * SEQ
QUERY_HEADS = 16
KEY_VALUE_HEADS = 8
HEADS_PER_GROUP = QUERY_HEADS // KEY_VALUE_HEADS
HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
QUERY_HIDDEN = QUERY_HEADS * HEAD_DIM
KEY_VALUE_HIDDEN = KEY_VALUE_HEADS * HEAD_DIM
TABLE_SIZE = SEQ * HEAD_DIM
ROWS_PER_PROGRAM = 8


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        config = next(iter(configs.values()))
        config = {
            "inputs": [
                {**spec, "device": str(device)}
                if isinstance(spec, dict) and spec.get("kind") == "tensor"
                else spec
                for spec in config["inputs"]
            ]
        }
        inputs = make_inputs_from_config(config)
    else:
        inputs = _load_repro_module().make_inputs()

    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    repro = module.Repro().eval()
    with torch.no_grad():
        return tuple(repro(*inputs))


@triton.jit
def _rotary_table_kernel(
    freq_ptr,
    sin_ptr,
    cos_ptr,
    kv_wgrad_accum_ptr,
    query_wgrad_accum_ptr,
    n_elements: tl.constexpr,
    HEAD_DIM_C: tl.constexpr,
    HALF_DIM_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    seq = offsets // HEAD_DIM_C
    dim = offsets - seq * HEAD_DIM_C
    freq_dim = dim % HALF_DIM_C
    angles = seq.to(tl.float32) * tl.load(freq_ptr + freq_dim, mask=mask, other=0.0)
    tl.store(sin_ptr + offsets, tl.sin(angles).to(tl.bfloat16), mask=mask)
    tl.store(cos_ptr + offsets, tl.cos(angles).to(tl.bfloat16), mask=mask)
    tl.store(kv_wgrad_accum_ptr + offsets, 0.0, mask=offsets < HEAD_DIM_C)
    tl.store(query_wgrad_accum_ptr + offsets, 0.0, mask=offsets < HEAD_DIM_C)


@triton.jit
def _kv_branch_kernel(
    rope_grad_ptr,
    weight_ptr,
    saved_ptr,
    inv_rstd_ptr,
    sin_ptr,
    cos_ptr,
    wgrad_accum_ptr,
    side_base_ptr,
    SEQ_C: tl.constexpr,
    QUERY_HEADS_C: tl.constexpr,
    KEY_VALUE_HEADS_C: tl.constexpr,
    HEADS_PER_GROUP_C: tl.constexpr,
    HEAD_DIM_C: tl.constexpr,
    HALF_DIM_C: tl.constexpr,
    KEY_VALUE_HIDDEN_C: tl.constexpr,
    TOTAL_ROWS_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < TOTAL_ROWS_C
    head = rows % KEY_VALUE_HEADS_C
    token = rows // KEY_VALUE_HEADS_C
    batch = token // SEQ_C
    seq = token - batch * SEQ_C
    dims = tl.arange(0, BLOCK_D)
    src_dims = tl.where(dims < HALF_DIM_C, dims + HALF_DIM_C, dims - HALF_DIM_C)
    signs = tl.where(dims < HALF_DIM_C, 1.0, -1.0)

    grouped_head = head * HEADS_PER_GROUP_C
    sum_dim = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    sum_src = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for group_offset in tl.static_range(0, HEADS_PER_GROUP_C):
        q_head = grouped_head + group_offset
        base = (
            (batch[:, None] * QUERY_HEADS_C + q_head[:, None]) * SEQ_C + seq[:, None]
        ) * HEAD_DIM_C
        sum_dim += tl.load(
            rope_grad_ptr + base + dims[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        sum_src += tl.load(
            rope_grad_ptr + base + src_dims[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)

    sum_dim = sum_dim.to(tl.bfloat16).to(tl.float32)
    sum_src = sum_src.to(tl.bfloat16).to(tl.float32)

    table_base = seq[:, None] * HEAD_DIM_C
    sin_src = tl.load(
        sin_ptr + table_base + src_dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    cos_dim = tl.load(
        cos_ptr + table_base + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    rotated = (sum_src * sin_src).to(tl.bfloat16).to(tl.float32) * signs[None, :]
    scaled = (sum_dim * cos_dim).to(tl.bfloat16).to(tl.float32)
    row_grad = (rotated + scaled).to(tl.bfloat16).to(tl.float32)

    hidden_offsets = head[:, None] * HEAD_DIM_C + dims[None, :]
    saved = tl.load(
        saved_ptr + token[:, None] * KEY_VALUE_HIDDEN_C + hidden_offsets,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    inv = tl.load(
        inv_rstd_ptr + token * KEY_VALUE_HEADS_C + head,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    xhat = (saved * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    wgrad_part = (row_grad * xhat).to(tl.bfloat16).to(tl.float32)
    wgrad_part = tl.where(row_mask[:, None], wgrad_part, 0.0)
    tl.atomic_add(wgrad_accum_ptr + dims, tl.sum(wgrad_part, axis=0), sem="relaxed")

    weight = tl.load(weight_ptr + dims).to(tl.float32)
    dy_weighted = (row_grad * weight[None, :]).to(tl.bfloat16).to(tl.float32)
    row_dot = tl.sum(dy_weighted * saved, axis=1)
    correction_scale = ((-0.5 * row_dot) * inv * inv * inv) / 128.0
    side = dy_weighted * inv[:, None] + correction_scale[:, None] * (saved * 2.0)
    tl.store(
        side_base_ptr + token[:, None] * KEY_VALUE_HIDDEN_C + hidden_offsets,
        side.to(tl.bfloat16),
        mask=row_mask[:, None],
    )


@triton.jit
def _query_branch_kernel(
    rope_grad_ptr,
    weight_ptr,
    saved_ptr,
    inv_rstd_ptr,
    sin_ptr,
    cos_ptr,
    wgrad_accum_ptr,
    side_base_ptr,
    SEQ_C: tl.constexpr,
    QUERY_HEADS_C: tl.constexpr,
    HEAD_DIM_C: tl.constexpr,
    HALF_DIM_C: tl.constexpr,
    QUERY_HIDDEN_C: tl.constexpr,
    TOTAL_ROWS_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < TOTAL_ROWS_C
    head = rows % QUERY_HEADS_C
    token = rows // QUERY_HEADS_C
    batch = token // SEQ_C
    seq = token - batch * SEQ_C
    dims = tl.arange(0, BLOCK_D)
    src_dims = tl.where(dims < HALF_DIM_C, dims + HALF_DIM_C, dims - HALF_DIM_C)
    signs = tl.where(dims < HALF_DIM_C, 1.0, -1.0)

    base = ((batch[:, None] * QUERY_HEADS_C + head[:, None]) * SEQ_C + seq[:, None]) * HEAD_DIM_C
    x_dim = tl.load(
        rope_grad_ptr + base + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    x_src = tl.load(
        rope_grad_ptr + base + src_dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    table_base = seq[:, None] * HEAD_DIM_C
    sin_src = tl.load(
        sin_ptr + table_base + src_dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    cos_dim = tl.load(
        cos_ptr + table_base + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    rotated = (x_src * sin_src).to(tl.bfloat16).to(tl.float32) * signs[None, :]
    scaled = (x_dim * cos_dim).to(tl.bfloat16).to(tl.float32)
    row_grad = (rotated + scaled).to(tl.bfloat16).to(tl.float32)

    hidden_offsets = head[:, None] * HEAD_DIM_C + dims[None, :]
    saved = tl.load(
        saved_ptr + token[:, None] * QUERY_HIDDEN_C + hidden_offsets,
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    inv = tl.load(
        inv_rstd_ptr + token * QUERY_HEADS_C + head,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    xhat = (saved * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    wgrad_part = (row_grad * xhat).to(tl.bfloat16).to(tl.float32)
    wgrad_part = tl.where(row_mask[:, None], wgrad_part, 0.0)
    tl.atomic_add(wgrad_accum_ptr + dims, tl.sum(wgrad_part, axis=0), sem="relaxed")

    weight = tl.load(weight_ptr + dims).to(tl.float32)
    dy_weighted = (row_grad * weight[None, :]).to(tl.bfloat16).to(tl.float32)
    row_dot = tl.sum(dy_weighted * saved, axis=1)
    correction_scale = ((-0.5 * row_dot) * inv * inv * inv) / 128.0
    side = dy_weighted * inv[:, None] + correction_scale[:, None] * (saved * 2.0)
    tl.store(
        side_base_ptr + token[:, None] * QUERY_HIDDEN_C + hidden_offsets,
        side.to(tl.bfloat16),
        mask=row_mask[:, None],
    )


@triton.jit
def _finish_wgrads_kernel(
    kv_accum_ptr,
    kv_out_ptr,
    query_accum_ptr,
    query_out_ptr,
    BLOCK: tl.constexpr,
):
    dims = tl.arange(0, BLOCK)
    kv_values = tl.load(kv_accum_ptr + dims)
    query_values = tl.load(query_accum_ptr + dims)
    tl.store(kv_out_ptr + dims, kv_values.to(tl.bfloat16))
    tl.store(query_out_ptr + dims, query_values.to(tl.bfloat16))


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for this oracle")


def oracle(
    getitem_1: torch.Tensor,
    arg2_1: torch.Tensor,
    getitem: torch.Tensor,
    arg304_1: torch.Tensor,
    arg858_1: torch.Tensor,
    arg859_1: torch.Tensor,
    arg302_1: torch.Tensor,
    arg856_1: torch.Tensor,
    arg857_1: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton()
    if not getitem.is_cuda:
        raise RuntimeError("this oracle implements the CUDA benchmark path only")

    sin_table = torch.empty((SEQ, HEAD_DIM), device=getitem.device, dtype=torch.bfloat16)
    cos_table = torch.empty_like(sin_table)
    kv_wgrad_accum = torch.empty((HEAD_DIM,), device=getitem.device, dtype=torch.float32)
    query_wgrad_accum = torch.empty_like(kv_wgrad_accum)
    table_grid = (triton.cdiv(TABLE_SIZE, 256),)
    _rotary_table_kernel[table_grid](
        arg2_1,
        sin_table,
        cos_table,
        kv_wgrad_accum,
        query_wgrad_accum,
        TABLE_SIZE,
        HEAD_DIM_C=HEAD_DIM,
        HALF_DIM_C=HALF_DIM,
        BLOCK=256,
    )

    kv_wgrad = torch.empty((HEAD_DIM,), device=getitem.device, dtype=torch.bfloat16)
    query_wgrad = torch.empty_like(kv_wgrad)
    kv_side_base = torch.empty(
        (TOKENS, KEY_VALUE_HIDDEN), device=getitem.device, dtype=torch.bfloat16
    )
    query_side_base = torch.empty(
        (TOKENS, QUERY_HIDDEN), device=getitem.device, dtype=torch.bfloat16
    )

    _kv_branch_kernel[(triton.cdiv(TOKENS * KEY_VALUE_HEADS, ROWS_PER_PROGRAM),)](
        getitem_1,
        arg304_1,
        arg858_1,
        arg859_1,
        sin_table,
        cos_table,
        kv_wgrad_accum,
        kv_side_base,
        SEQ_C=SEQ,
        QUERY_HEADS_C=QUERY_HEADS,
        KEY_VALUE_HEADS_C=KEY_VALUE_HEADS,
        HEADS_PER_GROUP_C=HEADS_PER_GROUP,
        HEAD_DIM_C=HEAD_DIM,
        HALF_DIM_C=HALF_DIM,
        KEY_VALUE_HIDDEN_C=KEY_VALUE_HIDDEN,
        TOTAL_ROWS_C=TOKENS * KEY_VALUE_HEADS,
        BLOCK_M=ROWS_PER_PROGRAM,
        BLOCK_D=HEAD_DIM,
        num_warps=4,
    )
    _query_branch_kernel[(triton.cdiv(TOKENS * QUERY_HEADS, ROWS_PER_PROGRAM),)](
        getitem,
        arg302_1,
        arg856_1,
        arg857_1,
        sin_table,
        cos_table,
        query_wgrad_accum,
        query_side_base,
        SEQ_C=SEQ,
        QUERY_HEADS_C=QUERY_HEADS,
        HEAD_DIM_C=HEAD_DIM,
        HALF_DIM_C=HALF_DIM,
        QUERY_HIDDEN_C=QUERY_HIDDEN,
        TOTAL_ROWS_C=TOKENS * QUERY_HEADS,
        BLOCK_M=ROWS_PER_PROGRAM,
        BLOCK_D=HEAD_DIM,
        num_warps=4,
    )
    _finish_wgrads_kernel[(1,)](
        kv_wgrad_accum,
        kv_wgrad,
        query_wgrad_accum,
        query_wgrad,
        BLOCK=HEAD_DIM,
    )

    return kv_wgrad, kv_side_base.t(), query_wgrad, query_side_base.t()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def _bench_callable(fn: Callable[[], object], warmup: int, rep: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1_000_000.0 / rep


def oracle_forward(inputs):
    return oracle_call(*inputs)


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
