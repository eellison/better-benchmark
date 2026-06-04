"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle builds the routed BF16 MoE accumulation once with a Triton scatter-add and feeds both the column reduction and row-gradient epilogue from that routed buffer, whereas Inductor currently emits separate index_put/where kernels followed by independent reduction and pointwise kernels over the materialized scatter result; Inductor cannot do this today because scheduler/codegen cannot represent index_put accumulate as a reusable scatter-reduce producer for sibling row/column reductions and a transpose-producing epilogue; the fix is SCATTER_REDUCE: add a routed index_put-accumulate lowering that schedules the scatter result directly into the dependent reductions and final transposed store."""
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
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_sum_c39c7b0d801a"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_Qwen_Qwen3-30B-A3B_001_f65095c5"

SRC_ROWS = 16384
ROWS = 2048
COLS = 2048
INV_COLS = 1.0 / COLS

COPY_BLOCK = 1024
SCATTER_BLOCK_M = 4
SCATTER_BLOCK_N = 512
REDUCE_BLOCK_M = 32
REDUCE_BLOCK_N = 128
FINAL_BLOCK_N = 64
ROW_TILES = math.ceil(ROWS / REDUCE_BLOCK_M)
COL_TILES = math.ceil(COLS / REDUCE_BLOCK_N)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


if triton is not None:

    @triton.jit
    def _copy_full_kernel(
        full_ptr,
        routed_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        vals = tl.load(full_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(routed_ptr + offsets, vals, mask=mask)

    @triton.jit
    def _scatter_where_add_kernel(
        mask_ptr,
        scalar_ptr,
        grouped_ptr,
        index_ptr,
        routed_ptr,
        mask_stride_m: tl.constexpr,
        grouped_stride_m: tl.constexpr,
        grouped_stride_n: tl.constexpr,
        index_stride_m: tl.constexpr,
        routed_stride_m: tl.constexpr,
        routed_stride_n: tl.constexpr,
        SRC_ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        active_m = m_offsets < SRC_ROWS_
        active = active_m[:, None] & (n_offsets[None, :] < COLS_)

        route = tl.load(index_ptr + m_offsets * index_stride_m, mask=active_m, other=0)
        route_mask = tl.load(mask_ptr + m_offsets * mask_stride_m, mask=active_m, other=0)
        scalar = tl.load(scalar_ptr).to(tl.float32)
        grouped = tl.load(
            grouped_ptr
            + m_offsets[:, None] * grouped_stride_m
            + n_offsets[None, :] * grouped_stride_n,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        vals = tl.where(route_mask[:, None], scalar, grouped)
        routed_offsets = route[:, None] * routed_stride_m + n_offsets[None, :] * routed_stride_n
        tl.atomic_add(routed_ptr + routed_offsets, vals, sem="relaxed", mask=active)

    @triton.jit
    def _partial_reductions_kernel(
        routed_ptr,
        mm_ptr,
        weight_ptr,
        x_ptr,
        scale_ptr,
        partial_col_ptr,
        partial_row_ptr,
        routed_stride_m: tl.constexpr,
        routed_stride_n: tl.constexpr,
        mm_stride_m: tl.constexpr,
        mm_stride_n: tl.constexpr,
        weight_stride_n: tl.constexpr,
        x_stride_m: tl.constexpr,
        x_stride_n: tl.constexpr,
        scale_stride_m: tl.constexpr,
        partial_col_stride_m: tl.constexpr,
        partial_col_stride_n: tl.constexpr,
        partial_row_stride_m: tl.constexpr,
        partial_row_stride_n: tl.constexpr,
        ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        active = (m_offsets[:, None] < ROWS_) & (n_offsets[None, :] < COLS_)

        routed = tl.load(
            routed_ptr + m_offsets[:, None] * routed_stride_m + n_offsets[None, :] * routed_stride_n,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        mm = tl.load(
            mm_ptr + m_offsets[:, None] * mm_stride_m + n_offsets[None, :] * mm_stride_n,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        a = (routed + mm).to(tl.bfloat16)

        x = tl.load(
            x_ptr + m_offsets[:, None] * x_stride_m + n_offsets[None, :] * x_stride_n,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(scale_ptr + m_offsets * scale_stride_m, mask=m_offsets < ROWS_, other=0.0).to(tl.float32)
        normalized = (x * scale[:, None]).to(tl.bfloat16)

        product = (a * normalized).to(tl.bfloat16).to(tl.float32)
        col_sum = tl.sum(tl.where(active, product, 0.0), axis=0)
        tl.store(
            partial_col_ptr + pid_m * partial_col_stride_m + n_offsets * partial_col_stride_n,
            col_sum,
            mask=n_offsets < COLS_,
        )

        weight = tl.load(weight_ptr + n_offsets * weight_stride_n, mask=n_offsets < COLS_, other=0.0).to(tl.float32)
        weighted = (a * weight[None, :]).to(tl.bfloat16).to(tl.float32)
        row_sum = tl.sum(tl.where(active, weighted * x, 0.0), axis=1)
        tl.store(
            partial_row_ptr + m_offsets * partial_row_stride_m + pid_n * partial_row_stride_n,
            row_sum,
            mask=m_offsets < ROWS_,
        )

    @triton.jit
    def _finalize_col_sum_kernel(
        partial_col_ptr,
        out_ptr,
        partial_col_stride_m: tl.constexpr,
        partial_col_stride_n: tl.constexpr,
        ROW_TILES_: tl.constexpr,
        COLS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tile_offsets = tl.arange(0, BLOCK_TILES)
        active = (tile_offsets[:, None] < ROW_TILES_) & (n_offsets[None, :] < COLS_)
        vals = tl.load(
            partial_col_ptr
            + tile_offsets[:, None] * partial_col_stride_m
            + n_offsets[None, :] * partial_col_stride_n,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        total = tl.sum(vals, axis=0)
        tl.store(out_ptr + n_offsets, total.to(tl.bfloat16), mask=n_offsets < COLS_)

    @triton.jit
    def _final_transposed_epilogue_kernel(
        routed_ptr,
        mm_ptr,
        weight_ptr,
        x_ptr,
        scale_ptr,
        add_ptr,
        partial_row_ptr,
        out_base_ptr,
        routed_stride_m: tl.constexpr,
        routed_stride_n: tl.constexpr,
        mm_stride_m: tl.constexpr,
        mm_stride_n: tl.constexpr,
        weight_stride_n: tl.constexpr,
        x_stride_m: tl.constexpr,
        x_stride_n: tl.constexpr,
        scale_stride_m: tl.constexpr,
        add_stride_m: tl.constexpr,
        add_stride_n: tl.constexpr,
        partial_row_stride_m: tl.constexpr,
        partial_row_stride_n: tl.constexpr,
        out_stride_m: tl.constexpr,
        out_stride_n: tl.constexpr,
        ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        COL_TILES_: tl.constexpr,
        INV_COLS_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_COL_TILES: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        active = (m_offsets[:, None] < ROWS_) & (n_offsets[None, :] < COLS_)

        col_tile_offsets = tl.arange(0, BLOCK_COL_TILES)
        row_partials = tl.load(
            partial_row_ptr
            + m_offsets[:, None] * partial_row_stride_m
            + col_tile_offsets[None, :] * partial_row_stride_n,
            mask=(m_offsets[:, None] < ROWS_) & (col_tile_offsets[None, :] < COL_TILES_),
            other=0.0,
        ).to(tl.float32)
        row_dot = tl.sum(row_partials, axis=1)

        routed = tl.load(
            routed_ptr + m_offsets[:, None] * routed_stride_m + n_offsets[None, :] * routed_stride_n,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        mm = tl.load(
            mm_ptr + m_offsets[:, None] * mm_stride_m + n_offsets[None, :] * mm_stride_n,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        a = (routed + mm).to(tl.bfloat16)
        weight = tl.load(weight_ptr + n_offsets * weight_stride_n, mask=n_offsets < COLS_, other=0.0).to(tl.float32)
        weighted = (a * weight[None, :]).to(tl.bfloat16).to(tl.float32)

        x = tl.load(
            x_ptr + m_offsets[:, None] * x_stride_m + n_offsets[None, :] * x_stride_n,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(scale_ptr + m_offsets * scale_stride_m, mask=m_offsets < ROWS_, other=0.0).to(tl.float32)
        scale3 = scale * scale * scale
        grad = weighted * scale[:, None] - row_dot[:, None] * scale3[:, None] * x * INV_COLS_
        grad_bf16 = grad.to(tl.bfloat16)

        add_val = tl.load(
            add_ptr + m_offsets[:, None] * add_stride_m + n_offsets[None, :] * add_stride_n,
            mask=active,
            other=0.0,
        )
        out = (add_val + grad_bf16).to(tl.bfloat16)
        tl.store(
            out_base_ptr + m_offsets[:, None] * out_stride_m + n_offsets[None, :] * out_stride_n,
            out,
            mask=active,
        )


def oracle_moe_index_put_reduce(
    arg107_1: torch.Tensor,
    full_3: torch.Tensor,
    _grouped_mm_11: torch.Tensor,
    full_5: torch.Tensor,
    arg105_1: torch.Tensor,
    mm_23: torch.Tensor,
    arg20_1: torch.Tensor,
    arg95_1: torch.Tensor,
    arg96_1: torch.Tensor,
    add_34: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3
    if triton is None:
        raise RuntimeError("triton is not available")
    if full_5.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    device = full_5.device
    routed = torch.empty((ROWS, COLS), device=device, dtype=full_5.dtype)
    partial_col = torch.empty((ROW_TILES, COLS), device=device, dtype=torch.float32)
    partial_row = torch.empty((ROWS, COL_TILES), device=device, dtype=torch.float32)
    out0 = torch.empty((COLS,), device=device, dtype=full_5.dtype)
    out_base = routed

    _copy_full_kernel[(triton.cdiv(ROWS * COLS, COPY_BLOCK),)](
        full_5,
        routed,
        n_elements=ROWS * COLS,
        BLOCK=COPY_BLOCK,
        num_warps=4,
    )
    _scatter_where_add_kernel[(triton.cdiv(SRC_ROWS, SCATTER_BLOCK_M), triton.cdiv(COLS, SCATTER_BLOCK_N))](
        arg107_1,
        full_3,
        _grouped_mm_11,
        arg105_1,
        routed,
        mask_stride_m=arg107_1.stride(0),
        grouped_stride_m=_grouped_mm_11.stride(0),
        grouped_stride_n=_grouped_mm_11.stride(1),
        index_stride_m=arg105_1.stride(0),
        routed_stride_m=routed.stride(0),
        routed_stride_n=routed.stride(1),
        SRC_ROWS_=SRC_ROWS,
        COLS_=COLS,
        BLOCK_M=SCATTER_BLOCK_M,
        BLOCK_N=SCATTER_BLOCK_N,
        num_warps=4,
    )
    _partial_reductions_kernel[(ROW_TILES, COL_TILES)](
        routed,
        mm_23,
        arg20_1,
        arg95_1,
        arg96_1,
        partial_col,
        partial_row,
        routed_stride_m=routed.stride(0),
        routed_stride_n=routed.stride(1),
        mm_stride_m=mm_23.stride(0),
        mm_stride_n=mm_23.stride(1),
        weight_stride_n=arg20_1.stride(0),
        x_stride_m=arg95_1.stride(1),
        x_stride_n=arg95_1.stride(2),
        scale_stride_m=arg96_1.stride(1),
        partial_col_stride_m=partial_col.stride(0),
        partial_col_stride_n=partial_col.stride(1),
        partial_row_stride_m=partial_row.stride(0),
        partial_row_stride_n=partial_row.stride(1),
        ROWS_=ROWS,
        COLS_=COLS,
        BLOCK_M=REDUCE_BLOCK_M,
        BLOCK_N=REDUCE_BLOCK_N,
        num_warps=4,
    )
    _finalize_col_sum_kernel[(triton.cdiv(COLS, FINAL_BLOCK_N),)](
        partial_col,
        out0,
        partial_col_stride_m=partial_col.stride(0),
        partial_col_stride_n=partial_col.stride(1),
        ROW_TILES_=ROW_TILES,
        COLS_=COLS,
        BLOCK_N=FINAL_BLOCK_N,
        BLOCK_TILES=triton.next_power_of_2(ROW_TILES),
        num_warps=2,
    )
    _final_transposed_epilogue_kernel[(ROW_TILES, COL_TILES)](
        routed,
        mm_23,
        arg20_1,
        arg95_1,
        arg96_1,
        add_34,
        partial_row,
        out_base,
        routed_stride_m=routed.stride(0),
        routed_stride_n=routed.stride(1),
        mm_stride_m=mm_23.stride(0),
        mm_stride_n=mm_23.stride(1),
        weight_stride_n=arg20_1.stride(0),
        x_stride_m=arg95_1.stride(1),
        x_stride_n=arg95_1.stride(2),
        scale_stride_m=arg96_1.stride(1),
        add_stride_m=add_34.stride(1),
        add_stride_n=add_34.stride(2),
        partial_row_stride_m=partial_row.stride(0),
        partial_row_stride_n=partial_row.stride(1),
        out_stride_m=out_base.stride(0),
        out_stride_n=out_base.stride(1),
        ROWS_=ROWS,
        COLS_=COLS,
        COL_TILES_=COL_TILES,
        INV_COLS_=INV_COLS,
        BLOCK_M=REDUCE_BLOCK_M,
        BLOCK_N=REDUCE_BLOCK_N,
        BLOCK_COL_TILES=triton.next_power_of_2(COL_TILES),
        num_warps=4,
    )
    return out0, out_base.t()


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    return module.Repro().to(device)(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_moe_index_put_reduce(*inputs)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        ok = ok and shape_ok and dtype_ok and stride_ok and value_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


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


def run_bench(device: torch.device, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    logical_read_bytes = (
        SRC_ROWS * COLS * 2
        + ROWS * COLS * 2 * 4
        + ROWS * COLS * 4
        + SRC_ROWS * 8
        + SRC_ROWS
    )
    logical_write_bytes = ROWS * COLS * 2 + COLS * 2
    print(
        f"oracle shape: routed=bf16[{SRC_ROWS}, {COLS}]->[{ROWS}, {COLS}] "
        f"shape={SHAPE_LABEL} device={device}"
    )
    print(f"direct logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_moe_index_put_reduce(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_moe_index_put_reduce(*inputs),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_moe_index_put_reduce: {oracle_us:.3f} us "
        f"impl=triton shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs and strides to repro.py")
    parser.add_argument("--bench", action="store_true", help="time the Triton oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-1)
    parser.add_argument("--atol", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
