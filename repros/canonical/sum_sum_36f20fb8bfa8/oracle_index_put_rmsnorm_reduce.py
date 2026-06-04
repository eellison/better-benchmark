"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle performs the masked MoE index_put(accumulate=True) scatter-reduce into the 2048x2048 matrix and feeds that accumulated matrix through both the column reduction and RMSNorm-backward transpose outputs, whereas Inductor currently materializes the generic scatter result and schedules the sibling reductions and transpose epilogue as separate kernels around it; Inductor cannot do this today because scheduler/codegen cannot represent an index_put scatter-reduce producer that is reused by dependent row reductions, column reductions, and a strided transpose store in one planned lowering; the fix is SCATTER_REDUCE: add a scatter-reduce lowering that keeps the accumulated rows available to the downstream RMSNorm reductions and transpose epilogue during scheduling."""
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


REPRO_ID = "sum_sum_36f20fb8bfa8"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "vllm_Qwen_Qwen3-30B-A3B_001_3b4ca287"

N_ROWS = 2048
HIDDEN = 2048
N_SCATTER = 16384
BLOCK_H = 128
BLOCK_M = 128
NUM_H_BLOCKS = HIDDEN // BLOCK_H
NUM_M_BLOCKS = N_ROWS // BLOCK_M


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
    def _partial_reduce_kernel(
        scatter_ptr,
        mm_ptr,
        activation_ptr,
        inv_ptr,
        partial_col_ptr,
        BLOCK_M_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
        HIDDEN_: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        h_offsets = pid_h * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        offsets = m_offsets[:, None] * HIDDEN_ + h_offsets[None, :]

        scatter = tl.load(scatter_ptr + offsets).to(tl.float32)
        mm = tl.load(mm_ptr + offsets).to(tl.float32)
        x = (scatter + mm).to(tl.bfloat16).to(tl.float32)

        activation = tl.load(activation_ptr + offsets).to(tl.float32)
        inv = tl.load(inv_ptr + m_offsets).to(tl.float32)

        normalized = (activation * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        col_terms = (x * normalized).to(tl.bfloat16).to(tl.float32)
        col_sums = tl.sum(col_terms, axis=0)
        tl.store(partial_col_ptr + pid_m * HIDDEN_ + h_offsets, col_sums)

    @triton.jit
    def _row_sum_kernel(
        scatter_ptr,
        mm_ptr,
        weight_ptr,
        activation_ptr,
        row_sum_ptr,
        BLOCK_H_FULL_: tl.constexpr,
        HIDDEN_: tl.constexpr,
    ):
        m = tl.program_id(0)
        h_offsets = tl.arange(0, BLOCK_H_FULL_)
        offsets = m * HIDDEN_ + h_offsets

        scatter = tl.load(scatter_ptr + offsets).to(tl.float32)
        mm = tl.load(mm_ptr + offsets).to(tl.float32)
        x = (scatter + mm).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(weight_ptr + h_offsets).to(tl.float32)
        activation = tl.load(activation_ptr + offsets).to(tl.float32)

        weighted = (x * weight).to(tl.bfloat16).to(tl.float32)
        row_sum = tl.sum(weighted * activation, axis=0)
        tl.store(row_sum_ptr + m, row_sum)

    @triton.jit
    def _finalize_col_kernel(
        partial_col_ptr,
        out0_ptr,
        BLOCK_H_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        NUM_M_BLOCKS_: tl.constexpr,
    ):
        h_offsets = tl.program_id(0) * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        m_blocks = tl.arange(0, NUM_M_BLOCKS_)
        values = tl.load(partial_col_ptr + m_blocks[:, None] * HIDDEN_ + h_offsets[None, :]).to(tl.float32)
        sums = tl.sum(values, axis=0)
        tl.store(out0_ptr + h_offsets, sums)

    @triton.jit
    def _rmsnorm_transpose_kernel(
        scatter_ptr,
        mm_ptr,
        weight_ptr,
        activation_ptr,
        inv_ptr,
        residual_ptr,
        row_sum_ptr,
        out_base_ptr,
        BLOCK_M_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
        HIDDEN_: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        h_offsets = pid_h * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        offsets = m_offsets[:, None] * HIDDEN_ + h_offsets[None, :]

        scatter = tl.load(scatter_ptr + offsets).to(tl.float32)
        mm = tl.load(mm_ptr + offsets).to(tl.float32)
        x = (scatter + mm).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(weight_ptr + h_offsets).to(tl.float32)
        activation = tl.load(activation_ptr + offsets).to(tl.float32)
        inv = tl.load(inv_ptr + m_offsets).to(tl.float32)
        row_sum = tl.load(row_sum_ptr + m_offsets).to(tl.float32)
        residual = tl.load(residual_ptr + offsets).to(tl.float32)

        weighted = (x * weight[None, :]).to(tl.bfloat16).to(tl.float32)
        mul_tensor_4 = weighted * inv[:, None]
        pow_tensor_scalar = inv * inv * inv
        mul_scalar = row_sum * -0.5
        mul_tensor_5 = mul_scalar * pow_tensor_scalar
        div_scalar = mul_tensor_5 / HIDDEN_
        mul_scalar_1 = activation * 2.0
        mul_tensor_6 = div_scalar[:, None] * mul_scalar_1
        update = mul_tensor_4 + mul_tensor_6
        update_bf16 = update.to(tl.bfloat16).to(tl.float32)
        out = (residual + update_bf16).to(tl.bfloat16)
        tl.store(out_base_ptr + offsets, out)


def oracle_index_put_rmsnorm_reduce(
    arg171_1: torch.Tensor,
    full_3: torch.Tensor,
    _grouped_mm_3: torch.Tensor,
    arg169_1: torch.Tensor,
    mm_3: torch.Tensor,
    arg42_1: torch.Tensor,
    arg159_1: torch.Tensor,
    arg160_1: torch.Tensor,
    convert_element_type_5: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm_3.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    where_self = torch.ops.aten.where.self(arg171_1, full_3, _grouped_mm_3)
    scatter = torch.ops.aten.index_put.default(
        torch.zeros((N_ROWS, HIDDEN), device=mm_3.device, dtype=mm_3.dtype),
        [arg169_1],
        where_self,
        True,
    )
    out0 = torch.empty((HIDDEN,), device=mm_3.device, dtype=mm_3.dtype)
    out_base = torch.empty((N_ROWS, HIDDEN), device=mm_3.device, dtype=mm_3.dtype)
    partial_col = torch.empty((NUM_M_BLOCKS, HIDDEN), device=mm_3.device, dtype=torch.float32)
    row_sum = torch.empty((N_ROWS,), device=mm_3.device, dtype=torch.float32)

    _partial_reduce_kernel[(NUM_H_BLOCKS, NUM_M_BLOCKS)](
        scatter,
        mm_3,
        arg159_1,
        arg160_1,
        partial_col,
        BLOCK_M_=BLOCK_M,
        BLOCK_H_=BLOCK_H,
        HIDDEN_=HIDDEN,
        num_warps=4,
    )
    _finalize_col_kernel[(NUM_H_BLOCKS,)](
        partial_col,
        out0,
        BLOCK_H_=BLOCK_H,
        HIDDEN_=HIDDEN,
        NUM_M_BLOCKS_=NUM_M_BLOCKS,
        num_warps=1,
    )
    _row_sum_kernel[(N_ROWS,)](
        scatter,
        mm_3,
        arg42_1,
        arg159_1,
        row_sum,
        BLOCK_H_FULL_=HIDDEN,
        HIDDEN_=HIDDEN,
        num_warps=8,
    )
    _rmsnorm_transpose_kernel[(NUM_H_BLOCKS, NUM_M_BLOCKS)](
        scatter,
        mm_3,
        arg42_1,
        arg159_1,
        arg160_1,
        convert_element_type_5,
        row_sum,
        out_base,
        BLOCK_M_=BLOCK_M,
        BLOCK_H_=BLOCK_H,
        HIDDEN_=HIDDEN,
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
        actual = oracle_index_put_rmsnorm_reduce(*inputs)
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
        N_SCATTER * HIDDEN * 2
        + N_SCATTER * 8
        + N_SCATTER
        + 3 * N_ROWS * HIDDEN * 2
        + N_ROWS * HIDDEN * 4
        + N_ROWS * 4
        + HIDDEN * 2
    )
    logical_write_bytes = 2 * N_ROWS * HIDDEN * 2 + HIDDEN * 2
    print(
        f"oracle shape: scatter=bf16[{N_SCATTER}, {HIDDEN}] "
        f"dense=bf16[{N_ROWS}, {HIDDEN}] shape={SHAPE_LABEL} device={device}"
    )
    print(f"direct logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_index_put_rmsnorm_reduce(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_index_put_rmsnorm_reduce(*inputs),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_index_put_rmsnorm_reduce: {oracle_us:.3f} us "
        f"impl=aten_scatter+triton shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs and strides to repro.py")
    parser.add_argument("--bench", action="store_true", help="time the Triton oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--atol", type=float, default=7e-2)
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
