"""
Full-scope Triton oracle for sum_sum_07989f9f05c5 (MT5 layer-norm backward tail).

Gap diagnosis (classification: SCHEDULER_FUSION): The oracle consumes the same
six tensor inputs and six shape parameters as repro.py, and returns the same
`[512]` column reduction plus `[512, 4096]` transposed epilogue with stride
`(1, 512)`. It differs from Inductor by fusing the per-row channel reduction
directly into the dropout-masked epilogue while separately computing the
orthogonal per-channel reduction with coalesced row tiles. Inductor cannot make
this a single ideal template today because the two sibling reductions reduce
different axes and one reduction feeds a full pointwise/layout epilogue, so the
scheduler keeps the reductions and transpose-shaped epilogue as separate
ordinary nodes instead of one cross-axis multi-output reduction plan. The fix
class is SCHEDULER_FUSION.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_07989f9f05c5"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 4096
C = 512
N = 32
T = 128
NUMEL = ROWS * C
DROPOUT_SCALE = 1.1111111111111112
INV_C = 1.0 / 512.0

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]



@triton.jit
def _row_epilogue_kernel(
    mm_279_ptr,
    mm_281_ptr,
    arg7_ptr,
    arg202_ptr,
    arg203_ptr,
    add_213_ptr,
    arg201_ptr,
    out_ptr,
    C_: tl.constexpr,
    INV_C_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    offsets = row * C_ + cols

    add_value = (
        tl.load(mm_279_ptr + offsets).to(tl.float32)
        + tl.load(mm_281_ptr + offsets).to(tl.float32)
    )
    arg7 = tl.load(arg7_ptr + cols).to(tl.float32)
    arg202 = tl.load(arg202_ptr + offsets).to(tl.float32)
    weighted = add_value * arg7
    row_sum = tl.sum(weighted * arg202, axis=0)

    arg203 = tl.load(arg203_ptr + row).to(tl.float32)
    arg203_cubed = arg203 * arg203 * arg203
    add_213 = tl.load(add_213_ptr + offsets).to(tl.float32)
    mask = tl.load(arg201_ptr + offsets).to(tl.int1)

    correction = row_sum * arg203_cubed * arg202 * INV_C_
    epilogue = (add_213 + weighted * arg203 - correction) * DROPOUT_SCALE_
    epilogue = tl.where(mask, epilogue, 0.0)
    tl.store(out_ptr + offsets, epilogue)


@triton.jit
def _col_reduce_partial_kernel(
    mm_279_ptr,
    mm_281_ptr,
    arg202_ptr,
    arg203_ptr,
    partial_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    num_row_tiles: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = col_tile * BLOCK_C + tl.arange(0, BLOCK_C)
    active = (rows[:, None] < ROWS_) & (cols[None, :] < C_)
    offsets = rows[:, None] * C_ + cols[None, :]

    add_value = (
        tl.load(mm_279_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(mm_281_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    )
    arg202 = tl.load(arg202_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    arg203 = tl.load(arg203_ptr + rows, mask=rows < ROWS_, other=0.0).to(tl.float32)

    values = add_value * arg202 * arg203[:, None]
    partial = tl.sum(values, axis=0)
    partial_offsets = cols * num_row_tiles + row_tile
    tl.store(partial_ptr + partial_offsets, partial, mask=cols < C_)


@triton.jit
def _col_reduce_finalize_kernel(
    partial_ptr,
    out0_ptr,
    num_row_tiles: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    col = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < num_row_tiles
    values = tl.load(partial_ptr + col * num_row_tiles + tiles, mask=active, other=0.0).to(tl.float32)
    tl.store(out0_ptr + col, tl.sum(values, axis=0))


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


def oracle_fused(
    mm_279: torch.Tensor,
    mm_281: torch.Tensor,
    arg7_1: torch.Tensor,
    arg202_1: torch.Tensor,
    arg203_1: torch.Tensor,
    add_213: torch.Tensor,
    arg201_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mm_279.shape == (ROWS, C)
    assert mm_281.shape == (ROWS, C)
    assert arg7_1.shape == (C,)
    assert arg202_1.shape == (N, T, C)
    assert arg203_1.shape == (N, T, 1)
    assert add_213.shape == (N, T, C)
    assert arg201_1.shape == (N, T, C)
    assert tuple(_shape_param_0) == (N, T, C)
    assert tuple(_shape_param_1) == (N, T, C)
    assert tuple(_shape_param_2) == (C,)
    assert tuple(_shape_param_3) == (N, T, C)
    assert tuple(_shape_param_4) == (ROWS, C)
    assert mm_279.is_contiguous()
    assert mm_281.is_contiguous()
    assert arg7_1.is_contiguous()
    assert arg202_1.is_contiguous()
    assert arg203_1.is_contiguous()
    assert add_213.is_contiguous()
    assert arg201_1.is_contiguous()

    out0 = torch.empty((C,), device=mm_279.device, dtype=torch.float32)
    out1 = torch.empty_strided((C, ROWS), (1, C), device=mm_279.device, dtype=torch.float32)

    block_rows = 16
    block_cols = 64
    num_row_tiles = triton.cdiv(ROWS, block_rows)
    partial = torch.empty((C, num_row_tiles), device=mm_279.device, dtype=torch.float32)

    _row_epilogue_kernel[(ROWS,)](
        mm_279,
        mm_281,
        arg7_1,
        arg202_1,
        arg203_1,
        add_213,
        arg201_1,
        out1,
        C_=C,
        INV_C_=INV_C,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_C=C,
        num_warps=8,
    )

    _col_reduce_partial_kernel[(num_row_tiles, triton.cdiv(C, block_cols))](
        mm_279,
        mm_281,
        arg202_1,
        arg203_1,
        partial,
        ROWS_=ROWS,
        C_=C,
        num_row_tiles=num_row_tiles,
        BLOCK_ROWS=block_rows,
        BLOCK_C=block_cols,
        num_warps=4,
    )

    block_tiles = triton.next_power_of_2(num_row_tiles)
    _col_reduce_finalize_kernel[(C,)](
        partial,
        out0,
        num_row_tiles=num_row_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=4,
    )

    return out0, out1


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
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
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        shape_ok = got.shape == expected.shape
        dtype_ok = got.dtype == expected.dtype
        stride_ok = got.stride() == expected.stride()
        close_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        output_ok = shape_ok and dtype_ok and stride_ok and close_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={close_ok} "
            f"shape_match={shape_ok} stride_match={stride_ok} dtype_match={dtype_ok}"
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
    print(f"oracle_fused full-scope cross-axis reductions + transposed epilogue: {oracle_us:.3f} us")

    if no_compile:
        return

    compile_times: list[tuple[str, float]] = []
    module = _load_repro_module()
    for label, config in COMPILE_CONFIGS:
        model = module.Repro().cuda()
        with torch.no_grad():
            compiled = _compile_with_config(model, inputs, config)
            compiled_us = triton.testing.do_bench(
                lambda: compiled(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            ) * 1000.0
        compile_times.append((label, compiled_us))
        print(f"torch.compile {label}: {compiled_us:.3f} us")

    if compile_times and all(oracle_us < compiled_us for _, compiled_us in compile_times):
        print("valid_floor: oracle is faster than both required compile configs")
    elif compile_times:
        print("diagnosis_only: oracle is not a true floor because a required compile config is faster")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
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
