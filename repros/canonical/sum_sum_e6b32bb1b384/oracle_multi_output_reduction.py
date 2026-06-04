"""
Diagnosis-only full-scope oracle candidate for sum_sum_e6b32bb1b384.

Gap diagnosis (classification: BANDWIDTH_BOUND): This diagnostic oracle
consumes the same six tensor inputs plus shape parameters as repro.py, computes
the row-wise hidden reduction used by the returned transposed tensor, and at
the same time emits row-block partials for the sibling column reduction before
a small Triton finalizer writes the returned [512] vector. Inductor does not
form this cross-axis fused schedule today because output[0] reduces over rows
per hidden column while output[1] reduces over hidden columns per row, so its
scheduler keeps a split column reduction separate from the row-reduction
materialization. The measured full-scope candidate is slower than both required
local compiled configs and the historical best, so the Inductor closure is no
new codegen floor for this shape: leave the main oracle path blank and keep
the current tuned split schedule, with any future cross-axis fusion rejected by
cost model when partial stores and register pressure outweigh saved input reads.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_e6b32bb1b384"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

M = 8192
H = 512
SCALE_MM = 0.04419417382415922
MASK_SCALE = 1.1111111111111112
INV_H = 0.001953125
HISTORICAL_BEST_COMPILE_US = 25.37599951028824
DEFAULT_BLOCK_M = 2

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _fused_row_and_column_partials_kernel(
    mm_ptr,
    mask0_ptr,
    weight_ptr,
    activ_ptr,
    row_scale_ptr,
    mask1_ptr,
    partial_col_ptr,
    transposed_out_ptr,
    BLOCK_M: tl.constexpr,
    H_: tl.constexpr,
    SCALE_MM_: tl.constexpr,
    MASK_SCALE_: tl.constexpr,
    INV_H_: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    h_offsets = tl.arange(0, H_)
    offsets = row_offsets[:, None] * H_ + h_offsets[None, :]

    mm = tl.load(mm_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    mask0 = tl.load(mask0_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    weight = tl.load(weight_ptr + h_offsets, eviction_policy="evict_last").to(tl.float32)
    activ = tl.load(activ_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    row_scale = tl.load(row_scale_ptr + row_offsets, eviction_policy="evict_last").to(tl.float32)
    mask1 = tl.load(mask1_ptr + offsets, eviction_policy="evict_first").to(tl.float32)

    mul = mm * SCALE_MM_
    mask0_scaled = mask0 * MASK_SCALE_
    scaled = mul * mask0_scaled
    weighted = scaled * weight[None, :]
    weighted_activ = weighted * activ
    row_sum = tl.sum(weighted_activ, axis=1)

    first_term = weighted * row_scale[:, None]
    neg_half = row_sum * -0.5
    row_scale_sq = row_scale * row_scale
    row_scale_cu = row_scale_sq * row_scale
    expanded = neg_half * row_scale_cu * INV_H_
    second_term = expanded[:, None] * (activ * 2.0)
    out = (first_term + second_term) * (mask1 * MASK_SCALE_)

    tl.store(transposed_out_ptr + offsets, out)

    col_term = scaled * (activ * row_scale[:, None])
    partial = tl.sum(col_term, axis=0)
    tl.store(partial_col_ptr + tl.program_id(0) * H_ + h_offsets, partial)


@triton.jit
def _finalize_column_sum_kernel(
    partial_col_ptr,
    vector_out_ptr,
    num_row_blocks: tl.constexpr,
    BLOCK_R: tl.constexpr,
    H_: tl.constexpr,
):
    h = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    mask = r < num_row_blocks
    vals = tl.load(partial_col_ptr + r * H_ + h, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(vals, axis=0)
    tl.store(vector_out_ptr + h, total)


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
    mm_1: torch.Tensor,
    arg344_1: torch.Tensor,
    arg130_1: torch.Tensor,
    arg342_1: torch.Tensor,
    arg343_1: torch.Tensor,
    arg341_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    *,
    block_m: int = DEFAULT_BLOCK_M,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3

    assert mm_1.shape == (M, H)
    assert arg344_1.shape == (8, 1024, H)
    assert arg130_1.shape == (H,)
    assert arg342_1.shape == (8, 1024, H)
    assert arg343_1.shape == (8, 1024, 1)
    assert arg341_1.shape == (8, 1024, H)
    assert mm_1.is_contiguous()
    assert arg344_1.is_contiguous()
    assert arg130_1.is_contiguous()
    assert arg342_1.is_contiguous()
    assert arg343_1.is_contiguous()
    assert arg341_1.is_contiguous()
    if M % block_m != 0:
        raise ValueError(f"block_m must divide {M}, got {block_m}")

    num_row_blocks = triton.cdiv(M, block_m)
    partial_col = torch.empty((num_row_blocks, H), device=mm_1.device, dtype=torch.float32)
    vector_out = torch.empty((H,), device=mm_1.device, dtype=torch.float32)
    transposed_out = torch.empty_strided((H, M), (1, H), device=mm_1.device, dtype=torch.float32)

    _fused_row_and_column_partials_kernel[(num_row_blocks,)](
        mm_1,
        arg344_1,
        arg130_1,
        arg342_1,
        arg343_1,
        arg341_1,
        partial_col,
        transposed_out,
        BLOCK_M=block_m,
        H_=H,
        SCALE_MM_=SCALE_MM,
        MASK_SCALE_=MASK_SCALE,
        INV_H_=INV_H,
        num_warps=8,
    )

    block_r = triton.next_power_of_2(num_row_blocks)
    _finalize_column_sum_kernel[(H,)](
        partial_col,
        vector_out,
        num_row_blocks=num_row_blocks,
        BLOCK_R=block_r,
        H_=H,
        num_warps=8,
    )

    return vector_out, transposed_out


def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float, block_m: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual = oracle_fused(*inputs, block_m=block_m)
        torch.cuda.synchronize()

    ok = True
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        output_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == expected.stride()
        dtype_ok = got.dtype == expected.dtype
        ok = ok and output_ok and stride_ok and dtype_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={output_ok} stride_match={stride_ok} dtype_match={dtype_ok}"
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


def run_bench(rep: int, warmup: int, no_compile: bool, block_m: int) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        oracle_fused(*inputs, block_m=block_m)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_fused(*inputs, block_m=block_m),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(
        f"oracle_fused full-scope cross-axis reduction partials "
        f"(BLOCK_M={block_m}): {oracle_us:.3f} us"
    )
    print(f"historical best compile: {HISTORICAL_BEST_COMPILE_US:.3f} us")

    results: dict[str, float] = {
        "oracle_us": oracle_us,
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
    }
    if no_compile:
        return results

    module = _load_repro_module()
    compile_configs = [
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
        results[label] = compiled_us
        print(f"torch.compile {label}: {compiled_us:.3f} us")

    best_gate = min(HISTORICAL_BEST_COMPILE_US, *(v for k, v in results.items() if k.startswith("coordinate") or k.startswith("combo")))
    true_floor = oracle_us < best_gate
    print(f"true_floor_candidate: {true_floor} (best gate {best_gate:.3f} us)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--block-m", type=int, default=DEFAULT_BLOCK_M)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol, block_m=args.block_m):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile, block_m=args.block_m)


if __name__ == "__main__":
    with torch.no_grad():
        main()
