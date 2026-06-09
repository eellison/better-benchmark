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


@oracle_impl(hardware="H100", shapes="(T([8192, 512], f32), T([8, 1024, 512], b8), T([512], f32), T([8, 1024, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 512], b8), S([8, 1024, 512]), S([512]), S([8, 1024, 512]), S([8192, 512]))")
def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def main() -> None:
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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
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
