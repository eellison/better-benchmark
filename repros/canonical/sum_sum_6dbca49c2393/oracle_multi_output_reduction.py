"""
Full-scope diagnosis oracle for sum_sum_6dbca49c2393 (T5 LN/dropout backward).

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle consumes the
same eight tensor inputs and shape parameters as `repro.py`, returns the same
`[512]` vector plus `[512, 4096]` transposed side output, and fuses the three
matmul-gradient adds, affine multiply, row-local dot reduction, dropout-mask
epilogue store, and sibling column reduction into one Triton producer pass plus
a small partial-reduction finalizer. Inductor cannot express that exact
cooperative schedule today because the scheduler has to mix a dependent row
reduction, a materialized side output, and a compatible column reduction, but
the measured full-scope Triton variant is slower than the historical compiled
best; the practical fix for this repro is no oracle-floor wiring, while a future
cooperative split-K template would be justified only if it beats the tuned
bandwidth/launch floor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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


REPRO_ID = "sum_sum_6dbca49c2393"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ
C = 512
DROPOUT_SCALE = 1.1111111111111112
INV_C = 1.0 / C
HISTORICAL_BEST_COMPILE_US = 24.480000138282776

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@triton.jit
def _full_scope_partial_kernel(
    mm0_ptr,
    mm1_ptr,
    mm2_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    residual_ptr,
    keep_ptr,
    out_base_ptr,
    partial_col_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    INV_C_: tl.constexpr,
    ROW_SPLIT: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C_
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    acc_col = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * C_ + c[None, :]

        x0 = tl.load(mm0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(mm1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(mm2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = x0 + x1 + x2
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weighted = x * weight[None, :]
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)

        row_scale = tl.load(row_scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        row_scale3 = row_scale * row_scale * row_scale
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)

        out = residual + row_scale[:, None] * weighted
        out -= row_dot[:, None] * row_scale3[:, None] * rhs * INV_C_
        out *= keep * DROPOUT_SCALE_
        tl.store(out_base_ptr + offsets, out, mask=mask)

        acc_col += tl.sum(tl.where(mask, x * rhs * row_scale[:, None], 0.0), axis=0)

    partial_offsets = pid * C_ + c
    tl.store(partial_col_ptr + partial_offsets, acc_col, mask=c_mask)


@triton.jit
def _finalize_partials_kernel(
    partial_col_ptr,
    out_vec_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    row_block = tl.arange(0, BLOCK_ROW_BLOCKS)
    mask = (row_block[:, None] < NUM_ROW_BLOCKS) & (c[None, :] < C_)
    offsets = row_block[:, None] * C_ + c[None, :]
    values = tl.load(partial_col_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_vec_ptr + c, tl.sum(values, axis=0), mask=c < C_)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def get_inputs() -> tuple[object, ...]:
    return make_inputs()


def get_repro_instance() -> torch.nn.Module:
    module = _load_repro_module()
    return module.Repro().cuda()


def oracle_fused(
    mm_271: torch.Tensor,
    mm_273: torch.Tensor,
    mm_275: torch.Tensor,
    arg11_1: torch.Tensor,
    arg210_1: torch.Tensor,
    arg211_1: torch.Tensor,
    add_208: torch.Tensor,
    arg209_1: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mm_271.shape == (ROWS, C)
    assert mm_273.shape == (ROWS, C)
    assert mm_275.shape == (ROWS, C)
    assert arg11_1.shape == (C,)
    assert arg210_1.shape == (BATCH, SEQ, C)
    assert arg211_1.shape == (BATCH, SEQ, 1)
    assert add_208.shape == (BATCH, SEQ, C)
    assert arg209_1.shape == (BATCH, SEQ, C)

    mm0 = mm_271.contiguous()
    mm1 = mm_273.contiguous()
    mm2 = mm_275.contiguous()
    weight = arg11_1.contiguous()
    rhs = arg210_1.contiguous().reshape(ROWS, C)
    row_scale = arg211_1.contiguous().reshape(ROWS)
    residual = add_208.contiguous().reshape(ROWS, C)
    keep = arg209_1.contiguous().reshape(ROWS, C)

    row_split = 6
    xblock = 1
    block_c = 512
    num_row_blocks = triton.cdiv(ROWS, row_split)

    out_base = torch.empty((ROWS, C), device=mm0.device, dtype=torch.float32)
    partial_col = torch.empty((num_row_blocks, C), device=mm0.device, dtype=torch.float32)

    _full_scope_partial_kernel[(num_row_blocks,)](
        mm0,
        mm1,
        mm2,
        weight,
        rhs,
        row_scale,
        residual,
        keep,
        out_base,
        partial_col,
        ROWS_=ROWS,
        C_=C,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        INV_C_=INV_C,
        ROW_SPLIT=row_split,
        XBLOCK=xblock,
        BLOCK_C=block_c,
        num_warps=4,
    )

    out_vec = torch.empty((C,), device=mm0.device, dtype=torch.float32)
    finalize_block_c = 16
    _finalize_partials_kernel[(triton.cdiv(C, finalize_block_c),)](
        partial_col,
        out_vec,
        NUM_ROW_BLOCKS=num_row_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=finalize_block_c,
        num_warps=8,
    )

    return out_vec, out_base.t()


@oracle_impl(hardware="H100", shapes="(T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([512], f32), T([32, 128, 512], f32), T([32, 128, 1], f32), T([32, 128, 512], f32), T([32, 128, 512], b8), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([512]), S([32, 128, 512]), S([4096, 512]))")
def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    model = get_repro_instance()
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
