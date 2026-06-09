"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer sliding-window attention-backward Repro.forward output by fusing the padded diagonal score assembly, key-mask/global-mask multiply, row sum/FMA softmax-backward reduction, boundary masks, and final diagonal scatter layout into Triton kernels that write the returned [288,512,512] tensor directly, whereas Inductor currently lowers the 122-op view/pad/slice_scatter/copy/where/sum/fma/select_scatter pipeline as separate generic layout, reduction, and scatter-style kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no structured Longformer diagonal scatter-reduce template that combines row-local softmax-backward reductions with the surrounding sliding-window slice_scatter layout transforms; the fix is SCATTER_REDUCE: add a Longformer sliding-window backward lowering that recomputes the cheap diagonal index maps, fuses boundary masks and row reductions, and scatters directly to the final diagonalized output layout."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch._inductor.inductor_prims  # noqa: F401
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


REPRO_ID = "sum_617cd87647d6"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

sys.path.insert(0, str(REPO_ROOT))

BATCH = 8
SEQ = 1024
HEADS = 12
CHUNKS = BATCH * HEADS
BLOCKS_PER_SEQ = 4
BLOCK_ROWS = 256
WINDOW = 513
BMM_K = 768
OUT_CHANNELS = CHUNKS * 3
OUT_ROWS = 512
OUT_COLS = 512
OUT_PLANE = OUT_ROWS * OUT_COLS
OUT_NUMEL = OUT_CHANNELS * OUT_PLANE
R_BLOCK = 1024
ZERO_BLOCK = 1024


@triton.jit
def _zero_output_kernel(
    out_ptr,
    NUMEL_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=offsets < NUMEL_)


@triton.jit
def _longformer_backward_scatter_kernel(
    bmm_ptr,
    key_mask_ptr,
    global_mask_ptr,
    probs_ptr,
    end_mask_ptr,
    start_mask_ptr,
    out_ptr,
    WINDOW_: tl.constexpr,
    BMM_K_: tl.constexpr,
    SEQ_: tl.constexpr,
    HEADS_: tl.constexpr,
    BLOCK_ROWS_: tl.constexpr,
    OUT_PLANE_: tl.constexpr,
    R_BLOCK_: tl.constexpr,
):
    chunk = tl.program_id(0)
    diagonal_block = tl.program_id(1)
    row_in_block = tl.program_id(2)

    cols = tl.arange(0, R_BLOCK_)
    col_mask = cols < WINDOW_
    cols_i64 = cols.to(tl.int64)

    batch = chunk // HEADS_
    head = chunk - batch * HEADS_
    seq_pos = diagonal_block * BLOCK_ROWS_ + row_in_block
    bmm_row = chunk * 4 + diagonal_block

    bmm_offsets = (bmm_row * BLOCK_ROWS_ + row_in_block) * BMM_K_ + row_in_block + cols
    prob_offsets = ((batch * SEQ_ + seq_pos) * HEADS_ + head) * WINDOW_ + cols

    bmm = tl.load(bmm_ptr + bmm_offsets, mask=col_mask, other=0.0).to(tl.float32)
    key_mask = tl.load(key_mask_ptr + prob_offsets, mask=col_mask, other=0).to(tl.float32)
    probs = tl.load(probs_ptr + prob_offsets, mask=col_mask, other=0.0).to(tl.float32)
    global_mask = tl.load(global_mask_ptr + batch * SEQ_ + seq_pos).to(tl.int1)

    masked_scores = bmm * key_mask
    masked_scores = tl.where(global_mask, 0.0, masked_scores)
    product = masked_scores * probs
    row_sum = tl.sum(tl.where(col_mask, product, 0.0), axis=0)
    grad = tl.fma(-probs, row_sum, product)

    start_mask = tl.load(
        start_mask_ptr + row_in_block * 257 + cols,
        mask=cols <= 256,
        other=0.0,
    ).to(tl.float32)
    end_mask = tl.load(
        end_mask_ptr + row_in_block * 257 + (cols - 256),
        mask=(cols >= 256) & col_mask,
        other=0.0,
    ).to(tl.float32)
    grad = tl.where((diagonal_block == 0) & (cols <= 256) & (start_mask != 0.0), 0.0, grad)
    grad = tl.where((diagonal_block == 3) & (cols >= 256) & (end_mask != 0.0), 0.0, grad)

    # clone10 path: source diagonal blocks 0..2, columns 256..512.
    bflat10 = row_in_block * 513 + (cols_i64 - 256)
    out10 = (chunk * 3 + diagonal_block) * OUT_PLANE_ + bflat10
    tl.store(
        out_ptr + out10,
        grad,
        mask=(diagonal_block <= 2) & (cols >= 256) & col_mask,
    )

    # clone7 path: source diagonal block 0, rows 1..255, columns 1..255.
    bflat7 = (row_in_block - 1) * 513 + (cols_i64 + 257)
    out7 = chunk * 3 * OUT_PLANE_ + bflat7
    tl.store(
        out_ptr + out7,
        grad,
        mask=(diagonal_block == 0)
        & (row_in_block >= 1)
        & (cols >= 1)
        & (cols <= 255),
    )

    # clone8 path: source diagonal blocks 1..3, columns 0..255.
    bflat8 = (255 + row_in_block) * 513 + (cols_i64 + 257)
    out8 = (chunk * 3 + diagonal_block - 1) * OUT_PLANE_ + bflat8
    tl.store(
        out_ptr + out8,
        grad,
        mask=(diagonal_block >= 1) & (cols <= 255),
    )

    # clone9 path: source diagonal block 3, columns 256..512.  The final
    # view/truncation keeps only the first 262144 flat B-buffer elements.
    bflat9 = (256 + row_in_block) * 513 + (cols_i64 - 256)
    out9 = (chunk * 3 + 2) * OUT_PLANE_ + bflat9
    tl.store(
        out_ptr + out9,
        grad,
        mask=(diagonal_block == 3)
        & (cols >= 256)
        & col_mask
        & (bflat9 < OUT_PLANE_),
    )


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def make_inputs() -> tuple[Any, ...]:
    module = _load_repro_module()
    return tuple(
        x.cuda() if isinstance(x, torch.Tensor) and not x.is_cuda else x
        for x in module.make_inputs()
    )


def reference_outputs(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _check_shape_params(*shape_params: object) -> None:
    expected = (
        [96, 4, 256, 768, 1],
        [96, 4, 196864],
        [96, 4, 256, 770],
        [8, 12, 1024, 513],
        [96, 4, 256, 513],
        [8, 12, 1024, 513],
        [96, 4, 256, 513],
        [8, 12, 1024, 513],
        [96, 4, 256, 513],
        [8, 256, 12, 257],
        [96, 4, 256, 513],
        [8, 12, 1024, 513],
        [96, 4, 256, 513],
        [8, 256, 12, 257],
        [96, 4, 256, 513],
        [96, 3, 513, 512],
        [96, 3, 512, 512, 1],
        [288, 512, 512],
    )
    assert len(shape_params) == len(expected)
    for idx, (got, exp) in enumerate(zip(shape_params, expected)):
        assert list(got) == exp, (idx, got, exp)


def _check_inputs(
    bmm_1: torch.Tensor,
    arg222_1: torch.Tensor,
    arg3_1: torch.Tensor,
    arg221_1: torch.Tensor,
    arg99_1: torch.Tensor,
    arg98_1: torch.Tensor,
) -> None:
    assert bmm_1.is_cuda and arg222_1.is_cuda and arg3_1.is_cuda
    assert arg221_1.is_cuda and arg99_1.is_cuda and arg98_1.is_cuda
    assert bmm_1.shape == (384, 256, 768) and bmm_1.dtype == torch.float32
    assert arg222_1.shape == (8, 1024, 12, 513) and arg222_1.dtype == torch.bool
    assert arg3_1.shape == (8, 1024) and arg3_1.dtype == torch.bool
    assert arg221_1.shape == (8, 1024, 12, 513) and arg221_1.dtype == torch.float32
    assert arg99_1.shape == (1, 256, 1, 257) and arg99_1.dtype == torch.float32
    assert arg98_1.shape == (1, 256, 1, 257) and arg98_1.dtype == torch.float32
    assert bmm_1.is_contiguous()
    assert arg222_1.is_contiguous()
    assert arg3_1.is_contiguous()
    assert arg221_1.is_contiguous()
    assert arg99_1.is_contiguous()
    assert arg98_1.is_contiguous()


def oracle_full_longformer_backward(
    bmm_1: torch.Tensor,
    arg222_1: torch.Tensor,
    arg3_1: torch.Tensor,
    arg221_1: torch.Tensor,
    arg99_1: torch.Tensor,
    arg98_1: torch.Tensor,
    *shape_params: object,
) -> torch.Tensor:
    _check_shape_params(*shape_params)
    _check_inputs(bmm_1, arg222_1, arg3_1, arg221_1, arg99_1, arg98_1)

    out = torch.empty(
        (OUT_CHANNELS, OUT_ROWS, OUT_COLS),
        device=bmm_1.device,
        dtype=torch.float32,
    )
    _zero_output_kernel[(triton.cdiv(OUT_NUMEL, ZERO_BLOCK),)](
        out,
        NUMEL_=OUT_NUMEL,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )
    _longformer_backward_scatter_kernel[(CHUNKS, BLOCKS_PER_SEQ, BLOCK_ROWS)](
        bmm_1,
        arg222_1,
        arg3_1,
        arg221_1,
        arg99_1,
        arg98_1,
        out,
        WINDOW_=WINDOW,
        BMM_K_=BMM_K,
        SEQ_=SEQ,
        HEADS_=HEADS,
        BLOCK_ROWS_=BLOCK_ROWS,
        OUT_PLANE_=OUT_PLANE,
        R_BLOCK_=R_BLOCK,
        num_warps=8,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([384, 256, 768], f32), T([8, 1024, 12, 513], b8), T([8, 1024], b8), T([8, 1024, 12, 513], f32), T([1, 256, 1, 257], f32), T([1, 256, 1, 257], f32), S([96, 4, 256, 768, 1]), S([96, 4, 196864]), S([96, 4, 256, 770]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([8, 256, 12, 257]), S([96, 4, 256, 513]), S([8, 12, 1024, 513]), S([96, 4, 256, 513]), S([8, 256, 12, 257]), S([96, 4, 256, 513]), S([96, 3, 513, 512]), S([96, 3, 512, 512, 1]), S([288, 512, 512]))")
def oracle_forward(inputs: tuple[Any, ...]) -> torch.Tensor:
    return oracle_full_longformer_backward(*inputs)


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
