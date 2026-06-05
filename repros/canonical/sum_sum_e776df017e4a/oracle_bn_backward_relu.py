"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete GhostNet BN-backward/ReLU-backward fragment returned by Repro.forward by folding `relu(affine) <= 0` to the equivalent affine predicate, hoisting channel-only mean/invstd/weight/bias terms, and using one shared dual-accumulator Triton reduction for `sum(where)` and `sum(where * centered)` before the full tensor epilogue and scale-gradient vector store, whereas Inductor lowers the decomposed unsqueeze/sub/mul/relu/le/where/sum/sum/epilogue graph as generic regions that reread and recompute the same masked producer; Inductor cannot do this today because its simplifier does not canonicalize this BN-backward algebra and ReLU-mask predicate before multi-output reduction scheduling; the fix is ALGEBRAIC_ELIMINATION: expose the channel-only coefficients and compatible sibling reductions to one multi-output reduction/epilogue template."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    get_shape_key,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C = 8
INPUT_C = 16
SLICE_START = 8
H = 112
W = 112
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
SCALE = 1.5570192920918366e-07


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _dual_reduce_partial_kernel(
        getitem_ptr,
        arg202_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partial_sum1_ptr,
        partial_sum2_ptr,
        num_tiles: tl.constexpr,
        C_: tl.constexpr,
        INPUT_C_: tl.constexpr,
        SLICE_START_: tl.constexpr,
        HW_: tl.constexpr,
        TOTAL_SPATIAL_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        active = k < TOTAL_SPATIAL_

        n = k // HW_
        hw = k - n * HW_
        arg202_offsets = n * (C_ * HW_) + c * HW_ + hw
        getitem_offsets = n * (INPUT_C_ * HW_) + (SLICE_START_ + c) * HW_ + hw

        x = tl.load(arg202_ptr + arg202_offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        weight = tl.load(weight_ptr + c).to(tl.float32)
        bias = tl.load(bias_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        centered = x - mean
        affine = centered * invstd * weight + bias
        use_full = affine <= 0.0
        use_slice = ~use_full
        slice_value = tl.load(
            getitem_ptr + getitem_offsets,
            mask=active & use_slice,
            other=0.0,
        ).to(tl.float32)
        where_value = tl.where(use_full, full_value, slice_value)

        partial_offset = c * num_tiles + tile
        tl.store(partial_sum1_ptr + partial_offset, tl.sum(where_value, axis=0))
        tl.store(partial_sum2_ptr + partial_offset, tl.sum(where_value * centered, axis=0))


    @triton.jit
    def _finalize_reduce_kernel(
        partial_sum1_ptr,
        partial_sum2_ptr,
        invstd_ptr,
        sum1_ptr,
        sum2_ptr,
        vector_out_ptr,
        num_tiles: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        c = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES)
        tile_mask = tiles < num_tiles
        partial_offsets = c * num_tiles + tiles

        sum1_values = tl.load(partial_sum1_ptr + partial_offsets, mask=tile_mask, other=0.0).to(tl.float32)
        sum2_values = tl.load(partial_sum2_ptr + partial_offsets, mask=tile_mask, other=0.0).to(tl.float32)
        sum1 = tl.sum(sum1_values, axis=0)
        sum2 = tl.sum(sum2_values, axis=0)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)

        tl.store(sum1_ptr + c, sum1)
        tl.store(sum2_ptr + c, sum2)
        tl.store(vector_out_ptr + c, sum2 * invstd)


    @triton.jit
    def _epilogue_kernel(
        getitem_ptr,
        arg202_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        sum1_ptr,
        sum2_ptr,
        tensor_out_ptr,
        C_: tl.constexpr,
        INPUT_C_: tl.constexpr,
        SLICE_START_: tl.constexpr,
        HW_: tl.constexpr,
        NUMEL_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
        active = offsets < NUMEL_

        hw = offsets % HW_
        c = (offsets // HW_) % C_
        n = offsets // (C_ * HW_)
        arg202_offsets = n * (C_ * HW_) + c * HW_ + hw
        getitem_offsets = n * (INPUT_C_ * HW_) + (SLICE_START_ + c) * HW_ + hw

        x = tl.load(arg202_ptr + arg202_offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
        sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)

        centered = x - mean
        affine = centered * invstd * weight + bias
        use_full = affine <= 0.0
        use_slice = ~use_full
        slice_value = tl.load(
            getitem_ptr + getitem_offsets,
            mask=active & use_slice,
            other=0.0,
        ).to(tl.float32)
        where_value = tl.where(use_full, full_value, slice_value)

        mean_term = sum1 * SCALE_
        variance_term = sum2 * SCALE_ * invstd * invstd
        affine_term = invstd * weight
        out = (where_value - centered * variance_term - mean_term) * affine_term
        tl.store(tensor_out_ptr + offsets, out, mask=active)


def _oracle_triton(
    getitem_273: torch.Tensor,
    arg202_1: torch.Tensor,
    arg203_1: torch.Tensor,
    arg204_1: torch.Tensor,
    arg6_1: torch.Tensor,
    arg7_1: torch.Tensor,
    full: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    assert getitem_273.shape == (N, INPUT_C, H, W)
    assert arg202_1.shape == (N, C, H, W)
    assert arg203_1.shape == (1, C, 1, 1)
    assert arg204_1.shape == (1, C, 1, 1)
    assert arg6_1.shape == (C,)
    assert arg7_1.shape == (C,)
    assert full.shape == ()
    assert getitem_273.is_contiguous()
    assert arg202_1.is_contiguous()
    assert arg203_1.is_contiguous()
    assert arg204_1.is_contiguous()
    assert arg6_1.is_contiguous()
    assert arg7_1.is_contiguous()

    block_k = 2048
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum1 = torch.empty((C, num_tiles), device=arg202_1.device, dtype=torch.float32)
    partial_sum2 = torch.empty((C, num_tiles), device=arg202_1.device, dtype=torch.float32)
    sum1 = torch.empty((C,), device=arg202_1.device, dtype=torch.float32)
    sum2 = torch.empty((C,), device=arg202_1.device, dtype=torch.float32)
    vector_out = torch.empty_strided((C,), (1,), device=arg202_1.device, dtype=torch.float32)

    _dual_reduce_partial_kernel[(C, num_tiles)](
        getitem_273,
        arg202_1,
        arg203_1,
        arg204_1,
        arg6_1,
        arg7_1,
        full,
        partial_sum1,
        partial_sum2,
        num_tiles=num_tiles,
        C_=C,
        INPUT_C_=INPUT_C,
        SLICE_START_=SLICE_START,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K=block_k,
        num_warps=8,
    )

    block_tiles = triton.next_power_of_2(num_tiles)
    _finalize_reduce_kernel[(C,)](
        partial_sum1,
        partial_sum2,
        arg204_1,
        sum1,
        sum2,
        vector_out,
        num_tiles=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=8,
    )

    tensor_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=arg202_1.device,
        dtype=torch.float32,
    )
    block_elems = 1024
    _epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        getitem_273,
        arg202_1,
        arg203_1,
        arg204_1,
        arg6_1,
        arg7_1,
        full,
        sum1,
        sum2,
        tensor_out,
        C_=C,
        INPUT_C_=INPUT_C,
        SLICE_START_=SLICE_START,
        HW_=HW,
        NUMEL_=NUMEL,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return tensor_out, vector_out


def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    return _oracle_triton(*inputs)


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

    inputs = get_inputs()
    instance = get_repro_instance()

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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
