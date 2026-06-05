"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full MobileBERT backward tuple by producing the dense transposed side output directly while fusing the sibling channel reductions and the two duplicate-preserving indexed accumulations into Triton reduction/scatter kernels; Inductor currently materializes the shared `[256,128,512]` intermediates and lowers the reductions, `index_put(accumulate=True)`, and transpose as separate generic scheduled work; Inductor cannot do this today because scheduler/codegen does not recognize this gather-mask-reduce/indexed-scatter family as one shared producer with several accumulator destinations; the fix is SCATTER_REDUCE: teach Inductor to fuse rowwise producers into direct duplicate-safe scatter-reduce epilogues and sibling reductions without materializing the full intermediates."""
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
BATCH = 256
SEQ = 128
HIDDEN = 512
ROWS = BATCH * SEQ
SCATTER_ROWS = 2
TOKEN_ROWS = 512

if triton is not None:

    @triton.jit
    def _zero_kernel(out_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        tl.store(out_ptr + offsets, tl.zeros([BLOCK], dtype=tl.float32), mask=mask)


    @triton.jit
    def _dense_partial_kernel(
        mm714_ptr,
        mul516_ptr,
        mm720_ptr,
        mm722_ptr,
        arg585_ptr,
        arg3_ptr,
        arg583_ptr,
        dense_ptr,
        partial_sum_ptr,
        partial_weighted_ptr,
        partial_scaled_ptr,
        partial_bucket0_ptr,
        partial_bucket1_ptr,
        N: tl.constexpr,
        C: tl.constexpr,
        SCATTER_N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        tile_n = tl.program_id(0)
        tile_c = tl.program_id(1)
        rows = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        cols = tile_c * BLOCK_C + tl.arange(0, BLOCK_C)
        offsets = rows[:, None] * C + cols[None, :]
        mask = (rows[:, None] < N) & (cols[None, :] < C)

        acc = tl.load(mul516_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.load(mm714_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.load(mm720_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.load(mm722_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        channel_scale = tl.load(arg3_ptr + cols, mask=cols < C, other=0.0).to(tl.float32)
        scaled = acc * channel_scale[None, :]
        tl.store(dense_ptr + offsets, scaled, mask=mask)

        weighted = tl.load(arg585_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_bucket = tl.load(arg583_ptr + rows, mask=rows < N, other=-99).to(tl.int64)
        row_bucket = tl.where(row_bucket < 0, row_bucket + SCATTER_N, row_bucket)
        valid_acc = tl.where(mask, acc, 0.0)
        valid_scaled = tl.where(mask, scaled, 0.0)

        partial_offsets = tile_n * C + cols
        channel_mask = cols < C
        tl.store(partial_sum_ptr + partial_offsets, tl.sum(valid_acc, axis=0), mask=channel_mask)
        tl.store(
            partial_weighted_ptr + partial_offsets,
            tl.sum(tl.where(mask, acc * weighted, 0.0), axis=0),
            mask=channel_mask,
        )
        tl.store(partial_scaled_ptr + partial_offsets, tl.sum(valid_scaled, axis=0), mask=channel_mask)
        tl.store(
            partial_bucket0_ptr + partial_offsets,
            tl.sum(tl.where(mask & (row_bucket[:, None] == 0), scaled, 0.0), axis=0),
            mask=channel_mask,
        )
        tl.store(
            partial_bucket1_ptr + partial_offsets,
            tl.sum(tl.where(mask & (row_bucket[:, None] == 1), scaled, 0.0), axis=0),
            mask=channel_mask,
        )


    @triton.jit
    def _finalize_partial_kernel(
        partial_sum_ptr,
        partial_weighted_ptr,
        partial_scaled_ptr,
        partial_bucket0_ptr,
        partial_bucket1_ptr,
        out_sum_ptr,
        out_weighted_ptr,
        out_bucket_ptr,
        out_scaled_sum_ptr,
        N_TILES: tl.constexpr,
        C: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        c = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES)
        mask = tiles < N_TILES
        offsets = tiles * C + c

        sum_acc = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0)
        weighted_acc = tl.load(partial_weighted_ptr + offsets, mask=mask, other=0.0)
        scaled_acc = tl.load(partial_scaled_ptr + offsets, mask=mask, other=0.0)
        bucket0_acc = tl.load(partial_bucket0_ptr + offsets, mask=mask, other=0.0)
        bucket1_acc = tl.load(partial_bucket1_ptr + offsets, mask=mask, other=0.0)

        tl.store(out_sum_ptr + c, tl.sum(sum_acc, axis=0))
        tl.store(out_weighted_ptr + c, tl.sum(weighted_acc, axis=0))
        tl.store(out_scaled_sum_ptr + c, tl.sum(scaled_acc, axis=0))
        tl.store(out_bucket_ptr + c, tl.sum(bucket0_acc, axis=0))
        tl.store(out_bucket_ptr + C + c, tl.sum(bucket1_acc, axis=0))


    @triton.jit
    def _token_scatter_kernel(
        dense_ptr,
        token_index_ptr,
        full_ptr,
        out_token_ptr,
        B: tl.constexpr,
        S: tl.constexpr,
        C: tl.constexpr,
        OUT_ROWS: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        token = tl.program_id(0)
        tile_c = tl.program_id(1)
        batch_offsets = tl.arange(0, BLOCK_B)
        cols = tile_c * BLOCK_C + tl.arange(0, BLOCK_C)
        rows = batch_offsets[:, None] * S + token
        offsets = rows * C + cols[None, :]
        mask = (batch_offsets[:, None] < B) & (cols[None, :] < C)
        values = tl.load(dense_ptr + offsets, mask=mask, other=0.0)
        token_sum = tl.sum(tl.where(mask, values, 0.0), axis=0)

        raw_index = tl.load(token_index_ptr + token).to(tl.int64)
        normalized_index = tl.where(raw_index < 0, raw_index + OUT_ROWS, raw_index)
        out_cols = cols
        out_offsets = normalized_index * C + out_cols
        full_value = tl.load(full_ptr).to(tl.float32)
        out_value = tl.where(raw_index == -1, full_value, token_sum)
        out_mask = (out_cols < C) & (normalized_index >= 0) & (normalized_index < OUT_ROWS)
        tl.atomic_add(out_token_ptr + out_offsets, out_value, sem="relaxed", mask=out_mask)


def _torch_fallback(inputs):
    (
        mm_714,
        mul_516,
        mm_720,
        mm_722,
        arg585_1,
        arg3_1,
        full_1,
        arg583_1,
        arg1_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3
    del _shape_param_4, _shape_param_5, _shape_param_6
    dense = mul_516 + mm_714.view(BATCH, SEQ, HIDDEN)
    dense = dense + mm_720.view(BATCH, SEQ, HIDDEN)
    dense = dense + mm_722.view(BATCH, SEQ, HIDDEN)
    out0 = dense.sum([0, 1])
    out1 = (dense * arg585_1).sum([0, 1])
    scaled = dense * arg3_1
    out2 = torch.zeros((SCATTER_ROWS, HIDDEN), device=mm_714.device, dtype=torch.float32)
    out2 = torch.ops.aten.index_put.default(out2, [arg583_1], scaled, True)
    token_slice = arg1_1[:, :SEQ]
    token_values = scaled.sum([0], keepdim=True)
    token_values = torch.where((token_slice == -1).unsqueeze(-1), full_1, token_values)
    out3 = torch.zeros((TOKEN_ROWS, HIDDEN), device=mm_714.device, dtype=torch.float32)
    out3 = torch.ops.aten.index_put.default(out3, [token_slice], token_values, True)
    dense_2d = scaled.view(ROWS, HIDDEN)
    out4 = dense_2d.permute(1, 0)
    out5 = dense_2d.sum([0])
    return out0, out1, out2, out3, out4, out5


def _check_shape(name, tensor, shape, dtype):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} expected shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} expected dtype {dtype}, got {tensor.dtype}")


def _validate_inputs(inputs):
    if len(inputs) != 16:
        raise ValueError(f"expected 16 inputs, got {len(inputs)}")
    (
        mm_714,
        mul_516,
        mm_720,
        mm_722,
        arg585_1,
        arg3_1,
        full_1,
        arg583_1,
        arg1_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3
    del _shape_param_4, _shape_param_5, _shape_param_6
    _check_shape("mm_714", mm_714, (ROWS, HIDDEN), torch.float32)
    _check_shape("mul_516", mul_516, (BATCH, SEQ, HIDDEN), torch.float32)
    _check_shape("mm_720", mm_720, (ROWS, HIDDEN), torch.float32)
    _check_shape("mm_722", mm_722, (ROWS, HIDDEN), torch.float32)
    _check_shape("arg585_1", arg585_1, (BATCH, SEQ, HIDDEN), torch.float32)
    _check_shape("arg3_1", arg3_1, (HIDDEN,), torch.float32)
    _check_shape("full_1", full_1, (), torch.float32)
    _check_shape("arg583_1", arg583_1, (BATCH, SEQ), torch.int64)
    _check_shape("arg1_1", arg1_1, (1, TOKEN_ROWS), torch.int64)
    if not mm_714.is_cuda:
        return None
    return mm_714, mul_516, mm_720, mm_722, arg585_1, arg3_1, full_1, arg583_1, arg1_1


def oracle_mobilebert_scatter_reduce(
    mm_714,
    mul_516,
    mm_720,
    mm_722,
    arg585_1,
    arg3_1,
    full_1,
    arg583_1,
    arg1_1,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
):
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3
    del _shape_param_4, _shape_param_5, _shape_param_6
    return _oracle_cuda(
        mm_714,
        mul_516,
        mm_720,
        mm_722,
        arg585_1,
        arg3_1,
        full_1,
        arg583_1,
        arg1_1,
    )


def _oracle_cuda(
    mm_714,
    mul_516,
    mm_720,
    mm_722,
    arg585_1,
    arg3_1,
    full_1,
    arg583_1,
    arg1_1,
):
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    device = mm_714.device
    out_sum = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_weighted = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_bucket = torch.empty((SCATTER_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_token = torch.empty((TOKEN_ROWS, HIDDEN), device=device, dtype=torch.float32)
    dense = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_scaled_sum = torch.empty((HIDDEN,), device=device, dtype=torch.float32)

    block_n = 64
    block_c = 32
    n_tiles = triton.cdiv(ROWS, block_n)
    partial_shape = (n_tiles, HIDDEN)
    partial_sum = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_weighted = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_scaled = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_bucket0 = torch.empty(partial_shape, device=device, dtype=torch.float32)
    partial_bucket1 = torch.empty(partial_shape, device=device, dtype=torch.float32)

    _dense_partial_kernel[(n_tiles, triton.cdiv(HIDDEN, block_c))](
        mm_714,
        mul_516,
        mm_720,
        mm_722,
        arg585_1,
        arg3_1,
        arg583_1,
        dense,
        partial_sum,
        partial_weighted,
        partial_scaled,
        partial_bucket0,
        partial_bucket1,
        N=ROWS,
        C=HIDDEN,
        SCATTER_N=SCATTER_ROWS,
        BLOCK_N=block_n,
        BLOCK_C=block_c,
        num_warps=4,
    )

    _finalize_partial_kernel[(HIDDEN,)](
        partial_sum,
        partial_weighted,
        partial_scaled,
        partial_bucket0,
        partial_bucket1,
        out_sum,
        out_weighted,
        out_bucket,
        out_scaled_sum,
        N_TILES=n_tiles,
        C=HIDDEN,
        BLOCK_TILES=triton.next_power_of_2(n_tiles),
        num_warps=8,
    )

    zero_block = 1024
    _zero_kernel[(triton.cdiv(TOKEN_ROWS * HIDDEN, zero_block),)](
        out_token,
        total=TOKEN_ROWS * HIDDEN,
        BLOCK=zero_block,
        num_warps=4,
    )
    _token_scatter_kernel[(SEQ, triton.cdiv(HIDDEN, 16))](
        dense,
        arg1_1,
        full_1,
        out_token,
        B=BATCH,
        S=SEQ,
        C=HIDDEN,
        OUT_ROWS=TOKEN_ROWS,
        BLOCK_B=triton.next_power_of_2(BATCH),
        BLOCK_C=16,
        num_warps=8,
    )

    return out_sum, out_weighted, out_bucket, out_token, dense.permute(1, 0), out_scaled_sum


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    validated = _validate_inputs(inputs)
    if validated is None:
        return _torch_fallback(inputs)
    return oracle_mobilebert_scatter_reduce(*inputs)


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
