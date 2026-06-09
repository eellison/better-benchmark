"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete DeBERTa dropout/add/layernorm-backward row scope, including both channel reductions, the [512,1536] accumulated index_put output, and the [128100,1536] base-plus-scatter output by cloning the live base and scattering updates directly, whereas Inductor currently materializes zero-filled scatter buffers and schedules the sibling reductions, pointwise normalization backward, index_puts, and final base add as separate generic kernels; Inductor cannot do this today because scheduler/codegen cannot represent these index_put(accumulate=True) producers as scatter-reduce nodes with reusable row reductions and side-output stores; the fix is SCATTER_REDUCE: add an index_put scatter-reduce lowering that fuses compatible row/channel reductions, sentinel where handling, and direct base-update epilogues."""
from __future__ import annotations

import argparse
import math
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
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


BATCH = 8
TOKENS = 512
HIDDEN = 1536
ROWS = BATCH * TOKENS
VOCAB_ROWS = 128100
DROPOUT_SCALE = 1.1111111111111112
INV_HIDDEN = 1.0 / HIDDEN

ROW_BLOCK_C = 2048
BLOCK_M = 32
BLOCK_C = 64
NUM_ROW_BLOCKS = math.ceil(ROWS / BLOCK_M)
FINAL_BLOCK_M = 128


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _row_stats_kernel(
        mm_286_ptr,
        mul_635_ptr,
        mm_288_ptr,
        mm_290_ptr,
        arg203_ptr,
        arg3_ptr,
        arg199_ptr,
        arg200_ptr,
        arg201_ptr,
        arg202_ptr,
        row_sum0_ptr,
        row_sum1_ptr,
        HIDDEN_: tl.constexpr,
        TOKENS_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        row = tl.program_id(0)
        channels = tl.arange(0, BLOCK_C_)
        active = channels < HIDDEN_
        token = row % TOKENS_
        offsets = row * HIDDEN_ + channels
        token_offsets = token * HIDDEN_ + channels

        summed = tl.load(mul_635_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        summed += tl.load(mm_286_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        summed += tl.load(mm_288_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        summed += tl.load(mm_290_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        keep = tl.load(arg203_ptr + offsets, mask=active, other=0).to(tl.float32)
        mul1 = summed * keep * DROPOUT_SCALE_
        gamma = tl.load(arg3_ptr + channels, mask=active, other=0.0).to(tl.float32)
        mul2 = mul1 * gamma

        arg199 = tl.load(arg199_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        arg200 = tl.load(arg200_ptr + token_offsets, mask=active, other=0.0).to(tl.float32)
        arg201 = tl.load(arg201_ptr + row).to(tl.float32)
        arg202 = tl.load(arg202_ptr + row).to(tl.float32)
        mul4 = (arg199 + arg200 - arg201) * arg202

        sum0 = tl.sum(tl.where(active, mul2, 0.0), axis=0)
        sum1 = tl.sum(tl.where(active, mul2 * mul4, 0.0), axis=0)
        tl.store(row_sum0_ptr + row, sum0)
        tl.store(row_sum1_ptr + row, sum1)

    @triton.jit
    def _scatter_reduce_kernel(
        mm_286_ptr,
        mul_635_ptr,
        mm_288_ptr,
        mm_290_ptr,
        arg203_ptr,
        arg3_ptr,
        arg199_ptr,
        arg200_ptr,
        arg201_ptr,
        arg202_ptr,
        arg1_ptr,
        full_ptr,
        arg0_ptr,
        row_sum0_ptr,
        row_sum1_ptr,
        out2_ptr,
        out3_ptr,
        partial0_ptr,
        partial1_ptr,
        HIDDEN_: tl.constexpr,
        TOKENS_: tl.constexpr,
        ROWS_: tl.constexpr,
        VOCAB_ROWS_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        INV_HIDDEN_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        rows = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        channels = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (rows[:, None] < ROWS_) & (channels[None, :] < HIDDEN_)
        row_active = rows < ROWS_
        token = rows - (rows // TOKENS_) * TOKENS_
        offsets = rows[:, None] * HIDDEN_ + channels[None, :]
        token_offsets = token[:, None] * HIDDEN_ + channels[None, :]

        summed = tl.load(mul_635_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        summed += tl.load(mm_286_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        summed += tl.load(mm_288_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        summed += tl.load(mm_290_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        keep = tl.load(arg203_ptr + offsets, mask=active, other=0).to(tl.float32)
        mul1 = summed * keep * DROPOUT_SCALE_
        gamma = tl.load(arg3_ptr + channels, mask=channels < HIDDEN_, other=0.0).to(tl.float32)
        mul2 = mul1 * gamma[None, :]

        arg199 = tl.load(arg199_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        arg200 = tl.load(arg200_ptr + token_offsets, mask=active, other=0.0).to(tl.float32)
        arg201 = tl.load(arg201_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
        arg202 = tl.load(arg202_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
        mul4 = (arg199 + arg200 - arg201[:, None]) * arg202[:, None]

        s0 = tl.load(row_sum0_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
        s1 = tl.load(row_sum1_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
        sub = (mul2 * HIDDEN_) - s0[:, None]
        sub = sub - mul4 * s1[:, None]
        mul7 = (arg202[:, None] * INV_HIDDEN_) * sub
        mul7 = tl.where(active, mul7, 0.0)

        target0_raw = tl.load(arg0_ptr + rows, mask=row_active, other=0).to(tl.int64)
        target0 = tl.where(target0_raw < 0, target0_raw + VOCAB_ROWS_, target0_raw)
        full = tl.load(full_ptr).to(tl.float32)
        out3_value = tl.where(target0_raw[:, None] == 0, full, mul7)
        out3_offsets = target0[:, None] * HIDDEN_ + channels[None, :]
        out3_live = active & (target0[:, None] >= 0) & (target0[:, None] < VOCAB_ROWS_)
        tl.atomic_add(out3_ptr + out3_offsets, out3_value, sem="relaxed", mask=out3_live)

        target1_raw = tl.load(arg1_ptr + token, mask=row_active, other=0).to(tl.int64)
        target1 = tl.where(target1_raw < 0, target1_raw + TOKENS_, target1_raw)
        batch = rows // TOKENS_
        out2_value = tl.where(
            target1_raw[:, None] == -1,
            tl.where(batch[:, None] == 0, full, 0.0),
            mul7,
        )
        out2_offsets = target1[:, None] * HIDDEN_ + channels[None, :]
        out2_live = active & (target1[:, None] >= 0) & (target1[:, None] < TOKENS_)
        tl.atomic_add(out2_ptr + out2_offsets, out2_value, sem="relaxed", mask=out2_live)

        sum0 = tl.sum(tl.where(active, mul1 * mul4, 0.0), axis=0)
        sum1 = tl.sum(tl.where(active, mul1, 0.0), axis=0)
        partial_offsets = pid_m * HIDDEN_ + channels
        channel_active = channels < HIDDEN_
        tl.store(partial0_ptr + partial_offsets, sum0, mask=channel_active)
        tl.store(partial1_ptr + partial_offsets, sum1, mask=channel_active)

    @triton.jit
    def _finalize_reductions_kernel(
        partial0_ptr,
        partial1_ptr,
        out0_ptr,
        out1_ptr,
        HIDDEN_: tl.constexpr,
        NUM_ROW_BLOCKS_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        channels = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        row_blocks = tl.arange(0, BLOCK_M_)
        active = (row_blocks[:, None] < NUM_ROW_BLOCKS_) & (channels[None, :] < HIDDEN_)
        offsets = row_blocks[:, None] * HIDDEN_ + channels[None, :]
        vals0 = tl.load(partial0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        vals1 = tl.load(partial1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        out0 = tl.sum(vals0, axis=0)
        out1 = tl.sum(vals1, axis=0)
        tl.store(out0_ptr + channels, out0, mask=channels < HIDDEN_)
        tl.store(out1_ptr + channels, out1, mask=channels < HIDDEN_)


def _check_shapes(
    mm_286: torch.Tensor,
    mul_635: torch.Tensor,
    mm_288: torch.Tensor,
    mm_290: torch.Tensor,
    arg203_1: torch.Tensor,
    arg3_1: torch.Tensor,
    arg199_1: torch.Tensor,
    arg200_1: torch.Tensor,
    arg201_1: torch.Tensor,
    arg202_1: torch.Tensor,
    arg1_1: torch.Tensor,
    full_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm_1: torch.Tensor,
) -> None:
    expected = {
        "mm_286": ((ROWS, HIDDEN), torch.float32),
        "mul_635": ((BATCH, TOKENS, HIDDEN), torch.float32),
        "mm_288": ((ROWS, HIDDEN), torch.float32),
        "mm_290": ((ROWS, HIDDEN), torch.float32),
        "arg203_1": ((BATCH, TOKENS, HIDDEN), torch.bool),
        "arg3_1": ((HIDDEN,), torch.float32),
        "arg199_1": ((BATCH, TOKENS, HIDDEN), torch.float32),
        "arg200_1": ((1, TOKENS, HIDDEN), torch.float32),
        "arg201_1": ((BATCH, TOKENS, 1), torch.float32),
        "arg202_1": ((BATCH, TOKENS, 1), torch.float32),
        "arg1_1": ((1, TOKENS), torch.int64),
        "full_1": ((), torch.float32),
        "arg0_1": ((BATCH, TOKENS), torch.int64),
        "mm_1": ((VOCAB_ROWS, HIDDEN), torch.float32),
    }
    values = {
        "mm_286": mm_286,
        "mul_635": mul_635,
        "mm_288": mm_288,
        "mm_290": mm_290,
        "arg203_1": arg203_1,
        "arg3_1": arg3_1,
        "arg199_1": arg199_1,
        "arg200_1": arg200_1,
        "arg201_1": arg201_1,
        "arg202_1": arg202_1,
        "arg1_1": arg1_1,
        "full_1": full_1,
        "arg0_1": arg0_1,
        "mm_1": mm_1,
    }
    for name, tensor in values.items():
        shape, dtype = expected[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} has dtype {tensor.dtype}, expected {dtype}")
        if tensor.device.type != "cuda":
            raise RuntimeError("triton oracle requires CUDA inputs")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for this captured layout")


def oracle_scatter_reduce_full_scope(
    mm_286: torch.Tensor,
    mul_635: torch.Tensor,
    mm_288: torch.Tensor,
    mm_290: torch.Tensor,
    arg203_1: torch.Tensor,
    arg3_1: torch.Tensor,
    arg199_1: torch.Tensor,
    arg200_1: torch.Tensor,
    arg201_1: torch.Tensor,
    arg202_1: torch.Tensor,
    arg1_1: torch.Tensor,
    full_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2
    if triton is None:
        raise RuntimeError("triton is not available")
    _check_shapes(
        mm_286,
        mul_635,
        mm_288,
        mm_290,
        arg203_1,
        arg3_1,
        arg199_1,
        arg200_1,
        arg201_1,
        arg202_1,
        arg1_1,
        full_1,
        arg0_1,
        mm_1,
    )

    row_sum0 = torch.empty((ROWS,), device=mm_286.device, dtype=torch.float32)
    row_sum1 = torch.empty_like(row_sum0)
    partial0 = torch.empty((NUM_ROW_BLOCKS, HIDDEN), device=mm_286.device, dtype=torch.float32)
    partial1 = torch.empty_like(partial0)
    out0 = torch.empty((HIDDEN,), device=mm_286.device, dtype=torch.float32)
    out1 = torch.empty_like(out0)
    out2 = torch.zeros((TOKENS, HIDDEN), device=mm_286.device, dtype=torch.float32)
    out3 = mm_1.clone()

    _row_stats_kernel[(ROWS,)](
        mm_286,
        mul_635,
        mm_288,
        mm_290,
        arg203_1,
        arg3_1,
        arg199_1,
        arg200_1,
        arg201_1,
        arg202_1,
        row_sum0,
        row_sum1,
        HIDDEN_=HIDDEN,
        TOKENS_=TOKENS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_C_=ROW_BLOCK_C,
        num_warps=8,
    )
    _scatter_reduce_kernel[(triton.cdiv(HIDDEN, BLOCK_C), NUM_ROW_BLOCKS)](
        mm_286,
        mul_635,
        mm_288,
        mm_290,
        arg203_1,
        arg3_1,
        arg199_1,
        arg200_1,
        arg201_1,
        arg202_1,
        arg1_1,
        full_1,
        arg0_1,
        row_sum0,
        row_sum1,
        out2,
        out3,
        partial0,
        partial1,
        HIDDEN_=HIDDEN,
        TOKENS_=TOKENS,
        ROWS_=ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        INV_HIDDEN_=INV_HIDDEN,
        BLOCK_M_=BLOCK_M,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
    )
    _finalize_reductions_kernel[(triton.cdiv(HIDDEN, BLOCK_C),)](
        partial0,
        partial1,
        out0,
        out1,
        HIDDEN_=HIDDEN,
        NUM_ROW_BLOCKS_=NUM_ROW_BLOCKS,
        BLOCK_C_=BLOCK_C,
        BLOCK_M_=FINAL_BLOCK_M,
        num_warps=4,
    )
    return out0, out1, out2, out3


@oracle_impl(hardware="H100", shapes="(T([4096, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], b8), T([1536], f32), T([8, 512, 1536], f32), T([1, 512, 1536], f32), T([8, 512, 1], f32), T([8, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([8, 512], i64, gen=Index(128100)), T([128100, 1536], f32), S([8, 512, 1536]), S([8, 512, 1536]), S([8, 512, 1536]))")
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
    return oracle_scatter_reduce_full_scope(*inputs)


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
