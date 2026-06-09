"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Qwen3 bf16 query/key projection epilogue returned by Repro.forward, including query RMSNorm over `mm.view([4,512,16,128])`, generated RoPE cos/sin from `arg2_1` and positions, query rotate-half application with the returned non-contiguous `[4,16,512,128]` layout, key RMSNorm over `mm_1.view([4,512,8,128])`, key RoPE, grouped KV expand/clone/view into contiguous `[4,16,512,128]`, and the adjacent-position bool side output, whereas Inductor currently lowers the decomposed reductions, RoPE table construction, rotate-half cats, layout transforms, and KV repeat materialization as generic scheduled regions; Inductor cannot do this today because its scheduler/pattern library does not recognize a Qwen RMSNorm-plus-RoPE projection epilogue with grouped-query KV expansion as one full-scope semantic lowering; the fix is NEW_PATTERN: add a guarded Qwen attention-projection epilogue template that fuses fixed-hidden RMSNorm, rotary embedding application, output layout writes, and repeat-kv materialization while preserving side-output metadata."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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

BATCH = 4
SEQ = 512
Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
ROPE_DIM = 64
EPS = 1.0e-6
Q_ROWS = BATCH * SEQ * Q_HEADS
KV_ROWS = BATCH * SEQ * KV_HEADS
Q_OUT_SHAPE = (BATCH, Q_HEADS, SEQ, HEAD_DIM)
Q_OUT_STRIDE = (Q_HEADS * SEQ * HEAD_DIM, HEAD_DIM, Q_HEADS * HEAD_DIM, 1)
KV_OUT_SHAPE = (BATCH, Q_HEADS, SEQ, HEAD_DIM)
NE_SHAPE = (BATCH, SEQ)

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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 16:
        raise ValueError(f"{REPRO_ID} expects 16 inputs, got {len(inputs)}")

    mm, arg5_1, arg2_1, mm_1, arg7_1 = inputs[:5]
    tensor_inputs = (mm, arg5_1, arg2_1, mm_1, arg7_1)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected = (
        ((2048, 2048), torch.bfloat16),
        ((HEAD_DIM,), torch.bfloat16),
        ((ROPE_DIM,), torch.float32),
        ((2048, 1024), torch.bfloat16),
        ((HEAD_DIM,), torch.bfloat16),
    )
    for index, (value, (shape, dtype)) in enumerate(zip(tensor_inputs, expected)):
        if tuple(value.shape) != shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {shape}")
        if value.dtype != dtype:
            raise TypeError(f"input {index} dtype {value.dtype} != {dtype}")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for this Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    device = mm.device
    if any(value.device != device for value in tensor_inputs[1:]):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    expected_shapes = (
        (BATCH, SEQ, Q_HEADS * HEAD_DIM),
        (BATCH, SEQ, -1, HEAD_DIM),
        (1, ROPE_DIM, 1),
        (1, 1, SEQ),
        (1, SEQ, 2, ROPE_DIM),
        (1, SEQ, HEAD_DIM),
        (BATCH, SEQ, KV_HEADS * HEAD_DIM),
        (BATCH, SEQ, -1, HEAD_DIM),
        (BATCH, KV_HEADS, 2, SEQ, HEAD_DIM),
        KV_OUT_SHAPE,
        (BATCH, -1),
    )
    for index, (value, expected_shape) in enumerate(zip(inputs[5:], expected_shapes), start=5):
        if _shape_tuple(value) != expected_shape:
            raise ValueError(
                f"shape parameter {index} value {_shape_tuple(value)} != {expected_shape}"
            )

    return mm, arg5_1, arg2_1, mm_1, arg7_1


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _round_bf16(x):
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _qwen_query_rmsnorm_rope_kernel(
        mm_ptr,
        weight_ptr,
        inv_freq_ptr,
        out_ptr,
        ne_ptr,
        N_ROWS: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        ROPE_D: tl.constexpr,
        EPS_VALUE: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_D)
        row_mask = rows < N_ROWS
        mask = row_mask[:, None]

        offsets = rows[:, None] * BLOCK_D + cols[None, :]
        x = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=1)
        inv_rms = tl.rsqrt(sum_sq * (1.0 / BLOCK_D) + EPS_VALUE)

        rot_cols = tl.where(cols < ROPE_D, cols + ROPE_D, cols - ROPE_D)
        rot_offsets = rows[:, None] * BLOCK_D + rot_cols[None, :]
        x_rot = tl.load(mm_ptr + rot_offsets, mask=mask, other=0.0).to(tl.float32)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        rot_weight = tl.load(weight_ptr + rot_cols).to(tl.float32)
        q = _round_bf16(_round_bf16(x * inv_rms[:, None]) * weight[None, :])
        q_rot = _round_bf16(_round_bf16(x_rot * inv_rms[:, None]) * rot_weight[None, :])
        q_rot = q_rot * tl.where(cols < ROPE_D, -1.0, 1.0)[None, :]

        pos = (rows // NUM_HEADS) - ((rows // (SEQ_LEN * NUM_HEADS)) * SEQ_LEN)
        freq = tl.load(inv_freq_ptr + (cols % ROPE_D)).to(tl.float32)
        theta = pos[:, None].to(tl.float32) * freq[None, :]
        cos_theta = _round_bf16(tl.cos(theta))
        sin_theta = _round_bf16(tl.sin(theta))

        out = _round_bf16(q * cos_theta) + _round_bf16(q_rot * sin_theta)
        tl.store(out_ptr + offsets, out, mask=mask)

        head = rows - (rows // NUM_HEADS) * NUM_HEADS
        batch = rows // (SEQ_LEN * NUM_HEADS)
        false_value = rows != rows
        tl.store(ne_ptr + batch * SEQ_LEN + pos, false_value, mask=row_mask & (head == 0))

    @triton.jit
    def _qwen_key_rmsnorm_rope_repeat_kernel(
        mm_ptr,
        weight_ptr,
        inv_freq_ptr,
        out_ptr,
        N_ROWS: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        OUT_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        ROPE_D: tl.constexpr,
        EPS_VALUE: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_D)
        row_mask = rows < N_ROWS
        mask = row_mask[:, None]

        offsets = rows[:, None] * BLOCK_D + cols[None, :]
        x = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=1)
        inv_rms = tl.rsqrt(sum_sq * (1.0 / BLOCK_D) + EPS_VALUE)

        rot_cols = tl.where(cols < ROPE_D, cols + ROPE_D, cols - ROPE_D)
        rot_offsets = rows[:, None] * BLOCK_D + rot_cols[None, :]
        x_rot = tl.load(mm_ptr + rot_offsets, mask=mask, other=0.0).to(tl.float32)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        rot_weight = tl.load(weight_ptr + rot_cols).to(tl.float32)
        k = _round_bf16(_round_bf16(x * inv_rms[:, None]) * weight[None, :])
        k_rot = _round_bf16(_round_bf16(x_rot * inv_rms[:, None]) * rot_weight[None, :])
        k_rot = k_rot * tl.where(cols < ROPE_D, -1.0, 1.0)[None, :]

        pos = (rows // NUM_HEADS) - ((rows // (SEQ_LEN * NUM_HEADS)) * SEQ_LEN)
        freq = tl.load(inv_freq_ptr + (cols % ROPE_D)).to(tl.float32)
        theta = pos[:, None].to(tl.float32) * freq[None, :]
        cos_theta = _round_bf16(tl.cos(theta))
        sin_theta = _round_bf16(tl.sin(theta))
        out = _round_bf16(k * cos_theta) + _round_bf16(k_rot * sin_theta)

        kv_head = rows - (rows // NUM_HEADS) * NUM_HEADS
        batch = rows // (SEQ_LEN * NUM_HEADS)
        out_base = (
            batch[:, None] * (OUT_HEADS * SEQ_LEN * BLOCK_D)
            + (kv_head[:, None] * 2) * (SEQ_LEN * BLOCK_D)
            + pos[:, None] * BLOCK_D
            + cols[None, :]
        )
        tl.store(out_ptr + out_base, out, mask=mask)
        tl.store(out_ptr + out_base + (SEQ_LEN * BLOCK_D), out, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([2048, 2048], bf16), T([128], bf16), T([64], f32), T([2048, 1024], bf16), T([128], bf16), S([4, 512, 2048]), S([4, 512, -1, 128]), S([1, 64, 1]), S([1, 1, 512]), S([1, 512, 2, 64]), S([1, 512, 128]), S([4, 512, 1024]), S([4, 512, -1, 128]), S([4, 8, 2, 512, 128]), S([4, 16, 512, 128]), S([4, -1]))")
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
    if triton is None:
        raise RuntimeError("Triton is required for oracle_qwen_rope_rmsnorm.py")

    mm, arg5_1, arg2_1, mm_1, arg7_1 = _validate_inputs(inputs)

    query_rope = torch.empty_strided(
        Q_OUT_SHAPE,
        Q_OUT_STRIDE,
        device=mm.device,
        dtype=torch.bfloat16,
    )
    key_rope_expanded = torch.empty(
        KV_OUT_SHAPE,
        device=mm.device,
        dtype=torch.bfloat16,
    )
    ne_scalar = torch.empty(NE_SHAPE, device=mm.device, dtype=torch.bool)

    block_rows = 2
    _qwen_query_rmsnorm_rope_kernel[(triton.cdiv(Q_ROWS, block_rows),)](
        mm,
        arg5_1,
        arg2_1,
        query_rope,
        ne_scalar,
        N_ROWS=Q_ROWS,
        NUM_HEADS=Q_HEADS,
        SEQ_LEN=SEQ,
        ROPE_D=ROPE_DIM,
        EPS_VALUE=EPS,
        BLOCK_ROWS=block_rows,
        BLOCK_D=HEAD_DIM,
        num_warps=4,
        num_stages=3,
    )
    _qwen_key_rmsnorm_rope_repeat_kernel[(triton.cdiv(KV_ROWS, block_rows),)](
        mm_1,
        arg7_1,
        arg2_1,
        key_rope_expanded,
        N_ROWS=KV_ROWS,
        NUM_HEADS=KV_HEADS,
        OUT_HEADS=Q_HEADS,
        SEQ_LEN=SEQ,
        ROPE_D=ROPE_DIM,
        EPS_VALUE=EPS,
        BLOCK_ROWS=block_rows,
        BLOCK_D=HEAD_DIM,
        num_warps=4,
        num_stages=3,
    )
    return (query_rope, key_rope_expanded, ne_scalar)


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
