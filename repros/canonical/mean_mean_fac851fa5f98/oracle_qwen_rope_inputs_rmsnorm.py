"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Qwen bf16 query/key RMSNorm plus precomputed-RoPE scope returned by Repro.forward, including query RMSNorm over `mm_189.view([B,S,Hq,128])`, rotary-half cos/sin application into the returned non-contiguous `[B,Hq,S,128]` query layout, key RMSNorm over `mm_190.view([B,S,Hkv,128])`, key RoPE, and grouped-KV expand/clone/view materialization directly into contiguous `[B,Hq,S,128]`, whereas Inductor currently emits separate generic reduction kernels plus a pointwise expand/clone kernel for the repeated key/value heads; Inductor cannot do this today because the reduction scheduler does not sink this grouped-KV repeat materialization into the RMSNorm/RoPE producer epilogue while preserving the query and key output layout contracts; the fix is SCHEDULER_FUSION: allow fixed-hidden normalization epilogues to write repeated grouped-query attention layouts directly instead of scheduling a later clone kernel."""
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

HEAD_DIM = 128
HALF_DIM = HEAD_DIM // 2
EPS = 1.0e-6

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    mm_q, q_weight, rope_cos, rope_sin, mm_k, k_weight = inputs[:6]
    tensor_inputs = (mm_q, q_weight, rope_cos, rope_sin, mm_k, k_weight)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first six repro inputs must be tensors")

    for index, value in enumerate(tensor_inputs):
        if value.dtype != torch.bfloat16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.bfloat16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for this Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    device = mm_q.device
    if any(value.device != device for value in tensor_inputs[1:]):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    if q_weight.shape != (HEAD_DIM,) or k_weight.shape != (HEAD_DIM,):
        raise ValueError("RMSNorm weights must both have shape [128]")
    if rope_cos.shape != rope_sin.shape or tuple(rope_cos.shape) != (1, rope_cos.shape[1], HEAD_DIM):
        raise ValueError("RoPE cos/sin inputs must both have shape [1, seq, 128]")
    if mm_q.ndim != 2 or mm_k.ndim != 2:
        raise ValueError("projection inputs must be rank-2 tensors")
    if mm_q.shape[0] != mm_k.shape[0]:
        raise ValueError("query and key projection inputs must have the same row count")
    if mm_q.shape[1] % HEAD_DIM != 0 or mm_k.shape[1] % HEAD_DIM != 0:
        raise ValueError("projection hidden sizes must be divisible by 128")

    seq_len = int(rope_cos.shape[1])
    if seq_len <= 0 or int(mm_q.shape[0]) % seq_len != 0:
        raise ValueError("projection rows must be an integer multiple of sequence length")
    batch = int(mm_q.shape[0]) // seq_len
    q_heads = int(mm_q.shape[1]) // HEAD_DIM
    kv_heads = int(mm_k.shape[1]) // HEAD_DIM
    if q_heads % kv_heads != 0:
        raise ValueError("query head count must be a multiple of key/value head count")
    repeat = q_heads // kv_heads

    expected_shapes = (
        (batch, seq_len, q_heads * HEAD_DIM),
        (batch, seq_len, -1, HEAD_DIM),
        (batch, seq_len, kv_heads * HEAD_DIM),
        (batch, seq_len, -1, HEAD_DIM),
        (batch, kv_heads, repeat, seq_len, HEAD_DIM),
        (batch, q_heads, seq_len, HEAD_DIM),
    )
    for index, (value, expected_shape) in enumerate(zip(inputs[6:], expected_shapes), start=6):
        if _shape_tuple(value) != expected_shape:
            raise ValueError(
                f"shape parameter {index} value {_shape_tuple(value)} != {expected_shape}"
            )

    return mm_q, q_weight, rope_cos, rope_sin, mm_k, k_weight, batch, seq_len, q_heads, kv_heads


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _round_bf16(x):
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _query_rmsnorm_rope_kernel(
        mm_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        N_ROWS: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        EPS_VALUE: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_D)
        row_mask = rows < N_ROWS
        mask = row_mask[:, None]

        base = rows[:, None] * BLOCK_D + cols[None, :]
        x = tl.load(mm_ptr + base, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=1)
        inv_rms = tl.rsqrt(sum_sq * (1.0 / BLOCK_D) + EPS_VALUE)

        rot_cols = tl.where(cols < BLOCK_HALF, cols + BLOCK_HALF, cols - BLOCK_HALF)
        x_rot = tl.load(
            mm_ptr + rows[:, None] * BLOCK_D + rot_cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        rot_weight = tl.load(weight_ptr + rot_cols).to(tl.float32)
        q = _round_bf16(_round_bf16(x * inv_rms[:, None]) * weight[None, :])
        q_rot = _round_bf16(_round_bf16(x_rot * inv_rms[:, None]) * rot_weight[None, :])
        q_rot = q_rot * tl.where(cols < BLOCK_HALF, -1.0, 1.0)[None, :]

        pos = (rows // NUM_HEADS) - ((rows // (SEQ_LEN * NUM_HEADS)) * SEQ_LEN)
        rope_base = pos[:, None] * BLOCK_D + cols[None, :]
        cos_value = tl.load(cos_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)
        sin_value = tl.load(sin_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)

        out = _round_bf16(q * cos_value) + _round_bf16(q_rot * sin_value)
        tl.store(out_ptr + base, out, mask=mask)

    @triton.jit
    def _key_rmsnorm_rope_repeat_kernel(
        mm_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        N_ROWS: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        OUT_HEADS: tl.constexpr,
        REPEAT: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        EPS_VALUE: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_D)
        row_mask = rows < N_ROWS
        mask = row_mask[:, None]

        base = rows[:, None] * BLOCK_D + cols[None, :]
        x = tl.load(mm_ptr + base, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=1)
        inv_rms = tl.rsqrt(sum_sq * (1.0 / BLOCK_D) + EPS_VALUE)

        rot_cols = tl.where(cols < BLOCK_HALF, cols + BLOCK_HALF, cols - BLOCK_HALF)
        x_rot = tl.load(
            mm_ptr + rows[:, None] * BLOCK_D + rot_cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        rot_weight = tl.load(weight_ptr + rot_cols).to(tl.float32)
        k = _round_bf16(_round_bf16(x * inv_rms[:, None]) * weight[None, :])
        k_rot = _round_bf16(_round_bf16(x_rot * inv_rms[:, None]) * rot_weight[None, :])
        k_rot = k_rot * tl.where(cols < BLOCK_HALF, -1.0, 1.0)[None, :]

        pos = (rows // NUM_HEADS) - ((rows // (SEQ_LEN * NUM_HEADS)) * SEQ_LEN)
        rope_base = pos[:, None] * BLOCK_D + cols[None, :]
        cos_value = tl.load(cos_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)
        sin_value = tl.load(sin_ptr + rope_base, mask=mask, other=0.0).to(tl.float32)

        out = _round_bf16(k * cos_value) + _round_bf16(k_rot * sin_value)

        kv_head = rows - (rows // NUM_HEADS) * NUM_HEADS
        batch = rows // (SEQ_LEN * NUM_HEADS)
        for repeat_idx in tl.static_range(0, REPEAT):
            out_head = kv_head * REPEAT + repeat_idx
            out_base = (
                batch[:, None] * (OUT_HEADS * SEQ_LEN * BLOCK_D)
                + out_head[:, None] * (SEQ_LEN * BLOCK_D)
                + pos[:, None] * BLOCK_D
                + cols[None, :]
            )
            tl.store(out_ptr + out_base, out, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([2048, 2048], bf16), T([128], bf16), T([1, 512, 128], bf16), T([1, 512, 128], bf16), T([2048, 1024], bf16), T([128], bf16), S([4, 512, 2048]), S([4, 512, -1, 128]), S([4, 512, 1024]), S([4, 512, -1, 128]), S([4, 8, 2, 512, 128]), S([4, 16, 512, 128]))")
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
        raise RuntimeError("Triton is required for oracle_qwen_rope_inputs_rmsnorm.py")

    (
        mm_q,
        q_weight,
        rope_cos,
        rope_sin,
        mm_k,
        k_weight,
        batch,
        seq_len,
        q_heads,
        kv_heads,
    ) = _validate_inputs(inputs)

    repeat = q_heads // kv_heads
    query_rows = batch * seq_len * q_heads
    key_rows = batch * seq_len * kv_heads
    query_shape = (batch, q_heads, seq_len, HEAD_DIM)
    query_stride = (q_heads * seq_len * HEAD_DIM, HEAD_DIM, q_heads * HEAD_DIM, 1)

    query_rope = torch.empty_strided(
        query_shape,
        query_stride,
        device=mm_q.device,
        dtype=torch.bfloat16,
    )
    key_rope_expanded = torch.empty(
        query_shape,
        device=mm_q.device,
        dtype=torch.bfloat16,
    )

    block_rows = 4
    _query_rmsnorm_rope_kernel[(triton.cdiv(query_rows, block_rows),)](
        mm_q,
        q_weight,
        rope_cos,
        rope_sin,
        query_rope,
        N_ROWS=query_rows,
        NUM_HEADS=q_heads,
        SEQ_LEN=seq_len,
        EPS_VALUE=EPS,
        BLOCK_ROWS=block_rows,
        BLOCK_HALF=HALF_DIM,
        BLOCK_D=HEAD_DIM,
        num_warps=4,
        num_stages=3,
    )
    _key_rmsnorm_rope_repeat_kernel[(triton.cdiv(key_rows, block_rows),)](
        mm_k,
        k_weight,
        rope_cos,
        rope_sin,
        key_rope_expanded,
        N_ROWS=key_rows,
        NUM_HEADS=kv_heads,
        OUT_HEADS=q_heads,
        REPEAT=repeat,
        SEQ_LEN=seq_len,
        EPS_VALUE=EPS,
        BLOCK_ROWS=block_rows,
        BLOCK_HALF=HALF_DIM,
        BLOCK_D=HEAD_DIM,
        num_warps=4,
        num_stages=3,
    )
    return (query_rope, key_rope_expanded)


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
    parser.add_argument("--atol", type=float, default=3e-2,
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
