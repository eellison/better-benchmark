"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle builds the generated Llama-style RoPE cos/sin table once and fuses both Q/K rotate-half consumers, the returned strided Q layout, the four-way expanded contiguous K layout, and the all-false position-difference mask, whereas Inductor currently schedules the iota/frequency table build, two duplicated RoPE pointwise branches, K expand/clone/view layout, and boolean side output as separate generic pointwise/layout kernels; Inductor cannot do this today because scheduler fusion does not keep the generated rotary table shared across sibling Q/K rotate-half consumers or sink the K head-repeat clone into the final output layout; the fix is SCHEDULER_FUSION: teach Inductor to fuse generated RoPE table construction, repeated rotate-half consumers, head-repeat materialization, and side-mask emission into one multi-output scheduler group."""
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
Q_HEADS = 32
KV_HEADS = 8
KV_GROUPS = 4
HEAD_DIM = 64
FREQ_DIM = 32
QK_NUMEL = BATCH * Q_HEADS * SEQ * HEAD_DIM
ROPE_TABLE_NUMEL = SEQ * HEAD_DIM
Q_OUT_SHAPE = (BATCH, Q_HEADS, SEQ, HEAD_DIM)
Q_OUT_STRIDE = (Q_HEADS * SEQ * HEAD_DIM, HEAD_DIM, Q_HEADS * HEAD_DIM, 1)
K_OUT_SHAPE = (BATCH, Q_HEADS, SEQ, HEAD_DIM)
K_OUT_STRIDE = (Q_HEADS * SEQ * HEAD_DIM, SEQ * HEAD_DIM, HEAD_DIM, 1)
NE_SHAPE = (BATCH, SEQ)
SHAPE_PARAMS = (
    (4, 512, 2048),
    (4, 512, -1, 64),
    (1, 32, 1),
    (1, 1, 512),
    (1, 512, 2, 32),
    (1, 512, 64),
    (4, 512, 512),
    (4, 512, -1, 64),
    (4, 8, 4, 512, 64),
    (4, 32, 512, 64),
    (4, -1),
)

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
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"expected a shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _validate_tensor(
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} {name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{REPRO_ID} {name} expects shape {shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{REPRO_ID} {name} expects {dtype}, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{REPRO_ID} {name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{REPRO_ID} {name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 14:
        raise ValueError(f"{REPRO_ID} expects 14 inputs, got {len(inputs)}")

    mm = _validate_tensor(inputs[0], (BATCH * SEQ, Q_HEADS * HEAD_DIM), torch.bfloat16, "mm")
    inv_freq = _validate_tensor(inputs[1], (FREQ_DIM,), torch.float32, "arg2_1")
    mm_1 = _validate_tensor(inputs[2], (BATCH * SEQ, KV_HEADS * HEAD_DIM), torch.bfloat16, "mm_1")
    if mm.device != inv_freq.device or mm.device != mm_1.device:
        raise ValueError(f"{REPRO_ID} tensor inputs must be on the same CUDA device")

    for index, expected in enumerate(SHAPE_PARAMS, start=3):
        actual = _shape_tuple(inputs[index])
        if actual != expected:
            raise ValueError(f"{REPRO_ID} _shape_param_{index - 3} expects {expected}, got {actual}")

    return mm, inv_freq, mm_1


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _rope_table_kernel(
        inv_freq_ptr,
        cos_ptr,
        sin_ptr,
        N: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        FREQ_DIM_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        dim = offsets % HEAD_DIM_
        pos = offsets // HEAD_DIM_
        freq = tl.load(inv_freq_ptr + (dim % FREQ_DIM_), mask=mask, other=0.0).to(tl.float32)
        theta = pos.to(tl.float32) * freq
        tl.store(cos_ptr + offsets, tl.cos(theta), mask=mask)
        tl.store(sin_ptr + offsets, tl.sin(theta), mask=mask)

    @triton.jit
    def _rope_qk_expand_kernel(
        q_in_ptr,
        k_in_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        ne_ptr,
        N_ROWS: tl.constexpr,
        SEQ_: tl.constexpr,
        Q_HEADS_: tl.constexpr,
        KV_HEADS_: tl.constexpr,
        KV_GROUPS_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        FREQ_DIM_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        dims = tl.arange(0, BLOCK_D)
        row_mask = rows < N_ROWS
        elem_mask = row_mask[:, None] & (dims[None, :] < HEAD_DIM_)

        batch = rows // (SEQ_ * Q_HEADS_)
        rem = rows - batch * (SEQ_ * Q_HEADS_)
        seq = rem // Q_HEADS_
        q_head = rem - seq * Q_HEADS_
        kv_head = q_head // KV_GROUPS_

        q_base = (
            batch[:, None] * (SEQ_ * Q_HEADS_ * HEAD_DIM_)
            + seq[:, None] * (Q_HEADS_ * HEAD_DIM_)
            + q_head[:, None] * HEAD_DIM_
        )
        k_in_base = (
            batch[:, None] * (SEQ_ * KV_HEADS_ * HEAD_DIM_)
            + seq[:, None] * (KV_HEADS_ * HEAD_DIM_)
            + kv_head[:, None] * HEAD_DIM_
        )
        k_out_base = (
            batch[:, None] * (Q_HEADS_ * SEQ_ * HEAD_DIM_)
            + q_head[:, None] * (SEQ_ * HEAD_DIM_)
            + seq[:, None] * HEAD_DIM_
        )
        table_base = seq[:, None] * HEAD_DIM_

        rot_dims = tl.where(dims < FREQ_DIM_, dims + FREQ_DIM_, dims - FREQ_DIM_)
        rot_sign = tl.where(dims < FREQ_DIM_, -1.0, 1.0)
        coeff_offsets = table_base + dims[None, :]
        cos_v = tl.load(cos_ptr + coeff_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        sin_v = tl.load(sin_ptr + coeff_offsets, mask=elem_mask, other=0.0).to(tl.float32)

        q = tl.load(q_in_ptr + q_base + dims[None, :], mask=elem_mask, other=0.0).to(tl.float32)
        q_rot = tl.load(q_in_ptr + q_base + rot_dims[None, :], mask=elem_mask, other=0.0).to(tl.float32)
        q_direct = (q * cos_v).to(tl.bfloat16).to(tl.float32)
        q_rotated = (q_rot * rot_sign[None, :] * sin_v).to(tl.bfloat16).to(tl.float32)
        q_value = q_direct + q_rotated
        tl.store(q_out_ptr + q_base + dims[None, :], q_value, mask=elem_mask)

        k = tl.load(k_in_ptr + k_in_base + dims[None, :], mask=elem_mask, other=0.0).to(tl.float32)
        k_rot = tl.load(k_in_ptr + k_in_base + rot_dims[None, :], mask=elem_mask, other=0.0).to(tl.float32)
        k_direct = (k * cos_v).to(tl.bfloat16).to(tl.float32)
        k_rotated = (k_rot * rot_sign[None, :] * sin_v).to(tl.bfloat16).to(tl.float32)
        k_value = k_direct + k_rotated
        tl.store(k_out_ptr + k_out_base + dims[None, :], k_value, mask=elem_mask)

        tl.store(ne_ptr + batch * SEQ_ + seq, rows != rows, mask=row_mask & (q_head == 0))


@oracle_impl(hardware="H100", shapes="(T([2048, 2048], bf16), T([32], f32), T([2048, 512], bf16), S([4, 512, 2048]), S([4, 512, -1, 64]), S([1, 32, 1]), S([1, 1, 512]), S([1, 512, 2, 32]), S([1, 512, 64]), S([4, 512, 512]), S([4, 512, -1, 64]), S([4, 8, 4, 512, 64]), S([4, 32, 512, 64]), S([4, -1]))")
def oracle_forward(inputs):
    """Run the complete generated-RoPE Q/K expand repro scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_rope_qk_expand.py")

    mm, inv_freq, mm_1 = _validate_inputs(inputs)
    cos_table = torch.empty((SEQ, HEAD_DIM), device=mm.device, dtype=torch.bfloat16)
    sin_table = torch.empty_like(cos_table)
    out_q = torch.empty_strided(Q_OUT_SHAPE, Q_OUT_STRIDE, device=mm.device, dtype=torch.bfloat16)
    out_k = torch.empty_strided(K_OUT_SHAPE, K_OUT_STRIDE, device=mm.device, dtype=torch.bfloat16)
    ne = torch.empty(NE_SHAPE, device=mm.device, dtype=torch.bool)

    table_block = 256
    _rope_table_kernel[(triton.cdiv(ROPE_TABLE_NUMEL, table_block),)](
        inv_freq,
        cos_table,
        sin_table,
        N=ROPE_TABLE_NUMEL,
        HEAD_DIM_=HEAD_DIM,
        FREQ_DIM_=FREQ_DIM,
        BLOCK_N=table_block,
        num_warps=4,
        num_stages=3,
    )

    block_rows = 2
    _rope_qk_expand_kernel[(triton.cdiv(BATCH * SEQ * Q_HEADS, block_rows),)](
        mm,
        mm_1,
        cos_table,
        sin_table,
        out_q,
        out_k,
        ne,
        N_ROWS=BATCH * SEQ * Q_HEADS,
        SEQ_=SEQ,
        Q_HEADS_=Q_HEADS,
        KV_HEADS_=KV_HEADS,
        KV_GROUPS_=KV_GROUPS,
        HEAD_DIM_=HEAD_DIM,
        FREQ_DIM_=FREQ_DIM,
        BLOCK_ROWS=block_rows,
        BLOCK_D=HEAD_DIM,
        num_warps=4,
        num_stages=3,
    )
    return (out_q, out_k, ne)


def _check_layout(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = True
    for index, (expected, output) in enumerate(zip(eager, actual)):
        layout_ok = (
            tuple(output.shape) == tuple(expected.shape)
            and tuple(output.stride()) == tuple(expected.stride())
            and output.dtype == expected.dtype
            and output.storage_offset() == expected.storage_offset()
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(output.shape)} stride={output.stride()} "
            f"dtype={output.dtype} storage_offset={output.storage_offset()})"
        )
        ok = ok and layout_ok
    return ok


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
        ok = _check_layout(instance, inputs) and ok
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
