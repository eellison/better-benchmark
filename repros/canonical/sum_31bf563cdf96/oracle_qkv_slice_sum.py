"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle materializes the returned QKV transpose and computes the returned q/v reduction slices directly from the original strided inputs with a fixed two-stage f32 sum, whereas Inductor first writes the full cloned QKV buffer then schedules full-width reduction kernels that reread it and compute the unreturned middle k columns; Inductor cannot do this today because its scheduler does not fuse a returned layout materialization with a reduction consumer or push final slices into the reduction domain; the fix is SCHEDULER_FUSION: allow mixed materialization-plus-reduction scheduling with slice-aware column pruning."""
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


ROWS = 25216
COLS = 2304
Q_OR_V_COLS = 1536
HEAD_COLS = 768
BATCH_STRIDE = 151296
TOKEN_STRIDE = 768
TOKENS = 197
PARTIAL_ROWS = 428
PARTIAL_CHUNKS = 59
K_MATERIALIZE_BLOCK = 1024
PARTIAL_COL_BLOCK = 64
FINAL_COL_BLOCK = 32
PARTIAL_RBLOCK = 8
FINAL_RBLOCK = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _materialize_k_kernel(
        k_ptr,
        out_ptr,
        total: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        head_col = offsets % 768
        row = offsets // 768
        batch = row // 197
        token = row - batch * 197
        src_offset = batch * 151296 + token * 768 + head_col
        value = tl.load(k_ptr + src_offset, mask, other=0.0)
        tl.store(out_ptr + row * 2304 + 768 + head_col, value, mask)

    @triton.jit
    def _materialize_qv_partial_sum_kernel(
        q_ptr,
        v_ptr,
        qkv_ptr,
        partial_ptr,
        BLOCK_COL: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        col_base = tl.program_id(1) * BLOCK_COL
        cols = col_base + tl.arange(0, BLOCK_COL)[:, None]
        out_cols = tl.where(cols < 768, cols, cols + 768)
        r_base = tl.arange(0, RBLOCK)[None, :]
        acc = tl.full([BLOCK_COL, RBLOCK], 0.0, tl.float32)

        for r_offset in tl.range(0, 428, RBLOCK):
            r = r_offset + r_base
            rows = chunk * 428 + r
            valid = (cols < 1536) & (r < 428) & (rows < 25216)

            head_col = cols % 768
            batch = rows // 197
            token = rows - batch * 197
            src_offset = batch * 151296 + token * 768 + head_col

            q_val = tl.load(
                q_ptr + src_offset,
                valid & (cols < 768),
                eviction_policy='evict_first',
                other=0.0,
            )
            v_val = tl.load(
                v_ptr + src_offset,
                valid & (cols >= 768),
                eviction_policy='evict_first',
                other=0.0,
            )
            values = tl.where(cols < 768, q_val, v_val)
            tl.store(qkv_ptr + rows * 2304 + out_cols, values, valid)
            next_acc = acc + values
            acc = tl.where((cols < 1536) & (r < 428), next_acc, acc)

        sums = tl.sum(acc, axis=1)[:, None]
        tl.store(partial_ptr + chunk * 1536 + cols, sums, cols < 1536)

    @triton.jit
    def _final_qv_sum_kernel(
        partial_ptr,
        sum_ptr,
        BLOCK_COL: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        col_base = tl.program_id(0) * BLOCK_COL
        cols = col_base + tl.arange(0, BLOCK_COL)[:, None]
        chunks = tl.arange(0, RBLOCK)[None, :]
        mask = (cols < 1536) & (chunks < 59)
        partials = tl.load(partial_ptr + chunks * 1536 + cols, mask, other=0.0)
        sums = tl.sum(partials, axis=1)[:, None].to(tl.float32)
        out_cols = tl.where(cols < 768, cols, cols + 768)
        tl.store(sum_ptr + out_cols, sums, cols < 1536)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is not available")
    if len(inputs) != 7:
        raise ValueError(f"expected 7 inputs, got {len(inputs)}")
    q, k, v = inputs[:3]
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tensor.dtype is not torch.float32:
            raise TypeError(f"{name} must be torch.float32")
        if tuple(tensor.shape) != (128, 12, 197, 64):
            raise ValueError(f"{name} has unexpected shape {tuple(tensor.shape)}")
        if tuple(tensor.stride()) != (151296, 64, 768, 1):
            raise ValueError(f"{name} has unexpected stride {tuple(tensor.stride())}")
    return q, k, v


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
    """
    q, k, v = _validate_inputs(inputs)
    qkv_base = torch.empty_strided(
        (128, 197, 3, 12, 64),
        (453888, 2304, 768, 64, 1),
        device=q.device,
        dtype=torch.float32,
    )
    sum_base = torch.empty_strided((COLS,), (1,), device=q.device, dtype=torch.float32)
    partials = torch.empty(
        (PARTIAL_CHUNKS, Q_OR_V_COLS),
        device=q.device,
        dtype=torch.float32,
    )

    _materialize_qv_partial_sum_kernel[
        (PARTIAL_CHUNKS, triton.cdiv(Q_OR_V_COLS, PARTIAL_COL_BLOCK))
    ](
        q,
        v,
        qkv_base,
        partials,
        BLOCK_COL=PARTIAL_COL_BLOCK,
        RBLOCK=PARTIAL_RBLOCK,
        num_warps=4,
    )
    _materialize_k_kernel[(triton.cdiv(ROWS * HEAD_COLS, K_MATERIALIZE_BLOCK),)](
        k,
        qkv_base,
        total=ROWS * HEAD_COLS,
        BLOCK=K_MATERIALIZE_BLOCK,
        num_warps=4,
    )
    _final_qv_sum_kernel[(triton.cdiv(Q_OR_V_COLS, FINAL_COL_BLOCK),)](
        partials,
        sum_base,
        BLOCK_COL=FINAL_COL_BLOCK,
        RBLOCK=FINAL_RBLOCK,
        num_warps=4,
    )

    out0 = torch.as_strided(qkv_base, (COLS, ROWS), (1, COLS), 0)
    out1 = torch.as_strided(sum_base, (HEAD_COLS,), (1,), 0)
    out2 = torch.as_strided(sum_base, (HEAD_COLS,), (1,), 2 * HEAD_COLS)
    return (out0, out1, out2)


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
