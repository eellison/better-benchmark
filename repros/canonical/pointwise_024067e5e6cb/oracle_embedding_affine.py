"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileBERT embedding-add-affine scope as one row/hidden Triton writer, including the dynamic sliced token-id embedding, the all-zero token-type embedding folded to direct `arg6[0, :]` loads, the per-hidden multiply/add affine, and the final contiguous `[32768, 512]` view, whereas tuned Inductor already reaches the same required memory-traffic envelope for the captured slice/full/embedding/broadcast/add/mul/add/view graph; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the remaining work is dominated by the mandatory activation load, embedding/affine parameter reads, and output store rather than avoidable intermediate traffic; the fix is BANDWIDTH_BOUND: record this as at floor unless a broader embedding-gather or memory-bandwidth codegen improvement moves both paths."""
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


BATCH = 256
SEQ = 128
HIDDEN = 512
ROWS = BATCH * SEQ
OUT_SHAPE = (ROWS, HIDDEN)
OUT_STRIDE = (HIDDEN, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 2, "BLOCK_N": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 512}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _embedding_affine_kernel(
        token_ids_ptr,
        addmm_ptr,
        word_table_ptr,
        token_type_table_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        N_ROWS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        row_mask = rows < N_ROWS
        col_mask = cols < HIDDEN_SIZE
        mask = row_mask[:, None] & col_mask[None, :]

        seq = rows % SEQ_LEN
        token_ids = tl.load(token_ids_ptr + seq, mask=row_mask, other=0)

        offsets = rows[:, None] * HIDDEN_SIZE + cols[None, :]
        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
        word = tl.load(
            word_table_ptr + token_ids[:, None] * HIDDEN_SIZE + cols[None, :],
            mask=mask,
            other=0.0,
        )
        token_type = tl.load(token_type_table_ptr + cols, mask=col_mask, other=0.0)
        scale = tl.load(scale_ptr + cols, mask=col_mask, other=0.0)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0)

        values = (addmm + word + token_type[None, :]) * scale[None, :] + bias[None, :]
        tl.store(out_ptr + offsets, values, mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_embedding_affine.py")
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    token_ids, addmm, word_table, token_type_table, scale, bias, shape0, shape1 = inputs
    tensor_inputs = (token_ids, addmm, word_table, token_type_table, scale, bias)
    if not all(isinstance(t, torch.Tensor) for t in tensor_inputs):
        raise TypeError(f"{REPRO_ID} expects the first six inputs to be tensors")
    if any(t.device.type != "cuda" for t in tensor_inputs):
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")

    if token_ids.dtype != torch.int64:
        raise TypeError(f"expected int64 token ids, got {token_ids.dtype}")
    if any(t.dtype != torch.float32 for t in (addmm, word_table, token_type_table, scale, bias)):
        raise TypeError("expected fp32 activation, embedding, scale, and bias tensors")
    if tuple(token_ids.shape) != (1, 512):
        raise ValueError(f"unexpected token id shape: {tuple(token_ids.shape)}")
    if tuple(addmm.shape) != OUT_SHAPE:
        raise ValueError(f"unexpected addmm shape: {tuple(addmm.shape)}")
    if tuple(word_table.shape) != (HIDDEN, HIDDEN):
        raise ValueError(f"unexpected word embedding shape: {tuple(word_table.shape)}")
    if tuple(token_type_table.shape) != (2, HIDDEN):
        raise ValueError(f"unexpected token-type embedding shape: {tuple(token_type_table.shape)}")
    if tuple(scale.shape) != (HIDDEN,) or tuple(bias.shape) != (HIDDEN,):
        raise ValueError(f"unexpected affine shapes: scale={tuple(scale.shape)} bias={tuple(bias.shape)}")
    if list(shape0) != [BATCH, SEQ, HIDDEN] or list(shape1) != [ROWS, HIDDEN]:
        raise ValueError(f"unexpected shape params: {shape0!r}, {shape1!r}")
    if not all(t.is_contiguous() for t in tensor_inputs):
        raise ValueError(f"{REPRO_ID} expects captured contiguous input layouts")

    return token_ids, addmm, word_table, token_type_table, scale, bias, shape0, shape1


def oracle_forward(inputs):
    """Run the full Repro.forward scope with fused embedding gathers and affine."""
    token_ids, addmm, word_table, token_type_table, scale, bias, _shape0, shape1 = _validate_inputs(inputs)
    output = torch.empty_strided(
        tuple(shape1),
        OUT_STRIDE,
        device=addmm.device,
        dtype=addmm.dtype,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_M"]),)
    _embedding_affine_kernel[grid](
        token_ids,
        addmm,
        word_table,
        token_type_table,
        scale,
        bias,
        output,
        N_ROWS=ROWS,
        SEQ_LEN=SEQ,
        HIDDEN_SIZE=HIDDEN,
    )
    return output


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
