"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BERT/RoBERTa attention backward row-reduction scope from Repro.forward in one Triton kernel, including dropout-mask scaling, broadcast attention-mask substitution, libdevice f32 natural-exp probability reconstruction, row product reduction, exact fma.rn.f32 epilogue, and the final contiguous f32[48,512,512] view over a f32[4,12,512,512] base; Inductor already emits one fused reduction with the same mandatory score/dropout/mask reads, exp/reduction work, and output stores; the fix is BANDWIDTH_BOUND: record the measured full-scope floor instead of assigning this repro to a missing fusion or new lowering pattern."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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


BATCH = 4
HEADS = 12
Q_LEN = 512
K_LEN = 512
FLAT_BH = BATCH * HEADS
N_ROWS = FLAT_BH * Q_LEN
OUT_SHAPE = (FLAT_BH, Q_LEN, K_LEN)
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)
VIEW_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
VIEW_STRIDE = (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1)
ATTN_MASK_SHAPE = (1, 1, Q_LEN, 1)
ATTN_MASK_STRIDE = (Q_LEN, Q_LEN, 1, 1)
EXPAND_MASK_SHAPE = (BATCH, -1, Q_LEN, K_LEN)
ROW_SHAPE = (BATCH, HEADS, Q_LEN, 1)
ROW_STRIDE = (HEADS * Q_LEN, Q_LEN, 1, 1)
SCALE = 1.1111111111111112
ROWS_PER_BLOCK = 4
NUM_WARPS = 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _fma_rn_f32(a, b, c):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=[a, b, c],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def oracle_kernel(
        bmm_ptr,
        dropout_mask_ptr,
        attn_mask_ptr,
        full_ptr,
        logits_ptr,
        row_shift_ptr,
        row_denom_ptr,
        prob_zero_mask_ptr,
        out_base_ptr,
        rows_per_block: tl.constexpr,
        block_k: tl.constexpr,
        scale: tl.constexpr,
    ):
        rows = tl.program_id(0) * rows_per_block + tl.arange(0, rows_per_block)[:, None]
        cols = tl.arange(0, block_k)[None, :]
        offsets = rows * block_k + cols
        row_offsets = rows
        q_index = rows % block_k

        zero_row = tl.load(prob_zero_mask_ptr + row_offsets, eviction_policy="evict_last").to(tl.int1)
        logits = tl.load(logits_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
        attn_keep = tl.load(attn_mask_ptr + q_index, eviction_policy="evict_last").to(tl.int1)
        full_value = tl.load(full_ptr + 0).to(tl.float32)
        row_shift = tl.load(row_shift_ptr + row_offsets, eviction_policy="evict_last").to(tl.float32)
        row_denom = tl.load(row_denom_ptr + row_offsets, eviction_policy="evict_last").to(tl.float32)
        bmm = tl.load(bmm_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
        dropout_keep = tl.load(dropout_mask_ptr + offsets, eviction_policy="evict_first").to(tl.float32)

        mask_bias = tl.where(attn_keep, full_value, -float("inf"))
        add_tensor = logits + mask_bias
        shifted = add_tensor - row_shift
        exp_values = libdevice.exp(shifted)
        probs = exp_values / row_denom
        probs = tl.where(zero_row, 0.0, probs)

        dropout_scaled = dropout_keep * scale
        scaled_bmm = bmm * dropout_scaled
        product = scaled_bmm * probs
        row_sum = tl.sum(product, 1)[:, None].to(tl.float32)
        out = _fma_rn_f32(-probs, row_sum, product)

        tl.store(out_base_ptr + offsets, out)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        bmm_45,
        arg112_1,
        arg106_1,
        full_1,
        arg108_1,
        arg109_1,
        arg110_1,
        arg111_1,
        shape0,
        shape1,
        shape2,
        shape3,
    ) = inputs

    tensor_inputs = (
        bmm_45,
        arg112_1,
        arg106_1,
        full_1,
        arg108_1,
        arg109_1,
        arg110_1,
        arg111_1,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first eight inputs must be tensors")
    if not all(value.device.type == "cuda" for value in tensor_inputs):
        raise RuntimeError("this Triton oracle requires CUDA tensor inputs")

    expected = (
        (OUT_SHAPE, OUT_STRIDE, torch.float32),
        (VIEW_SHAPE, VIEW_STRIDE, torch.bool),
        (ATTN_MASK_SHAPE, ATTN_MASK_STRIDE, torch.bool),
        ((), (), torch.float32),
        (OUT_SHAPE, OUT_STRIDE, torch.float32),
        (ROW_SHAPE, ROW_STRIDE, torch.float32),
        (ROW_SHAPE, ROW_STRIDE, torch.float32),
        (ROW_SHAPE, ROW_STRIDE, torch.bool),
    )
    for index, (tensor, (shape, stride, dtype)) in enumerate(zip(tensor_inputs, expected)):
        if tuple(tensor.shape) != shape or tuple(tensor.stride()) != stride or tensor.dtype != dtype:
            raise ValueError(
                f"input {index} expected shape={shape} stride={stride} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
                f"dtype={tensor.dtype}"
            )

    _validate_shape_param("_shape_param_0", shape0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", shape1, EXPAND_MASK_SHAPE)
    _validate_shape_param("_shape_param_2", shape2, VIEW_SHAPE)
    _validate_shape_param("_shape_param_3", shape3, OUT_SHAPE)

    return tensor_inputs


@oracle_impl(hardware="H100", shapes="(T([48, 512, 512], f32), T([4, 12, 512, 512], b8), T([1, 1, 512, 1], b8), T([], f32), T([48, 512, 512], f32), T([4, 12, 512, 1], f32), T([4, 12, 512, 1], f32), T([4, 12, 512, 1], b8), S([4, 12, 512, 512]), S([4, -1, 512, 512]), S([4, 12, 512, 512]), S([48, 512, 512]))")
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
    (
        bmm_45,
        arg112_1,
        arg106_1,
        full_1,
        arg108_1,
        arg109_1,
        arg110_1,
        arg111_1,
    ) = _validate_inputs(inputs)

    out_base = torch.empty_strided(VIEW_SHAPE, VIEW_STRIDE, device=bmm_45.device, dtype=torch.float32)
    oracle_kernel[(N_ROWS // ROWS_PER_BLOCK,)](
        bmm_45,
        arg112_1,
        arg106_1,
        full_1,
        arg108_1,
        arg109_1,
        arg110_1,
        arg111_1,
        out_base,
        rows_per_block=ROWS_PER_BLOCK,
        block_k=K_LEN,
        scale=SCALE,
        num_warps=NUM_WARPS,
    )
    return out_base.view(OUT_SHAPE)


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
