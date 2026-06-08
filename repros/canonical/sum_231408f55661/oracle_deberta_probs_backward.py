"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DeBERTa probabilities-backward row update returned by Repro.forward in one fixed-shape last-dimension Triton row kernel, including the [192,512,512] -> [8,24,512,512] view, bool dropout-mask conversion, f32 dropout scale, row product reduction, exact fma.rn.f32 epilogue, constant-false final where scope, and final contiguous [192,512,512] view; Inductor already has the same local row-reduction fusion available for this simple no-exp softmax-backward tail unless the measured oracle materially beats torch.compile; the fix is BANDWIDTH_BOUND: record as at_floor if CUDAGraph timing stays within harness noise, otherwise reclassify as SCHEDULER_FUSION for row-reduction-plus-epilogue fusion."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

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


BATCH = 8
HEADS = 24
Q_LEN = 512
K_LEN = 512
FLAT_BH = BATCH * HEADS
N_ROWS = FLAT_BH * Q_LEN
VIEW_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
VIEW_STRIDE = (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1)
OUT_SHAPE = (FLAT_BH, Q_LEN, K_LEN)
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)
FULL_MASK_SHAPE = (BATCH, 1, Q_LEN, K_LEN)
FULL_MASK_STRIDE = (Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1)
SCALE_F32 = 1.1111111640930176
BLOCK_M = 1
BLOCK_K = K_LEN


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
    def _deberta_probs_backward_kernel(
        bmm_ptr,
        dropout_mask_ptr,
        probs_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        k_len: tl.constexpr,
        scale: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_k)
        row_mask = rows < n_rows
        col_mask = cols < k_len
        tile_mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * k_len + cols[None, :]

        bmm = tl.load(bmm_ptr + offsets, mask=tile_mask, other=0.0).to(tl.float32)
        keep = tl.load(dropout_mask_ptr + offsets, mask=tile_mask, other=0).to(tl.float32)
        probs = tl.load(probs_ptr + offsets, mask=tile_mask, other=0.0).to(tl.float32)

        scale_f32 = tl.full((), scale, tl.float32)
        dropout_scaled = keep * scale_f32
        scaled_bmm = bmm * dropout_scaled
        product = scaled_bmm * probs
        row_sum = tl.sum(tl.where(col_mask[None, :], product, 0.0), axis=1)
        out = _fma_rn_f32(-probs, row_sum[:, None], product)

        tl.store(out_ptr + offsets, out, mask=tile_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    bmm_1, arg461_1, arg460_1, full_1, shape0, shape1 = inputs
    tensor_inputs = (bmm_1, arg461_1, arg460_1, full_1)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first four inputs must be tensors")

    expected = (
        (OUT_SHAPE, OUT_STRIDE, torch.float32),
        (VIEW_SHAPE, VIEW_STRIDE, torch.bool),
        (VIEW_SHAPE, VIEW_STRIDE, torch.float32),
        ((), (), torch.float32),
    )
    for index, (tensor, (shape, stride, dtype)) in enumerate(zip(tensor_inputs, expected)):
        if tuple(tensor.shape) != shape or tuple(tensor.stride()) != stride or tensor.dtype != dtype:
            raise ValueError(
                f"input {index} expected shape={shape} stride={stride} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
                f"dtype={tensor.dtype}"
            )

    if not (bmm_1.device == arg461_1.device == arg460_1.device == full_1.device):
        raise ValueError("all tensor inputs must be on the same device")

    _validate_shape_param("_shape_param_0", shape0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", shape1, OUT_SHAPE)
    return bmm_1, arg461_1, arg460_1, full_1, shape0, shape1


def _torch_full_scope(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    bmm_1, arg461_1, arg460_1, full_1, shape0, shape1 = inputs
    view_default = torch.ops.aten.view.default(bmm_1, shape0)
    convert_element_type_default = torch.ops.prims.convert_element_type.default(
        arg461_1, torch.float32
    )
    mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default, mul_tensor)
    mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, arg460_1)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
    neg_default = torch.ops.aten.neg.default(arg460_1)
    fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2)
    full_default = torch.ops.aten.full.default(
        list(FULL_MASK_SHAPE),
        False,
        dtype=torch.bool,
        layout=torch.strided,
        device=bmm_1.device,
        pin_memory=False,
    )
    where_self = torch.ops.aten.where.self(full_default, full_1, fma_default)
    return torch.ops.aten.view.default(where_self, shape1)


def oracle_deberta_probs_backward(
    bmm_1: torch.Tensor,
    arg461_1: torch.Tensor,
    arg460_1: torch.Tensor,
    full_1: torch.Tensor,
    shape0: Any,
    shape1: Any,
    *,
    block_m: int = BLOCK_M,
    block_k: int = BLOCK_K,
    num_warps: int = 8,
) -> torch.Tensor:
    if triton is None or bmm_1.device.type != "cuda":
        return _torch_full_scope((bmm_1, arg461_1, arg460_1, full_1, shape0, shape1))
    if block_k != K_LEN:
        raise ValueError(f"block_k must equal {K_LEN}, got {block_k}")
    if block_m <= 0 or N_ROWS % block_m != 0:
        raise ValueError(f"block_m must evenly divide {N_ROWS}, got {block_m}")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_1.device,
        dtype=torch.float32,
    )
    _deberta_probs_backward_kernel[(N_ROWS // block_m,)](
        bmm_1,
        arg461_1,
        arg460_1,
        out,
        n_rows=N_ROWS,
        k_len=K_LEN,
        scale=SCALE_F32,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


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
    return oracle_deberta_probs_backward(*_validate_inputs(inputs))


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
