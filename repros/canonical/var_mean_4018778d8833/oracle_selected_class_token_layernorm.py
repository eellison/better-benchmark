"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete returned DeiT infer residual-add LayerNorm scope in one fixed-hidden Triton row kernel by sinking `select(..., 1, 0).clone()` through the row-independent reshape, residual add, fp32 population var_mean, and affine LayerNorm epilogue, so only the live class-token rows are reduced and stored, whereas Inductor currently lowers the generic norm-template graph over all `[128,197]` token rows before selecting token zero; Inductor cannot do this today because norm-template canonicalization does not narrow a row-wise normalization domain from a downstream token select; the fix is ALGEBRAIC_ELIMINATION: commute token selects through row-local normalization and eliminate dead patch-token rows before scheduling the LayerNorm template."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
BATCH = 128
SEQ_LEN = 197
HIDDEN = 192
ROWS = BATCH * SEQ_LEN
EPS = 1.0e-6
BLOCK_H = 256
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=3),
        ],
        key=["hidden", "seq_len"],
    )
    @triton.jit
    def _selected_class_token_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch: tl.constexpr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        block_h: tl.constexpr,
    ):
        batches = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, block_h)
        batch_mask = batches < batch
        col_mask = cols < hidden
        mask = batch_mask[:, None] & col_mask[None, :]

        input_offsets = batches[:, None] * seq_len * hidden + cols[None, :]
        values = tl.load(addmm_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        values = tl.where(mask, values + residual, 0.0)

        mean = tl.sum(values, axis=1) / hidden
        sum_x2 = tl.sum(values * values, axis=1)
        variance = tl.maximum(sum_x2 / hidden - mean * mean, 0.0)
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        output = (values - mean[:, None]) * invstd[:, None] * weight[None, :] + bias[None, :]

        output_offsets = batches[:, None] * hidden + cols[None, :]
        tl.store(output_ptr + output_offsets, output, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_47 = _require_tensor(
        "addmm_47",
        inputs[0],
        (ROWS, HIDDEN),
        torch.float32,
        (HIDDEN, 1),
    )
    add_80 = _require_tensor(
        "add_80",
        inputs[1],
        (BATCH, SEQ_LEN, HIDDEN),
        torch.float32,
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
    )
    weight = _require_tensor("arg149_1", inputs[2], (HIDDEN,), torch.float32, (1,))
    bias = _require_tensor("arg150_1", inputs[3], (HIDDEN,), torch.float32, (1,))

    shape_param = _shape_tuple(inputs[4])
    if shape_param != (BATCH, SEQ_LEN, HIDDEN):
        raise ValueError(f"_shape_param_0 is {shape_param}, expected {(BATCH, SEQ_LEN, HIDDEN)}")

    device = addmm_47.device
    for tensor in (add_80, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm_47, add_80, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    """Run the complete Repro.forward computation for the returned class-token tensor.

    SCOPE INVARIANT: accepts the same five inputs as Repro.forward() and returns
    the same single contiguous float32 `[128, 192]` output. The reshape,
    residual add, row-wise LayerNorm, affine epilogue, select, and clone are
    represented by direct indexing into the live token-zero rows.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_selected_class_token_layernorm.py")

    addmm_47, add_80, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=addmm_47.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(BATCH, meta["BLOCK_M"]),)
    _selected_class_token_layernorm_kernel[grid](
        addmm_47,
        add_80,
        weight,
        bias,
        output,
        batch=BATCH,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        block_h=BLOCK_H,
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
