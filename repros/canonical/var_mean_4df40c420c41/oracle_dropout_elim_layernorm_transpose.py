"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle eliminates the degenerate seed-index-11 Inductor dropout producer (`rand > 1e-30` with scale 1.0) and computes the complete DistillGPT2 residual LayerNorm scope, including the `[16384, 768] -> [32, 512, 768]` view, residual add, population var_mean over hidden size 768, eps=1e-5 rsqrt, affine scale/bias, final `[768, 16384]` transposed output, and `rsqrt / 768` side output, whereas Inductor currently keeps the seeded RNG/mask/mul producer live and schedules it with the decomposed var_mean, affine epilogue, transpose materialization, and inverse-std side output; Inductor cannot do this today because its algebraic simplifier does not prove this near-zero-threshold dropout mask is an identity before normalization scheduling; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to fold degenerate dropout/mask/mul producers before the LayerNorm template and then emit the transposed normalized output plus rsqrt side output from one schedule."""
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
BATCH = 32
SEQ_LEN = 512
HIDDEN = 768
ROWS = BATCH * SEQ_LEN
SEED_COUNT = 13
INPUT_2D_SHAPE = (ROWS, HIDDEN)
INPUT_VIEW_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (HIDDEN, ROWS)
OUTPUT_STRIDE = (1, HIDDEN)
SIDE_SHAPE = (BATCH, SEQ_LEN, 1)
SIDE_STRIDE = (SEQ_LEN, 1, 1)
EPS = 1.0e-5
BLOCK_H = 1024


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=2, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=2, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=1, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=2, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _layernorm_transpose_side_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_t_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        projected = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = projected + residual
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = (tl.sum(x_for_reduce, axis=1) / hidden)[:, None]
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden
        invstd_vec = tl.rsqrt(variance + eps)
        invstd = invstd_vec[:, None]

        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_t_ptr + offsets, y, mask=mask)
        tl.store(invstd_div_ptr + row_ids, invstd_vec / hidden, mask=row_ids < total_rows)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _resolve_output_shape(value: Any) -> tuple[int, int]:
    shape = _shape_tuple(value)
    if len(shape) != 2 or shape[1] != HIDDEN:
        raise ValueError(f"unexpected output view shape parameter: {value!r}")
    if shape[0] == -1:
        return INPUT_2D_SHAPE
    if shape == INPUT_2D_SHAPE:
        return shape
    raise ValueError(f"unexpected output view shape parameter: {value!r}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_21", inputs[0], INPUT_2D_SHAPE, torch.float32)
    seeds = _require_tensor("inductor_seeds", inputs[1], (SEED_COUNT,), torch.int64)
    residual = _require_tensor("add_41", inputs[2], INPUT_VIEW_SHAPE, torch.float32)
    weight = _require_tensor("arg69_1", inputs[3], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg70_1", inputs[4], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[5]) != INPUT_VIEW_SHAPE:
        raise ValueError(f"unexpected input view shape parameter: {inputs[5]!r}")
    _resolve_output_shape(inputs[6])

    device = addmm.device
    for tensor in (seeds, residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm, residual, weight, bias


def oracle_forward(inputs):
    """Run the full repro scope after eliminating the identity dropout producer.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same transposed LayerNorm output plus the `rsqrt / 768` side output.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_dropout_elim_layernorm_transpose.py")

    addmm, residual, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    invstd_div = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _layernorm_transpose_side_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        invstd_div,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        total_rows=ROWS,
    )
    return output, invstd_div


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


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
    if _has_inductor_random():
        print("NOTE: contains Inductor RNG, but the degenerate mask is eliminated; equality is checked for all outputs")

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
