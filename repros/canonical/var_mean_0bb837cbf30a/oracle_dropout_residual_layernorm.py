"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Blenderbot seeded dropout-residual LayerNorm scope for `[4096, 2560]`, including internal Inductor seed generation, p=0.1 dropout of the addmm view, residual add, population var_mean over hidden size 2560, eps=1e-5 rsqrt, affine scale/bias, and the final `[4096, 2560]` view in one Triton row kernel, whereas Inductor currently lowers the seeded dropout producer and decomposed var_mean/affine chain through generic norm-template boundaries; Inductor cannot do this today because the normalization scheduler does not sink the internal Inductor RNG/dropout producer into the row LayerNorm template for this full output-only scope; the fix is SCHEDULER_FUSION: teach the norm scheduler to fuse internal seeded dropout, residual load, row reduction, affine epilogue, and flattened store into one guarded LayerNorm schedule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims RNG ops

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
ROWS = 4096
BATCH = 32
SEQ_LEN = 128
HIDDEN = 2560
ADDMM_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
SEED_COUNT = 2
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5
BLOCK_H = 4096

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=16, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _dropout_residual_layernorm_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        hidden: tl.constexpr,
        seed_index: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask_1d = row_ids < total_rows
        row_mask = row_mask_1d[:, None]
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        addmm = tl.load(
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

        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        dropped = tl.where(random > dropout_p, addmm * dropout_scale, 0.0)
        x = residual + dropped

        x_for_reduce = tl.where(mask, x, 0.0)
        sum_x = tl.sum(x_for_reduce, axis=1)
        sum_x2 = tl.sum(x_for_reduce * x_for_reduce, axis=1)
        mean_1d = sum_x / hidden
        variance = sum_x2 / hidden - mean_1d * mean_1d
        invstd_1d = tl.rsqrt(tl.maximum(variance, 0.0) + eps)

        mean = mean_1d[:, None]
        invstd = invstd_1d[:, None]
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
        output = (x - mean) * invstd * weight + bias
        tl.store(output_ptr + offsets, output, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


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
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_3", inputs[0], ADDMM_SHAPE, torch.float32)
    residual = _require_tensor("arg2_1", inputs[1], RESIDUAL_SHAPE, torch.float32)
    weight = _require_tensor("arg12_1", inputs[2], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg13_1", inputs[3], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[4]) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected first view shape parameter: {inputs[4]!r}")
    if _shape_tuple(inputs[5]) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected final view shape parameter: {inputs[5]!r}")
    if (
        residual.device != addmm.device
        or weight.device != addmm.device
        or bias.device != addmm.device
    ):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return addmm, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias = _validate_inputs(inputs)
    view = torch.ops.aten.view.default(addmm, inputs[4])
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=addmm.device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(RESIDUAL_SHAPE, seed, "rand")
    dropped = torch.ops.aten.mul.Tensor(
        torch.ops.aten.mul.Tensor(torch.ops.aten.gt.Scalar(random, DROPOUT_P), view),
        DROPOUT_SCALE,
    )
    x = torch.ops.aten.add.Tensor(residual, dropped)
    var, mean = torch.ops.aten.var_mean.correction(x, [2], correction=0, keepdim=True)
    normalized = torch.ops.aten.mul.Tensor(
        torch.ops.aten.sub.Tensor(x, mean),
        torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var, EPS)),
    )
    affine = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(normalized, weight),
        bias,
    )
    return torch.ops.aten.view.default(affine, inputs[5])


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
    addmm, residual, weight, bias = _validate_inputs(inputs)
    if triton is None:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        (HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=addmm.device)

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _dropout_residual_layernorm_kernel[grid](
        addmm,
        seeds,
        residual,
        weight,
        bias,
        output,
        hidden=HIDDEN,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        eps=EPS,
        block_h=BLOCK_H,
        total_rows=ROWS,
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
