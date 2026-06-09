"""Gap diagnosis (classification: BANDWIDTH_BOUND): this full-scope oracle computes the complete Longformer seed-index-35 random-mask residual LayerNorm in one fixed-hidden Triton row kernel, including the [8192,768] -> [8,1024,768] view, `inductor_random > 1e-30` mask, residual add, population var_mean, affine scale/bias, and rsqrt/768 side output, but it benchmarks as BAD_ORACLE because current Inductor already emits a faster single fused persistent-reduction kernel for the same stochastic producer, row statistics, affine epilogue, and side-output store; Inductor cannot be fixed by a local fusion change here because the remaining work is the mandatory RNG, input/residual/affine memory traffic, hidden-size-768 reduction, rsqrt, and stores, and dropping the near-zero random predicate would not be exact because the predicate can be false; the fix is BANDWIDTH_BOUND: treat this row as no valid full-scope optimization gap on this stack unless a broader RNG or row-reduction codegen improvement beats the existing persistent-reduction lowering."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims.* RNG ops

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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


# --- Oracle kernel(s) ---
BATCH = 8
SEQ_LEN = 1024
HIDDEN = 768
ROWS = BATCH * SEQ_LEN
ADDMM_SHAPE = (ROWS, HIDDEN)
VIEW_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = VIEW_SHAPE
OUTPUT_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
SIDE_SHAPE = (BATCH, SEQ_LEN, 1)
SIDE_STRIDE = (SEQ_LEN, 1, 1)
WEIGHT_SHAPE = (HIDDEN,)
SEED_COUNT = 36
SEED_INDEX = 35
MASK_THRESHOLD = 1.0e-30
EPS = 1.0e-5
BLOCK_H = 1024

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _identity_dropout_layernorm_side_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        side_ptr,
        hidden: tl.constexpr,
        seed_index: tl.constexpr,
        mask_threshold: tl.constexpr,
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
        keep = (random > mask_threshold).to(tl.float32)
        x = keep * addmm + residual

        x_masked = tl.where(mask, x, 0.0)
        sum_x = tl.sum(x_masked, axis=1)
        sum_x2 = tl.sum(x_masked * x_masked, axis=1)
        mean_1d = sum_x / hidden
        variance = sum_x2 / hidden - mean_1d * mean_1d
        invstd_1d = tl.rsqrt(tl.maximum(variance, 0.0) + eps)

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
        output = (x - mean_1d[:, None]) * invstd_1d[:, None] * weight + bias

        tl.store(output_ptr + offsets, output, mask=mask)
        tl.store(side_ptr + row_ids, invstd_1d / hidden, mask=row_mask_1d)


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
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_23", inputs[0], ADDMM_SHAPE, torch.float32)
    seeds = _require_tensor("inductor_seeds", inputs[1], (SEED_COUNT,), torch.int64)
    residual = _require_tensor("add_139", inputs[2], VIEW_SHAPE, torch.float32)
    weight = _require_tensor("arg193_1", inputs[3], WEIGHT_SHAPE, torch.float32)
    bias = _require_tensor("arg194_1", inputs[4], WEIGHT_SHAPE, torch.float32)

    if _shape_tuple(inputs[5]) != VIEW_SHAPE:
        raise ValueError(f"unexpected view shape parameter: {inputs[5]!r}")

    device = addmm.device
    for tensor in (seeds, residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same device")
    if device.type != "cuda":
        raise RuntimeError("CUDA tensors are required for this Triton oracle")

    return addmm, seeds, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    shape = _shape_tuple(inputs[5])
    viewed = torch.ops.aten.view.default(addmm, shape)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
    keep = torch.ops.aten.gt.Scalar(random, MASK_THRESHOLD)
    x = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(keep, viewed), residual)
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    invstd = torch.ops.aten.rsqrt.default(variance + EPS)
    output = (x - mean) * invstd * weight + bias
    return output, invstd / HIDDEN


@oracle_impl(hardware="H100", shapes="(T([8192, 768], f32), T([36], i64), T([8, 1024, 768], f32), T([768], f32), T([768], f32), S([8, 1024, 768]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    if triton is None:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _identity_dropout_layernorm_side_kernel[grid](
        addmm,
        seeds,
        residual,
        weight,
        bias,
        output,
        side,
        hidden=HIDDEN,
        seed_index=SEED_INDEX,
        mask_threshold=MASK_THRESHOLD,
        eps=EPS,
        block_h=BLOCK_H,
        total_rows=ROWS,
    )
    return output, side


def _repro_has_stochastic_ops() -> bool:
    return has_stochastic_ops(REPRO_PATH) or "inductor_random" in REPRO_PATH.read_text()


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if any(isinstance(value, torch.Tensor) and value.is_cuda for value in actual):
            torch.cuda.synchronize()

    expected_list = expected if isinstance(expected, (list, tuple)) else (expected,)
    actual_list = actual if isinstance(actual, (list, tuple)) else (actual,)
    ok = len(expected_list) == len(actual_list)
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected_list, actual_list)):
        layout_ok = (
            isinstance(expected_tensor, torch.Tensor)
            and isinstance(actual_tensor, torch.Tensor)
            and tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and actual_tensor.stride() == expected_tensor.stride()
            and actual_tensor.dtype == expected_tensor.dtype
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(expected_stride={expected_tensor.stride()}, oracle_stride={actual_tensor.stride()})"
        )
        ok = layout_ok and ok
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
    if _repro_has_stochastic_ops():
        print(
            f"NOTE: {REPRO_ID} contains input-seeded stochastic ops; "
            f"exact stochastic equality is skipped by the harness when outputs vary"
        )

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
