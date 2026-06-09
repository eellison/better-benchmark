"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BART generated-seed dropout-residual LayerNorm Repro.forward scope in one fixed-hidden Triton row kernel, including p=0.1 dropout on `addmm_3`, residual add, fp32 `var_mean(..., correction=0, keepdim=True)`, eps=1e-5 affine scale/bias, and the final `[2048,768]` view, whereas tuned Inductor already emits the same full stochastic normalization scope as a persistent reduction template and measures within floor noise; Inductor cannot materially improve this today because the required seed generation, addmm/residual/affine reads, fixed-width row reduction, rsqrt, and output write dominate rather than an avoidable materialization or missed fusion; the fix is BANDWIDTH_BOUND: record this as an at-floor stochastic LayerNorm artifact unless broader normalization-template or launch-overhead improvements move the family."""
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
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


ROWS = 4 * 512
BATCH = 4
SEQ_LEN = 512
HIDDEN = 768
EPS = 1.0e-5
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 5
SEED_INDEX = 0
SEED_LOW = -9223372036854775808
SEED_HIGH = 9223372036854775807
BLOCK_H = 1024
STOCHASTIC_OUTPUTS = (0,)
CLASSIFICATION = "BANDWIDTH_BOUND"


def _repro_has_stochastic_ops() -> bool:
    return has_stochastic_ops(REPRO_PATH) or "inductor_random" in REPRO_PATH.read_text()


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
    def _dropout_residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        seed_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        seed_index: tl.constexpr,
        block_h: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        col_mask = cols < hidden
        row_mask = rows < total_rows
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

        seed = tl.load(seed_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        keep = (random > dropout_p).to(tl.float32)
        dropped = keep * addmm * dropout_scale
        x = residual + dropped

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

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

        tl.store(out_ptr + offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_3, residual, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (addmm_3, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        (ROWS, HIDDEN),
        (BATCH, SEQ_LEN, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} must be torch.float32, got {value.dtype}")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    if shape0_tuple != (BATCH, SEQ_LEN, HIDDEN):
        raise ValueError(f"unexpected first view shape: {shape0!r}")
    if shape1_tuple != (ROWS, HIDDEN):
        raise ValueError(f"unexpected output view shape: {shape1!r}")

    return addmm_3, residual, weight, bias, shape0_tuple, shape1_tuple


def _make_inductor_seeds(device: torch.device) -> torch.Tensor:
    seeds = torch.empty_strided((SEED_COUNT,), (1,), device=device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(SEED_LOW, SEED_HIGH, [SEED_COUNT], out=seeds)
    return seeds


@oracle_impl(hardware="H100", shapes="(T([2048, 768], f32), T([4, 512, 768], f32), T([768], f32), T([768], f32), S([4, 512, 768]), S([2048, 768]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward scope.

    The returned tensor matches the single `[2048, 768]` fp32 contiguous view
    produced by Repro.forward. The dropout is generated inside the oracle path
    from the same Inductor seed primitive and consumed by the fused row
    normalization kernel.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_dropout_residual_layernorm.py")

    addmm_3, residual, weight, bias, _shape0, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        (HIDDEN, 1),
        device=addmm_3.device,
        dtype=torch.float32,
    )
    seeds = _make_inductor_seeds(addmm_3.device)

    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _dropout_residual_layernorm_kernel[grid](
        addmm_3,
        residual,
        weight,
        bias,
        seeds,
        output,
        hidden=HIDDEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        seed_index=SEED_INDEX,
        block_h=BLOCK_H,
        total_rows=ROWS,
    )
    return output


def _normalize_outputs(outputs):
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    return (outputs,)


def _tensor_metadata_matches(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    return (
        expected.shape == actual.shape
        and expected.dtype == actual.dtype
        and expected.stride() == actual.stride()
        and expected.device == actual.device
    )


def _check_dropout_residual_layernorm_oracle(
    instance,
    inputs,
    *,
    atol: float,
    rtol: float,
    skip_stochastic: bool,
) -> bool:
    with torch.no_grad():
        eager = _normalize_outputs(instance(*inputs))
        oracle_out = _normalize_outputs(oracle_forward(inputs))
        if any(isinstance(value, torch.Tensor) and value.is_cuda for value in oracle_out):
            torch.cuda.synchronize()

    if len(oracle_out) != len(eager):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_out)} outputs, "
            f"eager produces {len(eager)}"
        )
        return False

    all_pass = True
    for index, (expected, actual) in enumerate(zip(eager, oracle_out)):
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            ok = expected == actual
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (non-tensor)")
            all_pass = all_pass and bool(ok)
            continue

        metadata = (
            f"shape={list(expected.shape)} dtype={expected.dtype} "
            f"stride={list(expected.stride())}"
        )
        if not _tensor_metadata_matches(expected, actual):
            print(
                f"  output {index}: SCOPE_MISMATCH metadata "
                f"oracle_shape={list(actual.shape)} eager_shape={list(expected.shape)} "
                f"oracle_dtype={actual.dtype} eager_dtype={expected.dtype} "
                f"oracle_stride={list(actual.stride())} eager_stride={list(expected.stride())} "
                f"oracle_device={actual.device} eager_device={expected.device}"
            )
            all_pass = False
            continue

        if skip_stochastic and index in STOCHASTIC_OUTPUTS:
            print(f"  output {index}: SKIP values (stochastic dropout; metadata PASS {metadata})")
            continue

        if not expected.is_floating_point():
            ok = torch.equal(expected, actual)
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (exact, {metadata})")
            all_pass = all_pass and bool(ok)
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        max_diff = (expected_f32 - actual_f32).abs().max().item()
        ok = torch.allclose(expected_f32, actual_f32, atol=atol, rtol=rtol)
        print(
            f"  output {index}: {'PASS' if ok else 'FAIL'} "
            f"({metadata} max_diff={max_diff:.2e})"
        )
        all_pass = all_pass and bool(ok)

    return all_pass


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
                        help="Disable skipping the stochastic dropout output")
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
        if args.no_skip_stochastic:
            print(
                f"NOTE: {REPRO_ID} contains stochastic ops; --no-skip-stochastic "
                "requested, so dropout-dependent values will be compared"
            )
        else:
            print(
                f"NOTE: {REPRO_ID} contains stochastic ops; output 0 metadata is "
                "checked and output 0 values are skipped"
            )

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _check_dropout_residual_layernorm_oracle(
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
