"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MT5 GELU-gating-dropout-layout scope in one output-space Triton pointwise kernel, including both `[4096,1024] -> [32,128,1024]` views, tanh-approximate GELU on `mm_141`, multiply by the viewed `mm_142`, seed-index 81 Inductor RNG dropout with p=0.1 and scale 1/0.9, and the final contiguous `[4096,1024]` materialization, whereas Inductor already emits one fused pointwise kernel for the same full scope; Inductor cannot materially improve this case today because the remaining work is dominated by mandatory two-input reads, tanh math, stateless RNG, and one contiguous output store with no intermediate layout materialization left to remove; the fix is BANDWIDTH_BOUND: record the full-scope not_true_floor and only reopen for broader pointwise math, RNG, or launch-overhead improvements. Exact stochastic value equality is skipped by default, so this is a structural floor rather than a true-value floor."""
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
    from torch._inductor.runtime.triton_helpers import libdevice
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

INPUT_SHAPE = (4096, 1024)
VIEW_SHAPE = (32, 128, 1024)
OUTPUT_SHAPE = (4096, 1024)
OUTPUT_STRIDE = (1024, 1)
NUMEL = 4096 * 1024
SEED_COUNT = 84
SEED_INDEX = 81
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
STOCHASTIC_OUTPUTS = (0,)
EXACT_STOCHASTIC_EQUALITY = False

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        ],
        key=["numel"],
    )
    @triton.jit
    def _gelu_mul_dropout_kernel(
        mm_141_ptr,
        mm_142_ptr,
        seeds_ptr,
        out_ptr,
        numel: tl.constexpr,
        seed_index: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        x = tl.load(mm_141_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(mm_142_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        x_cubed = x * x * x
        gelu_arg = (x + x_cubed * 0.044715) * 0.7978845608028654
        gelu = x * 0.5 * (libdevice.tanh(gelu_arg) + 1.0)

        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.int32).to(tl.uint32))
        keep = (random > dropout_p).to(tl.float32)
        out = gelu * gate * keep * dropout_scale
        tl.store(out_ptr + offsets, out, mask=mask)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm_141 = _require_tensor("mm_141", inputs[0], INPUT_SHAPE, torch.float32)
    mm_142 = _require_tensor("mm_142", inputs[1], INPUT_SHAPE, torch.float32)
    seeds = _require_tensor("inductor_seeds", inputs[2], (SEED_COUNT,), torch.int64)

    if _shape_tuple(inputs[3]) != VIEW_SHAPE:
        raise ValueError(f"unexpected _shape_param_0: {inputs[3]!r}")
    if _shape_tuple(inputs[4]) != VIEW_SHAPE:
        raise ValueError(f"unexpected _shape_param_1: {inputs[4]!r}")
    if _shape_tuple(inputs[5]) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected _shape_param_2: {inputs[5]!r}")
    if seeds.device != mm_141.device or mm_142.device != mm_141.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mm_141, mm_142, seeds


@oracle_impl(hardware="H100", shapes="(T([4096, 1024], f32), T([4096, 1024], f32), T([84], i64), S([32, 128, 1024]), S([32, 128, 1024]), S([4096, 1024]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete GELU, gated multiply, seeded dropout, and final view."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gelu_dropout_layout.py")

    mm_141, mm_142, seeds = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=mm_141.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(NUMEL, meta["BLOCK_SIZE"]),)
    _gelu_mul_dropout_kernel[grid](
        mm_141,
        mm_142,
        seeds,
        output,
        numel=NUMEL,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
    )
    return output


def _repro_has_stochastic_ops() -> bool:
    return has_stochastic_ops(REPRO_PATH) or "inductor_random" in REPRO_PATH.read_text()


def _normalize_outputs(outputs: Any) -> tuple[Any, ...]:
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


def _check_oracle(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
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
            all_pass = bool(ok) and all_pass
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
            print(f"  output {index}: SKIP values (stochastic RNG; metadata PASS {metadata})")
            continue

        if not expected.is_floating_point():
            ok = torch.equal(expected, actual)
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (exact, {metadata})")
            all_pass = bool(ok) and all_pass
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        max_diff = (expected_f32 - actual_f32).abs().max().item()
        ok = torch.allclose(expected_f32, actual_f32, atol=atol, rtol=rtol)
        print(
            f"  output {index}: {'PASS' if ok else 'FAIL'} "
            f"({metadata} max_diff={max_diff:.2e})"
        )
        all_pass = bool(ok) and all_pass

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
        if args.no_skip_stochastic:
            print(
                f"NOTE: {REPRO_ID} contains input-seeded Inductor RNG; "
                "--no-skip-stochastic requested, so RNG-dependent values will be compared"
            )
        else:
            print(
                f"NOTE: {REPRO_ID} contains input-seeded Inductor RNG; output metadata "
                "is checked and RNG-dependent values are skipped"
            )
    if not EXACT_STOCHASTIC_EQUALITY:
        print("NOTE: exact stochastic equality is not established; floor status not_true_floor")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _check_oracle(
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
