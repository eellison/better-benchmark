"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Reformer dropout/ReLU layout materialization by folding the `[32768, 512] -> [8, 4096, 512] -> [32768, 512]` views, generated Inductor RNG mask, mask multiply, dropout scale, and ReLU into one flat Triton store of the final contiguous output, whereas Inductor lowers the random/gt/mul/scale/relu/view chain through its generic stochastic pointwise/layout scheduler; Inductor cannot do this today because generated RNG producers are not modeled as a reusable dropout epilogue that can be shape-specialized together with surrounding metadata-only layout views; the fix is SCHEDULER_FUSION: teach pointwise scheduling/codegen to thread generated seeds through a fused stochastic layout materialization template."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

BATCH = 8
SEQ_LEN = 4096
HIDDEN = 512
ROWS = BATCH * SEQ_LEN
NUMEL = ROWS * HIDDEN
VIEW_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUT_SHAPE = (ROWS, HIDDEN)
OUT_STRIDE = (HIDDEN, 1)
DROPOUT_P = 0.05
DROPOUT_SCALE = 1.0526315789473684

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=2),
        ],
        key=["N"],  # re-tune when N changes
    )
    @triton.jit
    def _dropout_relu_layout_kernel(
        addmm_ptr,
        seeds_ptr,
        out_ptr,
        N: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        x = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        seed = tl.load(seeds_ptr)
        keep = tl.rand(seed, offsets.to(tl.uint32)) > dropout_p

        masked = tl.where(keep, 1.0, 0.0) * x
        scaled = masked * dropout_scale
        out = tl.where(scaled <= 0.0, 0.0, scaled)
        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter, got {type(value).__name__}")


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_reformer_dropout_relu_layout.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    addmm, shape0, shape1 = inputs
    if not isinstance(addmm, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(addmm).__name__}")
    if tuple(addmm.shape) != OUT_SHAPE:
        raise ValueError(f"unexpected addmm shape: {tuple(addmm.shape)}")
    if tuple(addmm.stride()) != OUT_STRIDE:
        raise ValueError(f"unexpected addmm stride: {tuple(addmm.stride())}")
    if addmm.dtype != torch.float32:
        raise TypeError(f"expected float32 addmm input, got {addmm.dtype}")
    if not addmm.is_cuda:
        raise RuntimeError("CUDA input is required for this Triton oracle")
    if _shape_tuple(shape0) != VIEW_SHAPE:
        raise ValueError(f"unexpected first view shape: {_shape_tuple(shape0)}")
    if _shape_tuple(shape1) != OUT_SHAPE:
        raise ValueError(f"unexpected final view shape: {_shape_tuple(shape1)}")
    return addmm


def oracle_forward(inputs):
    """Run the full Repro.forward dropout/ReLU layout computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    addmm = _validate_inputs(inputs)
    seeds = torch.ops.prims.inductor_seeds.default(2, addmm.device)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(NUMEL, meta["BLOCK_N"]),)
    _dropout_relu_layout_kernel[grid](
        addmm,
        seeds,
        out,
        N=NUMEL,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
    )
    return out


def _check_output_metadata(instance, inputs) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)

    if not isinstance(eager, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  output 0 metadata: SCOPE_MISMATCH non-tensor output")
        return False

    metadata_ok = (
        tuple(actual.shape) == tuple(eager.shape)
        and actual.dtype == eager.dtype
        and tuple(actual.stride()) == tuple(eager.stride())
    )
    print(
        f"  output 0 metadata: {'PASS' if metadata_ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={tuple(actual.stride())} "
        f"dtype={actual.dtype})"
    )
    return metadata_ok


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
    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
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
        ok = _check_output_metadata(instance, inputs) and ok
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
