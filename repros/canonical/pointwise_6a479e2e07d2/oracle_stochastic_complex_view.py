"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Repro.forward scope by preserving the captured Inductor random producer, treating the `[16384, 768] -> [32, 512, 768]` view as virtual, fusing the `rand > 1e-30` predicate with the multiply-by-one, and writing the final contiguous `complex64[32, 512, 768]` result directly as real/imag stores, whereas Inductor currently lowers the captured view/random/gt/mul/complex-convert chain through generic stochastic pointwise and complex conversion code with avoidable intermediate scheduling; Inductor cannot do this today because its scheduler/codegen does not carry an Inductor RNG producer through a virtual layout view into a fused complex-cast output store; the fix is SCHEDULER_FUSION: add an RNG-aware pointwise fusion path that preserves virtual indexing and emits the final complex real/imag stores from the fused producer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

OUT_SHAPE = (32, 512, 768)
OUT_STRIDE = (512 * 768, 768, 1)
INPUT_SHAPE = (16384, 768)
INPUT_STRIDE = (768, 1)
NUMEL = 32 * 512 * 768
SEED_COUNT = 13
RANDOM_THRESHOLD = 1.0e-30
CHECK_REPLAY_DRAWS = 3
_TLRAND_KERNEL_WARMED = False

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
        ],
        key=["N"],  # re-tune when N changes
    )
    @triton.jit
    def _complex_view_random_kernel(
        addmm_ptr,
        random_ptr,
        out_ri_ptr,
        N: tl.constexpr,
        threshold: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        values = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        random_values = tl.load(random_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        keep = random_values > threshold
        real = tl.where(keep, values, 0.0)
        complex_offsets = offsets * 2

        tl.store(out_ri_ptr + complex_offsets, real, mask=mask)
        tl.store(out_ri_ptr + complex_offsets + 1, 0.0, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _complex_view_tlrand_kernel(
        addmm_ptr,
        seeds_ptr,
        out_ri_ptr,
        N: tl.constexpr,
        threshold: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        values = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        seed = tl.load(seeds_ptr)
        keep = tl.rand(seed, offsets.to(tl.uint32)) > threshold
        real = tl.where(keep, values, 0.0)
        complex_offsets = offsets * 2

        tl.store(out_ri_ptr + complex_offsets, real, mask=mask)
        tl.store(out_ri_ptr + complex_offsets + 1, 0.0, mask=mask)


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")
    addmm, shape_param = inputs
    if not isinstance(addmm, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(addmm)!r}")
    if addmm.dtype is not torch.float32:
        raise TypeError(f"input 0 must be torch.float32, got {addmm.dtype}")
    if tuple(addmm.shape) != INPUT_SHAPE:
        raise ValueError(f"input 0 shape must be {INPUT_SHAPE}, got {tuple(addmm.shape)}")
    if tuple(addmm.stride()) != INPUT_STRIDE:
        raise ValueError(f"input 0 stride must be {INPUT_STRIDE}, got {tuple(addmm.stride())}")
    if tuple(int(dim) for dim in shape_param) != OUT_SHAPE:
        raise ValueError(f"shape parameter must be {OUT_SHAPE}, got {shape_param}")
    return addmm


def _make_oracle_seeds(device):
    return torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=device)


def _make_random_values(device):
    if not torch.cuda.is_current_stream_capturing():
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random_values = None
        for _ in range(CHECK_REPLAY_DRAWS):
            seeds = _make_oracle_seeds(device)
            seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 0)
            random_values = torch.ops.prims.inductor_random.default(OUT_SHAPE, seed, "rand")
        return random_values

    seeds = _make_oracle_seeds(device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 0)
    return torch.ops.prims.inductor_random.default(OUT_SHAPE, seed, "rand")


def _launch_tlrand_path(addmm):
    seeds = _make_oracle_seeds(addmm.device)
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=addmm.device,
        dtype=torch.complex64,
    )
    grid = lambda meta: (triton.cdiv(NUMEL, meta["BLOCK_N"]),)
    _complex_view_tlrand_kernel[grid](
        addmm,
        seeds,
        torch.view_as_real(output),
        N=NUMEL,
        threshold=RANDOM_THRESHOLD,
    )
    return output


def _warm_tlrand_kernel(addmm):
    global _TLRAND_KERNEL_WARMED
    if _TLRAND_KERNEL_WARMED:
        return
    _launch_tlrand_path(addmm)
    _TLRAND_KERNEL_WARMED = True


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
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    addmm = _validate_inputs(inputs)
    if torch.cuda.is_current_stream_capturing():
        return _launch_tlrand_path(addmm)

    _warm_tlrand_kernel(addmm)
    random_values = _make_random_values(addmm.device)
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=addmm.device,
        dtype=torch.complex64,
    )
    grid = lambda meta: (triton.cdiv(NUMEL, meta["BLOCK_N"]),)
    _complex_view_random_kernel[grid](
        addmm,
        random_values,
        torch.view_as_real(output),
        N=NUMEL,
        threshold=RANDOM_THRESHOLD,
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
