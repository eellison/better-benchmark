"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete T5 training dropout-residual-RMSNorm-dropout alias scope in one fixed-hidden persistent Triton row kernel, including the `[8192, 768] -> [8, 1024, 768]` view, seed-index 48 dropout, residual add, fp32 mean-square reduction over hidden size 768, eps=1e-6 rsqrt, affine scale, seed-index 49 dropout, and all 24 aliasing `[8192, 768]` output views, whereas Inductor lowers the same graph through its generic stochastic fused normalization schedule and repeated alias-view epilogue; Inductor cannot emit this exact floor today because norm-template canonicalization does not recognize dropout-residual-RMSNorm-dropout with many sibling alias views as one reusable semantic lowering with tuned persistent-reduction metadata; the fix is NEW_PATTERN: add a guarded stochastic RMSNorm alias template that threads Inductor RNG seeds through the row normalization kernel and preserves repeated view aliases without materialized sibling work. Exact stochastic equality is skipped by the harness, so the floor status is not_true_floor."""
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
ROWS = 8192
BATCH = 8
SEQ_LEN = 1024
HIDDEN = 768
INPUT_SHAPE = (ROWS, HIDDEN)
VIEW_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
OUTPUT_COUNT = 24
BLOCK_H = 1024
EPS = 1.0e-6
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_INDEX_0 = 48
SEED_INDEX_1 = 49
CLASSIFICATION = "NEW_PATTERN"
FLOOR_STATUS = "not_true_floor"

if triton is not None:
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
    from torch._inductor.runtime.triton_helpers import libdevice

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.persistent_reduction(
        size_hints={"x": 8192, "r0_": 1024},
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta={
            "signature": {
                "out_ptr": "*fp32",
                "seeds_ptr": "*i64",
                "residual_ptr": "*fp32",
                "mm_ptr": "*fp32",
                "weight_ptr": "*fp32",
                "load_seed_offset": "i32",
                "load_seed_offset1": "i32",
                "xnumel": "i32",
                "r0_numel": "i32",
                "XBLOCK": "constexpr",
            },
            "device": DeviceProperties.create(torch.device("cuda", 0)),
            "constants": {},
            "native_matmul": False,
            "enable_fp_fusion": True,
            "launch_pdl": False,
            "disable_ftz": False,
            "configs": [
                {
                    (0,): [["tt.divisibility", 16]],
                    (1,): [["tt.divisibility", 16]],
                    (2,): [["tt.divisibility", 16]],
                    (3,): [["tt.divisibility", 16]],
                    (4,): [["tt.divisibility", 16]],
                    (7,): [["tt.divisibility", 16]],
                    (8,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "oracle_dropout_rmsnorm_aliases_kernel",
            "mutated_arg_names": ["out_ptr"],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 3,
            "num_store": 1,
            "num_reduction": 1,
            "autotune_hints": set(),
            "tiling_scores": {"x": 0, "r0_": 100666368},
            "backend_hash": "2FF7A49C450C045FF2AC7A730B5A8D4CD6DF40914BC38271CC70D96733C19B52",
            "assert_indirect_indexing": True,
            "autotune_local_cache": True,
            "autotune_pointwise": True,
            "autotune_remote_cache": None,
            "force_disable_caches": False,
            "dynamic_scale_rblock": True,
            "incremental_autotune": False,
            "max_autotune": False,
            "max_autotune_pointwise": False,
            "min_split_scan_rblock": 256,
            "spill_threshold": 16,
            "store_cubin": False,
            "deterministic": False,
            "batch_invariant": False,
            "force_filter_reduction_configs": False,
            "mix_order_reduction_allow_multi_stages": True,
            "dynamic_disable_pipelining": True,
            "are_deterministic_algorithms_enabled": False,
            "coordinate_descent_tuning": True,
            "coordinate_descent_search_radius": 1,
            "coordinate_descent_check_all_directions": False,
        },
    )
    @triton.jit
    def oracle_kernel(
        out_ptr,
        seeds_ptr,
        residual_ptr,
        mm_ptr,
        weight_ptr,
        load_seed_offset,
        load_seed_offset1,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 8192
        r0_numel = 768
        R0_BLOCK: tl.constexpr = 1024
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_mask = r0_index < r0_numel
        offsets = r0_index + 768 * xindex

        residual = tl.load(
            residual_ptr + offsets,
            mask=r0_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        mm = tl.load(
            mm_ptr + offsets,
            mask=r0_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + r0_index,
            mask=r0_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        seed0 = tl.load(seeds_ptr + load_seed_offset)
        rng_offsets = offsets.to(tl.int32).to(tl.uint32)
        keep0 = (tl.rand(seed0, rng_offsets) > 0.1).to(tl.float32)
        x = residual + keep0 * mm * 1.1111111111111112

        x_broadcast = tl.broadcast_to(x, [XBLOCK, R0_BLOCK])
        square = x_broadcast * x_broadcast
        sum_square = tl.sum(tl.where(r0_mask, square, 0.0), 1)[:, None].to(tl.float32)
        mean_square = sum_square / 768.0
        inv_rms = libdevice.rsqrt(mean_square + 1.0e-6)

        seed1 = tl.load(seeds_ptr + load_seed_offset1)
        keep1 = (tl.rand(seed1, rng_offsets) > 0.1).to(tl.float32)
        out = keep1 * weight * x * inv_rms * 1.1111111111111112
        tl.store(out_ptr + offsets, out, mask=r0_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]:
    expected_input_count = 4 + 1 + OUTPUT_COUNT
    if len(inputs) != expected_input_count:
        raise ValueError(f"{REPRO_ID} expects {expected_input_count} inputs, got {len(inputs)}")

    mm, seeds, residual, weight, shape0, *output_shape_args = inputs
    tensor_inputs = (mm, seeds, residual, weight)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (INPUT_SHAPE, None, VIEW_SHAPE, (HIDDEN,))
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if expected is not None and tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    if seeds.ndim != 1 or seeds.shape[0] <= SEED_INDEX_1:
        raise ValueError(
            f"inductor seed tensor must have at least {SEED_INDEX_1 + 1} values"
        )
    if mm.dtype != torch.float32 or residual.dtype != torch.float32:
        raise TypeError("activation inputs must be torch.float32")
    if weight.dtype != torch.float32:
        raise TypeError("RMSNorm weight must be torch.float32")
    if seeds.dtype != torch.int64:
        raise TypeError(f"inductor seeds must be torch.int64, got {seeds.dtype}")

    if _shape_tuple(shape0) != VIEW_SHAPE:
        raise ValueError(f"unexpected input view shape: {shape0!r}")

    output_shapes = tuple(_shape_tuple(shape) for shape in output_shape_args)
    for index, shape in enumerate(output_shapes):
        if shape != OUTPUT_SHAPE:
            raise ValueError(f"unexpected output view shape {index}: {shape!r}")

    return mm, seeds, residual, weight, output_shapes


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
        raise RuntimeError("Triton is required for oracle_dropout_rmsnorm_aliases.py")

    mm, seeds, residual, weight, output_shapes = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=mm.device,
        dtype=torch.float32,
    )

    device_index = (
        mm.device.index if mm.device.index is not None else torch.cuda.current_device()
    )
    with torch.cuda._DeviceGuard(device_index):
        torch.cuda.set_device(device_index)
        raw_stream = get_raw_stream(device_index)
        oracle_kernel.run(
            output,
            seeds,
            residual,
            mm,
            weight,
            SEED_INDEX_0,
            SEED_INDEX_1,
            ROWS,
            HIDDEN,
            stream=raw_stream,
        )
    return tuple(output.as_strided(shape, OUTPUT_STRIDE) for shape in output_shapes)


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
    if FLOOR_STATUS == "not_true_floor":
        print("NOTE: exact stochastic equality is not established; floor status not_true_floor")

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
