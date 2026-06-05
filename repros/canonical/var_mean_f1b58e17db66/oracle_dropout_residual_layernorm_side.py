"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GPT-2 training seeded-dropout residual LayerNorm scope in one fixed-hidden persistent Triton row kernel, including the `[8192,768] -> [8,1024,768]` view, seed-index 24 dropout, residual add, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 affine scale/bias, final `[8192,768]` view, and sibling `rsqrt / 768` side output, whereas CUDAGraph-measured Inductor already reaches the same persistent normalization schedule and memory-traffic envelope; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new pattern because the remaining work is the required activation/residual/affine reads, RNG mask, one 768-wide row reduction, rsqrt, output store, and side-output store; the fix is BANDWIDTH_BOUND: record this norm-template-canonicalization case as at floor unless broader normalization, RNG, or launch-overhead work moves both implementations."""
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
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    get_raw_stream = None
    triton_helpers = None
    triton_heuristics = None
    DeviceProperties = None
    ReductionHint = None
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
SEED_COUNT = 25
SEED_INDEX = 24
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5
BLOCK_H = 1024

INPUT_2D_SHAPE = (ROWS, HIDDEN)
INPUT_VIEW_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, SEQ_LEN, 1)
SIDE_STRIDE = (SEQ_LEN, 1, 1)

if triton is not None:
    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.persistent_reduction(
        size_hints={"x": ROWS, "r0_": BLOCK_H},
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta={
            "signature": {
                "out_ptr": "*fp32",
                "side_ptr": "*fp32",
                "seeds_ptr": "*i64",
                "residual_ptr": "*fp32",
                "addmm_ptr": "*fp32",
                "weight_ptr": "*fp32",
                "bias_ptr": "*fp32",
                "load_seed_offset": "i32",
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
                    (5,): [["tt.divisibility", 16]],
                    (6,): [["tt.divisibility", 16]],
                    (8,): [["tt.divisibility", 16]],
                    (9,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "oracle_dropout_residual_layernorm_side",
            "mutated_arg_names": ["out_ptr", "side_ptr"],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 5,
            "num_store": 2,
            "num_reduction": 4,
            "autotune_hints": set(),
            "tiling_scores": {"x": 0, "r0_": ROWS * BLOCK_H * 16},
            "backend_hash": "oracle",
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
        },
    )
    @triton.jit
    def _dropout_layernorm_side_kernel(
        out_ptr,
        side_ptr,
        seeds_ptr,
        residual_ptr,
        addmm_ptr,
        weight_ptr,
        bias_ptr,
        load_seed_offset,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 8192
        r0_numel = 768
        R0_BLOCK: tl.constexpr = 1024
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        r0_mask = r0_index < r0_numel
        offsets = r0_index + 768 * xindex

        addmm = tl.load(
            addmm_ptr + offsets,
            mask=r0_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=r0_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        seed = tl.load(seeds_ptr + load_seed_offset)
        random = tl.rand(seed, offsets.to(tl.uint32))
        dropped = tl.where(random > 0.1, addmm * 1.1111111111111112, 0.0)
        x = residual + dropped

        x_for_reduce = tl.where(r0_mask, x, 0.0)
        mean = (tl.sum(x_for_reduce, axis=1) / 768.0)[:, None]
        centered = x - mean
        variance = tl.sum(tl.where(r0_mask, centered * centered, 0.0), axis=1) / 768.0
        invstd_1d = libdevice.rsqrt(variance + 1.0e-5)
        invstd = invstd_1d[:, None]

        weight = tl.load(
            weight_ptr + r0_index,
            mask=r0_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + r0_index,
            mask=r0_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        output = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, output, mask=r0_mask)
        tl.store(side_ptr + xindex, invstd_1d[:, None] / 768.0)


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
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_47", inputs[0], INPUT_2D_SHAPE, torch.float32)
    seeds = _require_tensor("inductor_seeds", inputs[1], (SEED_COUNT,), torch.int64)
    residual = _require_tensor("add_92", inputs[2], INPUT_VIEW_SHAPE, torch.float32)
    weight = _require_tensor("arg147_1", inputs[3], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg148_1", inputs[4], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[5]) != INPUT_VIEW_SHAPE:
        raise ValueError(f"unexpected input view shape parameter: {inputs[5]!r}")
    if _shape_tuple(inputs[6]) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {inputs[6]!r}")

    device = addmm.device
    for tensor in (seeds, residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same device")
    return addmm, seeds, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    shape0 = _shape_tuple(inputs[5])
    shape1 = _shape_tuple(inputs[6])

    viewed = torch.ops.aten.view.default(addmm, shape0)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape0, seed, "rand")
    keep = torch.ops.aten.gt.Scalar(random, DROPOUT_P)
    dropped = torch.ops.aten.mul.Tensor(keep, viewed)
    dropped = torch.ops.aten.mul.Tensor(dropped, DROPOUT_SCALE)
    x = torch.ops.aten.add.Tensor(residual, dropped)
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    invstd = torch.ops.aten.rsqrt.default(variance + EPS)
    normalized = torch.ops.aten.mul.Tensor(x - mean, invstd)
    affine = torch.ops.aten.add.Tensor(normalized * weight, bias)
    return torch.ops.aten.view.default(affine, shape1), invstd / HIDDEN


def oracle_forward(inputs):
    """Run the complete Repro.forward seeded-dropout LayerNorm scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
    """
    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
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

    device_index = (
        addmm.device.index if addmm.device.index is not None else torch.cuda.current_device()
    )
    with torch.cuda._DeviceGuard(device_index):
        torch.cuda.set_device(device_index)
        raw_stream = get_raw_stream(device_index)
        _dropout_layernorm_side_kernel.run(
            output,
            side,
            seeds,
            residual,
            addmm,
            weight,
            bias,
            SEED_INDEX,
            ROWS,
            HIDDEN,
            stream=raw_stream,
        )
    return output, side


def _repro_has_stochastic_ops() -> bool:
    return has_stochastic_ops(REPRO_PATH) or "inductor_random" in REPRO_PATH.read_text()


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
