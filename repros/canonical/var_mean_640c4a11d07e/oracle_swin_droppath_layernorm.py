"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin window-reverse drop-path LayerNorm scope in one Triton row-reduction kernel by loading seed index 38 and generating the per-batch `tl.rand(seed, batch)` mask inside the reduction, whereas Inductor emits a separate pointwise seed/random kernel that materializes 128 random values before its fused window-reverse/drop-path/residual/LayerNorm/affine kernel; Inductor cannot do this today because the scheduler treats `prims.inductor_random` as an independent materialized producer and does not sink generated-seed, batch-broadcast RNG into a downstream inner reduction while preserving the seed offset and counter mapping; the fix is SCHEDULER_FUSION: allow generated-seed pointwise RNG producers with broadcasted consumers to fuse into the persistent row-reduction schedule that uses them."""
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


ROWS = 25088
HIDDEN = 512
BATCH = 128
WINDOW_ROWS = 14
WINDOW_COLS = 14
KEEP_PROB = 0.9130434766411781
DROP_SCALE = 1.0952380971809903
EPS = 1.0e-5
SEED_INDEX = 38

ADDMM_SHAPE = (25088, 512)
ADDMM_STRIDE = (512, 1)
SEEDS_SHAPE = (46,)
SEEDS_STRIDE = (1,)
RESIDUAL_SHAPE = (128, 14, 14, 512)
RESIDUAL_STRIDE = (100352, 7168, 512, 1)
AFFINE_SHAPE = (512,)
AFFINE_STRIDE = (1,)
OUT0_SHAPE = (25088, 512)
OUT0_STRIDE = (512, 1)
OUT1_SHAPE = (128, 196, 1)
OUT1_STRIDE = (196, 1, 1)

SHAPE_PARAMS = (
    (512, 49, 512),
    (-1, 7, 7, 512),
    (-1, 2, 2, 7, 7, 512),
    (-1, 14, 14, 512),
    (128, -1, 512),
    (25088, 512),
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
    from torch._inductor.runtime.triton_helpers import libdevice

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.reduction(
        size_hints={"x": 32768, "r0_": 512},
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta={
            "signature": {
                "residual_ptr": "*fp32",
                "addmm_ptr": "*fp32",
                "seeds_ptr": "*i64",
                "weight_ptr": "*fp32",
                "bias_ptr": "*fp32",
                "out0_ptr": "*fp32",
                "out1_ptr": "*fp32",
                "xnumel": "i32",
                "r0_numel": "i32",
                "XBLOCK": "constexpr",
                "R0_BLOCK": "constexpr",
            },
            "device": DeviceProperties(
                type="cuda",
                index=0,
                multi_processor_count=132,
                cc=90,
                major=9,
                regs_per_multiprocessor=65536,
                max_threads_per_multi_processor=2048,
                max_threads_per_block=1024,
                warp_size=32,
            ),
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
                    (7,): [["tt.divisibility", 16]],
                    (8,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "oracle_swin_droppath_layernorm",
            "mutated_arg_names": [],
            "optimize_mem": True,
            "no_x_dim": False,
            "atomic_add_found": False,
            "num_load": 7,
            "num_store": 2,
            "num_reduction": 2,
            "autotune_hints": set(),
            "tiling_scores": {"x": 200704, "r0_": 205524992},
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
    def _swin_droppath_layernorm_kernel(
        residual_ptr,
        addmm_ptr,
        seeds_ptr,
        weight_ptr,
        bias_ptr,
        out0_ptr,
        out1_ptr,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
        R0_BLOCK: tl.constexpr,
    ):
        xnumel = 25088
        r0_numel = 512
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]

        row = xindex
        spatial = xindex % 196
        batch = xindex // 196
        seed = tl.load(seeds_ptr + 38)
        random_value = tl.rand(seed, batch.to(tl.uint32))

        mean_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        m2_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        weight_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)

        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            channel = r0_index

            residual = tl.load(
                residual_ptr + channel + 512 * row,
                r0_mask & xmask,
                eviction_policy="evict_last",
                other=0.0,
            )
            window = tl.load(
                addmm_ptr
                + channel
                + 512 * (((spatial % 14) % 7))
                + 3584 * (((spatial // 14) % 7))
                + 25088 * (((spatial // 7) % 2))
                + 50176 * (spatial // 98)
                + 100352 * batch,
                r0_mask & xmask,
                eviction_policy="evict_last",
                other=0.0,
            )
            keep = random_value < tl.full([1, 1], 0.9130434766411781, tl.float32)
            drop = keep.to(tl.float32) * tl.full([1, 1], 1.0952380971809903, tl.float32)
            value = residual + window * drop
            value = tl.broadcast_to(value, [XBLOCK, R0_BLOCK])
            mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
                value,
                mean_acc,
                m2_acc,
                weight_acc,
                r0_offset == 0,
            )
            mean_acc = tl.where(r0_mask & xmask, mean_next, mean_acc)
            m2_acc = tl.where(r0_mask & xmask, m2_next, m2_acc)
            weight_acc = tl.where(r0_mask & xmask, weight_next, weight_acc)

        mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
        mean = mean[:, None]
        m2 = m2[:, None]

        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            channel = r0_index

            residual = tl.load(
                residual_ptr + channel + 512 * row,
                r0_mask & xmask,
                eviction_policy="evict_first",
                other=0.0,
            )
            window = tl.load(
                addmm_ptr
                + channel
                + 512 * (((spatial % 14) % 7))
                + 3584 * (((spatial // 14) % 7))
                + 25088 * (((spatial // 7) % 2))
                + 50176 * (spatial // 98)
                + 100352 * batch,
                r0_mask & xmask,
                eviction_policy="evict_first",
                other=0.0,
            )
            gamma = tl.load(
                weight_ptr + channel,
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            beta = tl.load(
                bias_ptr + channel,
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            keep = random_value < tl.full([1, 1], 0.9130434766411781, tl.float32)
            drop = keep.to(tl.float32) * tl.full([1, 1], 1.0952380971809903, tl.float32)
            value = residual + window * drop
            centered = value - mean
            variance = m2 / tl.full([1, 1], 512.0, tl.float32)
            invstd = libdevice.rsqrt(variance + tl.full([1, 1], 1.0e-5, tl.float32))
            normalized = centered * invstd
            out = normalized * gamma + beta
            tl.store(out0_ptr + channel + 512 * row, out, r0_mask & xmask)

        variance = m2 / tl.full([1, 1], 512.0, tl.float32)
        invstd = libdevice.rsqrt(variance + tl.full([1, 1], 1.0e-5, tl.float32))
        side = invstd * tl.full([1, 1], 0.001953125, tl.float32)
        tl.store(out1_ptr + row, side, xmask)


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
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    addmm = _require_tensor(
        "addmm_81", inputs[0], ADDMM_SHAPE, torch.float32, ADDMM_STRIDE
    )
    seeds = _require_tensor(
        "inductor_seeds_default", inputs[1], SEEDS_SHAPE, torch.int64, SEEDS_STRIDE
    )
    residual = _require_tensor(
        "view_547", inputs[2], RESIDUAL_SHAPE, torch.float32, RESIDUAL_STRIDE
    )
    weight = _require_tensor(
        "primals_310", inputs[3], AFFINE_SHAPE, torch.float32, AFFINE_STRIDE
    )
    bias = _require_tensor(
        "primals_311", inputs[4], AFFINE_SHAPE, torch.float32, AFFINE_STRIDE
    )

    for i, expected in enumerate(SHAPE_PARAMS, start=5):
        actual = _shape_tuple(inputs[i])
        if actual != expected:
            raise ValueError(f"_shape_param_{i - 5} has value {actual}, expected {expected}")

    device = addmm.device
    for tensor in (seeds, residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same CUDA device")
    return addmm, seeds, residual, weight, bias


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same two outputs with matching shapes, dtypes, and strides.

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_swin_droppath_layernorm.py")

    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    out0 = torch.empty_strided(
        OUT0_SHAPE,
        OUT0_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided(
        OUT1_SHAPE,
        OUT1_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    raw_stream = get_raw_stream(addmm.device.index or 0)
    _swin_droppath_layernorm_kernel.run(
        residual,
        addmm,
        seeds,
        weight,
        bias,
        out0,
        out1,
        ROWS,
        HIDDEN,
        stream=raw_stream,
    )
    return out0, out1


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
