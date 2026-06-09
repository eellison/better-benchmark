"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GPT2 dropout-residual-LayerNorm scope in one fixed-hidden persistent Triton row kernel, including seed-index-23 Inductor RNG dropout on the viewed matmul input, residual add, fp32 hidden-size-768 var_mean normalization, affine scale/bias, the final [768, 2048] permuted view layout, and the rsqrt/768 side output, whereas tuned Inductor already emits an equivalent full-scope persistent reduction kernel for this graph; Inductor cannot materially improve this today because the remaining cost is dominated by required RNG, input/residual/affine memory traffic, row reduction, rsqrt, and output stores rather than a missed scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, or recompute opportunity; the fix is BANDWIDTH_BOUND: record this as a full-scope bandwidth floor and only reopen for broad norm-template/runtime overhead improvements. Exact stochastic equality against eager is not established, so the floor status is not_true_floor."""
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
ROWS = 2048
HIDDEN = 768
BATCH = 4
SEQ_LEN = 512
SEED_COUNT = 25
SEED_INDEX = 23
EPS = 1.0e-5
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
BLOCK_N = 1024
EXACT_STOCHASTIC_EQUALITY = False

if triton is not None:
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
    from torch._inductor.runtime.triton_helpers import libdevice

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.persistent_reduction(
        size_hints={"x": 2048, "r0_": 1024},
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta={
            "signature": {
                "out_ptr": "*fp32",
                "seeds_ptr": "*i64",
                "addmm_ptr": "*fp32",
                "residual_ptr": "*fp32",
                "weight_ptr": "*fp32",
                "bias_ptr": "*fp32",
                "div_ptr": "*fp32",
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
            "kernel_name": "_dropout_residual_layernorm_transpose_kernel",
            "mutated_arg_names": ["out_ptr"],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 4,
            "num_store": 2,
            "num_reduction": 4,
            "autotune_hints": set(),
            "tiling_scores": {"x": 16384, "r0_": 25171968},
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
        },
    )
    @triton.jit
    def _dropout_residual_layernorm_transpose_kernel(
        out_ptr,
        seeds_ptr,
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        div_ptr,
        load_seed_offset,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 2048
        r0_numel = 768
        R0_BLOCK: tl.constexpr = 1024
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_mask = r0_index < r0_numel
        offsets = r0_index + 768 * xindex

        addmm = tl.load(addmm_ptr + offsets, r0_mask & xmask, eviction_policy="evict_first", other=0.0)
        residual = tl.load(residual_ptr + offsets, r0_mask & xmask, eviction_policy="evict_first", other=0.0)
        weight = tl.load(weight_ptr + r0_index, r0_mask, eviction_policy="evict_last", other=0.0)
        bias = tl.load(bias_ptr + r0_index, r0_mask, eviction_policy="evict_last", other=0.0)
        seed = tl.load(seeds_ptr + load_seed_offset)
        random = tl.rand(seed, offsets.to(tl.int32).to(tl.uint32))
        keep = (random > 0.1).to(tl.float32)
        x = keep * addmm * 1.1111111111111112 + residual

        x_broadcast = tl.broadcast_to(x, [XBLOCK, R0_BLOCK])
        mean = tl.sum(tl.where(r0_mask & xmask, x_broadcast, 0), 1)[:, None].to(tl.float32) / 768.0
        centered_for_var = x_broadcast - mean
        var = tl.sum(tl.where(r0_mask & xmask, centered_for_var * centered_for_var, 0), 1)[:, None].to(tl.float32) / 768.0
        invstd = libdevice.rsqrt(var + 1.0e-5)
        y = (x - mean) * invstd * weight + bias
        div = invstd * 0.0013020833333333333

        tl.store(out_ptr + offsets, y, r0_mask & xmask)
        tl.store(div_ptr + xindex, div, xmask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape sequence, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | None = None,
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
    if device is not None and value.device != device:
        raise ValueError(f"{name} is on {value.device}, expected {device}")
    return value


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_45", inputs[0], (ROWS, HIDDEN), torch.float32)
    seeds = _require_tensor("inductor_seeds", inputs[1], (SEED_COUNT,), torch.int64, addmm.device)
    residual = _require_tensor("add_89", inputs[2], (BATCH, SEQ_LEN, HIDDEN), torch.float32, addmm.device)
    weight = _require_tensor("arg141_1", inputs[3], (HIDDEN,), torch.float32, addmm.device)
    bias = _require_tensor("arg142_1", inputs[4], (HIDDEN,), torch.float32, addmm.device)
    shape0 = _shape_tuple(inputs[5])
    shape1 = _shape_tuple(inputs[6])
    if shape0 != (BATCH, SEQ_LEN, HIDDEN) or shape1 != (-1, HIDDEN):
        raise ValueError(f"unexpected shape parameters: {inputs[5]!r}, {inputs[6]!r}")
    return addmm, seeds, residual, weight, bias, shape0, shape1


@oracle_impl(hardware="H100", shapes="(T([2048, 768], f32), T([25], i64), T([4, 512, 768], f32), T([768], f32), T([768], f32), S([4, 512, 768]), S([-1, 768]))")
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

    addmm, seeds, residual, weight, bias, _, _ = _validate_inputs(inputs)
    out = torch.empty_strided(
        (HIDDEN, ROWS),
        (1, HIDDEN),
        device=addmm.device,
        dtype=torch.float32,
    )
    div = torch.empty_strided(
        (BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    with torch.cuda._DeviceGuard(addmm.device.index):
        torch.cuda.set_device(addmm.device)
        raw_stream = get_raw_stream(addmm.device.index)
        _dropout_residual_layernorm_transpose_kernel.run(
            out,
            seeds,
            addmm,
            residual,
            weight,
            bias,
            div,
            SEED_INDEX,
            ROWS,
            HIDDEN,
            stream=raw_stream,
        )
    return out, div


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
    if not EXACT_STOCHASTIC_EQUALITY:
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
                warmup=args.warmup, rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
