"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Reformer training affine LayerNorm plus generated-seed dropout scope in one fixed-hidden Triton row kernel, including fp32 `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, weight/bias epilogue, p=0.05 dropout, scale 1.0526315789473684, and the final contiguous `[8,4096,512]` output, whereas tuned Inductor already lowers this norm-template-canonicalized stochastic normalization to the same required input/affine reads, row reduction, RNG mask, rsqrt, and output store envelope; Inductor cannot materially improve it today through local fusion or algebraic rewrites because the remaining work is mandatory memory traffic plus fixed-width normalization and stochastic epilogue cost rather than an avoidable intermediate; the fix is BANDWIDTH_BOUND: record this as an at-floor timing result but not_true_floor for exact correctness because eager `inductor_random` is stochastic and the only output's values are skipped."""
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

ROWS = 8 * 4096
BATCH = 8
SEQ_LEN = 4096
HIDDEN = 512
INPUT_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
INPUT_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
WEIGHT_SHAPE = (HIDDEN,)
OUTPUT_SHAPE = INPUT_SHAPE
OUTPUT_STRIDE = INPUT_STRIDE
EPS = 1.0e-12
DROPOUT_P = 0.05
DROPOUT_SCALE = 1.0526315789473684
BLOCK_H = 512
SEED_LOW = -9223372036854775808
SEED_HIGH = 9223372036854775807


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
    from torch._inductor.runtime.triton_helpers import libdevice

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.persistent_reduction(
        size_hints={"x": ROWS, "r0_": HIDDEN},
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta={
            "signature": {
                "out_ptr": "*fp32",
                "x_ptr": "*fp32",
                "seed_ptr": "*i64",
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
                    (6,): [["tt.divisibility", 16]],
                    (7,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "oracle_reformer_dropout_layernorm_kernel",
            "mutated_arg_names": ["out_ptr"],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 3,
            "num_store": 1,
            "num_reduction": 4,
            "autotune_hints": set(),
            "tiling_scores": {"x": 0, "r0_": 201330688},
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
    def oracle_kernel(
        out_ptr,
        x_ptr,
        seed_ptr,
        weight_ptr,
        bias_ptr,
        load_seed_offset,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 32768
        r0_numel = 512
        R0_BLOCK: tl.constexpr = 512
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        offsets = r0_index + 512 * xindex

        x = tl.load(
            x_ptr + offsets,
            eviction_policy="evict_first",
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + r0_index,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + r0_index,
            eviction_policy="evict_last",
        ).to(tl.float32)

        x_broadcast = tl.broadcast_to(x, [XBLOCK, R0_BLOCK])
        mean = tl.sum(x_broadcast, 1)[:, None].to(tl.float32) / 512.0
        centered = x - mean
        centered_broadcast = tl.broadcast_to(centered * centered, [XBLOCK, R0_BLOCK])
        variance = tl.sum(centered_broadcast, 1)[:, None].to(tl.float32) / 512.0
        invstd = libdevice.rsqrt(variance + 1.0e-12)

        seed = tl.load(seed_ptr + load_seed_offset)
        random = tl.rand(seed, offsets.to(tl.int32).to(tl.uint32))
        keep = (random > 0.05).to(tl.float32)
        normalized = centered * invstd * weight + bias
        out = keep * normalized * 1.0526315789473684
        tl.store(out_ptr + offsets, out)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
    if stride is not None and tuple(value.stride()) != stride:
        raise ValueError(f"{name} stride {tuple(value.stride())} != {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} dtype {value.dtype} != torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    x = _require_f32_tensor("arg2_1", inputs[0], INPUT_SHAPE, INPUT_STRIDE)
    weight = _require_f32_tensor("arg0_1", inputs[1], WEIGHT_SHAPE, (1,))
    bias = _require_f32_tensor("arg1_1", inputs[2], WEIGHT_SHAPE, (1,))
    if weight.device != x.device or bias.device != x.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, weight, bias


def oracle_forward(inputs):
    """Run the full affine LayerNorm plus dropout scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_reformer_dropout_layernorm.py")

    x, weight, bias = _validate_inputs(inputs)
    seeds = torch.empty_strided((1,), (1,), device=x.device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(SEED_LOW, SEED_HIGH, [1], out=seeds)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )

    device_index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    with torch.cuda._DeviceGuard(device_index):
        torch.cuda.set_device(device_index)
        raw_stream = get_raw_stream(device_index)
        oracle_kernel.run(
            out,
            x,
            seeds,
            weight,
            bias,
            0,
            ROWS,
            HIDDEN,
            stream=raw_stream,
        )
    return out


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
