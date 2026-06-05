"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete M2M100 training dropout-residual LayerNorm scope in one Triton row-reduction kernel, including the `[8192,1024] -> [64,128,1024]` view, internally generated Inductor seed-index-0 dropout, residual add, fp32 population var_mean over hidden size 1024, eps=1e-5 rsqrt, affine scale/bias, and final contiguous `[8192,1024]` view, whereas tuned Inductor already emits a fused persistent norm kernel for the same full scope; Inductor cannot materially improve this repro through a local scheduler-fusion, scatter-reduce, split-K, algebraic, or recompute rewrite because the remaining work is the required activation/residual/affine reads, one stochastic mask, one fixed-hidden reduction, rsqrt, and output store; the fix is BANDWIDTH_BOUND: keep this norm-template-canonicalization case closed unless a broader normalization/RNG template moves both paths. Exact stochastic equality with eager is not established for the internally generated seed, so checks verify layout and skip stochastic values."""
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
BATCH = 64
SEQ_LEN = 128
HIDDEN = 1024
INPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SEED_COUNT = 4
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5
BLOCK_H = 1024
EXACT_STOCHASTIC_EQUALITY = False


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
                    (7,): [["tt.divisibility", 16]],
                    (8,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "triton_per_fused_add_gt_inductor_lookup_seed_inductor_random_mul_rsqrt_sub_var_mean_view_0",
            "mutated_arg_names": ["out_ptr"],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 4,
            "num_store": 1,
            "num_reduction": 4,
            "autotune_hints": set(),
            "tiling_scores": {"x": 0, "r0_": 134225920},
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
    def _dropout_residual_layernorm_kernel(
        out_ptr,
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
        r0_numel = 1024
        R0_BLOCK: tl.constexpr = 1024
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_offset = 0
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        x0 = xindex
        tmp3 = tl.load(residual_ptr + (r0_1 + 1024 * x0), None, eviction_policy="evict_first")
        tmp7 = tl.load(addmm_ptr + (r0_1 + 1024 * x0), None, eviction_policy="evict_first")
        tmp31 = tl.load(weight_ptr + (r0_1), None, eviction_policy="evict_last")
        tmp33 = tl.load(bias_ptr + (r0_1), None, eviction_policy="evict_last")
        tmp0 = tl.load(seeds_ptr + load_seed_offset)
        tmp1 = (r0_1 + 1024 * x0).to(tl.int32)
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp4 = tl.full([1, 1], 0.1, tl.float32)
        tmp5 = tmp2 > tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tl.full([1, 1], 1.1111111111111112, tl.float32)
        tmp10 = tmp8 * tmp9
        tmp11 = tmp3 + tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp16 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
        tmp17 = (tl.full([1, 1], 1024, tl.int32)).to(tl.float32)
        tmp18 = tmp16 / tmp17
        tmp19 = tmp12 - tmp18
        tmp20 = tmp19 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
        tmp24 = tmp11 - tmp18
        tmp25 = tl.full([1, 1], 1024.0, tl.float32)
        tmp26 = tmp23 / tmp25
        tmp27 = tl.full([1, 1], 1e-05, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp30 * tmp31
        tmp34 = tmp32 + tmp33
        tl.store(out_ptr + (r0_1 + 1024 * x0), tmp34, None)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_3", inputs[0], INPUT_SHAPE, torch.float32)
    residual = _require_tensor("arg2_1", inputs[1], RESIDUAL_SHAPE, torch.float32)
    weight = _require_tensor("arg13_1", inputs[2], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg14_1", inputs[3], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[4]) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected input view shape parameter: {inputs[4]!r}")
    if _shape_tuple(inputs[5]) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {inputs[5]!r}")
    if (
        residual.device != addmm.device
        or weight.device != addmm.device
        or bias.device != addmm.device
    ):
        raise ValueError("all tensor inputs must be on the same device")

    return addmm, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias = _validate_inputs(inputs)
    shape0 = _shape_tuple(inputs[4])
    shape1 = _shape_tuple(inputs[5])
    view = torch.ops.aten.view.default(addmm, shape0)
    seeds = torch.empty_strided(
        (SEED_COUNT,),
        (1,),
        device=addmm.device,
        dtype=torch.int64,
    )
    torch.ops.aten.randint.low_out(
        -9223372036854775808,
        9223372036854775807,
        [SEED_COUNT],
        out=seeds,
    )
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape0, seed, "rand")
    dropped = (random > DROPOUT_P) * view
    dropped = dropped * DROPOUT_SCALE
    x = residual + dropped
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    invstd = torch.ops.aten.rsqrt.default(variance + EPS)
    normalized = (x - mean) * invstd
    affine = normalized * weight + bias
    return torch.ops.aten.view.default(affine, shape1)


def oracle_forward(inputs):
    """Run the full dropout-residual LayerNorm computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    addmm, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=addmm.device)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    device_index = (
        addmm.device.index if addmm.device.index is not None else torch.cuda.current_device()
    )
    with torch.cuda._DeviceGuard(device_index):
        torch.cuda.set_device(device_index)
        raw_stream = get_raw_stream(device_index)
        _dropout_residual_layernorm_kernel.run(
            output,
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
    return output


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if isinstance(actual, torch.Tensor) and actual.is_cuda:
            torch.cuda.synchronize()

    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and tuple(actual.shape) == tuple(expected.shape)
        and actual.stride() == expected.stride()
        and actual.dtype == expected.dtype
    )
    expected_stride = expected.stride() if isinstance(expected, torch.Tensor) else None
    actual_stride = actual.stride() if isinstance(actual, torch.Tensor) else None
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected_stride}, oracle_stride={actual_stride})"
    )
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
    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")
    if not EXACT_STOCHASTIC_EQUALITY:
        print(
            "NOTE: exact stochastic equality is not established for the internally "
            "generated Inductor seed; output value checks rely on stochastic skipping"
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
