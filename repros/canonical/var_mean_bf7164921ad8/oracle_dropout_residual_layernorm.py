"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Bart/MBart training seeded-dropout residual LayerNorm scope for `[8192, 1024]`, including internal Inductor seed generation, p=0.1 dropout of the addmm view, residual add, population var_mean over hidden size 1024, eps=1e-5 rsqrt, affine scale/bias, and final `[8192, 1024]` view by replaying the tuned full-scope persistent-reduction row kernel, whereas measured Inductor already emits the same fused RNG/dropout/residual/LayerNorm/affine schedule and lands at the same practical floor; Inductor cannot materially improve this norm_template_canonicalization repro through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, recompute fusion, or a new local pattern because the remaining cost is dominated by mandatory f32 activation/residual/affine reads, one row reduction, RNG work, output traffic, and launch overhead; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader normalization-template, RNG, launch, or bandwidth work moves both paths together."""
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
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.async_compile import AsyncCompile
except ImportError:
    triton = None
    tl = None
    get_raw_stream = None
    AsyncCompile = None

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
HIDDEN = 1024
INPUT_2D_SHAPE = (ROWS, HIDDEN)
INPUT_VIEW_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SEED_COUNT = 2
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5
BLOCK_H = 1024
SEED_LOW = -9223372036854775808
SEED_HIGH = 9223372036854775807
CLASSIFICATION = "BANDWIDTH_BOUND"

if triton is not None:

    _async_compile = AsyncCompile()

    _dropout_residual_layernorm_kernel = _async_compile.triton(
        "oracle_bf716_dropout_residual_layernorm",
        '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.hints import ReductionHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'oracle_bf716_dropout_residual_layernorm', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 134225920}, 'backend_hash': 'oracle', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def oracle_bf716_dropout_residual_layernorm(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, load_seed_offset, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 8192
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = (r0_1 + 1024 * x0).to(tl.int32)
    tmp2 = tl.rand(tmp0, tmp1.to(tl.uint32))
    tmp3 = tl.load(in_ptr1 + (r0_1 + 1024 * x0), None, eviction_policy='evict_first')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 1024 * x0), None, eviction_policy='evict_first')
    tmp31 = tl.load(in_ptr3 + r0_1, None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + r0_1, None, eviction_policy='evict_last')
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
    tmp17 = tl.full([1, 1], 1024, tl.int32).to(tl.float32)
    tmp18 = tmp16 / tmp17
    tmp19 = tmp12 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp24 = tmp11 - tmp18
    tmp25 = tl.full([1, 1], 1024.0, tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tl.full([1, 1], 1.0e-5, tl.float32)
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tl.store(in_out_ptr0 + (r0_1 + 1024 * x0), tmp34, None)
''',
        device_str="cuda",
    )

    _async_compile.wait(globals())
    del _async_compile


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
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_3", inputs[0], INPUT_2D_SHAPE, torch.float32)
    residual = _require_tensor("arg0_1", inputs[1], INPUT_VIEW_SHAPE, torch.float32)
    weight = _require_tensor("arg10_1", inputs[2], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg11_1", inputs[3], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[4]) != INPUT_VIEW_SHAPE:
        raise ValueError(f"unexpected first view shape parameter: {inputs[4]!r}")
    if _shape_tuple(inputs[5]) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected final view shape parameter: {inputs[5]!r}")

    device = addmm.device
    for tensor in (residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same CUDA device")
    return addmm, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias = _validate_inputs(inputs)
    view_shape = _shape_tuple(inputs[4])
    output_shape = _shape_tuple(inputs[5])

    viewed = torch.ops.aten.view.default(addmm, view_shape)
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=addmm.device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(view_shape, seed, "rand")
    keep = torch.ops.aten.gt.Scalar(random, DROPOUT_P)
    dropped = torch.ops.aten.mul.Tensor(keep, viewed)
    dropped = torch.ops.aten.mul.Tensor(dropped, DROPOUT_SCALE)
    x = torch.ops.aten.add.Tensor(residual, dropped)
    var, mean = torch.ops.aten.var_mean.correction(x, [2], correction=0, keepdim=True)
    invstd = torch.ops.aten.rsqrt.default(var + EPS)
    normalized = torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(x, mean), invstd)
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    return torch.ops.aten.view.default(affine, output_shape)


def oracle_forward(inputs):
    """Run the complete Repro.forward scope in one shape-specialized row kernel.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same single contiguous f32 [8192, 1024] output.
    """
    addmm, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or get_raw_stream is None:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    seeds = torch.empty_strided(
        (SEED_COUNT,),
        (1,),
        device=addmm.device,
        dtype=torch.int64,
    )
    torch.ops.aten.randint.low_out(SEED_LOW, SEED_HIGH, [SEED_COUNT], out=seeds)

    raw_stream = get_raw_stream(addmm.device.index or 0)
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
