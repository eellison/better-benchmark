"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DistilBERT token embedding gather, sliced position embedding gather, add, population var_mean LayerNorm with eps=1e-12, affine epilogue, and three aliasing [32768,768] view returns in one shape-specialized Triton row kernel, whereas Inductor already emits one fused row-reduction kernel for the same gather/norm/affine scope; Inductor cannot materially improve this repro through a local scheduler rewrite because the remaining work is dominated by mandatory embedding, position, affine reads, row statistics, and full output stores; the fix is BANDWIDTH_BOUND: record this as an at-floor embedding LayerNorm case unless broader normalization-template tuning moves the family."""
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
    has_stochastic_ops,
)


BATCH = 256
SEQ_LEN = 128
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
TOKEN_VOCAB = 30522
POSITION_VOCAB = 512
POSITION_WIDTH = 512
BASE_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
BASE_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
VIEW_SHAPE = (ROWS, HIDDEN)
VIEW_STRIDE = (HIDDEN, 1)
EPS = 1.0e-12
BLOCK_N = 1024
OUTPUT_COUNT = 3


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
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
        size_hints={"x": 32768, "r0_": 1024},
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta={
            "signature": {
                "token_ids_ptr": "*i64",
                "token_table_ptr": "*fp32",
                "position_ids_ptr": "*i64",
                "position_table_ptr": "*fp32",
                "weight_ptr": "*fp32",
                "bias_ptr": "*fp32",
                "out_ptr": "*fp32",
                "xnumel": "i32",
                "r0_numel": "i32",
                "XBLOCK": "constexpr",
                "R0_BLOCK": "constexpr",
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
                    (7,): [["tt.divisibility", 16]],
                    (8,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "oracle_distilbert_embedding_layernorm_kernel",
            "mutated_arg_names": ["out_ptr"],
            "optimize_mem": True,
            "no_x_dim": False,
            "atomic_add_found": False,
            "num_load": 4,
            "num_store": 1,
            "num_reduction": 2,
            "autotune_hints": set(),
            "tiling_scores": {"x": 263168, "r0_": 201332736},
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
    def _distilbert_embedding_layernorm_kernel(
        token_ids_ptr,
        token_table_ptr,
        position_ids_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
        R0_BLOCK: tl.constexpr,
    ):
        xnumel = 32768
        r0_numel = 768
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        x3 = xindex
        token_ids = tl.load(token_ids_ptr + x3, None, eviction_policy="evict_last")
        seq_index = xindex % 128
        position_ids = tl.load(position_ids_ptr + seq_index, None, eviction_policy="evict_last")
        running_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        running_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        running_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            tl.device_assert((0 <= token_ids) & (token_ids < 30522), "index out of bounds: 0 <= token_id < 30522")
            token = tl.load(
                token_table_ptr + (r0_index + 768 * token_ids),
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            tl.device_assert((0 <= position_ids) & (position_ids < 512), "index out of bounds: 0 <= position_id < 512")
            position = tl.load(
                position_table_ptr + (r0_index + 768 * position_ids),
                r0_mask,
                eviction_policy="evict_last",
                other=0.0,
            )
            values = token + position
            next_mean, next_m2, next_weight = triton_helpers.welford_reduce(
                values,
                running_mean,
                running_m2,
                running_weight,
                r0_offset == 0,
            )
            running_mean = tl.where(r0_mask, next_mean, running_mean)
            running_m2 = tl.where(r0_mask, next_m2, running_m2)
            running_weight = tl.where(r0_mask, next_weight, running_weight)
        mean, m2, _ = triton_helpers.welford(running_mean, running_m2, running_weight, 1)
        mean = mean[:, None]
        variance_accum = m2[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            gamma = tl.load(weight_ptr + r0_index, r0_mask, eviction_policy="evict_last", other=0.0)
            beta = tl.load(bias_ptr + r0_index, r0_mask, eviction_policy="evict_last", other=0.0)
            tl.device_assert((0 <= token_ids) & (token_ids < 30522), "index out of bounds: 0 <= token_id < 30522")
            token = tl.load(
                token_table_ptr + (r0_index + 768 * token_ids),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tl.device_assert((0 <= position_ids) & (position_ids < 512), "index out of bounds: 0 <= position_id < 512")
            position = tl.load(
                position_table_ptr + (r0_index + 768 * position_ids),
                r0_mask,
                eviction_policy="evict_first",
                other=0.0,
            )
            values = token + position
            centered = values - mean
            variance = variance_accum / tl.full([1, 1], 768.0, tl.float32)
            invstd = libdevice.rsqrt(variance + tl.full([1, 1], 1e-12, tl.float32))
            normalized = centered * invstd
            scaled = normalized * gamma
            out = scaled + beta
            tl.store(out_ptr + (r0_index + 768 * x3), out, r0_mask)

    from torch._inductor.async_compile import AsyncCompile

    _async_compile = AsyncCompile()
    import torch._inductor.config as _inductor_config

    _inductor_config.coordinate_descent_tuning = True
    _distilbert_generated_kernel = _async_compile.triton(
        "oracle_generated_distilbert_embedding_layernorm_kernel",
        r'''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'oracle_generated_distilbert_embedding_layernorm_kernel', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 263168, 'r0_': 201332736}, 'backend_hash': '2FF7A49C450C045FF2AC7A730B5A8D4CD6DF40914BC38271CC70D96733C19B52', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False}
)
@triton.jit
def oracle_generated_distilbert_embedding_layernorm_kernel(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 768
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x0 = (xindex % 128)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tl.device_assert((0 <= tmp0) & (tmp0 < 30522), "index out of bounds: 0 <= tmp0 < 30522")
        tmp2 = tl.load(in_ptr1 + (r0_2 + 768*tmp0), r0_mask, eviction_policy='evict_last', other=0.0)
        tl.device_assert((0 <= tmp3) & (tmp3 < 512), "index out of bounds: 0 <= tmp3 < 512")
        tmp5 = tl.load(in_ptr3 + (r0_2 + 768*tmp3), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(r0_mask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(r0_mask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(r0_mask, tmp8_weight_next, tmp8_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp8_mean, tmp8_m2, tmp8_weight, 1)
    tmp8 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp26 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tl.device_assert((0 <= tmp0) & (tmp0 < 30522), "index out of bounds: 0 <= tmp0 < 30522")
        tmp15 = tl.load(in_ptr1 + (r0_2 + 768*tmp0), r0_mask, eviction_policy='evict_first', other=0.0)
        tl.device_assert((0 <= tmp3) & (tmp3 < 512), "index out of bounds: 0 <= tmp3 < 512")
        tmp17 = tl.load(in_ptr3 + (r0_2 + 768*tmp3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp18 = tmp15 + tmp17
        tmp19 = tmp18 - tmp8
        tmp20 = tl.full([1, 1], 768.0, tl.float32)
        tmp21 = (tmp12 / tmp20)
        tmp22 = tl.full([1, 1], 1e-12, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tl.store(out_ptr2 + (r0_2 + 768*x3), tmp29, r0_mask)
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


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    shape = _shape_tuple(value)
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[tuple[int, int], ...],
]:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects 9 inputs, got {len(inputs)}")

    token_table = _require_tensor("arg1_1", inputs[0], (TOKEN_VOCAB, HIDDEN), torch.float32)
    token_ids = _require_tensor("arg0_1", inputs[1], (BATCH, SEQ_LEN), torch.int64)
    position_ids = _require_tensor("arg2_1", inputs[2], (1, POSITION_WIDTH), torch.int64)
    position_table = _require_tensor("arg3_1", inputs[3], (POSITION_VOCAB, HIDDEN), torch.float32)
    weight = _require_tensor("arg4_1", inputs[4], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg5_1", inputs[5], (HIDDEN,), torch.float32)
    output_shapes = tuple(
        _require_shape(f"_shape_param_{index}", inputs[index + 6], VIEW_SHAPE)
        for index in range(OUTPUT_COUNT)
    )

    tensor_inputs = (token_table, token_ids, position_ids, position_table, weight, bias)
    if any(value.device != token_table.device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return token_table, token_ids, position_ids, position_table, weight, bias, output_shapes


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the complete DistilBERT embedding LayerNorm alias scope.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns
    the same three float32 [32768,768] views. The returned tensors are views
    of one contiguous [256,128,768] base buffer, matching the eager repeated
    `view(add_tensor_2, shape)` alias and layout contract.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_distilbert_embedding_layernorm.py")

    token_table, token_ids, position_ids, position_table, weight, bias, output_shapes = _validate_inputs(inputs)
    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=token_table.device,
        dtype=torch.float32,
    )
    device_index = (
        token_table.device.index
        if token_table.device.index is not None
        else torch.cuda.current_device()
    )
    with torch.cuda._DeviceGuard(device_index):
        torch.cuda.set_device(device_index)
        raw_stream = get_raw_stream(device_index)
        _distilbert_generated_kernel.run(
            token_ids,
            token_table,
            position_ids,
            position_table,
            weight,
            bias,
            out_base,
            ROWS,
            HIDDEN,
            stream=raw_stream,
        )
    return tuple(out_base.view(shape) for shape in output_shapes)


def _layout_alias_check(outputs: Any) -> bool:
    tensors = tuple(outputs)
    if len(tensors) != OUTPUT_COUNT:
        print(f"Layout/Alias: FAIL (expected {OUTPUT_COUNT} outputs, got {len(tensors)})")
        return False

    all_pass = True
    for index, tensor in enumerate(tensors):
        shape_ok = tuple(tensor.shape) == VIEW_SHAPE
        stride_ok = tuple(tensor.stride()) == VIEW_STRIDE
        offset_ok = tensor.storage_offset() == 0
        dtype_ok = tensor.dtype == torch.float32
        if not (shape_ok and stride_ok and offset_ok and dtype_ok):
            print(
                "Layout/Alias: FAIL "
                f"output {index} shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
                f"storage_offset={tensor.storage_offset()} dtype={tensor.dtype}"
            )
            all_pass = False

    data_ptrs = [tensor.data_ptr() for tensor in tensors]
    alias_ok = len(set(data_ptrs)) == 1
    if not alias_ok:
        print(f"Layout/Alias: FAIL data_ptrs={data_ptrs}")
        all_pass = False

    if all_pass:
        print(
            "Layout/Alias: PASS "
            f"(shape={list(VIEW_SHAPE)} stride={VIEW_STRIDE} shared_data_ptr={data_ptrs[0]})"
        )
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
        with torch.no_grad():
            layout_ok = _layout_alias_check(oracle_forward(inputs))
        ok = ok and layout_ok
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
