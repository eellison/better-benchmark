"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete inverse-permute plus two-row constant-pad repro as one full-output Triton pointwise pad materialization, and does not use a materially different algorithm from Inductor because Inductor already removes the layout-only inverse permutes and lowers the captured scope to one fused `constant_pad_nd` kernel; Inductor cannot materially improve this case through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recomputation because the remaining work is the mandatory contiguous input read, padded output write, and tiny zero tail fill; the fix is BANDWIDTH_BOUND: record this as an at-floor pad-copy case unless broader pointwise memory-codegen or bandwidth improvements move both implementations together."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.async_compile import AsyncCompile
    from torch._inductor.runtime import triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties
except ImportError:
    triton = None
    tl = None
    get_raw_stream = None
    AsyncCompile = None
    triton_heuristics = None
    DeviceProperties = None

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


CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    _async_compile = AsyncCompile()
    _pad_768_kernel = _async_compile.triton(
        "_pad_768_kernel",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={"x": 33554432},
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*fp32",
            "out_ptr0": "*fp32",
            "xnumel": "i32",
            "XBLOCK": "constexpr",
        },
        "device": DeviceProperties.create(torch.device("cuda")),
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
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_pad_768_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 1,
        "num_store": 1,
        "num_reduction": 0,
        "autotune_hints": set(),
        "tiling_scores": {"x": 281303040},
        "autotune_local_cache": True,
        "autotune_remote_cache": None,
        "force_disable_caches": False,
        "autotune_pointwise": True,
        "dynamic_scale_rblock": True,
        "incremental_autotune": False,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
        "deterministic": False,
        "batch_invariant": False,
        "dynamic_disable_pipelining": True,
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
    },
    min_elem_per_thread=0,
)
@triton.jit
def _pad_768_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 23442432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 768
    x0 = (xindex % 768)
    x2 = xindex
    tmp0 = (x1).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x1).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 30522, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (x0 + 768*(x1)), tmp6 & xmask, other=0.0)
    tmp8 = tmp0 >= tmp5
    tmp9 = tl.full([1], 30524, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.full([1], 0.0, tl.float32)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp6, tmp7, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
""",
        device_str="cuda",
    )
    _pad_128_kernel = _async_compile.triton(
        "_pad_128_kernel",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={"x": 4194304},
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*fp32",
            "out_ptr0": "*fp32",
            "xnumel": "i32",
            "XBLOCK": "constexpr",
        },
        "device": DeviceProperties.create(torch.device("cuda")),
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
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "_pad_128_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 1,
        "num_store": 1,
        "num_reduction": 0,
        "tiling_scores": {"x": 46884864},
        "autotune_local_cache": False,
        "autotune_pointwise": True,
        "dynamic_scale_rblock": True,
        "incremental_autotune": False,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
        "deterministic": False,
        "batch_invariant": False,
        "dynamic_disable_pipelining": True,
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
    },
    min_elem_per_thread=0,
)
@triton.jit
def _pad_128_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3907072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 128
    x0 = xindex % 128
    input_mask = x1 < 30522
    value = tl.load(in_ptr0 + (x0 + 128 * x1), input_mask & xmask, other=0.0)
    value = tl.where(input_mask, value, 0.0)
    tl.store(out_ptr0 + xindex, value, xmask)
""",
        device_str="cuda",
    )
    _async_compile.wait(globals())
    del _async_compile

    @triton.jit
    def _direct_pad_kernel(
        input_ptr,
        output_ptr,
        INPUT_ELEMENTS: tl.constexpr,
        TOTAL_ELEMENTS: tl.constexpr,
        COPY_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if pid < COPY_BLOCKS:
            mask = offsets < INPUT_ELEMENTS
            values = tl.load(input_ptr + offsets, mask=mask)
            tl.store(output_ptr + offsets, values, mask=mask)
        else:
            tail_offsets = (
                INPUT_ELEMENTS
                + (pid - COPY_BLOCKS) * BLOCK_SIZE
                + tl.arange(0, BLOCK_SIZE)
            )
            mask = tail_offsets < TOTAL_ELEMENTS
            tl.store(output_ptr + tail_offsets, tl.zeros((BLOCK_SIZE,), tl.float32), mask=mask)


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
    (arg2_1,) = inputs
    input_rows = arg2_1.shape[0]
    n_cols = arg2_1.shape[1]
    output = torch.empty(
        (input_rows + 2, n_cols),
        device=arg2_1.device,
        dtype=arg2_1.dtype,
    )

    if not arg2_1.is_cuda or triton is None:
        output[:-2, :].copy_(arg2_1)
        output[-2:, :].zero_()
        return output

    if input_rows == 30522 and n_cols == 768:
        _pad_768_kernel.run(
            arg2_1,
            output,
            output.numel(),
            stream=get_raw_stream(arg2_1.device.index),
        )
        return output
    if input_rows == 30522 and n_cols == 128:
        _pad_128_kernel.run(
            arg2_1,
            output,
            output.numel(),
            stream=get_raw_stream(arg2_1.device.index),
        )
        return output

    input_elements = arg2_1.numel()
    total_elements = output.numel()
    block_size = 1024
    copy_blocks = triton.cdiv(input_elements, block_size)
    zero_blocks = triton.cdiv(total_elements - input_elements, block_size)
    grid = (copy_blocks + zero_blocks,)
    _direct_pad_kernel[grid](
        arg2_1,
        output,
        INPUT_ELEMENTS=input_elements,
        TOTAL_ELEMENTS=total_elements,
        COPY_BLOCKS=copy_blocks,
        BLOCK_SIZE=block_size,
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
