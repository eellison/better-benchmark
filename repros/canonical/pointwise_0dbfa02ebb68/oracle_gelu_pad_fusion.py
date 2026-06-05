"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle does not use a materially different algorithm from Inductor; it hand-runs the same full-scope fused output-space Triton lowering for exact-erf GELU, scale, and right/bottom constant_pad_nd, whereas Inductor already emits one fused pointwise/pad kernel for the captured graph; Inductor cannot materially improve this layout/stencil case today because the full output still requires the interior input read, exact erf math, and contiguous padded output store with no remaining intermediate to eliminate; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope case unless broader pointwise math/codegen or launch-overhead improvements beat the existing fused kernel."""
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
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    get_raw_stream = None
    AsyncCompile = None
    triton_heuristics = None
    DeviceProperties = None
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
    _gelu_scale_pad_default_kernel = _async_compile.triton(
        "_gelu_scale_pad_default_kernel",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.hints import DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={"x": 16777216},
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*fp16",
            "out_ptr0": "*fp16",
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
        "kernel_name": "_gelu_scale_pad_default_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 1,
        "num_store": 1,
        "num_reduction": 0,
        "autotune_hints": set(),
        "tiling_scores": {"x": 94765056},
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
def _gelu_scale_pad_default_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16613376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x1 = ((xindex // 13) % 13)
    x0 = (xindex % 13)
    x2 = xindex // 169
    x3 = xindex
    tmp0 = (x1).to(tl.int32)
    tmp1 = tl.full([1], 12, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = (x0).to(tl.int32)
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + 12 * x1 + 144 * x2), tmp5, other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full([1], 0.5, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp11 = tmp7 * tmp10
    tmp12 = libdevice.erf(tmp11)
    tmp13 = tl.full([1], 1.0, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp9 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.full([1], 1.7015043497085571, tl.float32)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tl.store(out_ptr0 + (x3), tmp20, None)
""",
        device_str="cuda",
    )
    _async_compile.wait(globals())
    del _async_compile

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        ],
        key=["total_in", "H", "W"],
    )
    @triton.jit
    def _gelu_scale_interior_kernel(
        input_ptr,
        output_ptr,
        total_in: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_in

        out_h: tl.constexpr = H + 1
        out_w: tl.constexpr = W + 1
        out_hw: tl.constexpr = out_h * out_w

        w = offsets % W
        tmp = offsets // W
        h = tmp % H
        nc = tmp // H
        out_offsets = nc * out_hw + h * out_w + w

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        erf_term = tl.math.erf(x * 0.7071067811865476) + 1.0
        gelu = x * 0.5 * erf_term
        scaled = gelu * 1.7015043497085571
        tl.store(output_ptr + out_offsets, scaled, mask=mask)

    @triton.jit
    def _zero_pad_kernel(
        output_ptr,
        total_pad: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_pad

        out_h: tl.constexpr = H + 1
        out_w: tl.constexpr = W + 1
        out_hw: tl.constexpr = out_h * out_w
        pad_per_plane: tl.constexpr = H + out_w

        pos = offsets % pad_per_plane
        nc = offsets // pad_per_plane
        right_col = pos < H
        row_pad_offset = pos * out_w + W
        bottom_pad_offset = H * out_w + (pos - H)
        within_plane = tl.where(right_col, row_pad_offset, bottom_pad_offset)
        tl.store(output_ptr + nc * out_hw + within_plane, 0.0, mask=mask)


def _expected_output_layout(x: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    batch, channels, height, width = x.shape
    shape = (batch, channels, height + 1, width + 1)
    stride = (channels * (height + 1) * (width + 1), (height + 1) * (width + 1), width + 1, 1)
    return shape, stride


def oracle_forward(inputs):
    """Run the full Repro.forward GELU-scale plus constant_pad_nd scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gelu_pad_fusion.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects 1 input, got {len(inputs)}")

    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor")
    if x.dtype != torch.float16:
        raise TypeError(f"{REPRO_ID} expects f16 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA input")
    if x.dim() != 4:
        raise ValueError(f"{REPRO_ID} expects a 4D NCHW input, got shape {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects the captured contiguous input layout")

    batch, channels, height, width = x.shape
    out_shape, out_stride = _expected_output_layout(x)
    output = torch.empty_strided(
        out_shape,
        out_stride,
        device=x.device,
        dtype=torch.float16,
    )
    if tuple(x.shape) == (128, 768, 12, 12):
        _gelu_scale_pad_default_kernel.run(
            x,
            output,
            output.numel(),
            stream=get_raw_stream(x.device.index),
        )
        return output

    total_in = x.numel()
    interior_grid = lambda meta: (triton.cdiv(total_in, meta["BLOCK_SIZE"]),)
    _gelu_scale_interior_kernel[interior_grid](
        x,
        output,
        total_in=total_in,
        H=height,
        W=width,
    )
    total_pad = batch * channels * (height + width + 1)
    _zero_pad_kernel[(triton.cdiv(total_pad, 256),)](
        output,
        total_pad=total_pad,
        H=height,
        W=width,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return output


def _check_layout(output: torch.Tensor, x: torch.Tensor) -> bool:
    out_shape, out_stride = _expected_output_layout(x)
    return tuple(output.shape) == out_shape and tuple(output.stride()) == out_stride


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
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout(layout_out, inputs[0])
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_out.shape)} stride={layout_out.stride()})"
        )
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
