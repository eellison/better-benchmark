"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete attention-head scale and layout materialization by writing the fresh cloned `[B, H, D, S]` storage directly from the contiguous `[B*S, H*D]` projection with one scaled pointwise layout kernel and then returning the eager-compatible non-contiguous `[B*H, S, D]` view, whereas tuned Inductor already lowers the captured view/permute/mul/expand/clone/view/permute chain to the same single materialization over the same storage; Inductor cannot materially improve this local repro because the output contract requires reading and scaling every f32 element and writing the full transposed clone backing, so the fix is BANDWIDTH_BOUND: record this as at floor unless broader layout-copy bandwidth or launch-overhead work moves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.config as inductor_config

inductor_config.coordinate_descent_tuning = True

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

if triton is not None:

    _async_compile = AsyncCompile()
    _scaled_head_layout_inductor_kernel = _async_compile.triton(
        "_scaled_head_layout_inductor_kernel",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties, TileHint
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={"y": 32768, "x": 512},
    tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*fp32",
            "out_ptr0": "*fp32",
            "ynumel": "i32",
            "xnumel": "i32",
            "YBLOCK": "constexpr",
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
                (3,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid2D",
        "kernel_name": "_scaled_head_layout_inductor_kernel",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 1,
        "num_store": 1,
        "num_reduction": 0,
        "autotune_hints": set(),
        "tiling_scores": {"y": 67108864, "x": 134217728},
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
        "deterministic": False,
        "batch_invariant": False,
        "force_filter_reduction_configs": False,
        "mix_order_reduction_allow_multi_stages": True,
        "dynamic_disable_pipelining": True,
    },
    min_elem_per_thread=0,
)
@triton.jit
def _scaled_head_layout_inductor_kernel(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 32768
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 4096
    y1 = yindex // 4096
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4096 * x2 + 2097152 * y1), xmask)
    tmp1 = tl.full([1, 1], 0.3535533905932738, tl.float32)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + 512 * y3), tmp2, xmask)
""",
        device_str="cuda",
    )
    _async_compile.wait(globals())
    del _async_compile

    @triton.autotune(
        configs=[
            triton.Config({"YBLOCK": 1, "XBLOCK": 512}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 8, "XBLOCK": 128}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 64}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 128}, num_warps=8, num_stages=3),
            triton.Config({"YBLOCK": 32, "XBLOCK": 64}, num_warps=8, num_stages=3),
            triton.Config({"YBLOCK": 32, "XBLOCK": 128}, num_warps=8, num_stages=3),
        ],
        key=["YNUMEL", "XNUMEL", "HIDDEN"],
    )
    @triton.jit
    def _scaled_head_layout_2d_kernel(
        input_ptr,
        output_ptr,
        YNUMEL: tl.constexpr,
        XNUMEL: tl.constexpr,
        HIDDEN: tl.constexpr,
        YBLOCK: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        yoffset = tl.program_id(1) * YBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        mask = (yindex < YNUMEL) & (xindex < XNUMEL)

        inner = yindex % HIDDEN
        batch = yindex // HIDDEN
        input_offsets = inner + HIDDEN * xindex + HIDDEN * XNUMEL * batch
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0) * 0.3535533905932738
        tl.store(output_ptr + xindex + XNUMEL * yindex, values, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _resolve_view_shape(value: Any, numel: int) -> tuple[int, ...]:
    dims = list(_shape_tuple(value))
    neg_one_count = dims.count(-1)
    if neg_one_count > 1:
        raise ValueError(f"only one inferred dimension is valid, got {dims}")
    if neg_one_count == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer shape {dims} for numel={numel}")
        dims[dims.index(-1)] = numel // known
    resolved = tuple(dims)
    if _numel(resolved) != numel:
        raise ValueError(f"shape {resolved} has {_numel(resolved)} elements, expected {numel}")
    return resolved


def _validate_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int], int, int, int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    addmm_68, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(addmm_68, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not addmm_68.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if addmm_68.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {addmm_68.dtype}")
    if addmm_68.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects a rank-2 input, got shape={tuple(addmm_68.shape)}")
    if not addmm_68.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(addmm_68.stride())}")

    rows = int(addmm_68.shape[0])
    hidden = int(addmm_68.shape[1])
    total = int(addmm_68.numel())

    batch, seq, view_hidden = _resolve_view_shape(shape0, total)
    if (batch * seq, view_hidden) != (rows, hidden):
        raise ValueError(
            f"first view shape {(batch, seq, view_hidden)} does not match input "
            f"shape={tuple(addmm_68.shape)}"
        )

    batch1, seq1, heads, head_dim = _resolve_view_shape(shape1, total)
    if (batch1, seq1) != (batch, seq) or heads * head_dim != hidden:
        raise ValueError(
            f"head view shape {(batch1, seq1, heads, head_dim)} is incompatible "
            f"with first view {(batch, seq, hidden)}"
        )

    clone_shape = _resolve_view_shape(shape2, total)
    expected_clone_shape = (batch, heads, head_dim, seq)
    if clone_shape != expected_clone_shape:
        raise ValueError(f"unexpected clone shape {clone_shape}, expected {expected_clone_shape}")

    base_view_shape = _resolve_view_shape(shape3, total)
    expected_base_view_shape = (batch * heads, head_dim, seq)
    if base_view_shape != expected_base_view_shape:
        raise ValueError(
            f"unexpected base view shape {base_view_shape}, expected {expected_base_view_shape}"
        )

    output_shape = (batch * heads, seq, head_dim)
    output_stride = (head_dim * seq, 1, seq)
    return addmm_68, output_shape, output_stride, batch, seq, heads, head_dim


@oracle_impl(hardware="H100", shapes="(T([4096, 4096], f32), S([8, 512, 4096]), S([8, 512, -1, 64]), S([8, 64, 64, 512]), S([512, 64, 512]))")
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
    if triton is None or get_raw_stream is None:
        raise RuntimeError("Triton is required for this oracle")

    addmm_68, output_shape, output_stride, batch, seq, heads, head_dim = _validate_layout(inputs)
    output = torch.empty_strided(
        output_shape,
        output_stride,
        device=addmm_68.device,
        dtype=addmm_68.dtype,
    )
    ynumel = batch * heads * head_dim

    if batch == 8 and seq == 512 and heads == 64 and head_dim == 64:
        device_index = addmm_68.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        _scaled_head_layout_inductor_kernel.run(
            addmm_68,
            output,
            ynumel,
            seq,
            stream=get_raw_stream(device_index),
        )
        return output

    grid = lambda meta: (
        triton.cdiv(seq, meta["XBLOCK"]),
        triton.cdiv(ynumel, meta["YBLOCK"]),
    )
    _scaled_head_layout_2d_kernel[grid](
        addmm_68,
        output,
        YNUMEL=ynumel,
        XNUMEL=seq,
        HIDDEN=heads * head_dim,
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
