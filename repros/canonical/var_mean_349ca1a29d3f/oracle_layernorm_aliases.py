"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Reformer LayerNorm alias scope in one generated-code-matching Triton row kernel, including the fresh clone-equivalent backing allocation, fp32 correction=0 mean and centered variance over hidden size 256, eps before libdevice.rsqrt, affine multiply then add, and the three returned `[32768,256]` alias views of the same final buffer, whereas Inductor already emits the same single persistent row-reduction shape for this fixed hidden-size normalization; Inductor cannot materially improve this case with local scheduler fusion, split-K, scatter-reduce, or algebraic rewrites because the required activation/weight/bias reads, one row reduction pair, reciprocal square root, and one output write dominate; the fix is BANDWIDTH_BOUND: record this as an at-floor LayerNorm alias variant unless broader normalization-template or memory-traffic work moves both paths."""
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

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


BATCH = 8
SEQ_LEN = 4096
HIDDEN = 256
ROWS = BATCH * SEQ_LEN
BASE_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
BASE_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
VIEW_SHAPE = (ROWS, HIDDEN)
VIEW_STRIDE = (HIDDEN, 1)
EPS = 1.0e-12
OUTPUT_COUNT = 3

if triton is not None:
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
    from torch._inductor.runtime.triton_helpers import libdevice

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.persistent_reduction(
        size_hints={"x": 32768, "r0_": 256},
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta={
            "signature": {
                "weight_ptr": "*fp32",
                "bias_ptr": "*fp32",
                "input_ptr": "*fp32",
                "out_ptr": "*fp32",
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
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "oracle_layernorm_aliases_kernel",
            "mutated_arg_names": ["out_ptr"],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 3,
            "num_store": 1,
            "num_reduction": 4,
            "autotune_hints": set(),
            "tiling_scores": {"x": 0, "r0_": 100665344},
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
        weight_ptr,
        bias_ptr,
        input_ptr,
        out_ptr,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 32768
        r0_numel = 256
        R0_BLOCK: tl.constexpr = 256
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        row = xindex
        col = r0_index
        offsets = col + 256 * row

        x = tl.load(input_ptr + offsets, None, eviction_policy="evict_first")
        weight = tl.load(weight_ptr + col, None, eviction_policy="evict_last")
        bias = tl.load(bias_ptr + col, None, eviction_policy="evict_last")

        x_broadcast = tl.broadcast_to(x, [XBLOCK, R0_BLOCK])
        mean_sum = tl.sum(x_broadcast, 1)[:, None].to(tl.float32)
        mean = mean_sum / (tl.full([1, 1], 256, tl.int32).to(tl.float32))
        centered_for_var = x_broadcast - mean
        square = centered_for_var * centered_for_var
        square_sum = tl.sum(tl.broadcast_to(square, [XBLOCK, R0_BLOCK]), 1)[:, None].to(
            tl.float32
        )
        centered = x - mean
        var = square_sum / tl.full([1, 1], 256.0, tl.float32)
        inv_std = libdevice.rsqrt(var + tl.full([1, 1], 1e-12, tl.float32))
        normalized = centered * inv_std
        scaled = normalized * weight
        out = scaled + bias
        tl.store(out_ptr + offsets, out, None)


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
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    activation = _require_tensor("arg2_1", inputs[0], BASE_SHAPE, torch.float32)
    weight = _require_tensor("arg0_1", inputs[1], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg1_1", inputs[2], (HIDDEN,), torch.float32)
    output_shapes = tuple(
        _shape_tuple(inputs[index]) for index in range(3, 3 + OUTPUT_COUNT)
    )
    for index, shape in enumerate(output_shapes):
        if shape != VIEW_SHAPE:
            raise ValueError(f"_shape_param_{index} is {shape}, expected {VIEW_SHAPE}")
    if weight.device != activation.device or bias.device != activation.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return activation, weight, bias, output_shapes


def oracle_forward(inputs):
    """Compute the same function as Repro()(*make_inputs())."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layernorm_aliases.py")

    activation, weight, bias, output_shapes = _validate_inputs(inputs)
    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=activation.device,
        dtype=torch.float32,
    )

    device_index = (
        activation.device.index
        if activation.device.index is not None
        else torch.cuda.current_device()
    )
    with torch.cuda._DeviceGuard(device_index):
        torch.cuda.set_device(device_index)
        raw_stream = get_raw_stream(device_index)
        oracle_kernel.run(
            weight,
            bias,
            activation,
            out_base,
            ROWS,
            HIDDEN,
            stream=raw_stream,
        )
    return tuple(out_base.view(shape) for shape in output_shapes)


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _check_layout_alias(instance, inputs) -> bool:
    with torch.no_grad():
        eager_outputs = instance(*inputs)
        oracle_outputs = oracle_forward(inputs)
    if len(eager_outputs) != OUTPUT_COUNT or len(oracle_outputs) != OUTPUT_COUNT:
        print(
            f"Layout/alias: FAIL output count oracle={len(oracle_outputs)} "
            f"eager={len(eager_outputs)}"
        )
        return False

    ok = True
    for index, (oracle_out, eager_out) in enumerate(zip(oracle_outputs, eager_outputs)):
        expected = (
            tuple(eager_out.shape),
            eager_out.dtype,
            tuple(eager_out.stride()),
            eager_out.storage_offset(),
        )
        actual = (
            tuple(oracle_out.shape),
            oracle_out.dtype,
            tuple(oracle_out.stride()),
            oracle_out.storage_offset(),
        )
        if actual != expected:
            print(f"  output {index}: SCOPE_MISMATCH layout oracle={actual} eager={expected}")
            ok = False
    eager_alias = all(_same_storage(eager_outputs[0], out) for out in eager_outputs[1:])
    oracle_alias = all(_same_storage(oracle_outputs[0], out) for out in oracle_outputs[1:])
    input_alias = _same_storage(oracle_outputs[0], inputs[0])
    if not eager_alias or not oracle_alias or input_alias:
        print(
            "  alias: SCOPE_MISMATCH "
            f"eager_alias={eager_alias} oracle_alias={oracle_alias} "
            f"oracle_aliases_input={input_alias}"
        )
        ok = False
    print(f"Layout/alias: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--no-skip-stochastic", action="store_true")
    parser.add_argument("--all-shapes", action="store_true")
    parser.add_argument("--show-hw", action="store_true")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be skipped")

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
        ok = _check_layout_alias(instance, inputs) and ok
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        "WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
