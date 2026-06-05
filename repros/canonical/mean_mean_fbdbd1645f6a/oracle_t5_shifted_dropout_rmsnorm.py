"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete T5 token embedding gather plus shifted decoder-token construction, two independent Inductor-seeded dropout masks from seed indices 0 and 50 of 124 generated seeds, two fp32 hidden-size-768 RMSNorm mean(square)+rsqrt eps=1e-6 branches, separate affine weight multiplies, and the two returned [8192,768] views in one persistent Triton row kernel, matching the canonical fused scope Inductor already emits; exact stochastic value equality with the eager repro is not established, so this is full-scope structurally but true_floor=no/not_true_floor, whereas Inductor is already at the same one-kernel normalization-template shape; Inductor cannot be assigned a material local optimization gap today because the remaining work is dominated by mandatory indexed embedding reads, RNG, reductions, affine multiplies, and output stores rather than an avoidable scheduler split; the fix is BANDWIDTH_BOUND: mark this repro at floor for local norm-template work and only reopen for broader bandwidth/RNG/codegen improvements."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

BATCH = 8
SEQ_LEN = 1024
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
VOCAB = 32128
TOKEN_SHAPE = (BATCH, SEQ_LEN)
BASE_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
BASE_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
VIEW_SHAPE = (ROWS, HIDDEN)
EPS = 1.0e-6
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 124
SEED_INDEX_0 = 0
SEED_INDEX_1 = 50
IGNORE_INDEX = -100
BLOCK_M = 1
BLOCK_H = 1024
EXACT_STOCHASTIC_EQUALITY = False
TRUE_FLOOR = False

if triton is not None:
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
    from torch._inductor.runtime.triton_helpers import libdevice

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.persistent_reduction(
        size_hints={"x": 8192, "r0_": 1024},
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta={
            "signature": {
                "out0_ptr": "*fp32",
                "out1_ptr": "*fp32",
                "seeds_ptr": "*i64",
                "input_ids_ptr": "*i64",
                "embedding_ptr": "*fp32",
                "shifted_ids_ptr": "*i64",
                "weight0_ptr": "*fp32",
                "weight1_ptr": "*fp32",
                "seed_index0": "i32",
                "seed_index1": "i32",
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
                    (7,): [["tt.divisibility", 16]],
                    (10,): [["tt.divisibility", 16]],
                    (11,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "oracle_t5_shifted_dual_dropout_rmsnorm_kernel",
            "mutated_arg_names": ["out0_ptr", "out1_ptr"],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 4,
            "num_store": 2,
            "num_reduction": 2,
            "autotune_hints": set(),
            "tiling_scores": {"x": 131072, "r0_": 100669440},
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
    def _t5_shifted_dual_dropout_rmsnorm_kernel(
        out0_ptr,
        out1_ptr,
        seeds_ptr,
        input_ids_ptr,
        embedding_ptr,
        shifted_ids_ptr,
        weight0_ptr,
        weight1_ptr,
        seed_index0,
        seed_index1,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 8192
        r0_numel = 768
        R0_BLOCK: tl.constexpr = 1024
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_offset = 0
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        x0 = xindex
        x2 = xindex % 1024
        tmp8 = tl.load(input_ids_ptr + x0, None, eviction_policy="evict_last")
        tmp42 = tl.load(
            weight0_ptr + r0_1,
            r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        tmp50 = tl.load(
            weight1_ptr + r0_1,
            r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        tmp0 = tl.load(seeds_ptr + seed_index1)
        tmp1 = (r0_1 + 768 * x0).to(tl.int32)
        tmp2 = tl.rand(tmp0, tmp1.to(tl.uint32))
        tmp3 = tl.load(seeds_ptr + seed_index0)
        tmp4 = tl.rand(tmp3, tmp1.to(tl.uint32))
        tmp5 = tl.full([1, 1], 0.1, tl.float32)
        tmp6 = tmp4 > tmp5
        tmp7 = tmp6.to(tl.float32)
        tl.device_assert(
            (0 <= tmp8) & (tmp8 < 32128),
            "index out of bounds: 0 <= tmp8 < 32128",
        )
        tmp10 = tl.load(
            embedding_ptr + (r0_1 + 768 * tmp8),
            r0_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full([1, 1], 1.1111111111111112, tl.float32)
        tmp13 = tmp11 * tmp12
        tmp14 = tmp13 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = tl.where(r0_mask, tmp15, 0)
        tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
        tmp19 = tmp2 > tmp5
        tmp20 = tmp19.to(tl.float32)
        tmp21 = x2.to(tl.int32)
        tmp22 = tl.full([1, 1], 0, tl.int32)
        tmp23 = tmp21 == tmp22
        tmp24 = tl.full([1, 1], 1, tl.int64)
        tmp25 = tmp21 >= tmp24
        tmp26 = tl.load(
            shifted_ids_ptr + tl.broadcast_to((-1) + x0, [XBLOCK, R0_BLOCK]),
            r0_mask & tmp25,
            eviction_policy="evict_last",
            other=0.0,
        )
        tmp27 = tl.full([1, 1], 0, tl.int64)
        tmp28 = tl.where(tmp25, tmp26, tmp27)
        tmp29 = tl.where(tmp23, tmp27, tmp28)
        tmp30 = tl.full([1, 1], -100, tl.int64)
        tmp31 = tmp29 == tmp30
        tmp32 = tl.where(tmp31, tmp27, tmp29)
        tl.device_assert(
            ((0 <= tmp32) & (tmp32 < 32128)) | ~r0_mask,
            "index out of bounds: 0 <= tmp32 < 32128",
        )
        tmp34 = tl.load(
            embedding_ptr + (r0_1 + 768 * tmp32),
            r0_mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        tmp35 = tmp20 * tmp34
        tmp36 = tmp35 * tmp12
        tmp37 = tmp36 * tmp36
        tmp38 = tl.broadcast_to(tmp37, [XBLOCK, R0_BLOCK])
        tmp40 = tl.where(r0_mask, tmp38, 0)
        tmp41 = tl.sum(tmp40, 1)[:, None].to(tl.float32)
        tmp43 = tl.full([1, 1], 768.0, tl.float32)
        tmp44 = tmp18 / tmp43
        tmp45 = tl.full([1, 1], 1.0e-6, tl.float32)
        tmp46 = tmp44 + tmp45
        tmp47 = libdevice.rsqrt(tmp46)
        tmp48 = tmp13 * tmp47
        tmp49 = tmp42 * tmp48
        tmp51 = tmp41 / tmp43
        tmp52 = tmp51 + tmp45
        tmp53 = libdevice.rsqrt(tmp52)
        tmp54 = tmp36 * tmp53
        tmp55 = tmp50 * tmp54
        tl.store(out0_ptr + (r0_1 + 768 * x0), tmp49, r0_mask)
        tl.store(out1_ptr + (r0_1 + 768 * x0), tmp55, r0_mask)


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
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a shape sequence, got {value!r}") from exc
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
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    embedding, input_ids, weight0, shifted_ids, weight1, shape0, shape1 = inputs
    embedding_t = _require_tensor("arg1_1", embedding, (VOCAB, HIDDEN), torch.float32)
    input_ids_t = _require_tensor("arg0_1", input_ids, TOKEN_SHAPE, torch.int64)
    weight0_t = _require_tensor("arg2_1", weight0, (HIDDEN,), torch.float32)
    shifted_ids_t = _require_tensor("arg100_1", shifted_ids, TOKEN_SHAPE, torch.int64)
    weight1_t = _require_tensor("arg101_1", weight1, (HIDDEN,), torch.float32)

    if (
        input_ids_t.device != embedding_t.device
        or weight0_t.device != embedding_t.device
        or shifted_ids_t.device != embedding_t.device
        or weight1_t.device != embedding_t.device
    ):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    out_shape0 = _require_shape("_shape_param_0", shape0, VIEW_SHAPE)
    out_shape1 = _require_shape("_shape_param_1", shape1, VIEW_SHAPE)
    return embedding_t, input_ids_t, weight0_t, shifted_ids_t, weight1_t, out_shape0, out_shape1


def _make_inductor_seeds(device: torch.device) -> torch.Tensor:
    seeds = torch.empty_strided((SEED_COUNT,), (1,), device=device, dtype=torch.int64)
    torch.ops.aten.randint.low_out(
        -9223372036854775808,
        9223372036854775807,
        [SEED_COUNT],
        out=seeds,
    )
    return seeds


def oracle_forward(inputs):
    """Run the complete T5 shifted-token embedding + dual dropout + RMSNorm scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same two f32 [8192,768] view outputs. The second branch folds the exact
    decoder-token shift and -100 replacement before the embedding gather.

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_t5_shifted_dropout_rmsnorm.py")

    embedding, input_ids, weight0, shifted_ids, weight1, shape0, shape1 = _validate_inputs(inputs)
    out0 = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=embedding.device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=embedding.device,
        dtype=torch.float32,
    )
    seeds = _make_inductor_seeds(embedding.device)

    device_index = (
        embedding.device.index
        if embedding.device.index is not None
        else torch.cuda.current_device()
    )
    with torch.cuda._DeviceGuard(device_index):
        torch.cuda.set_device(device_index)
        raw_stream = get_raw_stream(device_index)
        _t5_shifted_dual_dropout_rmsnorm_kernel.run(
            out0,
            out1,
            seeds,
            input_ids,
            embedding,
            shifted_ids,
            weight0,
            weight1,
            SEED_INDEX_0,
            SEED_INDEX_1,
            ROWS,
            HIDDEN,
            stream=raw_stream,
        )
    return (out0.view(shape0), out1.view(shape1))


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

    # Report if stochastic ops detected in source.
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
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
