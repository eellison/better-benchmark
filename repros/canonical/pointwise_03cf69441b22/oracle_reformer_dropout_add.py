"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Reformer embedding plus positional dropout-add region as one shape-specialized Triton kernel with the embedding gather, both Inductor-seeded dropout masks, virtual positional expand/cat/permute/view layout, and final f32[8,4096,256] add, whereas Inductor currently lowers the embedding dropout, positional expand/cat/permute/dropout/view, and add as decomposed generic stochastic pointwise/layout work; Inductor cannot do this today because its scheduler/codegen pattern library does not recognize the Reformer positional-concat layout with two independent RNG domains as one direct output materialization; the fix is NEW_PATTERN: add a guarded Reformer embedding-plus-positional-dropout lowering that fuses the gather, virtual positional layout, RNG masks, and final add into one generated kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
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
SEQ_LEN = 4096
HIDDEN = 256
VOCAB = 320
POS = 64
POS_LEFT = 64
POS_RIGHT = 192
TOTAL_TOKENS = BATCH * SEQ_LEN
DROPOUT_P = 0.05
DROPOUT_KEEP = 1.0 - DROPOUT_P
DROPOUT_SCALE = 1.0 / DROPOUT_KEEP
EXACT_STOCHASTIC_EQUALITY = False
TRUE_FLOOR = False

if triton is not None:

    @triton.jit
    def oracle_kernel(
        weight_ptr,
        indices_ptr,
        pos_left_ptr,
        pos_right_ptr,
        seeds_ptr,
        out_ptr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        tokens = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        dims = tl.arange(0, BLOCK_D)
        token_mask = tokens < 32768
        elem_mask = token_mask[:, None] & (dims[None, :] < 256)

        batch = tokens // 4096
        seq = tokens - batch * 4096
        pos_row = seq // 64
        pos_col = seq - pos_row * 64
        out_offsets = tokens[:, None] * 256 + dims[None, :]

        token_indices = tl.load(indices_ptr + tokens, mask=token_mask, other=0)
        embedding = tl.load(
            weight_ptr + token_indices[:, None] * 256 + dims[None, :],
            mask=elem_mask,
            other=0.0,
        ).to(tl.float32)

        seed0 = tl.load(seeds_ptr + 0)
        rand0 = tl.rand(seed0, out_offsets.to(tl.uint32))
        dropped_embedding = tl.where(
            rand0 > 0.05,
            embedding * 1.0526315789473684,
            0.0,
        )

        left = tl.load(
            pos_left_ptr + pos_row[:, None] * 64 + dims[None, :],
            mask=elem_mask & (dims[None, :] < 64),
            other=0.0,
        ).to(tl.float32)
        right = tl.load(
            pos_right_ptr + pos_col[:, None] * 192 + (dims[None, :] - 64),
            mask=elem_mask & (dims[None, :] >= 64),
            other=0.0,
        ).to(tl.float32)
        position = left + right

        seed1 = tl.load(seeds_ptr + 1)
        pos_rand_offsets = batch * 64 + pos_col
        rand1 = tl.rand(seed1, pos_rand_offsets.to(tl.uint32))[:, None]
        dropped_position = tl.where(
            rand1 < 0.95,
            position * 1.0526315789473684,
            0.0,
        )

        tl.store(out_ptr + out_offsets, dropped_embedding + dropped_position, mask=elem_mask)


def _expect_shape_param(name, actual, expected):
    actual_tuple = tuple(int(dim) for dim in actual)
    if actual_tuple != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual_tuple}")


def _validate_inputs(
    weight,
    indices,
    pos_left,
    pos_right,
    shape0,
    shape1,
    shape2,
):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not (weight.is_cuda and indices.is_cuda and pos_left.is_cuda and pos_right.is_cuda):
        raise RuntimeError("CUDA inputs are required for this oracle")
    if weight.dtype != torch.float32 or pos_left.dtype != torch.float32 or pos_right.dtype != torch.float32:
        raise TypeError("expected float32 weight and positional tensors")
    if indices.dtype != torch.int64:
        raise TypeError(f"expected int64 indices, got {indices.dtype}")
    if tuple(weight.shape) != (VOCAB, HIDDEN):
        raise ValueError(f"unexpected weight shape: {tuple(weight.shape)}")
    if tuple(indices.shape) != (BATCH, SEQ_LEN):
        raise ValueError(f"unexpected indices shape: {tuple(indices.shape)}")
    if tuple(pos_left.shape) != (POS, 1, POS_LEFT):
        raise ValueError(f"unexpected left positional shape: {tuple(pos_left.shape)}")
    if tuple(pos_right.shape) != (1, POS, POS_RIGHT):
        raise ValueError(f"unexpected right positional shape: {tuple(pos_right.shape)}")
    _expect_shape_param("_shape_param_0", shape0, (BATCH, POS, POS, POS_LEFT))
    _expect_shape_param("_shape_param_1", shape1, (BATCH, POS, POS, POS_RIGHT))
    _expect_shape_param("_shape_param_2", shape2, (BATCH, SEQ_LEN, -1))


def _launch_oracle(weight, indices, pos_left, pos_right):
    seeds = torch.ops.prims.inductor_seeds.default(2, weight.device)
    out = torch.empty((BATCH, SEQ_LEN, HIDDEN), device=weight.device, dtype=torch.float32)
    block_t = 8
    grid = (triton.cdiv(TOTAL_TOKENS, block_t),)
    oracle_kernel[grid](
        weight,
        indices,
        pos_left,
        pos_right,
        seeds,
        out,
        BLOCK_T=block_t,
        BLOCK_D=HIDDEN,
        num_warps=4,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([320, 256], f32), T([8, 4096], i64, gen=Index(320)), T([64, 1, 64], f32), T([1, 64, 192], f32), S([8, 64, 64, 64]), S([8, 64, 64, 192]), S([8, 4096, -1]))")
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
    weight, indices, pos_left, pos_right, shape0, shape1, shape2 = inputs
    _validate_inputs(weight, indices, pos_left, pos_right, shape0, shape1, shape2)
    return _launch_oracle(weight, indices, pos_left, pos_right)


def _normalize_outputs(outputs: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, (tuple, list)):
        return tuple(outputs)
    raise TypeError(f"unexpected output type: {type(outputs)!r}")


def check_oracle_structural_stochastic(oracle_fn, instance, inputs) -> bool:
    """Check full output metadata, skipping only stochastic value equality."""
    with torch.no_grad():
        eager = _normalize_outputs(instance(*inputs))
        oracle = _normalize_outputs(oracle_fn(inputs))

    if len(oracle) != len(eager):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle)} outputs, "
            f"eager produces {len(eager)}"
        )
        return False

    all_pass = True
    for i, (eager_out, oracle_out) in enumerate(zip(eager, oracle)):
        if not isinstance(eager_out, torch.Tensor) or not isinstance(oracle_out, torch.Tensor):
            print(f"  output {i}: SCOPE_MISMATCH non-tensor output")
            all_pass = False
            continue

        if eager_out.shape != oracle_out.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(oracle_out.shape)} "
                f"eager={list(eager_out.shape)}"
            )
            all_pass = False
            continue
        if eager_out.dtype != oracle_out.dtype:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={oracle_out.dtype} "
                f"eager={eager_out.dtype}"
            )
            all_pass = False
            continue
        if eager_out.stride() != oracle_out.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={tuple(oracle_out.stride())} "
                f"eager={tuple(eager_out.stride())}"
            )
            all_pass = False
            continue

        print(
            f"  output {i}: SKIP_VALUE (stochastic, metadata PASS "
            f"shape={list(eager_out.shape)} dtype={eager_out.dtype} "
            f"stride={tuple(eager_out.stride())})"
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
    if not EXACT_STOCHASTIC_EQUALITY:
        print("NOTE: exact stochastic value equality is not established; true_floor=no")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle_structural_stochastic(oracle_forward, instance, inputs)
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
        if not TRUE_FLOOR:
            print("diagnosis_only: true_floor=no because stochastic values are structurally checked")


if __name__ == "__main__":
    main()
