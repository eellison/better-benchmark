"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BERT attention softmax/dropout return from Repro.forward, including folding the always-true iota/ge broadcast mask to zero, the [384,512,512] to [32,12,512,512] view, stable last-dimension softmax with the all-minus-inf row guard, seeded Inductor-style dropout and scale, expand/view, and the final non-contiguous [384,512,512] permuted output view, whereas Inductor's compiled path is already within the at-floor band for this full scope despite lowering the decomposed attention-softmax/dropout graph through its generic reduction/RNG schedule; Inductor cannot be assigned a material local gap today because the remaining work is dominated by mandatory score reads, exp/reduction math, dropout RNG, and output stores rather than an avoidable missing fusion; the fix classification is BANDWIDTH_BOUND: keep this repro as at-floor/not_true_floor unless broader online-softmax/dropout throughput improvements move the shared template. Exact stochastic equality is skipped by the harness, so the floor status is not_true_floor."""
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

BATCH = 32
N_HEADS = 12
Q_LEN = 512
K_LEN = 512
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
BMM_SHAPE = (BH, Q_LEN, K_LEN)
VIEW_SHAPE = (BATCH, N_HEADS, Q_LEN, K_LEN)
MASK_VIEW_SHAPE = (BATCH, -1, Q_LEN, K_LEN)
OUT_BASE_SHAPE = BMM_SHAPE
OUT_BASE_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
SEED_INDEX = 1
CLASSIFICATION = "BANDWIDTH_BOUND"
FLOOR_STATUS = "not_true_floor"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _softmax_any_dropout_kernel(
        bmm_ptr,
        seeds_ptr,
        out_base_ptr,
        n_rows: tl.constexpr,
        k_len: tl.constexpr,
        block_m: tl.constexpr,
        block_k: tl.constexpr,
        seed_index: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_k)
        row_mask = rows < n_rows
        col_mask = cols < k_len
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * k_len + cols[None, :]

        scores = tl.load(
            bmm_ptr + offsets,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        row_max = tl.max(scores, axis=1)
        row_has_value = row_max != -float("inf")
        safe_max = tl.where(row_has_value, row_max, 0.0)
        numer = tl.exp2((scores - safe_max[:, None]) * 1.4426950408889634)
        numer = tl.where(mask, numer, 0.0)
        denom = tl.sum(numer, axis=1)
        safe_denom = tl.where(row_has_value, denom, 1.0)
        softmax = tl.where(row_has_value[:, None], numer / safe_denom[:, None], 0.0)

        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        out = tl.where(random > dropout_p, softmax * dropout_scale, 0.0)

        tl.store(out_base_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    bmm, inductor_seeds, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(bmm, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(bmm).__name__}")
    if not isinstance(inductor_seeds, torch.Tensor):
        raise TypeError(f"input 1 must be a tensor, got {type(inductor_seeds).__name__}")
    if not bmm.is_cuda or not inductor_seeds.is_cuda:
        raise RuntimeError("this Triton oracle requires CUDA inputs")
    if bmm.dtype != torch.float32:
        raise TypeError(f"input 0 has dtype {bmm.dtype}, expected torch.float32")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"input 1 has dtype {inductor_seeds.dtype}, expected torch.int64")
    if tuple(bmm.shape) != BMM_SHAPE:
        raise ValueError(f"input 0 has shape {tuple(bmm.shape)}, expected {BMM_SHAPE}")
    if not bmm.is_contiguous():
        raise ValueError("input 0 must be contiguous")
    if inductor_seeds.numel() <= SEED_INDEX:
        raise ValueError(f"input 1 must contain seed index {SEED_INDEX}")
    if _shape_tuple(shape0) != MASK_VIEW_SHAPE:
        raise ValueError(f"_shape_param_0 mismatch: expected {MASK_VIEW_SHAPE}, got {shape0!r}")
    if _shape_tuple(shape1) != VIEW_SHAPE:
        raise ValueError(f"_shape_param_1 mismatch: expected {VIEW_SHAPE}, got {shape1!r}")
    if _shape_tuple(shape2) != VIEW_SHAPE:
        raise ValueError(f"_shape_param_2 mismatch: expected {VIEW_SHAPE}, got {shape2!r}")
    if _shape_tuple(shape3) != OUT_BASE_SHAPE:
        raise ValueError(f"_shape_param_3 mismatch: expected {OUT_BASE_SHAPE}, got {shape3!r}")
    return bmm, inductor_seeds


@oracle_impl(hardware="H100", shapes="(T([384, 512, 512], f32), T([37], i64), S([32, -1, 512, 512]), S([32, 12, 512, 512]), S([32, 12, 512, 512]), S([384, 512, 512]))")
def oracle_forward(inputs, *, block_m: int = 8, num_warps: int = 8, num_stages: int = 2):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_full_bert_attention_softmax_dropout.py")

    bmm, inductor_seeds = _validate_inputs(inputs)
    out_base = torch.empty_strided(
        OUT_BASE_SHAPE,
        OUT_BASE_STRIDE,
        device=bmm.device,
        dtype=torch.float32,
    )
    _softmax_any_dropout_kernel[(triton.cdiv(N_ROWS, block_m),)](
        bmm,
        inductor_seeds,
        out_base,
        n_rows=N_ROWS,
        k_len=K_LEN,
        block_m=block_m,
        block_k=K_LEN,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_base.permute(0, 2, 1)


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
    if FLOOR_STATUS == "not_true_floor":
        print("NOTE: exact stochastic equality is skipped; floor status not_true_floor")

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
