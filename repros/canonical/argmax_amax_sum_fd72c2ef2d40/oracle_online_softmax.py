"""
Oracle for argmax_amax_sum_fd72c2ef2d40

Gap diagnosis:
  Classification: NEW_PATTERN
  What oracle does differently: This oracle computes the full last-token classification loss in one Triton launch, including last nonzero-token argmax over the attention mask, logits gather, online two-class logsumexp, ignore-index masking, and scalar mean reduction.
  What Inductor change would fix: Add a canonical lowering for sequence-classification cross entropy that recognizes mask-derived last-token gather feeding log-softmax/NLL mean and emits one fused reduction kernel instead of materializing the generic argmax, index, softmax, gather, and mean stages.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _last_token_xent_mean_kernel(
        labels_ptr,
        logits_ptr,
        mask_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        block_b: tl.constexpr,
        block_n: tl.constexpr,
    ):
        batches = tl.arange(0, block_b)
        positions = tl.arange(0, block_n)
        pos_mask = positions < seq_len

        token_offsets = batches[:, None] * seq_len + positions[None, :]
        tokens = tl.load(mask_ptr + token_offsets, mask=pos_mask[None, :], other=0)
        candidate_positions = tl.where(tokens != 0, positions[None, :], 0)
        last_positions = tl.max(candidate_positions, axis=1)

        labels = tl.load(labels_ptr + batches)
        is_valid = labels != -100
        safe_labels = tl.where(is_valid, labels, 0)

        row_offsets = (batches * seq_len + last_positions) * 2
        logit0 = tl.load(logits_ptr + row_offsets).to(tl.float32)
        logit1 = tl.load(logits_ptr + row_offsets + 1).to(tl.float32)
        target_logit = tl.where(safe_labels == 0, logit0, logit1)

        running_max = tl.full((block_b,), -float("inf"), tl.float32)
        running_sum = tl.full((block_b,), 0.0, tl.float32)

        next_max = tl.maximum(running_max, logit0)
        running_sum = running_sum * tl.exp(running_max - next_max) + tl.exp(logit0 - next_max)
        running_max = next_max

        next_max = tl.maximum(running_max, logit1)
        running_sum = running_sum * tl.exp(running_max - next_max) + tl.exp(logit1 - next_max)
        running_max = next_max

        nll = running_max + tl.log(running_sum) - target_logit
        valid_f = tl.where(is_valid, 1.0, 0.0)
        total_loss = tl.sum(tl.where(is_valid, nll, 0.0), axis=0)
        total_valid = tl.sum(valid_f, axis=0)
        tl.store(out_ptr, total_loss / total_valid)


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _next_power_of_2(value: int) -> int:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    return triton.next_power_of_2(value)


@oracle_impl(hardware="H100", shapes="(T([8], i64, gen=Index(2)), T([8192, 2], f32), T([8, 1024], i64), S([8, 1024, 2]))")
def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    _require_triton_cuda()

    labels, logits_flat, token_mask, shape_param = inputs
    batch = labels.shape[0]
    seq_len = token_mask.shape[1]
    if batch != 8 or logits_flat.shape != (batch * seq_len, 2):
        raise ValueError(
            "oracle_online_softmax expects labels [8], logits [8192, 2], "
            f"and token mask [8, 1024], got {tuple(labels.shape)}, "
            f"{tuple(logits_flat.shape)}, {tuple(token_mask.shape)}"
        )
    if tuple(shape_param) != (batch, seq_len, 2):
        raise ValueError(f"unexpected shape parameter: {shape_param!r}")

    out = torch.empty((), device=logits_flat.device, dtype=torch.float32)
    _last_token_xent_mean_kernel[(1,)](
        labels,
        logits_flat,
        token_mask,
        out,
        seq_len=seq_len,
        block_b=_next_power_of_2(batch),
        block_n=_next_power_of_2(seq_len),
        num_warps=8,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
