"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete grouped-query sum, full_2-backed RoPE half-rotation, and final [512, 2048] transposed layout directly in one Triton output-layout kernel, whereas Inductor currently emits a grouped-head reduction kernel followed by generic slice_scatter/add/mul/clone/permute layout work; Inductor cannot do this today because its scheduler does not keep a reduction producer fused through rotary slice-scatter reconstruction into a non-contiguous transpose-layout store; the fix is SCHEDULER_FUSION: teach the scheduler a grouped RoPE reduction fusion that accumulates the four source heads once, folds the full-base half-rotation epilogue, and writes the required strided output directly."""
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

BATCH = 4
HEADS = 32
GROUPS = 8
HEADS_PER_GROUP = HEADS // GROUPS
SEQ = 512
HEAD_DIM = 64
HALF_DIM = HEAD_DIM // 2
TOKENS = BATCH * SEQ
GROUP_HIDDEN = GROUPS * HEAD_DIM
PAIR_HIDDEN = GROUPS * HALF_DIM
PAIR_TOTAL = TOKENS * PAIR_HIDDEN
BLOCK_PAIRS = 1024


if triton is not None and tl is not None:

    @triton.jit
    def _round_bf16_to_f32(value):
        bits = value.to(tl.int32, bitcast=True)
        lsb = (bits >> 16) & 1
        rounded = (bits + 0x7FFF + lsb) & -65536
        return rounded.to(tl.float32, bitcast=True)

    @triton.jit
    def _grouped_rope_layout_kernel(
        getitem_ptr,
        sin_ptr,
        full_ptr,
        cos_ptr,
        out_ptr,
        PAIR_TOTAL_: tl.constexpr,
        PAIR_HIDDEN_: tl.constexpr,
        GROUP_HIDDEN_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        HALF_DIM_: tl.constexpr,
        HEADS_PER_GROUP_: tl.constexpr,
        SEQ_: tl.constexpr,
        BLOCK_PAIRS_: tl.constexpr,
    ):
        pair_offsets = tl.program_id(0) * BLOCK_PAIRS_ + tl.arange(0, BLOCK_PAIRS_)
        active = pair_offsets < PAIR_TOTAL_

        pair_hidden = pair_offsets % PAIR_HIDDEN_
        token = pair_offsets // PAIR_HIDDEN_
        batch = token // SEQ_
        seq = token - batch * SEQ_
        group = pair_hidden // HALF_DIM_
        dim_lo = pair_hidden - group * HALF_DIM_
        dim_hi = dim_lo + HALF_DIM_
        head_base = group * HEADS_PER_GROUP_

        base0 = ((batch * 32 + head_base + 0) * SEQ_ + seq) * HEAD_DIM_
        base1 = ((batch * 32 + head_base + 1) * SEQ_ + seq) * HEAD_DIM_
        base2 = ((batch * 32 + head_base + 2) * SEQ_ + seq) * HEAD_DIM_
        base3 = ((batch * 32 + head_base + 3) * SEQ_ + seq) * HEAD_DIM_

        sum_lo = (
            tl.load(getitem_ptr + base0 + dim_lo, mask=active, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + base1 + dim_lo, mask=active, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + base2 + dim_lo, mask=active, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + base3 + dim_lo, mask=active, other=0.0).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        sum_hi = (
            tl.load(getitem_ptr + base0 + dim_hi, mask=active, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + base1 + dim_hi, mask=active, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + base2 + dim_hi, mask=active, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + base3 + dim_hi, mask=active, other=0.0).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)

        table_base = seq * HEAD_DIM_
        sin_lo = tl.load(sin_ptr + table_base + dim_lo, mask=active, other=0.0).to(tl.float32)
        sin_hi = tl.load(sin_ptr + table_base + dim_hi, mask=active, other=0.0).to(tl.float32)
        cos_lo = tl.load(cos_ptr + table_base + dim_lo, mask=active, other=0.0).to(tl.float32)
        cos_hi = tl.load(cos_ptr + table_base + dim_hi, mask=active, other=0.0).to(tl.float32)

        full_base = ((batch * 8 + group) * SEQ_ + seq) * HEAD_DIM_
        full_lo = tl.load(full_ptr + full_base + dim_lo, mask=active, other=0.0).to(tl.float32)
        full_hi = tl.load(full_ptr + full_base + dim_hi, mask=active, other=0.0).to(tl.float32)

        rotated_lo = (sum_hi * sin_hi).to(tl.bfloat16).to(tl.float32)
        rotated_hi = -((sum_lo * sin_lo).to(tl.bfloat16).to(tl.float32))
        scaled_lo = (sum_lo * cos_lo).to(tl.bfloat16).to(tl.float32)
        scaled_hi = (sum_hi * cos_hi).to(tl.bfloat16).to(tl.float32)

        add_lo = _round_bf16_to_f32(full_lo + rotated_lo)
        add_hi = _round_bf16_to_f32(rotated_hi + full_hi)
        out_lo = (add_lo + scaled_lo).to(tl.bfloat16)
        out_hi = (add_hi + scaled_hi).to(tl.bfloat16)

        hidden_lo = group * HEAD_DIM_ + dim_lo
        hidden_hi = hidden_lo + HALF_DIM_
        out_base = token * GROUP_HIDDEN_
        tl.store(out_ptr + out_base + hidden_lo, out_lo, mask=active)
        tl.store(out_ptr + out_base + hidden_hi, out_hi, mask=active)


def _check_shape_params(
    shape_param_0,
    shape_param_1,
    shape_param_2,
) -> None:
    assert list(shape_param_0) == [BATCH, GROUPS, HEADS_PER_GROUP, SEQ, HEAD_DIM]
    assert list(shape_param_1) == [BATCH, SEQ, GROUP_HIDDEN]
    assert list(shape_param_2) == [TOKENS, GROUP_HIDDEN]


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} expected shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} expected dtype torch.bfloat16, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} expected contiguous captured layout")


def _torch_oracle(
    getitem_46: torch.Tensor,
    unsqueeze_6: torch.Tensor,
    full_2: torch.Tensor,
    unsqueeze_7: torch.Tensor,
) -> torch.Tensor:
    grouped = getitem_46.view(BATCH, GROUPS, HEADS_PER_GROUP, SEQ, HEAD_DIM).sum(
        dim=2,
        keepdim=True,
    ).squeeze(2)
    sin_scaled = grouped * unsqueeze_6
    lower = torch.ops.aten.slice.Tensor(sin_scaled, 3, 0, HALF_DIM)
    upper = torch.ops.aten.slice.Tensor(sin_scaled, 3, HALF_DIM, HEAD_DIM)
    scattered_hi = torch.ops.aten.slice_scatter.default(
        full_2,
        torch.ops.aten.neg.default(lower),
        3,
        HALF_DIM,
        9223372036854775807,
    )
    scattered_lo = torch.ops.aten.slice_scatter.default(full_2, upper, 3, 0, HALF_DIM)
    rotated = scattered_hi + scattered_lo
    result = rotated + grouped * unsqueeze_7
    return (
        result.permute(0, 2, 1, 3)
        .clone(memory_format=torch.contiguous_format)
        .view(BATCH, SEQ, GROUP_HIDDEN)
        .view(TOKENS, GROUP_HIDDEN)
        .permute(1, 0)
    )


@oracle_impl(hardware="H100", shapes="(T([4, 32, 512, 64], bf16), T([1, 1, 512, 64], bf16), T([4, 8, 512, 64], bf16), T([1, 1, 512, 64], bf16), S([4, 8, 4, 512, 64]), S([4, 512, 512]), S([2048, 512]))")
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
    (
        getitem_46,
        unsqueeze_6,
        full_2,
        unsqueeze_7,
        shape_param_0,
        shape_param_1,
        shape_param_2,
    ) = inputs
    _check_shape_params(shape_param_0, shape_param_1, shape_param_2)
    _check_tensor("getitem_46", getitem_46, (BATCH, HEADS, SEQ, HEAD_DIM))
    _check_tensor("unsqueeze_6", unsqueeze_6, (1, 1, SEQ, HEAD_DIM))
    _check_tensor("full_2", full_2, (BATCH, GROUPS, SEQ, HEAD_DIM))
    _check_tensor("unsqueeze_7", unsqueeze_7, (1, 1, SEQ, HEAD_DIM))

    if getitem_46.device.type != "cuda":
        return _torch_oracle(getitem_46, unsqueeze_6, full_2, unsqueeze_7)
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    out = torch.empty_strided(
        (GROUP_HIDDEN, TOKENS),
        (1, GROUP_HIDDEN),
        device=getitem_46.device,
        dtype=torch.bfloat16,
    )
    _grouped_rope_layout_kernel[(triton.cdiv(PAIR_TOTAL, BLOCK_PAIRS),)](
        getitem_46,
        unsqueeze_6,
        full_2,
        unsqueeze_7,
        out,
        PAIR_TOTAL_=PAIR_TOTAL,
        PAIR_HIDDEN_=PAIR_HIDDEN,
        GROUP_HIDDEN_=GROUP_HIDDEN,
        HEAD_DIM_=HEAD_DIM,
        HALF_DIM_=HALF_DIM,
        HEADS_PER_GROUP_=HEADS_PER_GROUP,
        SEQ_=SEQ,
        BLOCK_PAIRS_=BLOCK_PAIRS,
        num_warps=4,
    )
    return out


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
