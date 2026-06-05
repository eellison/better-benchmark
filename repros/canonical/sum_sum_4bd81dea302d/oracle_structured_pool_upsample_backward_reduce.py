"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer overlapping-window return tuple by decoding the generated iota/as_strided duplicate-index scatters into direct source loads, writing both live contiguous [8192, 768] bases used by the returned [768, 8192] transposes, applying the /8 epilogue on the second branch, and accumulating both sibling [768] sums from the same Triton tile, whereas Inductor currently materializes two zero-filled [6291456] scatter buffers, runs generic accumulate=True index_puts, reinterprets them through as_strided/view/permute chains, and schedules the reductions and transposed side outputs as separate work; Inductor cannot do this today because scheduler/codegen does not recognize the structured overlapping-window scatter-add aliases as one scatter-reduce producer with multiple reduction and layout-changing side-output epilogues; the fix is SCATTER_REDUCE: add a Longformer indexed scatter-reduce lowering that targets the final live layout directly, fuses scale/reduction epilogues, and emits the required transposed stores without materializing the scatter buffers."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None



from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_sum_4bd81dea302d"
SHAPE_LABEL = "hf_allenailongformerbase_train_005_5509b72a"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
HEADS = 12
SOURCE_A = BATCH * HEADS
OVERLAP_BLOCKS = 3
WINDOW = 512
STEP = 256
TOKENS = 1024
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM
ROWS = TOKENS * BATCH
SOURCE_NUMEL = SOURCE_A * OVERLAP_BLOCKS * WINDOW * HEAD_DIM
SCALE = 0.125

BLOCK_ROWS = 32
BLOCK_HIDDEN = 32
NUM_ROW_TILES = math.ceil(ROWS / BLOCK_ROWS)



if triton is not None:

    @triton.jit
    def _materialize_and_partial_sum_kernel(
        bmm_transposed_src_ptr,
        bmm_direct_src_ptr,
        out_transposed_base_ptr,
        out_direct_scaled_base_ptr,
        partial_transposed_ptr,
        partial_direct_scaled_ptr,
        ROWS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BATCH_: tl.constexpr,
        HEADS_: tl.constexpr,
        OVERLAP_BLOCKS_: tl.constexpr,
        WINDOW_: tl.constexpr,
        STEP_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HIDDEN_: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_r = tl.program_id(1)
        rows = pid_r * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        hidden = pid_h * BLOCK_HIDDEN_ + tl.arange(0, BLOCK_HIDDEN_)

        token = rows // BATCH_
        batch = rows - token * BATCH_
        head = hidden // HEAD_DIM_
        k = hidden - head * HEAD_DIM_
        a = batch[:, None] * HEADS_ + head[None, :]
        active = (rows[:, None] < ROWS_) & (hidden[None, :] < HIDDEN_)

        source_a_stride = OVERLAP_BLOCKS_ * HEAD_DIM_ * WINDOW_
        source_block_stride = HEAD_DIM_ * WINDOW_
        base = a * source_a_stride

        row0 = token[:, None]
        valid0 = active & (token[:, None] < WINDOW_)
        transposed_value = tl.load(
            bmm_transposed_src_ptr + base + k[None, :] * WINDOW_ + row0,
            mask=valid0,
            other=0.0,
        ).to(tl.float32)
        direct_value = tl.load(
            bmm_direct_src_ptr + base + row0 * HEAD_DIM_ + k[None, :],
            mask=valid0,
            other=0.0,
        ).to(tl.float32)

        row1 = token[:, None] - STEP_
        valid1 = active & (token[:, None] >= STEP_) & (
            token[:, None] < STEP_ + WINDOW_
        )
        transposed_value += tl.load(
            bmm_transposed_src_ptr
            + base
            + source_block_stride
            + k[None, :] * WINDOW_
            + row1,
            mask=valid1,
            other=0.0,
        ).to(tl.float32)
        direct_value += tl.load(
            bmm_direct_src_ptr
            + base
            + source_block_stride
            + row1 * HEAD_DIM_
            + k[None, :],
            mask=valid1,
            other=0.0,
        ).to(tl.float32)

        row2 = token[:, None] - 2 * STEP_
        valid2 = active & (token[:, None] >= 2 * STEP_)
        transposed_value += tl.load(
            bmm_transposed_src_ptr
            + base
            + 2 * source_block_stride
            + k[None, :] * WINDOW_
            + row2,
            mask=valid2,
            other=0.0,
        ).to(tl.float32)
        direct_value += tl.load(
            bmm_direct_src_ptr
            + base
            + 2 * source_block_stride
            + row2 * HEAD_DIM_
            + k[None, :],
            mask=valid2,
            other=0.0,
        ).to(tl.float32)

        direct_scaled = direct_value * SCALE_
        out_offsets = rows[:, None] * HIDDEN_ + hidden[None, :]
        tl.store(out_transposed_base_ptr + out_offsets, transposed_value, mask=active)
        tl.store(out_direct_scaled_base_ptr + out_offsets, direct_scaled, mask=active)

        transposed_sum = tl.sum(tl.where(active, transposed_value, 0.0), axis=0)
        direct_scaled_sum = tl.sum(tl.where(active, direct_scaled, 0.0), axis=0)
        partial_offsets = pid_r * HIDDEN_ + hidden
        tl.store(
            partial_transposed_ptr + partial_offsets,
            transposed_sum,
            mask=hidden < HIDDEN_,
        )
        tl.store(
            partial_direct_scaled_ptr + partial_offsets,
            direct_scaled_sum,
            mask=hidden < HIDDEN_,
        )

    @triton.jit
    def _finalize_sums_kernel(
        partial_transposed_ptr,
        partial_direct_scaled_ptr,
        out_transposed_sum_ptr,
        out_direct_scaled_sum_ptr,
        HIDDEN_: tl.constexpr,
        NUM_ROW_TILES_: tl.constexpr,
        BLOCK_HIDDEN_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        hidden = pid_h * BLOCK_HIDDEN_ + tl.arange(0, BLOCK_HIDDEN_)
        tiles = tl.arange(0, BLOCK_TILES_)
        active = (tiles[:, None] < NUM_ROW_TILES_) & (hidden[None, :] < HIDDEN_)
        offsets = tiles[:, None] * HIDDEN_ + hidden[None, :]

        transposed_partials = tl.load(
            partial_transposed_ptr + offsets,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        direct_scaled_partials = tl.load(
            partial_direct_scaled_ptr + offsets,
            mask=active,
            other=0.0,
        ).to(tl.float32)

        tl.store(
            out_transposed_sum_ptr + hidden,
            tl.sum(transposed_partials, axis=0),
            mask=hidden < HIDDEN_,
        )
        tl.store(
            out_direct_scaled_sum_ptr + hidden,
            tl.sum(direct_scaled_partials, axis=0),
            mask=hidden < HIDDEN_,
        )


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda() if isinstance(value, torch.Tensor) else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _torch_direct_oracle(
    bmm_2: torch.Tensor,
    bmm_3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_t = bmm_2.view(SOURCE_A, OVERLAP_BLOCKS, HEAD_DIM, WINDOW)
    source_d = bmm_3.view(SOURCE_A, OVERLAP_BLOCKS, WINDOW, HEAD_DIM)
    live_t = torch.empty((SOURCE_A, TOKENS, HEAD_DIM), device=bmm_2.device, dtype=torch.float32)
    live_d = torch.empty((SOURCE_A, TOKENS, HEAD_DIM), device=bmm_3.device, dtype=torch.float32)

    live_t[:, 0:STEP, :] = source_t[:, 0, :, 0:STEP].transpose(1, 2)
    live_t[:, STEP:WINDOW, :] = (
        source_t[:, 0, :, STEP:WINDOW].transpose(1, 2)
        + source_t[:, 1, :, 0:STEP].transpose(1, 2)
    )
    live_t[:, WINDOW:WINDOW + STEP, :] = (
        source_t[:, 1, :, STEP:WINDOW].transpose(1, 2)
        + source_t[:, 2, :, 0:STEP].transpose(1, 2)
    )
    live_t[:, WINDOW + STEP:TOKENS, :] = source_t[:, 2, :, STEP:WINDOW].transpose(1, 2)

    live_d[:, 0:STEP, :] = source_d[:, 0, 0:STEP, :]
    live_d[:, STEP:WINDOW, :] = source_d[:, 0, STEP:WINDOW, :] + source_d[:, 1, 0:STEP, :]
    live_d[:, WINDOW:WINDOW + STEP, :] = (
        source_d[:, 1, STEP:WINDOW, :] + source_d[:, 2, 0:STEP, :]
    )
    live_d[:, WINDOW + STEP:TOKENS, :] = source_d[:, 2, STEP:WINDOW, :]

    out_t = (
        live_t.view(BATCH, HEADS, TOKENS, HEAD_DIM)
        .permute(2, 0, 1, 3)
        .reshape(ROWS, HIDDEN)
    )
    out_d = (
        live_d.view(BATCH, HEADS, TOKENS, HEAD_DIM)
        .permute(2, 0, 1, 3)
        .reshape(ROWS, HIDDEN)
        * SCALE
    )
    return out_t.sum(dim=0), out_t.t(), out_d.sum(dim=0), out_d.t()


def oracle_structured_pool_upsample_backward_reduce(
    bmm_2: torch.Tensor,
    bmm_3: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
    _shape_param_8,
    _shape_param_9,
    _shape_param_10,
    _shape_param_11,
    _shape_param_12,
    _shape_param_13,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
        _shape_param_10,
        _shape_param_11,
        _shape_param_12,
        _shape_param_13,
    )
    if bmm_2.shape != (SOURCE_A * OVERLAP_BLOCKS, HEAD_DIM, WINDOW):
        raise ValueError(f"unexpected bmm_2 shape: {tuple(bmm_2.shape)}")
    if bmm_3.shape != (SOURCE_A * OVERLAP_BLOCKS, WINDOW, HEAD_DIM):
        raise ValueError(f"unexpected bmm_3 shape: {tuple(bmm_3.shape)}")
    if bmm_2.dtype != torch.float32 or bmm_3.dtype != torch.float32:
        raise ValueError(f"unexpected dtypes: {bmm_2.dtype}, {bmm_3.dtype}")
    if not bmm_2.is_contiguous() or not bmm_3.is_contiguous():
        raise ValueError("oracle expects the captured contiguous bmm layouts")

    if bmm_2.device.type != "cuda":
        return _torch_direct_oracle(bmm_2, bmm_3)
    if triton is None:
        raise RuntimeError("triton is required for CUDA oracle execution")

    out_t_base = torch.empty((ROWS, HIDDEN), device=bmm_2.device, dtype=torch.float32)
    out_d_base = torch.empty((ROWS, HIDDEN), device=bmm_2.device, dtype=torch.float32)
    partial_t = torch.empty((NUM_ROW_TILES, HIDDEN), device=bmm_2.device, dtype=torch.float32)
    partial_d = torch.empty((NUM_ROW_TILES, HIDDEN), device=bmm_2.device, dtype=torch.float32)

    grid = (triton.cdiv(HIDDEN, BLOCK_HIDDEN), NUM_ROW_TILES)
    _materialize_and_partial_sum_kernel[grid](
        bmm_2,
        bmm_3,
        out_t_base,
        out_d_base,
        partial_t,
        partial_d,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        BATCH_=BATCH,
        HEADS_=HEADS,
        OVERLAP_BLOCKS_=OVERLAP_BLOCKS,
        WINDOW_=WINDOW,
        STEP_=STEP,
        HEAD_DIM_=HEAD_DIM,
        SCALE_=SCALE,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HIDDEN_=BLOCK_HIDDEN,
        num_warps=4,
    )

    out_t_sum = torch.empty((HIDDEN,), device=bmm_2.device, dtype=torch.float32)
    out_d_sum = torch.empty((HIDDEN,), device=bmm_2.device, dtype=torch.float32)
    _finalize_sums_kernel[(triton.cdiv(HIDDEN, BLOCK_HIDDEN),)](
        partial_t,
        partial_d,
        out_t_sum,
        out_d_sum,
        HIDDEN_=HIDDEN,
        NUM_ROW_TILES_=NUM_ROW_TILES,
        BLOCK_HIDDEN_=BLOCK_HIDDEN,
        BLOCK_TILES_=triton.next_power_of_2(NUM_ROW_TILES),
        num_warps=8,
    )

    return out_t_sum, out_t_base.t(), out_d_sum, out_d_base.t()


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def oracle_forward(inputs):
    return oracle_structured_pool_upsample_backward_reduce(*inputs)


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
