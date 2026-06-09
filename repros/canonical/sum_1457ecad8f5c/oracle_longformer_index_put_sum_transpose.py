"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer duplicate-index index_put(accumulate=True) into the live [8192, 768] layout, its scaled [768] hidden sum, and scaled [768, 8192] transposed side output directly with Triton initialization, scatter, and reduction kernels, whereas Inductor currently lowers the duplicate-index index_put as a generic atomic scatter and then schedules the as_strided/view/permute/div/sum/permute consumers as separate materialization, layout, and reduction kernels; Inductor cannot do this today because scheduler/codegen does not recognize a one-dimensional indexed scatter-add whose as_strided alias feeds both a reduction epilogue and a layout-changing full side-output store; the fix is SCATTER_REDUCE: add an indexed scatter-reduce lowering that targets the captured live layout, applies the scale, emits the transposed side store, and accumulates the hidden-dimension sum from the same scatter/load tile."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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

REPRO_ID = "sum_1457ecad8f5c"
SHAPE_LABEL = "hf_allenailongformerbase_train_005_e98f6a22"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

SOURCE_BLOCKS = 288
SOURCE_SEQ = 512
HEAD_DIM = 64
SOURCE_NUMEL = SOURCE_BLOCKS * SOURCE_SEQ * HEAD_DIM
ROWS = 8192
HIDDEN = 768
FULL_NUMEL = ROWS * HIDDEN
SCALE = 0.125
BLOCK_COPY = 256
BLOCK_SCATTER = 256
BLOCK_ROWS = 128
BLOCK_HIDDEN = 16
NUM_ROW_TILES = math.ceil(ROWS / BLOCK_ROWS)



if triton is not None:

    @triton.jit
    def _copy_full_kernel(
        full_ptr,
        out_ptr,
        FULL_NUMEL_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < FULL_NUMEL_
        values = tl.load(full_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, values * SCALE_, mask=active)

    @triton.jit
    def _scatter_raw_kernel(
        bmm_ptr,
        index_ptr,
        out_ptr,
        SOURCE_NUMEL_: tl.constexpr,
        FULL_NUMEL_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < SOURCE_NUMEL_
        target = tl.load(index_ptr + offsets, mask=active, other=0).to(tl.int64)
        values = tl.load(bmm_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        valid = active & (target >= 0) & (target < FULL_NUMEL_)
        tl.atomic_add(out_ptr + target, values * SCALE_, sem="relaxed", mask=valid)

    @triton.jit
    def _partial_reduce_kernel(
        out_ptr,
        partial_ptr,
        ROWS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HIDDEN_: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_r = tl.program_id(1)
        rows = pid_r * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        hidden = pid_h * BLOCK_HIDDEN_ + tl.arange(0, BLOCK_HIDDEN_)
        mask = (rows[:, None] < ROWS_) & (hidden[None, :] < HIDDEN_)
        offsets = rows[:, None] * HIDDEN_ + hidden[None, :]
        values = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sums = tl.sum(values, axis=0)
        tl.store(
            partial_ptr + pid_r * HIDDEN_ + hidden,
            sums,
            mask=hidden < HIDDEN_,
        )

    @triton.jit
    def _finalize_reduce_kernel(
        partial_ptr,
        sum_ptr,
        HIDDEN_: tl.constexpr,
        NUM_ROW_TILES_: tl.constexpr,
        BLOCK_HIDDEN_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        hidden = pid_h * BLOCK_HIDDEN_ + tl.arange(0, BLOCK_HIDDEN_)
        tiles = tl.arange(0, BLOCK_TILES_)
        mask = (hidden[None, :] < HIDDEN_) & (tiles[:, None] < NUM_ROW_TILES_)
        offsets = tiles[:, None] * HIDDEN_ + hidden[None, :]
        values = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sums = tl.sum(values, axis=0)
        tl.store(sum_ptr + hidden, sums, mask=hidden < HIDDEN_)


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        config = next(iter(configs.values()))
        config = {
            "inputs": [
                {**spec, "device": str(device)}
                if isinstance(spec, dict) and spec.get("kind") == "tensor"
                else spec
                for spec in config["inputs"]
            ]
        }
        inputs = make_inputs_from_config(config)
    else:
        inputs = _load_repro_module().make_inputs()

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def oracle_longformer_index_put_sum_transpose(
    bmm_47: torch.Tensor,
    full_15: torch.Tensor,
    view_38: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5
    if triton is None:
        raise RuntimeError("triton is not available")
    if bmm_47.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")
    if bmm_47.shape != (SOURCE_BLOCKS, SOURCE_SEQ, HEAD_DIM):
        raise ValueError(f"unexpected bmm_47 shape: {tuple(bmm_47.shape)}")
    if full_15.numel() != FULL_NUMEL or view_38.numel() != SOURCE_NUMEL:
        raise ValueError("unexpected Longformer scatter buffer/index size")
    if not bmm_47.is_contiguous() or not full_15.is_contiguous() or not view_38.is_contiguous():
        raise ValueError("oracle expects the captured contiguous input layouts")

    out_base = torch.empty((ROWS, HIDDEN), device=bmm_47.device, dtype=torch.float32)
    _copy_full_kernel[(triton.cdiv(FULL_NUMEL, BLOCK_COPY),)](
        full_15,
        out_base,
        FULL_NUMEL_=FULL_NUMEL,
        SCALE_=SCALE,
        BLOCK=BLOCK_COPY,
        num_warps=4,
    )

    _scatter_raw_kernel[(triton.cdiv(SOURCE_NUMEL, BLOCK_SCATTER),)](
        bmm_47,
        view_38,
        out_base,
        SOURCE_NUMEL_=SOURCE_NUMEL,
        FULL_NUMEL_=FULL_NUMEL,
        SCALE_=SCALE,
        BLOCK=BLOCK_SCATTER,
        num_warps=4,
    )

    partial = torch.empty((NUM_ROW_TILES, HIDDEN), device=bmm_47.device, dtype=torch.float32)
    grid = (triton.cdiv(HIDDEN, BLOCK_HIDDEN), NUM_ROW_TILES)
    _partial_reduce_kernel[grid](
        out_base,
        partial,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HIDDEN_=BLOCK_HIDDEN,
        num_warps=4,
    )

    out_sum = torch.empty((HIDDEN,), device=bmm_47.device, dtype=torch.float32)
    _finalize_reduce_kernel[(triton.cdiv(HIDDEN, BLOCK_HIDDEN),)](
        partial,
        out_sum,
        HIDDEN_=HIDDEN,
        NUM_ROW_TILES_=NUM_ROW_TILES,
        BLOCK_HIDDEN_=BLOCK_HIDDEN,
        BLOCK_TILES_=triton.next_power_of_2(NUM_ROW_TILES),
        num_warps=1,
    )
    return out_sum, out_base.t()


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = _load_repro_module().Repro().to(device)
    return model(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(max(0, warmup)):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


@oracle_impl(hardware="H100", shapes="(T([288, 512, 64], f32), T([6291456], f32), T([9437184], i64, gen=Index(6291456)), S([96, 3, 512, 64, 1]), S([96, 1024, 64]), S([8, 12, 1024, 64]), S([1024, 8, 768]), S([768]), S([8192, 768]))")
def oracle_forward(inputs):
    return oracle_longformer_index_put_sum_transpose(*inputs)


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
