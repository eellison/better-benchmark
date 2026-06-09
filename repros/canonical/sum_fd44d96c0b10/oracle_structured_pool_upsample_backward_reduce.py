"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer transposed-clone index_put(accumulate=True) return tuple by applying the captured [96,3,512,64] source layout through the arbitrary one-dimensional index tensor into the live [8192,768] scatter base, then returning both the hidden-dimension sum and required [768,8192] transpose side output, whereas Inductor currently lowers the view/permute/squeeze/clone/index_put/as_strided/view/permute/sum chain as separate generic layout, scatter, aliasing, and reduction work over a materialized scatter buffer; Inductor cannot do this today because scheduler/codegen does not model indexed scatter-add producers whose as_strided aliases feed both a reduction epilogue and a layout-changing full side-output store; the fix is SCATTER_REDUCE: add an indexed scatter-reduce lowering that fuses captured source-layout reads, scatter accumulation, final-layout materialization, and compatible reduction/transpose epilogues for the full return tuple."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

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

REPRO_ID = "sum_fd44d96c0b10"
SHAPE_LABEL = "hf_allenailongformerbase_train_005_aae62a6c"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

SOURCE_A = 96
OVERLAP_BLOCKS = 3
SOURCE_BLOCKS = SOURCE_A * OVERLAP_BLOCKS
SOURCE_HEAD_DIM = 64
SOURCE_SEQ = 512
SOURCE_NUMEL = SOURCE_A * OVERLAP_BLOCKS * SOURCE_SEQ * SOURCE_HEAD_DIM

BATCH = 8
HEADS = 12
TOKENS = 1024
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM
ROWS = TOKENS * BATCH
FULL_NUMEL = ROWS * HIDDEN

BLOCK_COPY = 1024
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
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        active = offsets < FULL_NUMEL_
        values = tl.load(full_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, values, mask=active)

    @triton.jit
    def _scatter_transposed_source_kernel(
        bmm_ptr,
        index_ptr,
        out_ptr,
        SOURCE_NUMEL_: tl.constexpr,
        FULL_NUMEL_: tl.constexpr,
        SOURCE_SEQ_: tl.constexpr,
        SOURCE_HEAD_DIM_: tl.constexpr,
        OVERLAP_BLOCKS_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        source_offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        active = source_offsets < SOURCE_NUMEL_

        k = source_offsets % SOURCE_HEAD_DIM_
        q = source_offsets // SOURCE_HEAD_DIM_
        seq = q % SOURCE_SEQ_
        q = q // SOURCE_SEQ_
        overlap_block = q % OVERLAP_BLOCKS_
        source_a = q // OVERLAP_BLOCKS_

        bmm_block = source_a * OVERLAP_BLOCKS_ + overlap_block
        bmm_offsets = (bmm_block * SOURCE_HEAD_DIM_ + k) * SOURCE_SEQ_ + seq
        targets = tl.load(index_ptr + source_offsets, mask=active, other=0).to(tl.int64)
        values = tl.load(bmm_ptr + bmm_offsets, mask=active, other=0.0).to(tl.float32)

        valid = active & (targets >= 0) & (targets < FULL_NUMEL_)
        tl.atomic_add(out_ptr + targets, values, sem="relaxed", mask=valid)

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
        active = (rows[:, None] < ROWS_) & (hidden[None, :] < HIDDEN_)
        offsets = rows[:, None] * HIDDEN_ + hidden[None, :]
        values = tl.load(out_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        sums = tl.sum(tl.where(active, values, 0.0), axis=0)
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
        active = (tiles[:, None] < NUM_ROW_TILES_) & (hidden[None, :] < HIDDEN_)
        offsets = tiles[:, None] * HIDDEN_ + hidden[None, :]
        values = tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        sums = tl.sum(values, axis=0)
        tl.store(sum_ptr + hidden, sums, mask=hidden < HIDDEN_)


def _config_on_device(config: dict, device: torch.device) -> dict:
    return {
        "inputs": [
            {**spec, "device": str(device)}
            if isinstance(spec, dict) and spec.get("kind") == "tensor"
            else spec
            for spec in config["inputs"]
        ]
    }


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        inputs = make_inputs_from_config(
            _config_on_device(next(iter(configs.values())), device)
        )
    else:
        inputs = _load_repro_module().make_inputs()

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def _torch_direct_oracle(
    bmm_46: torch.Tensor,
    full_15: torch.Tensor,
    view_38: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = (
        bmm_46.view(SOURCE_A, OVERLAP_BLOCKS, SOURCE_HEAD_DIM, SOURCE_SEQ, 1)
        .permute(0, 1, 4, 3, 2)
        .permute(0, 1, 3, 4, 2)
        .squeeze(4)
        .contiguous()
        .view(SOURCE_NUMEL)
    )
    scattered = full_15.clone()
    scattered.index_put_((view_38,), source, accumulate=True)
    out_base = (
        torch.as_strided(
            scattered,
            (SOURCE_A, 2, SOURCE_SEQ, HEAD_DIM),
            (HEAD_DIM, SOURCE_A * SOURCE_SEQ * HEAD_DIM, SOURCE_A * HEAD_DIM, 1),
            0,
        )
        .view(SOURCE_A, TOKENS, HEAD_DIM)
        .view(BATCH, HEADS, TOKENS, HEAD_DIM)
        .permute(0, 2, 1, 3)
        .permute(1, 0, 2, 3)
        .view(TOKENS, BATCH, HIDDEN)
        .view(ROWS, HIDDEN)
    )
    return out_base.sum(dim=0), out_base.t()


def oracle_structured_pool_upsample_backward_reduce(
    bmm_46: torch.Tensor,
    full_15: torch.Tensor,
    view_38: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    )
    if bmm_46.shape != (SOURCE_BLOCKS, SOURCE_HEAD_DIM, SOURCE_SEQ):
        raise ValueError(f"unexpected bmm_46 shape: {tuple(bmm_46.shape)}")
    if full_15.numel() != FULL_NUMEL or view_38.numel() != SOURCE_NUMEL:
        raise ValueError("unexpected Longformer scatter buffer/index size")
    if bmm_46.dtype != torch.float32 or full_15.dtype != torch.float32:
        raise ValueError(f"unexpected floating dtypes: {bmm_46.dtype}, {full_15.dtype}")
    if view_38.dtype != torch.int64:
        raise ValueError(f"unexpected index dtype: {view_38.dtype}")
    if not bmm_46.is_contiguous() or not full_15.is_contiguous() or not view_38.is_contiguous():
        raise ValueError("oracle expects the captured contiguous input layouts")

    if bmm_46.device.type != "cuda":
        return _torch_direct_oracle(bmm_46, full_15, view_38)
    if triton is None:
        raise RuntimeError("triton is required for CUDA oracle execution")

    out_base = torch.empty((ROWS, HIDDEN), device=bmm_46.device, dtype=torch.float32)
    _copy_full_kernel[(triton.cdiv(FULL_NUMEL, BLOCK_COPY),)](
        full_15,
        out_base,
        FULL_NUMEL_=FULL_NUMEL,
        BLOCK_=BLOCK_COPY,
        num_warps=4,
    )

    _scatter_transposed_source_kernel[(triton.cdiv(SOURCE_NUMEL, BLOCK_SCATTER),)](
        bmm_46,
        view_38,
        out_base,
        SOURCE_NUMEL_=SOURCE_NUMEL,
        FULL_NUMEL_=FULL_NUMEL,
        SOURCE_SEQ_=SOURCE_SEQ,
        SOURCE_HEAD_DIM_=SOURCE_HEAD_DIM,
        OVERLAP_BLOCKS_=OVERLAP_BLOCKS,
        BLOCK_=BLOCK_SCATTER,
        num_warps=4,
    )

    partial = torch.empty((NUM_ROW_TILES, HIDDEN), device=bmm_46.device, dtype=torch.float32)
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

    out_sum = torch.empty((HIDDEN,), device=bmm_46.device, dtype=torch.float32)
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
) -> tuple[torch.Tensor, ...]:
    model = _load_repro_module().Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs to repro.py")
    parser.add_argument("--bench", action="store_true", help="time the oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    return parser.parse_args()


@oracle_impl(hardware="H100", shapes="(T([288, 64, 512], f32), T([6291456], f32), T([9437184], i64, gen=Index(6291456)), S([96, 3, 64, 512, 1]), S([9437184]), S([96, 1024, 64]), S([8, 12, 1024, 64]), S([1024, 8, 768]), S([768]), S([8192, 768]))")
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
