"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer accumulated index_put output scope by seeding the live cropped [8192, 768] base from the input scatter buffer, atomically adding only source values whose arbitrary indices land in that cropped region, then returning both the [768] hidden reduction and required [768, 8192] transpose view, whereas Inductor currently lowers the view/squeeze/index_put/as_strided/negative-pad/permute/clone/sum chain as separate generic scatter, layout, and reduction kernels over a materialized [9437184] scatter buffer; Inductor cannot do this today because scheduler/codegen does not model an indexed scatter-add producer whose cropped as_strided alias feeds both a full layout-changing side-output store and a compatible reduction epilogue; the fix is SCATTER_REDUCE: add an indexed scatter-reduce lowering that targets the final cropped layout directly, preserves arbitrary accumulate=True indices, and emits the side-output transpose plus hidden reduction from the same producer scope."""
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


REPRO_ID = "sum_2ccb3947c2c9"
SHAPE_LABEL = "hf_allenailongformerbase_train_005_4b43a1bf"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
HEADS = 12
SOURCE_A = BATCH * HEADS
OVERLAP_BLOCKS = 4
WINDOW = 768
PADDED_TOKENS = 1536
CROP_START = 256
TOKENS = 1024
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM
ROWS = BATCH * TOKENS
SOURCE_ROWS = SOURCE_A * OVERLAP_BLOCKS
SOURCE_NUMEL = SOURCE_ROWS * WINDOW * HEAD_DIM
FULL_NUMEL = SOURCE_A * PADDED_TOKENS * HEAD_DIM

BLOCK_SCATTER = 1024
BLOCK_ROWS = 32
BLOCK_HIDDEN = 32
NUM_ROW_TILES = math.ceil(ROWS / BLOCK_ROWS)

sys.path.insert(0, str(REPO_ROOT))


if triton is not None:

    @triton.jit
    def _init_cropped_base_kernel(
        full_ptr,
        out_base_ptr,
        ROWS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BATCH_: tl.constexpr,
        HEADS_: tl.constexpr,
        PADDED_TOKENS_: tl.constexpr,
        CROP_START_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
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
        padded_token = token[:, None] + CROP_START_
        active = (rows[:, None] < ROWS_) & (hidden[None, :] < HIDDEN_)

        full_offsets = (a * PADDED_TOKENS_ + padded_token) * HEAD_DIM_ + k[None, :]
        values = tl.load(full_ptr + full_offsets, mask=active, other=0.0).to(tl.float32)
        out_offsets = rows[:, None] * HIDDEN_ + hidden[None, :]
        tl.store(out_base_ptr + out_offsets, values, mask=active)

    @triton.jit
    def _scatter_to_cropped_base_kernel(
        bmm_ptr,
        index_ptr,
        out_base_ptr,
        SOURCE_NUMEL_: tl.constexpr,
        FULL_NUMEL_: tl.constexpr,
        BATCH_: tl.constexpr,
        HEADS_: tl.constexpr,
        PADDED_TOKENS_: tl.constexpr,
        CROP_START_: tl.constexpr,
        TOKENS_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        source_offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        active = source_offsets < SOURCE_NUMEL_
        targets = tl.load(index_ptr + source_offsets, mask=active, other=0).to(tl.int64)
        values = tl.load(bmm_ptr + source_offsets, mask=active, other=0.0).to(tl.float32)

        k = targets % HEAD_DIM_
        q = targets // HEAD_DIM_
        padded_token = q % PADDED_TOKENS_
        a = q // PADDED_TOKENS_
        token = padded_token - CROP_START_
        head = a % HEADS_
        batch = a // HEADS_
        row = token * BATCH_ + batch
        hidden = head * HEAD_DIM_ + k

        live = (
            active
            & (targets >= 0)
            & (targets < FULL_NUMEL_)
            & (padded_token >= CROP_START_)
            & (padded_token < CROP_START_ + TOKENS_)
        )
        out_offsets = row * HIDDEN_ + hidden
        tl.atomic_add(out_base_ptr + out_offsets, values, sem="relaxed", mask=live)

    @triton.jit
    def _partial_reduce_kernel(
        out_base_ptr,
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
        values = tl.load(out_base_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        sums = tl.sum(tl.where(active, values, 0.0), axis=0)
        tl.store(
            partial_ptr + pid_r * HIDDEN_ + hidden,
            sums,
            mask=hidden < HIDDEN_,
        )

    @triton.jit
    def _finalize_reduce_kernel(
        partial_ptr,
        out_sum_ptr,
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
        tl.store(out_sum_ptr + hidden, sums, mask=hidden < HIDDEN_)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    bmm_44: torch.Tensor,
    full_2: torch.Tensor,
    view_19: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = bmm_44.view(SOURCE_A, OVERLAP_BLOCKS, WINDOW, HEAD_DIM, 1).squeeze(4)
    scattered = full_2.clone()
    scattered.index_put_((view_19,), source.view(SOURCE_NUMEL), accumulate=True)
    padded = torch.as_strided(
        scattered,
        (SOURCE_A, PADDED_TOKENS, HEAD_DIM),
        (PADDED_TOKENS * HEAD_DIM, HEAD_DIM, 1),
        0,
    )
    cropped = torch.ops.aten.constant_pad_nd.default(
        padded,
        [0, 0, -CROP_START, -CROP_START],
    )
    out_base = (
        cropped.view(BATCH, HEADS, TOKENS, HEAD_DIM)
        .permute(0, 2, 1, 3)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(TOKENS, BATCH, HIDDEN)
        .view(ROWS, HIDDEN)
    )
    return out_base.sum(dim=0), out_base.t()


def _triton_oracle(
    bmm_44: torch.Tensor,
    full_2: torch.Tensor,
    view_19: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")

    out_base = torch.empty((ROWS, HIDDEN), device=bmm_44.device, dtype=torch.float32)
    grid = (triton.cdiv(HIDDEN, BLOCK_HIDDEN), NUM_ROW_TILES)
    _init_cropped_base_kernel[grid](
        full_2,
        out_base,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        BATCH_=BATCH,
        HEADS_=HEADS,
        PADDED_TOKENS_=PADDED_TOKENS,
        CROP_START_=CROP_START,
        HEAD_DIM_=HEAD_DIM,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HIDDEN_=BLOCK_HIDDEN,
        num_warps=4,
    )

    _scatter_to_cropped_base_kernel[(triton.cdiv(SOURCE_NUMEL, BLOCK_SCATTER),)](
        bmm_44,
        view_19,
        out_base,
        SOURCE_NUMEL_=SOURCE_NUMEL,
        FULL_NUMEL_=FULL_NUMEL,
        BATCH_=BATCH,
        HEADS_=HEADS,
        PADDED_TOKENS_=PADDED_TOKENS,
        CROP_START_=CROP_START,
        TOKENS_=TOKENS,
        HEAD_DIM_=HEAD_DIM,
        HIDDEN_=HIDDEN,
        BLOCK_=BLOCK_SCATTER,
        num_warps=8,
    )

    partial = torch.empty((NUM_ROW_TILES, HIDDEN), device=bmm_44.device, dtype=torch.float32)
    _partial_reduce_kernel[grid](
        out_base,
        partial,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HIDDEN_=BLOCK_HIDDEN,
        num_warps=4,
    )

    out_sum = torch.empty((HIDDEN,), device=bmm_44.device, dtype=torch.float32)
    _finalize_reduce_kernel[(triton.cdiv(HIDDEN, BLOCK_HIDDEN),)](
        partial,
        out_sum,
        HIDDEN_=HIDDEN,
        NUM_ROW_TILES_=NUM_ROW_TILES,
        BLOCK_HIDDEN_=BLOCK_HIDDEN,
        BLOCK_TILES_=triton.next_power_of_2(NUM_ROW_TILES),
        num_warps=8,
    )
    return out_sum, out_base.t()


def oracle_structured_pool_upsample_backward_reduce(
    bmm_44: torch.Tensor,
    full_2: torch.Tensor,
    view_19: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
    )
    if bmm_44.shape != (SOURCE_ROWS, WINDOW, HEAD_DIM):
        raise ValueError(f"unexpected bmm_44 shape: {tuple(bmm_44.shape)}")
    if full_2.numel() != FULL_NUMEL or view_19.numel() != SOURCE_NUMEL:
        raise ValueError("unexpected Longformer scatter buffer/index size")
    if bmm_44.dtype != torch.float32 or full_2.dtype != torch.float32:
        raise ValueError(f"unexpected floating dtypes: {bmm_44.dtype}, {full_2.dtype}")
    if view_19.dtype != torch.int64:
        raise ValueError(f"unexpected index dtype: {view_19.dtype}")
    if not bmm_44.is_contiguous() or not full_2.is_contiguous() or not view_19.is_contiguous():
        raise ValueError("oracle expects the captured contiguous input layouts")

    if bmm_44.device.type == "cuda" and triton is not None:
        return _triton_oracle(bmm_44, full_2, view_19)
    return _torch_direct_oracle(bmm_44, full_2, view_19)


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


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


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = _as_tuple(oracle_structured_pool_upsample_backward_reduce(*inputs))
        synchronize(device)

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        ok = ok and shape_ok and dtype_ok and stride_ok and value_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} expected_dtype={ref.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


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


def run_bench(device: torch.device, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    partial_numel = NUM_ROW_TILES * HIDDEN
    logical_read_bytes = (
        ROWS * HIDDEN * 4
        + SOURCE_NUMEL * 8
        + SOURCE_NUMEL * 4
        + ROWS * HIDDEN * 4
        + partial_numel * 4
    )
    logical_write_bytes = (
        ROWS * HIDDEN * 4
        + SOURCE_NUMEL * 4
        + partial_numel * 4
        + HIDDEN * 4
    )
    print(
        f"oracle shape: bmm=f32[{SOURCE_ROWS}, {WINDOW}, {HEAD_DIM}], "
        f"full=f32[{FULL_NUMEL}], index=i64[{SOURCE_NUMEL}], "
        f"rows={ROWS}, hidden={HIDDEN} shape={SHAPE_LABEL} device={device}"
    )
    print(f"direct-crop logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_structured_pool_upsample_backward_reduce(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_structured_pool_upsample_backward_reduce(*inputs),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_structured_pool_upsample_backward_reduce: {oracle_us:.3f} us "
        f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


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


def main() -> None:
    args = parse_args()
    if not args.check and not args.bench:
        args.check = True

    device = torch.device(args.device)
    with torch.no_grad():
        if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
            sys.exit(1)
        if args.bench:
            run_bench(device=device, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
