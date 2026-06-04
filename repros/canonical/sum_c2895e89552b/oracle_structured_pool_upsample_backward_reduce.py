"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer structured overlapping-window index_put(accumulate=True) output scope by decoding the generated iota/as_strided indices into direct overlapping source loads, writing the final contiguous [8192, 768] base used by the returned [768, 8192] transpose, and accumulating the sibling [768] sum in the same tiled producer, whereas Inductor currently materializes the zero-filled [96, 1536, 64] scatter buffer, crops it with negative padding, performs the clone/view/permute layout chain, and schedules the final sum as separate generic scatter, layout, and reduction kernels; Inductor cannot do this today because scheduler/codegen does not recognize the structured duplicate-index scatter-add plus crop/transpose aliases as one scatter-reduce producer with a reduction epilogue and full side-output store; the fix is SCATTER_REDUCE: add an indexed structured pool/upsample scatter-reduce lowering that targets the live cropped output layout directly and emits compatible reduction epilogues from the same tile."""
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


REPRO_ID = "sum_c2895e89552b"
SHAPE_LABEL = "hf_allenailongformerbase_train_005_2e3ec0c0"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
HEADS = 12
OVERLAP_BLOCKS = 4
WINDOW = 768
HEAD_DIM = 64
TOKENS = 1024
CROP_START = 256
PADDED_TOKENS = 1536
SOURCE_A = BATCH * HEADS
SOURCE_ROWS = SOURCE_A * OVERLAP_BLOCKS
HIDDEN = HEADS * HEAD_DIM
OUTPUT_ROWS = TOKENS * BATCH
SOURCE_NUMEL = SOURCE_ROWS * WINDOW * HEAD_DIM

BLOCK_ROWS = 32
BLOCK_HIDDEN = 32
NUM_ROW_TILES = math.ceil(OUTPUT_ROWS / BLOCK_ROWS)

sys.path.insert(0, str(REPO_ROOT))


if triton is not None:

    @triton.jit
    def _materialize_and_partial_sum_kernel(
        bmm_ptr,
        out_base_ptr,
        partial_ptr,
        OUTPUT_ROWS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BATCH_: tl.constexpr,
        HEADS_: tl.constexpr,
        OVERLAP_BLOCKS_: tl.constexpr,
        WINDOW_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        CROP_START_: tl.constexpr,
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
        active = (rows[:, None] < OUTPUT_ROWS_) & (hidden[None, :] < HIDDEN_)

        base = a * OVERLAP_BLOCKS_ * WINDOW_ * HEAD_DIM_ + k[None, :]

        c0 = padded_token
        offset0 = base + c0 * HEAD_DIM_
        valid0 = active & (token[:, None] < 512)
        value = tl.load(bmm_ptr + offset0, mask=valid0, other=0.0).to(tl.float32)

        c1 = padded_token - 256
        offset1 = base + (WINDOW_ * HEAD_DIM_) + c1 * HEAD_DIM_
        valid1 = active & (token[:, None] < 768)
        value += tl.load(bmm_ptr + offset1, mask=valid1, other=0.0).to(tl.float32)

        c2 = padded_token - 512
        offset2 = base + (2 * WINDOW_ * HEAD_DIM_) + c2 * HEAD_DIM_
        valid2 = active & (token[:, None] >= 256)
        value += tl.load(bmm_ptr + offset2, mask=valid2, other=0.0).to(tl.float32)

        c3 = padded_token - 768
        offset3 = base + (3 * WINDOW_ * HEAD_DIM_) + c3 * HEAD_DIM_
        valid3 = active & (token[:, None] >= 512)
        value += tl.load(bmm_ptr + offset3, mask=valid3, other=0.0).to(tl.float32)

        out_offsets = rows[:, None] * HIDDEN_ + hidden[None, :]
        tl.store(out_base_ptr + out_offsets, value, mask=active)

        sums = tl.sum(tl.where(active, value, 0.0), axis=0)
        tl.store(
            partial_ptr + pid_r * HIDDEN_ + hidden,
            sums,
            mask=hidden < HIDDEN_,
        )

    @triton.jit
    def _finalize_sum_kernel(
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


def _torch_direct_oracle(bmm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    source = bmm.view(BATCH, HEADS, OVERLAP_BLOCKS, WINDOW, HEAD_DIM)
    out4 = torch.empty(
        (TOKENS, BATCH, HEADS, HEAD_DIM),
        device=bmm.device,
        dtype=torch.float32,
    )

    out4[0:256] = (
        source[:, :, 0, 256:512, :] + source[:, :, 1, 0:256, :]
    ).permute(2, 0, 1, 3)
    out4[256:512] = (
        source[:, :, 0, 512:768, :]
        + source[:, :, 1, 256:512, :]
        + source[:, :, 2, 0:256, :]
    ).permute(2, 0, 1, 3)
    out4[512:768] = (
        source[:, :, 1, 512:768, :]
        + source[:, :, 2, 256:512, :]
        + source[:, :, 3, 0:256, :]
    ).permute(2, 0, 1, 3)
    out4[768:1024] = (
        source[:, :, 2, 512:768, :] + source[:, :, 3, 256:512, :]
    ).permute(2, 0, 1, 3)

    out_base = out4.reshape(OUTPUT_ROWS, HIDDEN)
    out_sum = out_base.sum(dim=0)
    return out_sum, out_base.t()


def _triton_oracle(bmm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    out_base = torch.empty((OUTPUT_ROWS, HIDDEN), device=bmm.device, dtype=torch.float32)
    partial = torch.empty(
        (NUM_ROW_TILES, HIDDEN),
        device=bmm.device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(HIDDEN, BLOCK_HIDDEN), NUM_ROW_TILES)
    _materialize_and_partial_sum_kernel[grid](
        bmm,
        out_base,
        partial,
        OUTPUT_ROWS_=OUTPUT_ROWS,
        HIDDEN_=HIDDEN,
        BATCH_=BATCH,
        HEADS_=HEADS,
        OVERLAP_BLOCKS_=OVERLAP_BLOCKS,
        WINDOW_=WINDOW,
        HEAD_DIM_=HEAD_DIM,
        CROP_START_=CROP_START,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HIDDEN_=BLOCK_HIDDEN,
        num_warps=4,
    )

    out_sum = torch.empty((HIDDEN,), device=bmm.device, dtype=torch.float32)
    _finalize_sum_kernel[(triton.cdiv(HIDDEN, BLOCK_HIDDEN),)](
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
    bmm: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
    )
    if bmm.shape != (SOURCE_ROWS, WINDOW, HEAD_DIM):
        raise ValueError(f"unexpected bmm shape: {tuple(bmm.shape)}")
    if bmm.dtype != torch.float32:
        raise ValueError(f"unexpected bmm dtype: {bmm.dtype}")
    if not bmm.is_contiguous():
        raise ValueError("oracle expects the captured contiguous bmm layout")

    if bmm.device.type == "cuda" and triton is not None:
        return _triton_oracle(bmm)
    return _torch_direct_oracle(bmm)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    if device.type != "cuda" or (device.index is not None and device.index != 0):
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    return model(*inputs)


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
        actual = oracle_structured_pool_upsample_backward_reduce(*inputs)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple_length_match=False actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        shape_ok = got.shape == ref.shape
        ok = ok and value_ok and dtype_ok and stride_ok and shape_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} "
            f"mean_abs={mean_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok} shape_match={shape_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def _compile_with_config(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    config: dict[str, object],
    device: torch.device,
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        synchronize(device)
    return compiled


def run_bench(
    device: torch.device,
    warmup: int,
    rep: int,
    compare_compile: bool,
) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)

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

    if not compare_compile:
        return

    module = _load_repro_module()
    if device.type != "cuda" or (device.index is not None and device.index != 0):
        module.device = lambda *unused_args, **unused_kwargs: device
    compile_configs = [
        ("default", {}),
        ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    ]
    print("torch.compile full repro timings:")
    for label, config in compile_configs:
        try:
            model = module.Repro().to(device)
            with torch.no_grad():
                compiled = _compile_with_config(model, inputs, config, device)
                compiled_us = benchmark(lambda: compiled(*inputs), device, warmup, rep)
            print(f"torch.compile {label}: {compiled_us:.3f} us")
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--compare-compile",
        action="store_true",
        help="include torch.compile full-repro timings in --bench output",
    )
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
            run_bench(
                device=device,
                warmup=args.warmup,
                rep=args.rep,
                compare_compile=args.compare_compile,
            )


if __name__ == "__main__":
    main()
