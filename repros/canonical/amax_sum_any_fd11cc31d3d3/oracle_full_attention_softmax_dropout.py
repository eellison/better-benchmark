"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT attention softmax/dropout return from Repro.forward, including the always-true iota/ge broadcast mask folded to zero, the [1024,128,128] to [256,4,128,128] view, stable last-dimension softmax with all-minus-inf row zeroing, exact Inductor RNG dropout and scale, and final [1024,128,128] transpose view, whereas Inductor currently lowers the decomposed iota/expand/where/view/add/amax/sub/exp/sum/div/eq/any/where/inductor_random/dropout/expand/view/permute graph as generic reductions, pointwise RNG/dropout, and layout work over materialized intermediates; Inductor cannot do this today because its pattern matcher and scheduler do not canonicalize a removable structured attention mask, row-all-masked guard, stochastic dropout, and trailing layout-only transpose into one fused online-softmax template; the fix is NEW_PATTERN: add an Inductor lowering for attention softmax/dropout that recognizes removable structured masks, preserves the all-masked-row guard, and fuses dropout plus the output-layout epilogue into the row softmax kernel."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl


REPRO_ID = "amax_sum_any_fd11cc31d3d3"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 256
N_HEADS = 4
Q_LEN = 128
K_LEN = 128
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
N_SEEDS = 24
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)


@triton.jit
def _softmax_any_dropout_kernel(
    bmm_ptr,
    random_ptr,
    out_base_ptr,
    K: tl.constexpr,
    N_TOTAL_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DROPOUT_P_CONST: tl.constexpr,
    DROPOUT_SCALE_CONST: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)

    row_mask = rows < N_TOTAL_ROWS
    col_mask = cols < K
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * K + cols[None, :]

    scores = tl.load(
        bmm_ptr + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    row_max = tl.max(scores, axis=1)
    row_has_value = row_max != -float("inf")
    safe_max = tl.where(row_has_value, row_max, 0.0)
    numer = tl.exp(scores - safe_max[:, None])
    denom = tl.sum(numer, axis=1)
    safe_denom = tl.where(row_has_value, denom, 1.0)
    softmax = numer / safe_denom[:, None]
    value = tl.where(row_has_value[:, None], softmax, 0.0)

    random = tl.load(random_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    keep = (random > DROPOUT_P_CONST).to(tl.float32)
    out = keep * value * DROPOUT_SCALE_CONST

    tl.store(out_base_ptr + offsets, out, mask=mask)


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _check_power_of_two(value: int, name: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def _check_shape_params(
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    _shape_param_3: object,
) -> None:
    assert list(_shape_param_0) == [BATCH, -1, Q_LEN, K_LEN]
    assert list(_shape_param_1) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_2) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_3) == [BH, Q_LEN, K_LEN]


def _inductor_random_like_repro(dev: torch.device) -> torch.Tensor:
    inductor_seeds = torch.ops.prims.inductor_seeds.default(N_SEEDS, dev)
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    return torch.ops.prims.inductor_random.default(
        [BATCH, N_HEADS, Q_LEN, K_LEN],
        seed,
        "rand",
    )


def _launch_oracle(
    bmm: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    assert bmm.is_cuda and random_values.is_cuda and out_base.is_cuda
    assert bmm.dtype == torch.float32 and bmm.shape == (BH, Q_LEN, K_LEN)
    assert random_values.dtype == torch.float32
    assert random_values.shape == (BATCH, N_HEADS, Q_LEN, K_LEN)
    assert out_base.dtype == torch.float32 and out_base.shape == bmm.shape
    assert bmm.is_contiguous()
    assert random_values.is_contiguous()
    assert out_base.is_contiguous()
    _check_power_of_two(block_m, "block_m")
    _check_power_of_two(block_k, "block_k")
    assert block_k >= K_LEN

    grid = (triton.cdiv(N_ROWS, block_m),)
    _softmax_any_dropout_kernel[grid](
        bmm,
        random_values,
        out_base,
        K=K_LEN,
        N_TOTAL_ROWS=N_ROWS,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        num_warps=num_warps,
    )
    return out_base.permute(0, 2, 1)


def oracle_full_attention_softmax_dropout(
    bmm: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    _shape_param_3: object,
    *,
    out_base: torch.Tensor | None = None,
    random_values: torch.Tensor | None = None,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 4,
) -> torch.Tensor:
    _check_shape_params(
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )

    if random_values is None:
        random_values = _inductor_random_like_repro(bmm.device)
    if out_base is None:
        out_base = torch.empty_like(bmm)
    return _launch_oracle(
        bmm,
        random_values,
        out_base,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )


def _make_inputs(module: Any, seed: int, inject_all_inf: bool) -> tuple[object, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = list(module.make_inputs())
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

    if inject_all_inf:
        bmm = inputs[0].clone()
        bmm[0, 0, :] = -float("inf")
        bmm[1, 17, 23] = -float("inf")
        inputs[0] = bmm
    return tuple(inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    finite_diff = diff[torch.isfinite(diff)]
    max_abs = finite_diff.max().item() if finite_diff.numel() else float("nan")
    rel = diff / expected.float().abs().clamp_min(1e-8)
    finite_rel = rel[torch.isfinite(rel)]
    max_rel = finite_rel.max().item() if finite_rel.numel() else float("nan")
    return max_abs, max_rel


def run_check(
    rtol: float,
    atol: float,
    block_m: int,
    block_k: int,
    num_warps: int,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=1234, inject_all_inf=True)
    model = module.Repro().cuda()
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()

    with torch.no_grad():
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        expected = _as_tuple(model(*inputs))
        torch.cuda.synchronize()

        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        actual = _as_tuple(
            oracle_full_attention_softmax_dropout(
                *inputs,
                block_m=block_m,
                block_k=block_k,
                num_warps=num_warps,
            )
        )
        torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"output arity mismatch: oracle={len(actual)} ref={len(expected)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (got_item, ref_item) in enumerate(zip(actual, expected)):
        if not isinstance(got_item, torch.Tensor) or not isinstance(ref_item, torch.Tensor):
            item_ok = got_item == ref_item
            print(f"output[{idx}]: non-tensor equal={item_ok}")
            ok = ok and bool(item_ok)
            continue

        metadata_ok = (
            got_item.shape == ref_item.shape
            and got_item.dtype == ref_item.dtype
            and got_item.stride() == ref_item.stride()
        )
        max_abs, max_rel = _max_diff(got_item, ref_item)
        value_ok = torch.allclose(
            got_item.float(),
            ref_item.float(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        item_ok = metadata_ok and value_ok
        ok = ok and bool(item_ok)
        print(
            f"output[{idx}]: shape={list(got_item.shape)} dtype={got_item.dtype} "
            f"stride={list(got_item.stride())} ref_stride={list(ref_item.stride())} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} metadata={metadata_ok}"
        )

    print(
        "check compared against full Repro.forward return value, including "
        "tautological mask elimination, any-mask zeroing, Inductor RNG dropout, "
        "expand/view, and final permute"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_cuda(fn: object, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def run_bench(
    warmup: int,
    rep: int,
    block_m: int,
    block_k: int,
    num_warps: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321, inject_all_inf=False)
    bmm, *_shape_params = inputs
    random_values = _inductor_random_like_repro(bmm.device)
    out_base = torch.empty_like(bmm)
    out_base_full = torch.empty_like(bmm)

    elems = N_ROWS * K_LEN
    logical_bytes = elems * 4 * 3
    print(f"oracle shape: bmm=f32[{BH}, {Q_LEN}, {K_LEN}], output=permute([{BH}, {Q_LEN}, {K_LEN}])")
    print(
        f"logical fused steady traffic: {logical_bytes / 1e6:.1f} MB "
        "(bmm + random + output; structured mask is folded away)"
    )

    holder: list[torch.Tensor | None] = [None]
    with torch.no_grad():
        kernel_us = _bench_cuda(
            lambda: holder.__setitem__(
                0,
                _launch_oracle(
                    bmm,
                    random_values,
                    out_base,
                    block_m=block_m,
                    block_k=block_k,
                    num_warps=num_warps,
                ),
            ),
            warmup=warmup,
            rep=rep,
        )
        full_us = _bench_cuda(
            lambda: holder.__setitem__(
                0,
                oracle_full_attention_softmax_dropout(
                    *inputs,
                    out_base=out_base_full,
                    block_m=block_m,
                    block_k=block_k,
                    num_warps=num_warps,
                ),
            ),
            warmup=warmup,
            rep=rep,
        )

    kernel_bw = logical_bytes / (kernel_us * 1e-6) / 1e12
    full_bw = logical_bytes / (full_us * 1e-6) / 1e12
    print(
        f"oracle fused full return, precomputed random: {kernel_us:.3f} us "
        f"({kernel_bw:.3f} TB/s logical bytes)"
    )
    print(
        f"oracle full return including eager inductor_random: {full_us:.3f} us "
        f"({full_bw:.3f} TB/s logical bytes)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=25, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="benchmark repetitions")
    parser.add_argument("--block-m", type=int, default=4, help="rows per Triton program")
    parser.add_argument("--block-k", type=int, default=K_LEN, help="Triton reduction tile size")
    parser.add_argument("--num-warps", type=int, default=4, help="Triton warps for row kernel")
    parser.add_argument("--rtol", type=float, default=5e-5)
    parser.add_argument("--atol", type=float, default=5e-6)
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    block_k = max(args.block_k, triton.next_power_of_2(K_LEN))
    if args.check and not run_check(
        rtol=args.rtol,
        atol=args.atol,
        block_m=args.block_m,
        block_k=block_k,
        num_warps=args.num_warps,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            warmup=args.warmup,
            rep=args.rep,
            block_m=args.block_m,
            block_k=block_k,
            num_warps=args.num_warps,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
