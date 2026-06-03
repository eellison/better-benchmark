"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MT5 dual attention softmax/dropout return from Repro.forward by recomputing both relative-position bucket tables and the causal/noncausal structured masks inside Triton row kernels, adding the viewed BMM scores, performing each last-dimension softmax and dropout with exact Inductor random values, and returning both required [192, 128, 128] transposed views, whereas Inductor currently lowers the decomposed iota/bucket/embedding/where/view/add/amax/sub/exp/sum/div/random/dropout/permute graph as generic mask and relative-bias producers feeding separate softmax, RNG, and layout kernels; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize T5/MT5 relative-position attention with two sibling stochastic softmax epilogues and trailing transpose views into persistent row-softmax templates that recompute cheap structured bias/mask predicates at point of use; the fix is NEW_PATTERN: add an Inductor lowering for T5-style relative-position attention softmax/dropout that fuses bucket lookup, structured masks, dropout, and output-layout epilogues into the row kernels."""
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


REPRO_ID = "amax_sum_amax_9faf4a02126a"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
N_HEADS = 6
Q_LEN = 128
K_LEN = 128
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
ENCODER_SEED_INDEX = 1
DECODER_SEED_INDEX = 35

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


@triton.jit
def _mt5_relative_softmax_dropout_kernel(
    bmm_ptr,
    rel_bias_ptr,
    random_ptr,
    out_base_ptr,
    H: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_BIDIRECTIONAL_BUCKET: tl.constexpr,
    DROPOUT_P_CONST: tl.constexpr,
    DROPOUT_SCALE_CONST: tl.constexpr,
):
    row = tl.program_id(0)
    bh = row // Q
    head = bh - (bh // H) * H
    q = row - bh * Q

    cols = tl.arange(0, BLOCK_K)
    col_mask = cols < K
    rel = cols.to(tl.int32) - q

    if IS_BIDIRECTIONAL_BUCKET:
        distance = tl.where(rel < 0, -rel, rel)
        bucket = distance
        bucket = tl.where(distance >= 8, 8, bucket)
        bucket = tl.where(distance >= 12, 9, bucket)
        bucket = tl.where(distance >= 16, 10, bucket)
        bucket = tl.where(distance >= 23, 11, bucket)
        bucket = tl.where(distance >= 32, 12, bucket)
        bucket = tl.where(distance >= 46, 13, bucket)
        bucket = tl.where(distance >= 64, 14, bucket)
        bucket = tl.where(distance >= 91, 15, bucket)
        bucket += tl.where(rel > 0, 16, 0)
    else:
        distance = tl.where(rel < 0, -rel, 0)
        bucket = distance
        bucket = tl.where(distance >= 16, 16, bucket)
        bucket = tl.where(distance >= 19, 17, bucket)
        bucket = tl.where(distance >= 21, 18, bucket)
        bucket = tl.where(distance >= 24, 19, bucket)
        bucket = tl.where(distance >= 27, 20, bucket)
        bucket = tl.where(distance >= 31, 21, bucket)
        bucket = tl.where(distance >= 35, 22, bucket)
        bucket = tl.where(distance >= 40, 23, bucket)
        bucket = tl.where(distance >= 46, 24, bucket)
        bucket = tl.where(distance >= 52, 25, bucket)
        bucket = tl.where(distance >= 59, 26, bucket)
        bucket = tl.where(distance >= 67, 27, bucket)
        bucket = tl.where(distance >= 77, 28, bucket)
        bucket = tl.where(distance >= 87, 29, bucket)
        bucket = tl.where(distance >= 99, 30, bucket)
        bucket = tl.where(distance >= 113, 31, bucket)

    row_offsets = row * K + cols
    bias = tl.load(
        rel_bias_ptr + bucket * H + head,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    bmm = tl.load(
        bmm_ptr + row_offsets,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    scores = bmm + bias
    if IS_CAUSAL:
        scores = tl.where(cols <= q, scores, -3.4028234663852886e38)
    scores = tl.where(col_mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=0)
    numer = tl.exp(scores - row_max)
    denom = tl.sum(numer, axis=0)
    softmax = numer / denom

    random = tl.load(random_ptr + row_offsets, mask=col_mask, other=0.0).to(tl.float32)
    keep = random > DROPOUT_P_CONST
    out = tl.where(keep, softmax * DROPOUT_SCALE_CONST, 0.0)

    tl.store(out_base_ptr + row_offsets, out, mask=col_mask)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _make_inputs(module, seed: int) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = module.make_inputs()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)


def _inductor_random_like_repro(
    inductor_seeds: torch.Tensor,
    seed_index: int,
) -> torch.Tensor:
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, seed_index)
    return torch.ops.prims.inductor_random.default(
        [BATCH, N_HEADS, Q_LEN, K_LEN],
        seed,
        "rand",
    )


def _check_shape_params(
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
) -> None:
    assert list(_shape_param_0) == [BATCH, -1, Q_LEN, K_LEN]
    assert list(_shape_param_1) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_2) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_3) == [BH, Q_LEN, K_LEN]
    assert list(_shape_param_4) == [BATCH, -1, Q_LEN, K_LEN]
    assert list(_shape_param_5) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_6) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_7) == [BH, Q_LEN, K_LEN]


def _check_kernel_inputs(
    bmm: torch.Tensor,
    rel_bias: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
) -> None:
    assert bmm.is_cuda and rel_bias.is_cuda and random_values.is_cuda and out_base.is_cuda
    assert bmm.dtype == torch.float32 and bmm.shape == (BH, Q_LEN, K_LEN)
    assert rel_bias.dtype == torch.float32 and rel_bias.shape == (32, N_HEADS)
    assert random_values.dtype == torch.float32
    assert random_values.shape == (BATCH, N_HEADS, Q_LEN, K_LEN)
    assert out_base.dtype == torch.float32 and out_base.shape == (BH, Q_LEN, K_LEN)
    assert bmm.is_contiguous()
    assert rel_bias.is_contiguous()
    assert random_values.is_contiguous()
    assert out_base.is_contiguous()


def _launch_attention_branch(
    bmm: torch.Tensor,
    rel_bias: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
    *,
    is_causal: bool,
    is_bidirectional_bucket: bool,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    _check_kernel_inputs(bmm, rel_bias, random_values, out_base)
    assert block_k >= K_LEN and (block_k & (block_k - 1)) == 0

    _mt5_relative_softmax_dropout_kernel[(N_ROWS,)](
        bmm,
        rel_bias,
        random_values,
        out_base,
        H=N_HEADS,
        Q=Q_LEN,
        K=K_LEN,
        BLOCK_K=block_k,
        IS_CAUSAL=is_causal,
        IS_BIDIRECTIONAL_BUCKET=is_bidirectional_bucket,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        num_warps=num_warps,
    )
    return out_base.permute(0, 2, 1)


def oracle_full_mt5_attention_softmax_dropout(
    bmm: torch.Tensor,
    arg6_1: torch.Tensor,
    inductor_seeds: torch.Tensor,
    bmm_16: torch.Tensor,
    arg81_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
    *,
    encoder_random: torch.Tensor | None = None,
    decoder_random: torch.Tensor | None = None,
    encoder_out_base: torch.Tensor | None = None,
    decoder_out_base: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_shape_params(
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
    )

    bmm_view = bmm.view(BH, Q_LEN, K_LEN)
    bmm_16_view = bmm_16.view(BH, Q_LEN, K_LEN)

    if encoder_random is None:
        encoder_random = _inductor_random_like_repro(inductor_seeds, ENCODER_SEED_INDEX)
    if decoder_random is None:
        decoder_random = _inductor_random_like_repro(inductor_seeds, DECODER_SEED_INDEX)
    if encoder_out_base is None:
        encoder_out_base = torch.empty_like(bmm_view)
    if decoder_out_base is None:
        decoder_out_base = torch.empty_like(bmm_16_view)

    encoder_out = _launch_attention_branch(
        bmm_view,
        arg6_1,
        encoder_random,
        encoder_out_base,
        is_causal=False,
        is_bidirectional_bucket=True,
        block_k=block_k,
        num_warps=num_warps,
    )
    decoder_out = _launch_attention_branch(
        bmm_16_view,
        arg81_1,
        decoder_random,
        decoder_out_base,
        is_causal=True,
        is_bidirectional_bucket=False,
        block_k=block_k,
        num_warps=num_warps,
    )
    return decoder_out, encoder_out


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / expected.float().abs().clamp_min(1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float, block_k: int, num_warps: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=1234)
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
            oracle_full_mt5_attention_softmax_dropout(
                *inputs,
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
        ok = ok and item_ok
        print(
            f"output[{idx}]: shape={list(got_item.shape)} dtype={got_item.dtype} "
            f"stride={list(got_item.stride())} ref_stride={list(ref_item.stride())} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} metadata={metadata_ok}"
        )

    print("check compared against full Repro.forward return value, including both relative-position biases, masks, dropout, and transpose views")
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_cuda(fn, warmup: int, rep: int) -> float:
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


def _compile_with_config(
    module,
    inputs: tuple[Any, ...],
    config: dict[str, object],
    warmup: int,
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = module.Repro().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(
    warmup: int,
    rep: int,
    block_k: int,
    num_warps: int,
    no_compile: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321)
    (
        bmm,
        arg6_1,
        inductor_seeds,
        bmm_16,
        arg81_1,
        *_shape_params,
    ) = inputs

    encoder_random = _inductor_random_like_repro(inductor_seeds, ENCODER_SEED_INDEX)
    decoder_random = _inductor_random_like_repro(inductor_seeds, DECODER_SEED_INDEX)
    encoder_out_base = torch.empty_like(bmm)
    decoder_out_base = torch.empty_like(bmm_16)

    elems_per_branch = N_ROWS * K_LEN
    logical_bytes = 2 * elems_per_branch * 4 * 4
    print(
        f"oracle shape: two branches of bmm=f32[{BH}, {Q_LEN}, {K_LEN}], "
        f"relative_bias=f32[32, {N_HEADS}]"
    )
    print(
        f"steady-row logical fused read+write traffic: {logical_bytes / 1e6:.1f} MB "
        "(two branches: bmm + random + output + approximate bias-table read)"
    )

    holder: list[Any] = [None]
    with torch.no_grad():
        kernel_us = _bench_cuda(
            lambda: holder.__setitem__(
                0,
                (
                    _launch_attention_branch(
                        bmm.view(BH, Q_LEN, K_LEN),
                        arg6_1,
                        encoder_random,
                        encoder_out_base,
                        is_causal=False,
                        is_bidirectional_bucket=True,
                        block_k=block_k,
                        num_warps=num_warps,
                    ),
                    _launch_attention_branch(
                        bmm_16.view(BH, Q_LEN, K_LEN),
                        arg81_1,
                        decoder_random,
                        decoder_out_base,
                        is_causal=True,
                        is_bidirectional_bucket=False,
                        block_k=block_k,
                        num_warps=num_warps,
                    ),
                ),
            ),
            warmup=warmup,
            rep=rep,
        )
        full_us = _bench_cuda(
            lambda: holder.__setitem__(
                0,
                oracle_full_mt5_attention_softmax_dropout(
                    *inputs,
                    encoder_out_base=encoder_out_base,
                    decoder_out_base=decoder_out_base,
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
        f"oracle full return, precomputed random: {kernel_us:.3f} us "
        f"({kernel_bw:.3f} TB/s logical bytes)"
    )
    print(
        f"oracle full return including eager inductor_random: {full_us:.3f} us "
        f"({full_bw:.3f} TB/s logical bytes)"
    )

    if no_compile:
        return

    print("torch.compile full repro timings include relative-position bucket construction, masks, RNG dropout, and final permutes")
    compiled_holder: list[Any] = [None]
    for label, config in COMPILE_CONFIGS:
        try:
            compiled = _compile_with_config(module, inputs, config, warmup)
            us = _bench_cuda(
                lambda: compiled_holder.__setitem__(0, compiled(*inputs)),
                warmup=warmup,
                rep=rep,
            )
            print(f"torch.compile {label}: {us:.3f} us")
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=25, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="benchmark repetitions")
    parser.add_argument("--block-k", type=int, default=K_LEN, help="Triton reduction tile size")
    parser.add_argument("--num-warps", type=int, default=1, help="Triton warps for each row kernel")
    parser.add_argument("--rtol", type=float, default=5e-5)
    parser.add_argument("--atol", type=float, default=5e-6)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile baselines for the requested configs",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    block_k = max(args.block_k, triton.next_power_of_2(K_LEN))
    if args.check and not run_check(
        rtol=args.rtol,
        atol=args.atol,
        block_k=block_k,
        num_warps=args.num_warps,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            warmup=args.warmup,
            rep=args.rep,
            block_k=block_k,
            num_warps=args.num_warps,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
