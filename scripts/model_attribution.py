#!/usr/bin/env python3
"""Per-model perf attribution: standalone parts vs end-to-end, exactly.

Closes the accounting identity on real hardware:

    e2e  ~=  SUM(fusible standalone_us x occurrences)
           + SUM(extern  standalone_us x occurrences)
           - G x (total_occurrences - 1)

where G is the CUDAGraph launch floor: the intercept of replay_us vs
n_kernels, measured at runtime on graphs of k no-op kernels (~4.5us on
B200). Every standalone point pays one graph launch; the real model pays
it once, so G is removed per occurrence and added back once. The
per-kernel-in-graph slope (~0.9us) appears identically on both sides
(same kernels, same counts) and cancels — only the intercept is
corrected.

Validated 2026-06-11 on 10 timm infer models:
raw parts/e2e 1.14-1.34 -> corrected 1.00-1.09. The two undershoots are
real perf findings, not accounting error: deit 0.75 (in-model layernorm
runs up to 4x slower than the identical standalone shape) and nfnet 0.92
(dominant conv slightly faster standalone). Residual +2-9% above 1.0 is
compile-context effects (epilogue fusion the standalone repro can't see).

Methodology (must match repro_harness.benchmark_repro):
  - coordinate_descent_tuning ON
  - fresh torch._dynamo.reset() + torch.compile per shape point
    (without it, dynamo marks dims dynamic after the 2nd shape and every
    later point gets a slow dynamic-shape kernel — the bug that produced
    0.12x-4.23x attribution scatter in the first validation round)
  - CUDAGraph capture + do_bench(replay, return_mode="min")

Dedup: nothing is benched twice within a run.
  - fusible: one bench per (pattern_hash, shape_hash) canonical point,
    shared across all models via occurrence-count joins
  - extern: one bench per (target, exact input signature incl. stride),
    run-level cache shared across models

Usage:
    python scripts/model_attribution.py --corpus-root /tmp/recapture_corpus/repros \
        --suite timm --mode infer --models mobilenetv2_100,resnet18
    python scripts/model_attribution.py --corpus-root ... --models all \
        --output results/attribution_b200.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.fx as fx
from triton.testing import do_bench

from model_graph_accounting import analyze_graph, trace_full_graph
from full_graph_harness import load_full_graph
from repro_harness import parse_shapes_config

N_WARMUP = 10
N_REP = 50


# ============================================================================
# CUDAGraph launch floor (measured per run, never hardcoded)
# ============================================================================

def measure_graph_launch_floor() -> tuple[float, float]:
    """(intercept_us, slope_us) of replay_us = intercept + slope * n_kernels.

    Graphs of k no-op kernels (1-element add_). The intercept is the
    per-graph launch cost each standalone measurement pays; the slope is
    per-kernel-in-graph overhead, identical on both sides of the
    accounting identity and therefore not corrected.
    """
    x = torch.zeros(1, device="cuda")
    ks = [1, 2, 5, 10, 20, 50, 100]
    ts = []
    for k in ks:
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(k):
                x.add_(1.0)
        torch.cuda.synchronize()
        ts.append(do_bench(lambda: g.replay(), warmup=25, rep=100,
                           return_mode="min") * 1000)
    n = len(ks)
    mk = sum(ks) / n
    mt = sum(ts) / n
    slope = (sum((k - mk) * (t - mt) for k, t in zip(ks, ts))
             / sum((k - mk) ** 2 for k in ks))
    return mt - slope * mk, slope


# ============================================================================
# Shared bench primitive
# ============================================================================

def _bench_replay(fn) -> float:
    """CUDAGraph-capture fn() and return min replay time in us."""
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
    return do_bench(lambda: g.replay(), warmup=N_WARMUP, rep=N_REP,
                    return_mode="min") * 1000


# ============================================================================
# Fusible side: canonical repro points
# ============================================================================

def bench_fusible_points(needed: set[tuple[str, str]],
                         canonical_dir: Path) -> dict[tuple[str, str], float]:
    """Bench each needed (pattern_hash, shape_hash) point exactly once."""
    times: dict[tuple[str, str], float] = {}
    needed_patterns = {p for p, _ in needed}
    for d in sorted(canonical_dir.iterdir()):
        phash = d.name.rsplit("_", 1)[-1]
        if phash not in needed_patterns or not (d / "repro.py").exists():
            continue
        spec = importlib.util.spec_from_file_location("r", d / "repro.py")
        mod = importlib.util.module_from_spec(spec)
        mod.device = torch.device
        mod.inf = math.inf
        mod.nan = float("nan")
        spec.loader.exec_module(mod)
        for pt in json.loads((d / "shapes.json").read_text())["points"]:
            key = (phash, pt["shape_hash"])
            if key not in needed or key in times:
                continue
            from input_codec import spec_from_compact
            from repro_harness import make_inputs_from_config
            specs = [spec_from_compact(e) for e in pt["inputs"]]
            inputs = [t.cuda() if isinstance(t, torch.Tensor) else t
                      for t in make_inputs_from_config({"inputs": specs})]
            torch._dynamo.reset()
            compiled = torch.compile(mod.Repro())
            times[key] = _bench_replay(lambda: compiled(*inputs))
    return times


# ============================================================================
# Extern side: non-fusible ops at exact input metas
# ============================================================================

def _meta_sig(val) -> str:
    if torch.is_tensor(val):
        return f"T({list(val.shape)},{val.dtype},{list(val.stride())})"
    return repr(val)


def _generation_kind_for_arg(node, arg_pos, sub_pos=None) -> dict | None:
    """Generation spec for an integer tensor arg of an extern op.

    Same op semantics as full_graph_harness.infer_index_bounds_from_gm —
    here the consumer IS the node being benched, so inference is direct:
    embedding(weight, indices) bounds indices by weight.shape[0];
    gather/index_select/scatter by input.shape[dim]; _embedding_bag
    offsets are a sorted-"offsets" kind. Returns a make_inputs_from_config
    generation dict, or None (harness default: small non-negative ints).

    sub_pos selects an element of a list-valued arg (e.g. the j-th index
    tensor in index.Tensor's / _unsafe_masked_index_put_accumulate's
    Tensor?[] indices list): element j indexes self.shape[j], so it must be
    bounded by that dim, not the harness default (randint 0..99) which
    overruns any indexed dim smaller than 100 and trips an out-of-bounds
    device-side assert.
    """
    def _arg_shape(i):
        if i >= len(node.args) or not isinstance(node.args[i], fx.Node):
            return None
        v = node.args[i].meta.get("val")
        return list(v.shape) if torch.is_tensor(v) else None

    t = node.target
    a = torch.ops.aten
    if t in (a.embedding.default, a._embedding_bag.default) and arg_pos == 1:
        w = _arg_shape(0)
        return {"kind": "index", "low": 0, "high": w[0]} if w else None
    if t == a._embedding_bag.default and arg_pos == 2:
        idx = _arg_shape(1)
        return {"kind": "offsets", "high": idx[0]} if idx else None
    # _grouped_mm(self, mat2, offs?, ...): offs (arg 2) is an i32 tensor of
    # inclusive cumulative END-offsets partitioning the M dim (self.shape[0])
    # into len(offs) groups — i.e. cumsum(group_sizes), last element == M.
    # The CUDA kernel (GroupMMCommon.cuh) computes delta = offs[i]-offs[i-1]
    # (start 0) as each group's GEMM dim and asserts delta>=0 and offs[i]<=M,
    # so random ints device-assert. Build a valid monotonic partition.
    if t == a._grouped_mm.default and arg_pos == 2:
        self_shape = _arg_shape(0)
        if self_shape:
            return {"kind": "cumulative_offsets", "total": self_shape[0]}
    if t in (a.gather.default, a.index_select.default,
             a.scatter.src, a.scatter.value, a.scatter_add.default):
        if arg_pos == 2:  # (input, dim, index, ...)
            inp = _arg_shape(0)
            dim = node.args[1] if len(node.args) > 1 else 0
            if inp and isinstance(dim, int):
                return {"kind": "index", "low": 0, "high": inp[dim]}
    # _unsafe_masked_index_put_accumulate(self, mask, indices, values):
    # indices is a Tensor?[] at arg 2; the j-th index tensor addresses
    # self.shape[j]. "unsafe" = the kernel skips bounds checks, but a
    # compiled lowering (or a checked build) device-asserts on overrun, so
    # bound each index by its own dim.
    if t == a._unsafe_masked_index_put_accumulate.default and arg_pos == 2:
        self_shape = _arg_shape(0)
        dim = sub_pos if sub_pos is not None else 0
        if self_shape and dim < len(self_shape):
            return {"kind": "index", "low": 0, "high": self_shape[dim]}
    if t == a.index.Tensor and arg_pos == 1:
        inp = _arg_shape(0)
        if not inp:
            return None
        # index.Tensor's indices is a Tensor?[]; the j-th index addresses
        # self.shape[j]. Bound each by its own dim when we know which
        # element we are fabricating; else fall back to the tightest dim.
        if sub_pos is not None and sub_pos < len(inp):
            return {"kind": "index", "low": 0, "high": inp[sub_pos]}
        return {"kind": "index", "low": 0, "high": min(inp)}
    return None


def _fabricate(val, dev="cuda", node=None, arg_pos=None, sub_pos=None):
    """Materialize a bench input from a fake-tensor meta val.

    REUSES repro_harness.make_inputs_from_config — the project's single
    input-generation implementation (randn floats, bounded randint index
    ints, permutation/offsets kinds) — by building the same spec dict the
    shapes.json pipeline produces. No value semantics live here.

    sub_pos is the element index when this val came from a list-valued arg
    (e.g. an index tensor inside a Tensor?[] indices list), so its bound can
    be inferred per-element.
    """
    if not torch.is_tensor(val):
        return val
    from repro_harness import make_inputs_from_config
    spec = {
        "shape": list(val.shape),
        "dtype": str(val.dtype),
        "stride": list(val.stride()),
        "device": dev,
    }
    if node is not None and arg_pos is not None:
        gen = _generation_kind_for_arg(node, arg_pos, sub_pos)
        if gen and gen.get("kind") == "cumulative_offsets":
            # Build the offs tensor here (kept local to the attribution
            # path, not in the shared repro_harness): a sorted i32 vector of
            # inclusive cumulative END-offsets — values in [0, total],
            # monotonically non-decreasing, last element == total — so every
            # group size offs[i]-offs[i-1] is >=0 and the partition covers
            # all `total` rows (grouped_mm's kernel invariant).
            n = int(val.shape[0])
            total = int(gen["total"])
            offs = torch.sort(
                torch.randint(0, total + 1, (n,), dtype=val.dtype, device=dev)
            ).values
            if n:
                offs[-1] = total
            return offs
        if gen:
            spec["gen"] = gen
    return make_inputs_from_config({"inputs": [spec]})[0]


def _arg_sig(a) -> str:
    """Exact signature of one arg: tensor metas for Nodes, repr for
    literals, recursing into containers. (fx.node.map_arg only visits
    Nodes — literals like conv padding would be silently dropped, merging
    padding-only-different calls into one point. Caught by
    test_canonical_invariants.)"""
    if isinstance(a, fx.Node):
        return _meta_sig(a.meta.get("val"))
    if isinstance(a, (list, tuple)):
        inner = ",".join(_arg_sig(x) for x in a)
        return f"[{inner}]" if isinstance(a, list) else f"({inner})"
    if isinstance(a, dict):
        return "{" + ",".join(f"{k}:{_arg_sig(v)}"
                              for k, v in sorted(a.items())) + "}"
    return repr(a)


def collect_extern_points(gm, graph=None) -> dict[tuple[str, str], dict]:
    """(target, exact input signature) -> {count, node[, graph]} for
    non-fusible ops.

    The signature includes shape, dtype AND stride of every tensor arg
    plus the repr of every literal arg — two conv calls differing only in
    padding or layout are distinct points.

    When `graph` (the full_graph_*.py path) is given, each point also
    records ("graph", node.name): a stable, reproducible handle so the op
    can be re-found and benched in an isolated subprocess (make_fx node
    names are deterministic across traces of the same graph).
    """
    from capture_hook import get_fusion_partitions, partition_node_is_supported

    partitioned: set = set()
    for comp in get_fusion_partitions(gm):
        partitioned.update(comp)
    points: dict[tuple[str, str], dict] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function" or node in partitioned:
            continue
        if partition_node_is_supported(node):
            continue
        key = (str(node.target), _arg_sig((node.args, node.kwargs)))
        if key in points:
            points[key]["count"] += 1
        else:
            entry = {"count": 1, "node": node}
            if graph is not None:
                entry["graph"] = (str(graph), node.name)
            points[key] = entry
    return points


def _fabricate_arg(node, arg_pos, a):
    """Fabricate one positional arg, carrying (arg_pos, element index) so
    index tensors get inferred bounds even inside a Tensor?[] list arg
    (e.g. _unsafe_masked_index_put_accumulate / index.Tensor indices).
    """
    if isinstance(a, fx.Node):
        return _fabricate(a.meta.get("val"), node=node, arg_pos=arg_pos)
    if isinstance(a, (list, tuple)):
        out = [
            _fabricate(e.meta.get("val"), node=node, arg_pos=arg_pos, sub_pos=j)
            if isinstance(e, fx.Node)
            else fx.node.map_arg(e, lambda x: _fabricate(x.meta.get("val")))
            for j, e in enumerate(a)
        ]
        return out if isinstance(a, list) else tuple(out)
    return fx.node.map_arg(a, lambda x: _fabricate(x.meta.get("val")))


# SDPA-backward forward-synthesis -----------------------------------------
#
# The *_attention_backward ops take aux tensors (out, logsumexp, philox_seed,
# philox_offset, and for flash/cudnn cum_seq/rng_state) that are FORWARD
# outputs — softmax statistics and RNG state — not fabricable from
# shape/dtype metas. Fed random values they SEGFAULT under dropout_p>0 (the
# captured dropout_p is 0.1). The fix is to fabricate q/k/v/grad_out from the
# node metas, run the MATCHING forward to synthesize VALID aux, then bind the
# backward over (fabricated grads + real aux + the captured scalar args) and
# time ONLY the backward via the same _bench_replay (CUDAGraph) path.
#
# Each variant's branch in _bench_sdpa_backward maps the captured backward
# node's positional args to roles (which arg is grad_out / q / k / v /
# attn_bias, and where the verbatim scalars dropout_p / is_causal / scale /
# grad_input_mask live), runs the matching forward for the real aux, then
# assembles the backward call from (fabricated grads + real aux + scalars).
# The three variants order their args differently (e.g. attn_bias is arg 4
# for efficient but arg 8 for cudnn), so each is spelled out explicitly.

_SDPA_BWD = {
    "aten._scaled_dot_product_efficient_attention_backward": "efficient",
    "aten._scaled_dot_product_flash_attention_backward": "flash",
    "aten._scaled_dot_product_cudnn_attention_backward": "cudnn",
}


def _node_scalar(node, pos, default=None):
    """Verbatim literal positional arg at `pos` (scale lives in kwargs)."""
    if pos < len(node.args) and not isinstance(node.args[pos], fx.Node):
        return node.args[pos]
    return default


def _sdpa_inductor_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Apply the SDPA constraint behavior needed by flash-backward replay."""
    if tensor.ndim not in (3, 4):
        return tensor
    non_last_aligned = all(stride % 8 == 0 for stride in tensor.stride()[:-1])
    last_dense = tensor.stride(-1) == 1 or tensor.size(-1) <= 1
    if non_last_aligned and last_dense:
        return tensor

    # sdpa_constraint derives an order from the captured FX strides. An input
    # already satisfies that order, so require_stride_order() preserves it.
    # Only a non-fastest last dimension switches the requirement to contiguous.
    fill_order = sorted(
        range(tensor.ndim), key=lambda dim: (tensor.stride(dim), dim)
    )
    if fill_order[0] == tensor.ndim - 1:
        return tensor

    # require_stride_order ignores singleton dimensions. A captured tensor may
    # therefore already satisfy contiguous order even when a size-1 dimension
    # owns the smallest physical stride.
    non_singleton_dims = [
        dim for dim in reversed(range(tensor.ndim)) if tensor.size(dim) != 1
    ]
    ordered_strides = [tensor.stride(dim) for dim in non_singleton_dims]
    if all(
        left <= right
        for left, right in zip(ordered_strides, ordered_strides[1:])
    ):
        return tensor
    return tensor.contiguous()


def _bench_sdpa_backward(node, kind: str) -> float:
    """Forward-synthesize valid aux, then CUDAGraph-time only the backward.

    grad_out/q/k/v (+ efficient/cudnn attn_bias) are fabricated from the
    captured node's tensor metas; the matching forward runs once on q/k/v to
    produce real out/logsumexp/philox/cum_seq aux; the backward is then bound
    over (grads + real aux + verbatim captured scalars) and benched.
    """
    a = torch.ops.aten
    fab = lambda i: _fabricate(node.args[i].meta.get("val"))  # noqa: E731
    grad_out, q, k, v = fab(0), fab(1), fab(2), fab(3)
    scale = node.kwargs.get("scale")

    if kind == "efficient":
        # bwd args: (grad_out, q, k, v, attn_bias, out, logsumexp,
        #            philox_seed, philox_offset, dropout_p, grad_input_mask,
        #            is_causal=False, *, scale)
        # Preserve the captured padded/broadcast layout. Efficient attention
        # validates bias stride alignment, so a fresh contiguous tensor can
        # be invalid even though the captured expanded view is accepted.
        bias_val = node.args[4].meta.get("val") if isinstance(
            node.args[4], fx.Node) else None
        attn_bias = None
        if torch.is_tensor(bias_val):
            attn_bias = fab(4)
        dropout_p = _node_scalar(node, 9, 0.0)
        grad_input_mask = _node_scalar(node, 10, [True, True, True, False])
        is_causal = _node_scalar(node, 11, False)
        out, logsumexp, philox_seed, philox_offset = \
            a._scaled_dot_product_efficient_attention.default(
                q, k, v, attn_bias, True, dropout_p, is_causal, scale=scale)
        bwd_args = (grad_out, q, k, v, attn_bias, out, logsumexp,
                    philox_seed, philox_offset, dropout_p, grad_input_mask,
                    is_causal)
        return _bench_replay(
            lambda: a._scaled_dot_product_efficient_attention_backward.default(
                *bwd_args, scale=scale))

    if kind == "flash":
        # bwd args: (grad_out, q, k, v, out, logsumexp, cum_seq_q, cum_seq_k,
        #            max_q, max_k, dropout_p, is_causal, philox_seed,
        #            philox_offset, *, scale)
        dropout_p = _node_scalar(node, 10, 0.0)
        is_causal = _node_scalar(node, 11, False)
        # flash fwd returns (out, logsumexp, cum_seq_q, cum_seq_k, max_q,
        # max_k, rng_state, unused, debug_mask); the backward's philox_seed /
        # philox_offset ARE the forward's rng_state / unused tensors verbatim
        # (the autograd formula passes them straight through). Random ints
        # there segfault under dropout, so the real fwd aux is required.
        forward_q, forward_k, forward_v = (
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
        )
        (forward_out, forward_logsumexp, cum_q, cum_k, max_q, max_k, philox_seed,
         philox_offset, _dbg) = \
            a._scaled_dot_product_flash_attention.default(
                forward_q, forward_k, forward_v,
                dropout_p, is_causal, False, scale=scale)
        # Restore the captured output layout. Layout conversions belong inside
        # the timed closure because Inductor inserts them immediately before
        # the fallback call. Aligned BSHD-physical layouts pass through, while
        # stride-6 K/V/out become canonical BHSD copies.
        out, logsumexp = fab(4), fab(5)
        out.copy_(forward_out)
        logsumexp.copy_(forward_logsumexp)

        def flash_backward():
            return a._scaled_dot_product_flash_attention_backward.default(
                *(
                    _sdpa_inductor_layout(tensor)
                    for tensor in (grad_out, q, k, v, out, logsumexp)
                ),
                cum_q, cum_k, max_q, max_k, dropout_p, is_causal,
                philox_seed, philox_offset, scale=scale)

        return _bench_replay(
            flash_backward
        )

    # cudnn bwd args: (grad_out, q, k, v, out, logsumexp, philox_seed,
    #                  philox_offset, attn_bias, cum_seq_q, cum_seq_k,
    #                  max_q, max_k, dropout_p, is_causal, *, scale)
    # attn_bias is at arg 8 (NOT 4 — cudnn orders it after the philox aux);
    # pass it to BOTH the forward (so the synthesized aux matches a biased
    # attention) and the backward, materialized dense from its meta.
    bias_val = node.args[8].meta.get("val") if (
        len(node.args) > 8 and isinstance(node.args[8], fx.Node)) else None
    attn_bias = None
    if torch.is_tensor(bias_val):
        attn_bias = torch.randn(
            list(bias_val.shape), dtype=bias_val.dtype, device="cuda")
    dropout_p = _node_scalar(node, 13, 0.0)
    is_causal = _node_scalar(node, 14, False)
    (out, logsumexp, cum_q, cum_k, max_q, max_k, philox_seed, philox_offset,
     _dbg) = a._scaled_dot_product_cudnn_attention.default(
        q, k, v, attn_bias, True, dropout_p, is_causal, False, scale=scale)
    bwd_args = (grad_out, q, k, v, out, logsumexp, philox_seed, philox_offset,
                attn_bias, cum_q, cum_k, max_q, max_k, dropout_p, is_causal)
    return _bench_replay(
        lambda: a._scaled_dot_product_cudnn_attention_backward.default(
            *bwd_args, scale=scale))


def bench_extern_point(node) -> float:
    """Bench one non-fusible op standalone at its exact input metas.

    Top-level positional args carry their position into _fabricate so
    integer index tensors get inferred valid bounds (embedding indices
    bounded by the weight's vocab dim, gather indices by input.shape[dim],
    each index tensor in a Tensor?[] list by its own indexed dim).

    SDPA-backward ops are special-cased: their aux args are forward outputs,
    so the matching forward is run on fabricated q/k/v to synthesize valid
    aux before timing the backward (see _bench_sdpa_backward).
    """
    target_name = str(node.target).rsplit(".", 1)[0]  # strip ".default"
    bwd_kind = _SDPA_BWD.get(target_name)
    if bwd_kind is not None:
        return _bench_sdpa_backward(node, bwd_kind)

    args = tuple(_fabricate_arg(node, i, a) for i, a in enumerate(node.args))
    kwargs = fx.node.map_arg(
        node.kwargs, lambda x: _fabricate(x.meta.get("val")))
    return _bench_replay(lambda: node.target(*args, **kwargs))


# ============================================================================
# Fatal-error-safe extern benching (subprocess isolation)
# ============================================================================
#
# Some extern ops cannot be benched standalone from shape/dtype metas alone
# and FATALLY crash the process (not a catchable Python exception):
#
#   - _scaled_dot_product_efficient_attention_backward: its logsumexp /
#     philox_seed / philox_offset args are forward outputs (softmax stats +
#     RNG state). Fabricated as random floats/ints, the kernel SEGFAULTs.
#   - any out-of-bounds index op on a checked/compiled lowering -> CUDA
#     device-side assert (which also wedges the whole process).
#
# A try/except around bench_extern_point CANNOT catch these — they kill the
# interpreter. So the standalone op-bench runs in a SUBPROCESS: a fatal
# child error becomes a non-zero/negative exit code the parent records into
# extern_bench_failures, and the model run continues. The child runs the
# EXACT same bench_extern_point/_bench_replay on the same node (re-found by
# its deterministic make_fx name), so the normal-path timing is unchanged.

# Child streams one JSON object per benched node on a dedicated result fd,
# tagged so it survives any native stdout/stderr noise from torch/triton.
_EXTERN_RESULT_PREFIX = "__EXTERN_BENCH__ "


def _run_extern_nodes_child(graph_path: str, node_names: list[str]) -> int:
    """Child entrypoint: trace `graph_path` once, bench each named node with
    bench_extern_point, and stream one result line per node.

    Results go to a dedicated fd (the original stdout) prefixed with
    _EXTERN_RESULT_PREFIX; fd 1 is then pointed at stderr so any native
    print from torch/triton/CUDA cannot corrupt the result stream (the
    same isolation bench_parallel's persistent worker uses). A node is
    flushed to the parent BEFORE the next one runs, so a later segfault
    never loses an already-measured result.
    """
    result_fd = os.dup(1)
    result_file = os.fdopen(result_fd, "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def _emit(obj):
        result_file.write(_EXTERN_RESULT_PREFIX + json.dumps(obj) + "\n")
        result_file.flush()

    gm = trace_full_graph(Path(graph_path))
    by_name = {n.name: n for n in gm.graph.nodes}
    for name in node_names:
        node = by_name.get(name)
        if node is None:
            _emit({"node": name, "error": "node not found in re-traced graph"})
            continue
        us = bench_extern_point(node)
        _emit({"node": name, "us": us})
    return 0


def _bench_extern_graph_isolated(
        graph_path: str, node_names: list[str],
        results: dict[str, float], failures: dict[str, str]) -> None:
    """Parent side: bench `node_names` (from one graph) in a subprocess,
    recording successes into `results[name]` and fatal failures into
    `failures[name]`.

    The child streams results as it goes. If it dies (segfault / device
    assert / timeout) the node it was working on — the first requested node
    with no result yet — is recorded as a fatal failure with the child's
    exit signal, and the remaining nodes are retried in a fresh subprocess
    (worker-recovery: one poison node never blocks the rest).
    """
    pending = list(node_names)
    while pending:
        env = os.environ.copy()
        env.setdefault("INDUCTOR_GPU_BENCH_LOCK", "1")
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--bench-extern-node", graph_path, *pending]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              cwd=str(ROOT))
        got: list[str] = []
        for line in proc.stdout.splitlines():
            if not line.startswith(_EXTERN_RESULT_PREFIX):
                continue
            try:
                obj = json.loads(line[len(_EXTERN_RESULT_PREFIX):])
            except json.JSONDecodeError:
                continue
            name = obj.get("node")
            if name is None:
                continue
            got.append(name)
            if "us" in obj:
                results[name] = float(obj["us"])
            else:
                failures[name] = obj.get("error", "unknown")
        # Drop everything the child reported on (success or per-node error).
        remaining = [n for n in pending if n not in got]
        if proc.returncode == 0:
            # Clean exit: any node still unreported genuinely produced no
            # result; record once and stop (don't respin indefinitely).
            for n in remaining:
                failures.setdefault(
                    n, "child exited 0 without emitting a result")
            return
        if not remaining:
            return
        # Fatal child crash: the first still-unreported node is the one it
        # died on. Record it with the signal and retry the rest.
        rc = proc.returncode
        sig = f"signal {-rc}" if rc < 0 else f"exit {rc}"
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        poison = remaining[0]
        failures[poison] = (
            f"fatal crash benching standalone ({sig}); "
            f"not fabricable from shape/dtype metas: {tail[0][:200]}")
        pending = remaining[1:]


# ============================================================================
# End-to-end reference
# ============================================================================

def bench_e2e(graph_path: Path) -> float:
    inst, inputs, _ = load_full_graph(graph_path, default_device="cuda")
    inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in inputs]
    torch._dynamo.reset()
    compiled = torch.compile(inst)
    return _bench_replay(lambda: compiled(*inputs))


def _run_e2e_child(graph_path: str) -> int:
    """Child entrypoint: compile + time one full graph and stream the result.

    Same dedicated-fd isolation as the extern child. A backward graph whose
    SDPA op consumes fabricated philox/logsumexp inputs segfaults HERE too
    (the compiled e2e graph runs the same kernel), so e2e is isolated for
    the same reason bench_extern_point is.
    """
    result_fd = os.dup(1)
    result_file = os.fdopen(result_fd, "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    us = bench_e2e(Path(graph_path))
    result_file.write(_EXTERN_RESULT_PREFIX + json.dumps({"us": us}) + "\n")
    result_file.flush()
    return 0


def bench_e2e_isolated(graph_path: str) -> tuple[float | None, str | None]:
    """Parent side: bench one full graph in a subprocess.

    Returns (us, None) on success or (None, reason) on a fatal child crash
    (segfault / device assert) — so one un-e2e-benchable graph (e.g. an
    SDPA-backward graph fed fabricated RNG state) is recorded instead of
    killing the whole model run. Identical _bench_replay methodology runs
    in the child, so successful-graph timing is unchanged.
    """
    env = os.environ.copy()
    env.setdefault("INDUCTOR_GPU_BENCH_LOCK", "1")
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--bench-e2e", str(graph_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith(_EXTERN_RESULT_PREFIX):
            try:
                obj = json.loads(line[len(_EXTERN_RESULT_PREFIX):])
            except json.JSONDecodeError:
                continue
            if "us" in obj:
                return float(obj["us"]), None
    rc = proc.returncode
    sig = f"signal {-rc}" if rc < 0 else f"exit {rc}"
    tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
    return None, f"e2e fatal crash ({sig}): {tail[0][:200]}"


# ============================================================================
# Per-model attribution
# ============================================================================

def attribute_model(model_dir: Path, canonical_dir: Path,
                    fusible_cache: dict[tuple[str, str], float],
                    extern_cache: dict[tuple[str, str], float],
                    launch_floor_us: float) -> dict:
    graphs = sorted(model_dir.glob("full_graph_*.py"))
    fus_occ: Counter = Counter()
    extern_occ: Counter = Counter()
    # key -> (graph_path, node_name): a stable handle to re-find + bench the
    # op in an isolated subprocess (make_fx node names are deterministic).
    extern_handle: dict[tuple[str, str], tuple[str, str]] = {}
    e2e_total = 0.0
    e2e_failed: list[str] = []
    n_e2e_ok = 0
    for g in graphs:
        gm = trace_full_graph(g)
        acc = analyze_graph(gm, str(g), g.stem)
        fus_occ.update((o.pattern_hash, o.shape_hash) for o in acc.occurrences)
        for key, info in collect_extern_points(gm, graph=g).items():
            extern_occ[key] += info["count"]
            extern_handle.setdefault(key, info["graph"])
        # e2e in a subprocess: a backward graph whose SDPA op consumes
        # fabricated philox/logsumexp inputs segfaults the compiled graph,
        # so a fatal crash is recorded (not fatal to the run).
        us, reason = bench_e2e_isolated(str(g))
        if us is None:
            e2e_failed.append(f"{g.name}: {reason}")
        else:
            e2e_total += us
            n_e2e_ok += 1

    # Fusible: bench only points not already in the run cache.
    missing = set(fus_occ) - set(fusible_cache)
    fusible_cache.update(bench_fusible_points(missing, canonical_dir))
    fus_unmatched = {k: n for k, n in fus_occ.items() if k not in fusible_cache}
    fus_sum = sum(fusible_cache[k] * n for k, n in fus_occ.items()
                  if k in fusible_cache)

    # Extern: run-level cache keyed on (target, exact signature). Each
    # uncached point is benched in an isolated subprocess so a fatal CUDA
    # error (SDPA-backward segfault, OOB device assert) is recorded instead
    # of killing the whole run. Group by graph so the child traces once and
    # benches every node from that graph in a single process.
    extern_failed: list[str] = []
    by_graph: dict[str, list[str]] = {}
    node_to_key: dict[tuple[str, str], tuple[str, str]] = {}
    for key in extern_occ:
        if key in extern_cache:
            continue
        handle = extern_handle.get(key)
        if handle is None:  # no graph handle (shouldn't happen) -> record
            extern_failed.append(f"{key[0]}: no graph handle to bench")
            continue
        graph_path, node_name = handle
        by_graph.setdefault(graph_path, []).append(node_name)
        node_to_key[(graph_path, node_name)] = key

    for graph_path, node_names in by_graph.items():
        node_us: dict[str, float] = {}
        node_fail: dict[str, str] = {}
        _bench_extern_graph_isolated(graph_path, node_names, node_us, node_fail)
        for name, us in node_us.items():
            extern_cache[node_to_key[(graph_path, name)]] = us
        for name, reason in node_fail.items():
            key = node_to_key[(graph_path, name)]
            extern_failed.append(f"{key[0]}: {reason}")
    ext_unmatched = {k: n for k, n in extern_occ.items()
                     if k not in extern_cache}
    ext_sum = sum(extern_cache[k] * n for k, n in extern_occ.items()
                  if k in extern_cache)

    n_occ = sum(fus_occ.values()) + sum(extern_occ.values())
    parts = fus_sum + ext_sum
    # Each standalone occurrence embeds one graph-launch intercept G; the
    # e2e side embeds G per FULL GRAPH (e2e_total sums one bench_e2e per
    # graph). Correction is therefore G*(n_occ - n_graphs) — using (n_occ-1)
    # over-subtracts on multi-graph models (opus verifier, 2026-06-11;
    # coincidentally equal for the 1-graph timm infer validation set).
    corrected = parts - launch_floor_us * (n_occ - len(graphs))
    # Ratios are only meaningful when EVERY graph's e2e benched: a partial
    # e2e_total (some graph segfaulted, e.g. SDPA-backward fed fabricated
    # RNG state) would divide parts by an undersized denominator. Report
    # null ratios + e2e_complete=False in that case rather than a wrong one.
    e2e_complete = not e2e_failed
    ratio_raw = round(parts / e2e_total, 3) if (e2e_complete and e2e_total) else None
    ratio_corrected = (round(corrected / e2e_total, 3)
                       if (e2e_complete and e2e_total) else None)
    return {
        "e2e_us": round(e2e_total, 1),
        "e2e_complete": e2e_complete,
        "n_e2e_graphs_ok": n_e2e_ok,
        "sum_fusible_us": round(fus_sum, 1),
        "sum_extern_us": round(ext_sum, 1),
        "sum_parts_us": round(parts, 1),
        "corrected_parts_us": round(corrected, 1),
        "ratio_raw": ratio_raw,
        "ratio_corrected": ratio_corrected,
        "n_graphs": len(graphs),
        "n_fusible_occurrences": sum(fus_occ.values()),
        "n_fusible_points": len(fus_occ),
        "n_extern_occurrences": sum(extern_occ.values()),
        "n_extern_points": len(extern_occ),
        "fusible_unmatched": [f"{p}/{s} x{n}"
                              for (p, s), n in sorted(fus_unmatched.items())],
        "extern_unmatched": [f"{t} x{n}"
                             for (t, _), n in sorted(ext_unmatched.items())],
        "extern_bench_failures": extern_failed,
        "e2e_bench_failures": e2e_failed,
    }


def main():
    # Isolated bench children: invoked by the parent's *_isolated helpers as
    #   `--bench-extern-node <graph_path> <node_name>...`  (extern op bench)
    #   `--bench-e2e <graph_path>`                          (full-graph e2e)
    # Same inductor config as the normal path so the standalone/e2e bench is
    # identical; a fatal CUDA error here is the parent's signal (non-zero /
    # negative exit), never a crash of the parent run.
    if len(sys.argv) >= 2 and sys.argv[1] in ("--bench-extern-node",
                                              "--bench-e2e"):
        import torch._inductor.config as inductor_config
        inductor_config.coordinate_descent_tuning = True
        if sys.argv[1] == "--bench-e2e":
            sys.exit(_run_e2e_child(sys.argv[2]))
        sys.exit(_run_extern_nodes_child(sys.argv[2], list(sys.argv[3:])))

    parser = argparse.ArgumentParser(
        description="Standalone-parts vs e2e attribution per model")
    parser.add_argument("--corpus-root", type=Path, required=True,
                        help="Corpus root containing models/ and canonical/")
    parser.add_argument("--suite", default="timm")
    parser.add_argument("--mode", default="infer")
    parser.add_argument("--models", required=True,
                        help="Comma-separated model names, or 'all'")
    parser.add_argument("--output", "-o", type=Path, default=None)
    args = parser.parse_args()

    import torch._inductor.config as inductor_config
    inductor_config.coordinate_descent_tuning = True

    suite_dir = args.corpus_root / "models" / args.suite / args.mode
    canonical_dir = args.corpus_root / "canonical"
    if args.models == "all":
        model_dirs = sorted(d for d in suite_dir.iterdir() if d.is_dir())
    else:
        model_dirs = [suite_dir / m for m in args.models.split(",") if m]

    floor_us, slope_us = measure_graph_launch_floor()
    print(f"CUDAGraph floor: replay = {floor_us:.2f} + {slope_us:.3f} * "
          f"n_kernels (us)", file=sys.stderr)

    fusible_cache: dict[tuple[str, str], float] = {}
    extern_cache: dict[tuple[str, str], float] = {}
    out = {"_launch_floor_us": round(floor_us, 2),
           "_launch_slope_us": round(slope_us, 3),
           "_device": torch.cuda.get_device_name(0),
           "models": {}}
    for md in model_dirs:
        if not md.is_dir():
            out["models"][md.name] = {"error": "model dir not found"}
            continue
        try:
            r = attribute_model(md, canonical_dir, fusible_cache,
                                extern_cache, floor_us)
            out["models"][md.name] = r
            ratio = (f"{r['ratio_corrected']:.2f}"
                     if r["ratio_corrected"] is not None else "n/a(e2e partial)")
            n_fail = len(r["extern_bench_failures"]) + len(r["e2e_bench_failures"])
            print(f"{md.name}: e2e={r['e2e_us']:.0f}us "
                  f"parts={r['sum_parts_us']:.0f}us "
                  f"corrected={r['corrected_parts_us']:.0f}us "
                  f"ratio={ratio} "
                  f"unmatched={len(r['fusible_unmatched']) + len(r['extern_unmatched'])}"
                  + (f" bench_failures={n_fail}" if n_fail else ""),
                  flush=True)
        except Exception as e:  # noqa: BLE001 -- per-model isolation
            out["models"][md.name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{md.name}: ERROR {type(e).__name__}: {e}", flush=True)

    print(f"run cache: {len(fusible_cache)} fusible points, "
          f"{len(extern_cache)} extern points benched once", file=sys.stderr)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(f"written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
