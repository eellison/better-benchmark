"""Regression tests for the canonicalize-before-serialization invariants.

These tests are the executable documentation of the corpus identity
invariants (CORPUS_MIGRATION_PLAN §1). Everything here was landed (and
burned by at least once) during the wave-1 prep sessions of 2026-06:

  1. RETRACE IDEMPOTENCE: canonicalize(canonicalize(g)) == canonicalize(g)
     — one retrace before serialization, then every subsequent trace+hash
     is a fixed point.
  2. STRIDE IN SHAPE HASH: same shapes+dtypes at different layouts
     (contiguous vs channels-last) hash to DIFFERENT shape_hashes; same
     layout hashes the same. Contiguous spells as stride=[] canonically.
  3. STRIDE FIDELITY THROUGH CANONICALIZATION: a channels-last placeholder
     survives the retrace with its layout intact in placeholder_info
     (wrong fabricated strides would change which view ops make_fx emits),
     and layout is shape-identity, never pattern-identity.
  4. SCHEMA-TYPED SHAPE LIFT: SymInt[]-slot args (expand/view sizes) are
     lifted to _shape_param_N placeholders; int[]-slot args (reduction
     dims, permute orders) are NEVER lifted; trivial shapes skipped;
     the lifted graph still executes with the params passed in.
  5. EXTERN SIGNATURE EXACTNESS: model_attribution's extern point
     signature distinguishes stride-only and literal-arg-only differences
     (two convs differing only in padding or input layout are distinct
     bench points), and dedupes true duplicates.

CPU/fake-mode only — safe to run during GPU sweeps.

Usage:
    pytest tests/test_canonical_invariants.py -x
"""
from __future__ import annotations

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import (
    canonicalize_subgraph,
    compute_partition_pattern,
    get_fusion_partitions,
    lift_shape_params,
    pattern_hash_for_subgraph,
    shape_hash_for_placeholders,
)


def _traced(fn, *inputs):
    """make_fx fake-mode trace with metas, like the capture pipeline sees."""
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fakes = [mode.from_tensor(t) if torch.is_tensor(t) else t
                 for t in inputs]
        return make_fx(fn, tracing_mode="fake")(*fakes)


def _first_pattern(gm) -> dict:
    comps = get_fusion_partitions(gm)
    assert comps, "no fusible partition found in test graph"
    comp = max(comps, key=len)
    pat = compute_partition_pattern(comp, gm)
    assert pat is not None, "compute_partition_pattern failed on test graph"
    return pat


# ============================================================================
# 1. Retrace idempotence
# ============================================================================

def test_retrace_idempotence():
    def f(a, b):
        # reshape spelling: the retrace normalizes to view — canonicalize
        # twice and the second pass must be a no-op for the hash.
        x = torch.reshape(a, [2, 12])
        y = (x + b).relu()
        return torch.reshape(y, [4, 6]) * 2

    gm = _traced(f, torch.randn(4, 6), torch.randn(2, 12))
    pat1 = _first_pattern(gm)

    sub2, info2, sp2 = canonicalize_subgraph(
        pat1["sub_gm"], pat1["placeholder_info"], pat1["shape_params"])
    assert pattern_hash_for_subgraph(sub2) == pat1["pattern_hash"]
    assert shape_hash_for_placeholders(info2) == pat1["shape_hash"]

    sub3, _info3, _ = canonicalize_subgraph(sub2, info2, sp2)
    assert pattern_hash_for_subgraph(sub3) == pat1["pattern_hash"]


def test_canonical_form_normalizes_reshape():
    def f(a, b):
        x = torch.reshape(a, [2, 12])
        return (x + b).relu()

    gm = _traced(f, torch.randn(4, 6), torch.randn(2, 12))
    pat = _first_pattern(gm)
    ops = {str(n.target) for n in pat["sub_gm"].graph.nodes
           if n.op == "call_function"}
    assert "aten.reshape.default" not in ops, f"ops={sorted(ops)}"


# ============================================================================
# 2. Stride participates in shape_hash
# ============================================================================

def test_stride_in_shape_hash():
    contig = {"a": {"shape": [8, 16, 4, 4], "stride": [],
                    "dtype": "torch.bfloat16"}}
    channels_last = {"a": {"shape": [8, 16, 4, 4],
                           "stride": [256, 1, 64, 16],
                           "dtype": "torch.bfloat16"}}
    assert (shape_hash_for_placeholders(contig)
            != shape_hash_for_placeholders(channels_last))


def test_same_layout_same_hash():
    a = {"a": {"shape": [8, 16, 4, 4], "stride": [],
               "dtype": "torch.bfloat16"}}
    b = {"a": {"shape": [8, 16, 4, 4], "stride": [],
               "dtype": "torch.bfloat16"}}
    assert shape_hash_for_placeholders(a) == shape_hash_for_placeholders(b)


def test_dtype_still_distinguishes_hash():
    bf16 = {"a": {"shape": [8, 16, 4, 4], "stride": [],
                  "dtype": "torch.bfloat16"}}
    f16 = {"a": {"shape": [8, 16, 4, 4], "stride": [],
                 "dtype": "torch.float16"}}
    assert shape_hash_for_placeholders(bf16) != shape_hash_for_placeholders(f16)


# ============================================================================
# 3. Stride fidelity through canonicalization
# ============================================================================

def _pointwise(a, b):
    return (a + b).relu() * 0.5


def test_channels_last_survives_retrace():
    x = torch.randn(2, 8, 4, 4).to(memory_format=torch.channels_last)
    y = torch.randn(2, 8, 4, 4).to(memory_format=torch.channels_last)
    gm = _traced(_pointwise, x, y)
    pat = _first_pattern(gm)
    strides = [tuple(i.get("stride") or ()) for i in
               pat["placeholder_info"].values()]
    assert all(s == tuple(x.stride()) for s in strides), strides


def test_layout_is_shape_identity_not_pattern_identity():
    x_cl = torch.randn(2, 8, 4, 4).to(memory_format=torch.channels_last)
    gm_cl = _traced(_pointwise, x_cl, x_cl.clone())
    gm_c = _traced(_pointwise, torch.randn(2, 8, 4, 4),
                   torch.randn(2, 8, 4, 4))
    pat_cl = _first_pattern(gm_cl)
    pat_c = _first_pattern(gm_c)
    # different layout -> different shape point...
    assert pat_cl["shape_hash"] != pat_c["shape_hash"]
    # ...but the SAME canonical pattern (layout never forks pattern identity)
    assert pat_cl["pattern_hash"] == pat_c["pattern_hash"]


# ============================================================================
# 4. Schema-typed shape lifting
# ============================================================================

def test_symint_slots_lifted_int_slots_not():
    def f(a):
        b = a.expand(4, 3, 5)          # expand sizes: SymInt[] -> lifted
        c = b.reshape(4, 15)           # view/reshape sizes: SymInt[] -> lifted
        d = c.sum(dim=[1])             # reduction dims: int[] -> NEVER lifted
        return d.view(2, 2)

    gm = _traced(f, torch.randn(1, 3, 5))
    gm, params = lift_shape_params(gm)

    assert len(params) >= 2, params
    assert all(isinstance(v, list) and v for v in params.values()), params

    sum_nodes = [n for n in gm.graph.nodes
                 if n.op == "call_function" and "sum" in str(n.target)]
    assert sum_nodes
    for n in sum_nodes:
        dims = n.args[1] if len(n.args) > 1 else []
        assert not any(isinstance(x, fx.Node) for x in dims), \
            "reduction dims were lifted"

    ph_names = [n.target for n in gm.graph.nodes if n.op == "placeholder"]
    assert all(name in ph_names for name in params), ph_names

    out = gm(torch.randn(1, 3, 5), *params.values())
    assert tuple(out.shape) == (2, 2)


def test_trivial_shapes_not_lifted():
    def g(a):
        return a.view(-1)

    gm = _traced(g, torch.randn(3, 4))
    _gm, params = lift_shape_params(gm)
    assert not params, params


# ============================================================================
# 5. Extern point signature exactness (model_attribution)
# ============================================================================

def test_extern_literal_args_distinguish_points():
    from model_attribution import collect_extern_points

    def f(x, w, y, w2):
        a = torch.ops.aten.convolution.default(
            x, w, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        b = torch.ops.aten.convolution.default(
            y, w2, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)  # padding differs
        c = torch.ops.aten.convolution.default(
            x, w, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)  # dup of a
        return a, b, c

    x = torch.randn(2, 8, 8, 8)
    w = torch.randn(8, 8, 3, 3)
    gm = _traced(f, x, w, x.clone(), w.clone())
    conv_points = {k: v for k, v in collect_extern_points(gm).items()
                   if "convolution" in k[0]}
    # padding-only difference -> 2 distinct points; exact dup -> count 2
    assert len(conv_points) == 2, list(conv_points)
    assert sorted(p["count"] for p in conv_points.values()) == [1, 2]


def test_extern_stride_only_difference_distinguishes_points():
    from model_attribution import collect_extern_points

    def g(x):
        return (torch.ops.aten.mm.default(x, x.transpose(0, 1).contiguous()),
                torch.ops.aten.mm.default(x, x.transpose(0, 1)))

    gm = _traced(g, torch.randn(4, 4))
    mm_points = [k for k in collect_extern_points(gm) if "mm" in k[0]]
    assert len(mm_points) == 2, mm_points
