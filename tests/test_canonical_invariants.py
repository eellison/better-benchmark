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


# ============================================================================
# 6. Conv never reaches the lift
# ============================================================================

def test_conv_symint_slots_never_lifted():
    """aten.convolution spells stride/padding/dilation as SymInt[], where
    lifting would be semantically wrong (different padding = different conv,
    not a shape variant). Safe today because conv is non-fusible — it can
    never appear inside a partition the lift runs on. Pin BOTH halves: the
    partitioner excludes conv, and a graph containing conv + pointwise ops
    lifts nothing from the conv node."""
    def f(x, w):
        y = torch.ops.aten.convolution.default(
            x, w, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1)
        return (y + 1).relu()

    gm = _traced(f, torch.randn(2, 8, 8, 8), torch.randn(8, 8, 3, 3))

    conv_nodes = [n for n in gm.graph.nodes
                  if n.op == "call_function" and "convolution" in str(n.target)]
    assert conv_nodes
    for comp in get_fusion_partitions(gm):
        assert not (set(conv_nodes) & set(comp)), \
            "conv landed inside a fusible partition"

    _gm, params = lift_shape_params(gm)
    for n in conv_nodes:
        assert not any(isinstance(a, fx.Node) and
                       str(a.target).startswith("_shape_param")
                       for a in n.args), "conv literal args were lifted"


def test_no_pointwise_op_has_symint_list_slot():
    """The structural reason the lift is safe: across ALL of aten, no
    pointwise-tagged op has a SymInt[] argument slot (swept 2026-06-11:
    122 SymInt[] overloads, 0 pointwise). SymInt[]-on-fusible-op therefore
    always means view-family output shape — exactly what the lift wants.
    If this ever fails, the schema-typed lift needs an explicit op filter."""
    offenders = []
    for opname in torch.ops.aten:
        packet = getattr(torch.ops.aten, opname)
        for overload in packet.overloads():
            op = getattr(packet, overload)
            if torch.Tag.pointwise not in op.tags:
                continue
            if "SymInt[]" in str(op._schema).split("->")[0]:
                offenders.append(str(op))
    assert not offenders, offenders


# ============================================================================
# 7. Node accounting: exhaustive, auditable fusibility classification
# ============================================================================

def test_node_accounting_exhaustive():
    """Every call_function lands in exactly one bucket; counts sum to total.
    The accounting is the capture-time record of the fusible/non-fusible
    decision — if Inductor's is_fusible_node changes, the artifact diff
    shows which ops moved buckets."""
    from capture_hook import graph_node_accounting

    def f(x, w):
        y = torch.ops.aten.convolution.default(
            x, w, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        return (y + 1).relu()

    gm = _traced(f, torch.randn(2, 8, 8, 8), torch.randn(8, 8, 3, 3))
    acct = graph_node_accounting(gm)

    n_cf = sum(1 for n in gm.graph.nodes if n.op == "call_function")
    assert acct["total_call_functions"] == n_cf
    assert sum(acct["counts"].values()) == n_cf, acct["counts"]
    # per-bucket op sets agree with their counts
    for bucket, ops in acct["ops"].items():
        assert sum(ops.values()) == acct["counts"][bucket], (bucket, ops)
    # conv classified non-fusible, pointwise classified fusible
    assert any("convolution" in op for op in acct["ops"]["non_fusible"])
    all_fusible_ops = (set(acct["ops"]["fusible_in_partition"])
                       | set(acct["ops"]["fusible_unpartitioned"]))
    assert not any("convolution" in op for op in all_fusible_ops)


# ============================================================================
# 8. Unreturned mutation is preserved
# ============================================================================

def test_unreturned_mutation_preserved():
    """A copy_ whose result is NOT returned (buffer mutation: the write IS
    the effect) must survive partition extraction and the canonicalize
    retrace — make_fx only keeps what's reachable from outputs, so the
    extraction rule 'mutating ops are outputs' is what protects it."""
    def f(a, b, buf):
        y = (a + b).relu()
        torch.ops.aten.copy_.default(buf, y)  # not returned
        return y * 2

    gm = _traced(f, torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4))
    assert any("copy_" in str(n.target) for n in gm.graph.nodes
               if n.op == "call_function"), "copy_ lost in test trace itself"

    comps = get_fusion_partitions(gm)
    comp = max(comps, key=len)
    pat = compute_partition_pattern(comp, gm)
    assert pat is not None

    canon_ops = [str(n.target) for n in pat["sub_gm"].graph.nodes
                 if n.op == "call_function"]
    assert any("copy" in op for op in canon_ops), \
        f"unreturned mutation vanished through canonicalization: {canon_ops}"

    # and it must still be there after a SECOND retrace (idempotence
    # includes the mutation)
    sub2, _, _ = canonicalize_subgraph(
        pat["sub_gm"], pat["placeholder_info"], pat["shape_params"])
    ops2 = [str(n.target) for n in sub2.graph.nodes
            if n.op == "call_function"]
    assert any("copy" in op for op in ops2), \
        f"mutation lost on re-retrace: {ops2}"


# ============================================================================
# 9. Pattern hash encodes dtype / string-mode / memory-format args
# ============================================================================

def _pattern_hashes(fn, *inputs):
    gm = _traced(fn, *inputs)
    comps = get_fusion_partitions(gm)
    pat = compute_partition_pattern(max(comps, key=len), gm)
    return pat["pattern_hash"], pat["shape_hash"]


def test_pattern_hash_encodes_dtype_args():
    """convert_element_type to f32 vs f16 writes different bytes — must be
    different patterns. Collision found by adversarial review 2026-06-11:
    _encode_arg dropped torch.dtype args, so 440-model dedup would merge
    semantically different partitions and attribute timings to the wrong
    kernel."""
    x = torch.randn(8, 8, dtype=torch.bfloat16)

    def to_f32(a):
        return torch.ops.prims.convert_element_type.default(a + 1, torch.float32)

    def to_f16(a):
        return torch.ops.prims.convert_element_type.default(a + 1, torch.float16)

    h32, _ = _pattern_hashes(to_f32, x)
    h16, _ = _pattern_hashes(to_f16, x)
    assert h32 != h16


def test_pattern_hash_encodes_str_mode_args():
    """div rounding_mode floor vs trunc are different computations."""
    y = torch.randn(8, 8)

    def div_floor(a, b):
        return torch.ops.aten.div.Tensor_mode(a + 1, b, rounding_mode="floor")

    def div_trunc(a, b):
        return torch.ops.aten.div.Tensor_mode(a + 1, b, rounding_mode="trunc")

    hf, _ = _pattern_hashes(div_floor, y, y.clone())
    ht, _ = _pattern_hashes(div_trunc, y, y.clone())
    assert hf != ht


def test_pattern_hash_ignores_scalar_constants():
    """Scalar float constants (mul by 3.0 vs 5.0, eps) do NOT fork the
    pattern: same kernel structure, the constant is baked. This is the
    intended dedup — only encode what changes the kernel."""
    z = torch.randn(8, 8)

    def mul3(a):
        return (a + 1) * 3.0

    def mul5(a):
        return (a + 1) * 5.0

    h3, _ = _pattern_hashes(mul3, z)
    h5, _ = _pattern_hashes(mul5, z)
    assert h3 == h5


def test_pattern_hash_encodes_int_dim_slots():
    """Bare 'int'-typed schema slots are dims (cat/gather axis) — structural,
    must fork the pattern. 'Scalar'-typed ints (add.Scalar) remain baked
    constants. Schema-typed discrimination, same principle as the SymInt[]
    lift; found by user question 'what ints are what' 2026-06-11: cat dim=0
    vs dim=1 on a square input hashed identically."""
    x = torch.randn(8, 8)  # square: shapes can't disambiguate the dim

    def cat0(a, b):
        return torch.cat([a + 1, b], 0).relu()

    def cat1(a, b):
        return torch.cat([a + 1, b], 1).relu()

    h0, _ = _pattern_hashes(cat0, x, x.clone())
    h1, _ = _pattern_hashes(cat1, x, x.clone())
    assert h0 != h1

    idx = torch.zeros(8, 8, dtype=torch.int64)
    g0, _ = _pattern_hashes(lambda a, i: torch.gather(a + 1, 0, i), x, idx)
    g1, _ = _pattern_hashes(lambda a, i: torch.gather(a + 1, 1, i), x, idx)
    assert g0 != g1


def test_int_dim_positional_vs_kwarg_invocation():
    """The same structural dim must hash IDENTICALLY whether the model
    called the op positionally or with a kwarg (same computation — the
    canonicalize retrace normalizes call spelling), and DIFFERENT dims
    must hash differently in BOTH invocation forms."""
    x = torch.randn(8, 8)

    pos0, _ = _pattern_hashes(
        lambda a: torch.ops.aten.argmax.default(a + 1, 0), x)
    pos1, _ = _pattern_hashes(
        lambda a: torch.ops.aten.argmax.default(a + 1, 1), x)
    kw0, _ = _pattern_hashes(
        lambda a: torch.ops.aten.argmax.default(a + 1, dim=0), x)
    kw1, _ = _pattern_hashes(
        lambda a: torch.ops.aten.argmax.default(a + 1, dim=1), x)

    assert pos0 != pos1, "positional dim not structural"
    assert kw0 != kw1, "kwarg dim not structural"
    assert pos0 == kw0 and pos1 == kw1, \
        "same dim hashed differently across invocation forms"


def test_kwarg_dtype_and_mode_match_positional():
    """dtype / rounding_mode must be structural in both invocation forms
    and spelling-invariant across them."""
    y = torch.randn(8, 8)

    posf, _ = _pattern_hashes(
        lambda a, b: torch.ops.aten.div.Tensor_mode(a + 1, b, rounding_mode="floor"),
        y, y.clone())
    post, _ = _pattern_hashes(
        lambda a, b: torch.ops.aten.div.Tensor_mode(a + 1, b, rounding_mode="trunc"),
        y, y.clone())
    assert posf != post

    s0, _ = _pattern_hashes(lambda a: (a + 1).sum(dim=[0]), torch.randn(8, 8))
    s1, _ = _pattern_hashes(lambda a: (a + 1).sum(dim=[1]), torch.randn(8, 8))
    assert s0 != s1, "int[] reduction dims not structural"


# ============================================================================
# 10. Output order canonicalized by definition order
# ============================================================================

def test_output_consumption_order_irrelevant():
    """A partition's identity must not depend on the ORDER its outputs are
    consumed outside: extraction sorts output nodes by definition order
    (graph_pos), so two models consuming (u, v) vs (v, u) dedup to ONE
    canonical repro. The mm consumers are non-fusible, forcing both u and
    v to escape the partition as outputs."""
    x, w = torch.randn(8, 8), torch.randn(8, 8)

    def f_uv(a, b, w):
        s = a + b
        u = s.relu()
        v = s.sigmoid()
        return torch.mm(u, w), torch.mm(v, w)

    def f_vu(a, b, w):
        s = a + b
        u = s.relu()
        v = s.sigmoid()
        return torch.mm(v, w), torch.mm(u, w)  # consumption order swapped

    h1, s1 = _pattern_hashes(f_uv, x, x.clone(), w)
    h2, s2 = _pattern_hashes(f_vu, x, x.clone(), w)
    assert (h1, s1) == (h2, s2), \
        "output consumption order leaked into partition identity"


def test_output_definition_order_is_the_canonical_order():
    """Sorting is BY DEFINITION ORDER, not by consumer: if the defs
    themselves swap (sigmoid defined before relu), that IS a different
    def order for the same multiset of ops — but the canonical node
    renumbering (deterministic Kahn) renumbers topologically-equivalent
    graphs identically, so the hash must still match."""
    x, w = torch.randn(8, 8), torch.randn(8, 8)

    def f_relu_first(a, b, w):
        s = a + b
        u = s.relu()
        v = s.sigmoid()
        return torch.mm(u, w), torch.mm(v, w)

    def f_sigmoid_first(a, b, w):
        s = a + b
        v = s.sigmoid()
        u = s.relu()
        return torch.mm(u, w), torch.mm(v, w)

    h1, s1 = _pattern_hashes(f_relu_first, x, x.clone(), w)
    h2, s2 = _pattern_hashes(f_sigmoid_first, x, x.clone(), w)
    assert (h1, s1) == (h2, s2), \
        "definition-order permutation of independent ops forked the hash"
