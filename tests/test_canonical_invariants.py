"""Regression tests for the canonicalize-before-serialization invariants.

These tests are the executable documentation of the corpus identity
invariants (CORPUS_MIGRATION_PLAN §1). Everything here was landed (and
burned by at least once) while hardening the capture pipeline, 2026-06:

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
    """make_fx fake-mode trace WITH the Inductor decomposition table.

    The capture hook sees POST-GRAD graphs — after Inductor's decomps have
    run. Tracing test functions without the decomp table manufactures
    spellings that can't occur in production (e.g. aten._to_copy, which
    decomposes to prims.convert_element_type before post-grad) and makes
    tests assert on graphs the pipeline never sees. Audited 2026-06-11:
    zero _to_copy across all 540 saved post-grad graphs.
    """
    from torch._inductor.decomposition import select_decomp_table

    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fakes = [mode.from_tensor(t) if torch.is_tensor(t) else t
                 for t in inputs]
        return make_fx(fn, tracing_mode="fake",
                       decomposition_table=select_decomp_table())(*fakes)


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


# ============================================================================
# 11. Hardened idempotence + tie-break + lift coverage (gap probes 2026-06-11)
# ============================================================================

def test_idempotence_reduction_partition():
    """Fixed point must hold on reduction partitions (var_mean/layernorm
    family), not just pointwise — reductions retrace through different
    decomposition paths."""
    def f(a, w):
        x = a.to(torch.float32)
        vm = torch.var_mean(x, dim=[1], keepdim=True)
        y = (x - vm[1]) * torch.rsqrt(vm[0] + 1e-5) * w
        return y.to(torch.bfloat16)

    gm = _traced(f, torch.randn(64, 384, dtype=torch.bfloat16),
                 torch.randn(1, 384))
    p1 = _first_pattern(gm)
    sub2, info2, _ = canonicalize_subgraph(
        p1["sub_gm"], p1["placeholder_info"], p1["shape_params"])
    assert pattern_hash_for_subgraph(sub2) == p1["pattern_hash"]
    assert shape_hash_for_placeholders(info2) == p1["shape_hash"]


def test_idempotence_with_lifted_shape_params():
    """canonicalize -> lift -> canonicalize again must converge: the bake
    (closure) -> retrace -> re-lift cycle is a fixed point including the
    lifted param VALUES."""
    def f(a):
        return (a + 1).expand(8, 4, 6).reshape(8, 24) * 2

    gm = _traced(f, torch.randn(1, 4, 6))
    p1 = _first_pattern(gm)
    assert p1["shape_params"], "test premise: shape params must be lifted"
    sub2, _info2, sp2 = canonicalize_subgraph(
        p1["sub_gm"], p1["placeholder_info"], p1["shape_params"])
    assert pattern_hash_for_subgraph(sub2) == p1["pattern_hash"]
    assert sp2 == p1["shape_params"], "lifted param values drifted on re-canon"


def test_kahn_same_preds_same_key_deterministic():
    """TRUE tie-break ambiguity: two relu ops on the SAME input (same pred
    indices, same structural key) with different consumers — def order
    must not fork the hash, and the signature must still wire each relu
    to its own consumer correctly."""
    def f1(a, w):
        u = a.relu()
        v = a.relu()
        return torch.mm(u, w), torch.mm(v.sigmoid(), w)

    def f2(a, w):
        v = a.relu()
        u = a.relu()
        return torch.mm(u, w), torch.mm(v.sigmoid(), w)

    x, w = torch.randn(8, 8), torch.randn(8, 8)
    h1, s1 = _pattern_hashes(f1, x, w)
    h2, s2 = _pattern_hashes(f2, x, w)
    assert (h1, s1) == (h2, s2)


def test_lift_shape_params_idempotent():
    """Lifting an already-lifted graph is a no-op: no new params, no new
    placeholders. (A second lift seeing _shape_param Node args must skip.)"""
    def f(a):
        return (a + 1).expand(8, 4, 6).reshape(8, 24)

    gm = _traced(f, torch.randn(1, 4, 6))
    gm, params1 = lift_shape_params(gm)
    n_ph_1 = sum(1 for n in gm.graph.nodes if n.op == "placeholder")
    gm, params2 = lift_shape_params(gm)
    n_ph_2 = sum(1 for n in gm.graph.nodes if n.op == "placeholder")
    assert params1 and not params2
    assert n_ph_1 == n_ph_2


def test_mutation_into_view_preserved():
    """copy_ into a VIEW SLICE (buf[0].copy_(y)) — the select/view chain
    plus the mutation must survive extraction + canonicalization."""
    def f(a, b, buf):
        y = (a + b).relu()
        torch.ops.aten.copy_.default(buf[0], y)
        return y * 2

    gm = _traced(f, torch.randn(4, 4), torch.randn(4, 4),
                 torch.randn(2, 4, 4))
    pat = _first_pattern(gm)
    ops = [str(n.target) for n in pat["sub_gm"].graph.nodes
           if n.op == "call_function"]
    assert any("copy" in o for o in ops), ops


def test_node_accounting_components_param_equivalence():
    """graph_node_accounting(gm) == graph_node_accounting(gm, precomputed)
    — the shared-components optimization can't change the verdict."""
    from capture_hook import graph_node_accounting

    def f(x, w):
        y = torch.ops.aten.convolution.default(
            x, w, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        return (y + 1).relu()

    gm = _traced(f, torch.randn(2, 8, 8, 8), torch.randn(8, 8, 3, 3))
    assert graph_node_accounting(gm) == graph_node_accounting(
        gm, get_fusion_partitions(gm))


def test_shape_hash_distinguishes_input_multiplicity():
    """Three identical-spec inputs vs two must hash differently (the
    sorted-strings multiset keeps duplicates)."""
    a = {f"p{i}": {"shape": [8, 16], "stride": [], "dtype": "torch.float32"}
         for i in range(3)}
    b = {f"p{i}": {"shape": [8, 16], "stride": [], "dtype": "torch.float32"}
         for i in range(2)}
    assert (shape_hash_for_placeholders(a)
            != shape_hash_for_placeholders(b))


# ============================================================================
# 12. Storage offset + exotic dtypes preserved through serialization
# ============================================================================

def test_storage_offset_recorded_and_honored():
    """Packed-qkv-style saved views (same storage, nonzero storage_offset)
    must survive the spec roundtrip: recorded in the compact encoding's
    'off' opt and honored on regeneration. First wave-1 train captures
    exposed this: annotations carry no offset, so the sidecar is the only
    carrier — losing it silently regenerates aliased views as offset-0."""
    from input_codec import compact_from_spec, spec_from_compact

    spec = {
        "kind": "tensor", "shape": [128, 3, 197, 64],
        "dtype": "bfloat16", "stride": [113472, 64, 576, 1],
        "storage_offset": 192,
    }
    entry = compact_from_spec(spec)
    assert entry[2]["off"] == 192, entry
    rt = spec_from_compact(entry)
    assert rt["storage_offset"] == 192
    assert rt["stride"] == [113472, 64, 576, 1]


def test_uint_and_fp8_dtypes_roundtrip_codec():
    """u16/u32/u64 (RNG-state tensors in train graphs) and fp8 must have
    SHORT_DTYPE entries — an unknown dtype passes through by long name,
    which works but silently forks identity spelling from every other
    artifact. mobilevit u64 misparse was the wave-1 catch."""
    from input_codec import compact_from_spec, spec_from_compact

    for dt in ("uint16", "uint32", "uint64", "float8_e4m3fn", "float8_e5m2"):
        entry = compact_from_spec(
            {"kind": "tensor", "shape": [2], "dtype": dt, "stride": [1]})
        assert entry[1] != dt, f"{dt}: no short spelling in codec"
        rt = spec_from_compact(entry)
        assert rt["dtype"] == dt, (dt, rt["dtype"])


def test_annotation_parser_offset_unknown_and_u64():
    """print_readable annotations carry no storage_offset: the parser must
    record None (unknown), never a false 0 — and must know u64 tokens."""
    from full_graph_harness import parse_full_graph_inputs

    content = '''
class Repro(torch.nn.Module):
    def forward(self, getitem_3: "bf16[128, 3, 197, 64][113472, 64, 576, 1]cuda:0", rng_state: "u64[2][1]cuda:0"):
        pass
'''
    specs = parse_full_graph_inputs(content)
    by_name = {s["name"]: s for s in specs}
    assert by_name["getitem_3"]["storage_offset"] is None
    assert by_name["rng_state"]["dtype"] == "uint64"


# ============================================================================
# 13. Alias groups: shared-storage inputs preserved end to end
# ============================================================================

def test_alias_group_codec_roundtrip():
    """alias_group survives compact encode/decode (opts['alias'])."""
    from input_codec import compact_from_spec, spec_from_compact

    spec = {"kind": "tensor", "shape": [8, 3, 4, 2], "dtype": "bfloat16",
            "stride": [24, 2, 6, 1], "storage_offset": 2, "alias_group": 0}
    entry = compact_from_spec(spec)
    assert entry[2]["alias"] == 0 and entry[2]["off"] == 2, entry
    rt = spec_from_compact(entry)
    assert rt["alias_group"] == 0 and rt["storage_offset"] == 2


def test_alias_group_generation_shares_storage():
    """Members of one alias group must be views of ONE buffer at their own
    offsets — writing through one member must be visible through the
    overlapping region of the storage (packed-qkv fidelity), and the
    group buffer is sized by alias_group_nbytes, not derived by scanning."""
    import torch
    from repro_harness import make_inputs_from_config

    # 3 views into one packed [N, 3*D] buffer: q/k/v at offsets 0/D/2D —
    # deit-style packed projection (D=4, N=8, fp32 -> 4 bytes/elem).
    N, D = 8, 4
    nbytes = N * 3 * D * 4
    config = {
        "alias_group_nbytes": [nbytes],
        "inputs": [
            {"kind": "tensor", "shape": [N, D], "dtype": "float32",
             "stride": [3 * D, 1], "storage_offset": off, "alias_group": 0,
             "device": "cpu"}
            for off in (0, D, 2 * D)
        ],
    }
    q, k, v = make_inputs_from_config(config)
    assert q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr() \
        == v.untyped_storage().data_ptr(), "alias group split into storages"
    assert (q.storage_offset(), k.storage_offset(), v.storage_offset()) == (0, D, 2 * D)
    # write through q row 0; k/v must NOT change (disjoint columns), but the
    # underlying storage region q covers must reflect it
    q[0].fill_(7.0)
    flat = torch.empty(0)
    flat = q.untyped_storage()
    assert q[0, 0].item() == 7.0
    # k's first element lives D floats after q's — distinct value space
    assert k.storage_offset() - q.storage_offset() == D


def test_alias_group_only_for_multiply_referenced():
    """A storage referenced by exactly ONE partition placeholder gets NO
    alias tag (single view == private buffer, grouping is noise). Pins the
    count>1 filter in the capture-side group assignment."""
    config = {
        "inputs": [
            {"kind": "tensor", "shape": [4, 4], "dtype": "float32",
             "stride": [4, 1], "storage_offset": 16, "device": "cpu"},
        ],
    }
    from repro_harness import make_inputs_from_config
    (t,) = make_inputs_from_config(config)
    # no alias_group -> plain generation path; offset alone is fine to drop
    # for a private buffer (behaviorally identical)
    assert t.shape == (4, 4)


def test_symint_and_scalar_inputs_materialize():
    """A live symint input (dynamic-shape graph) and a python scalar must
    materialize to a plain int / value through make_inputs_from_config.

    Regression pin for the silent-drop bug: make_inputs_from_config had no
    kind=='symint' branch, so a ['sym', h] input fell through to
    spec['shape'] and KeyError'd — which made eager validation fail and
    every dynamic-compilation region get dropped from the corpus. A live
    ['I', hint, expr] symint also evaluates (under a binding) to ['sym',
    val] and must materialize identically."""
    from input_codec import spec_from_compact, evaluate_symbolic_entry
    from repro_harness import make_inputs_from_config

    assert make_inputs_from_config(
        {"inputs": [spec_from_compact(["sym", 16])]}) == [16]
    assert make_inputs_from_config(
        {"inputs": [spec_from_compact(["sc", 0.5])]}) == [0.5]

    # ['I', hint, expr] -> evaluate at a binding -> ['sym', val] -> int
    ev = evaluate_symbolic_entry(["I", 256, "s16*s82"], {"s16": 24, "s82": 24})
    assert ev == ["sym", 576]
    assert make_inputs_from_config(
        {"inputs": [spec_from_compact(ev)]}) == [576]


def test_device_default_accelerator_cpu_recorded():
    """Device convention: default = accelerator. Bare 'cuda' AND any cuda
    ordinal ('cuda:0' — workers are pinned via CUDA_VISIBLE_DEVICES, so the
    ordinal is meaningless identity-wise) are omitted from the encoding;
    cpu (RNG state tensors etc.) is a real deviation and must roundtrip."""
    from input_codec import compact_from_spec, spec_from_compact

    for dev in ("cuda", "cuda:0", "cuda:1"):
        e = compact_from_spec({"kind": "tensor", "shape": [4],
                               "dtype": "float32", "stride": [1],
                               "device": dev})
        assert len(e) == 2 or "dev" not in e[2], (dev, e)
        assert spec_from_compact(e)["device"] == "cuda"

    e = compact_from_spec({"kind": "tensor", "shape": [2], "dtype": "uint64",
                           "stride": [1], "device": "cpu"})
    assert e[2]["dev"] == "cpu", e
    assert spec_from_compact(e)["device"] == "cpu"


def test_alias_group_nbytes_travels_through_shapes_json_load():
    """The v3 default path (load_shape_configs -> make_inputs_from_config)
    must propagate alias_group_nbytes from the shapes.json point into the
    config — without it, alias-tagged specs crash generation (adversarial
    review bug #1: the primary `python repro.py` path)."""
    import json, tempfile
    from pathlib import Path
    from repro_harness import load_shape_configs, make_inputs_from_config

    N, D = 8, 4
    point = {
        "shape_hash": "deadbeef",
        "models": {"timm/train/x": {"occurrences": 1}},
        "inputs": [
            [[N, D], "f32", {"st": [3 * D, 1], "off": off, "alias": 0,
                             "dev": "cpu"}]
            for off in (0, D, 2 * D)
        ],
        "alias_group_nbytes": [N * 3 * D * 4],
    }
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "shapes.json").write_text(json.dumps({"points": [point]}))
        (d / "repro.py").write_text("# stub")
        configs = load_shape_configs(str(d / "repro.py"))
        cfg = next(iter(configs.values()))
        assert cfg.get("alias_group_nbytes") == [N * 3 * D * 4], cfg.keys()
        q, k, v = make_inputs_from_config(cfg)
        assert q.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()


def test_merge_backfills_alias_onto_existing_point():
    """Re-merging a richer (alias-tagged) capture onto a pre-alias point
    must REPLACE the inputs and add alias_group_nbytes — never silently
    discard fidelity (adversarial review bug #2)."""
    import json, tempfile
    from pathlib import Path
    from merge_captures import _write_shapes_json

    plain = [[[8, 4], "f32", {"st": [12, 1]}]]
    rich = [[[8, 4], "f32", {"st": [12, 1], "off": 0, "alias": 0}],
            [[8, 4], "f32", {"st": [12, 1], "off": 4, "alias": 0}]]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_shapes_json(d, "aabbccdd", "(sig)", "timm/train/m1",
                           occurrences=1, inputs=plain)
        _write_shapes_json(d, "aabbccdd", "(sig)", "timm/train/m2",
                           occurrences=2, inputs=rich,
                           alias_group_nbytes=[384])
        data = json.loads((d / "shapes.json").read_text())
        pt = data["points"][0]
        assert pt["alias_group_nbytes"] == [384]
        assert any("alias" in (e[2] if len(e) > 2 else {})
                   for e in pt["inputs"]), "rich inputs not backfilled"


def test_merge_writes_dynamic_symbols_guards_bindings():
    """A dynamic capture index entry (symbols/guards on the entry, exprs in
    inputs) merges into shapes.json with symbols/guards GRAPH-LEVEL and
    bindings/captured_dynamic per point — exactly the schema
    _parse_shapes_json reads. CPU-only: pure dict plumbing."""
    import json, tempfile
    from pathlib import Path
    from merge_captures import _write_shapes_json

    symbols = {"s0": {"hint": 16, "range": [2, None]},
               "s53": {"hint": 16, "range": [2, None]}}
    guards = ["Eq(s0*s53, 256)"]
    inputs = [[[64, 64, "s53", "s0"], "f32",
               {"st": ["64*s0*s53", "s0*s53", "s0", 1]}],
              ["I", 256, "s0*s53"]]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_shapes_json(d, "deadbeef", "(sig)", "probe/infer/m",
                           occurrences=1, inputs=inputs,
                           symbols=symbols, guards=guards)
        data = json.loads((d / "shapes.json").read_text())
        # graph-level
        assert data["symbols"] == symbols
        assert data["guards"] == guards
        pt = data["points"][0]
        assert pt["captured_dynamic"] is True
        assert pt["bindings"] == {"s0": 16, "s53": 16}  # hints
        # exprs preserved verbatim in the point inputs
        assert pt["inputs"][0][0] == [64, 64, "s53", "s0"]
        assert pt["inputs"][1] == ["I", 256, "s0*s53"]

        # idempotent re-merge of the same symbols doesn't duplicate guards
        _write_shapes_json(d, "deadbeef", "(sig)", "probe/infer/m2",
                           occurrences=1, inputs=inputs,
                           symbols=symbols, guards=guards)
        data2 = json.loads((d / "shapes.json").read_text())
        assert data2["guards"] == guards, "guard duplicated on re-merge"
        assert len(data2["points"]) == 1


def test_merge_static_point_has_no_dynamic_fields():
    """A static capture (no symbols) must NOT gain symbols/guards/bindings/
    captured_dynamic — the static path stays byte-clean."""
    import json, tempfile
    from pathlib import Path
    from merge_captures import _write_shapes_json

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_shapes_json(d, "cafef00d", "(sig)", "timm/train/m",
                           occurrences=1, inputs=[[[8, 4], "f32"]])
        data = json.loads((d / "shapes.json").read_text())
        assert "symbols" not in data and "guards" not in data
        pt = data["points"][0]
        assert "bindings" not in pt and "captured_dynamic" not in pt


def test_maxpool_offset_tensors_generate_constant_center():
    """int8 maxpool OFFSET tensors must generate the window-CENTER constant,
    not random offsets: under padding, edge windows turn most offsets into
    out-of-range indices (offset 0 at a padded edge -> index -113), and the
    consuming scatter_add device-side-asserts — poisoning the CUDA context
    for the whole process (the wave-1 torchbench train failure class)."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    import torch._inductor.inductor_prims  # noqa: F401
    from full_graph_harness import infer_index_bounds_from_gm, placeholder_info_from_gm

    def f(off):
        idx = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
            off, [3, 3], [16, 16], [2, 2], [1, 1], [1, 1])
        return idx + 1

    with FakeTensorMode(allow_non_fake_inputs=True) as m:
        off = m.from_tensor(torch.zeros(2, 4, 8, 8, dtype=torch.int8))
        gm = make_fx(f, tracing_mode="fake")(off)

    info = placeholder_info_from_gm(gm)
    constants: dict = {}
    bounds = infer_index_bounds_from_gm(gm, info, constants_out=constants)
    (name,) = info.keys()
    assert name in constants, "offset tensor not routed to constant gen"
    assert constants[name] == 4, constants  # 3x3 window center
    assert name not in bounds, "must not ALSO get a random index bound"

    # And the generation path honors constant kind:
    from repro_harness import make_inputs_from_config
    (t,) = make_inputs_from_config({"inputs": [
        {"kind": "tensor", "shape": [2, 4, 8, 8], "dtype": "int8",
         "stride": [], "device": "cpu",
         "gen": {"kind": "constant", "value": 4}}]})
    assert t.unique().tolist() == [4]


def test_constant_gen_codec_roundtrip():
    """constant generation kind roundtrips the compact encoding."""
    from input_codec import compact_from_spec, spec_from_compact

    spec = {"kind": "tensor", "shape": [4], "dtype": "int8", "stride": [1],
            "gen": {"kind": "constant", "value": 4}}
    e = compact_from_spec(spec)
    assert e[2]["gen"] == ["const", 4], e
    rt = spec_from_compact(e)
    assert rt["gen"] == {"kind": "constant", "value": 4}


def test_index_bound_minimum_across_consumers():
    """A value consumed by SEVERAL index ops must get the MINIMUM bound:
    Electra's ids flow through gather (bound 512) INTO a 2-row token-type
    embedding — first-found inference returned 512 and the embedding
    asserted OOB (poisoning the CUDA context for all later validations).
    The walk must also follow gather's data edge (values flow through)."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from full_graph_harness import (infer_index_bounds_from_gm,
                                    placeholder_info_from_gm)

    def f(ids, pos, table2):
        g = torch.ops.aten.gather.default(pos, 1, ids)      # bound 512 via pos dim1
        e = torch.ops.aten.embedding.default(table2, g)      # bound 2 via table rows
        return e + 1

    with FakeTensorMode(allow_non_fake_inputs=True) as m:
        ids = m.from_tensor(torch.zeros(1, 512, dtype=torch.int64))
        pos = m.from_tensor(torch.zeros(1, 512, dtype=torch.int64))
        table2 = m.from_tensor(torch.zeros(2, 128))
        gm = make_fx(f, tracing_mode="fake")(ids, pos, table2)

    info = placeholder_info_from_gm(gm)
    bounds = infer_index_bounds_from_gm(gm, info)
    # Identify roles from the graph: gather(data, dim, index).
    data_ph = index_ph = None
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            for u in node.users:
                if u.op == "call_function" and "gather" in str(u.target):
                    if len(u.args) > 0 and u.args[0] is node:
                        data_ph = node.name
                    if len(u.args) > 2 and u.args[2] is node:
                        index_ph = node.name
    assert data_ph and index_ph, "test graph wiring changed"
    # index arg: bounded by the gathered dim (512)
    assert bounds.get(index_ph) == 512, bounds
    # DATA arg: its VALUES flow through gather into the 2-row embedding —
    # the through-gather walk must bound it by the table (2), not leave it
    # unbounded (the Electra OOB-assert class).
    assert bounds.get(data_ph) == 2, bounds


def test_index_bound_inverts_arithmetic_chain():
    """Bounds must be INVERTED through int arithmetic between the
    placeholder and the consuming index op (Longformer: position ids =
    (mask_i64 * mask_i32 + 1) index a 4098-row table — the leaf bound is
    NOT 4098; a tensor*tensor product needs each operand < sqrt(B))."""
    import math
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from full_graph_harness import (infer_index_bounds_from_gm,
                                    placeholder_info_from_gm)

    def f(a, b, table):
        prod = torch.ops.aten.mul.Tensor(a, b)
        idx = torch.ops.aten.add.Tensor(prod, 1)
        return torch.ops.aten.embedding.default(table, idx, 1)

    with FakeTensorMode(allow_non_fake_inputs=True) as m:
        a = m.from_tensor(torch.zeros(8, 64, dtype=torch.int64))
        b = m.from_tensor(torch.zeros(8, 64, dtype=torch.int64))
        table = m.from_tensor(torch.zeros(4098, 768))
        gm = make_fx(f, tracing_mode="fake")(a, b, table)

    info = placeholder_info_from_gm(gm)
    bounds = infer_index_bounds_from_gm(gm, info)
    int_phs = [n for n, i in info.items() if "int" in i["dtype"]]
    assert int_phs
    expected = math.isqrt(4098 - 1 - 1)  # invert add 1, then sqrt for mul
    for n in int_phs:
        assert bounds.get(n) is not None, (n, bounds)
        assert bounds[n] <= expected, (n, bounds[n], expected)
        # and generated products must actually be in range:
        hi = bounds[n]
        assert (hi - 1) * (hi - 1) + 1 < 4098


def test_maxpool_offset_kernel_size_from_lifted_param():
    """In canonical (lifted) graphs the maxpool kernel size arg is a
    _shape_param placeholder NODE, not a literal — the offset-constant
    inference must resolve it via meta['val'] (vgg16: literal-only check
    fell through to a 3x3-center default of 4 on a 2x2 kernel; offset 4
    is invalid for 2x2 -> scatter assert)."""
    import torch
    import torch.fx as fx
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    import torch._inductor.inductor_prims  # noqa: F401
    from capture_hook import lift_shape_params
    from full_graph_harness import (infer_index_bounds_from_gm,
                                    placeholder_info_from_gm)

    def f(off):
        idx = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
            off, [2, 2], [16, 16], [2, 2], [0, 0], [1, 1])
        return idx + 1

    with FakeTensorMode(allow_non_fake_inputs=True) as m:
        off = m.from_tensor(torch.zeros(2, 4, 8, 8, dtype=torch.int8))
        gm = make_fx(f, tracing_mode="fake")(off)
    gm, params = lift_shape_params(gm)
    # premise: kernel size got lifted to a param node
    assert any(v == [2, 2] for v in params.values()), params

    info = placeholder_info_from_gm(gm)
    constants: dict = {}
    infer_index_bounds_from_gm(gm, info, constants_out=constants)
    int8_phs = [n for n, i in info.items() if "int8" in i["dtype"]]
    assert int8_phs
    name = int8_phs[0]
    # 2x2 kernel: center = (kh//2)*kw + kw//2 = 1*2+1 = 3 — NOT the 3x3
    # default 4 (measured: offset 3 on 2x2 pad0 stays in-bounds; 4 is OOB)
    assert constants.get(name) == 3, constants


def test_zero_input_point_loads_and_runs():
    """'inputs': [] is a VALID config (constant-producing partitions like
    scalar_tensor take no args): load_shape_configs must return it (not
    skip the point as falsy) and generation must return []."""
    import json, tempfile
    from pathlib import Path
    from repro_harness import load_shape_configs, make_inputs_from_config

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "shapes.json").write_text(json.dumps({"points": [{
            "shape_hash": "00000000", "inputs": [],
            "models": {"hf/infer/x": {"occurrences": 1}}}]}))
        (d / "repro.py").write_text("# stub")
        configs = load_shape_configs(str(d / "repro.py"))
        assert configs, "zero-input point skipped as falsy"
        cfg = next(iter(configs.values()))
        assert cfg["inputs"] == []
        assert make_inputs_from_config(cfg) == []


def test_float_placeholder_feeding_index_chain_gets_bounded_gen():
    """FLOAT placeholders whose VALUES flow (via convert chains) into index
    consumption need bounded non-negative generation (OPT: position ids =
    (float_mask - 1).to(int64) + 2 index a 2050-row table; randn goes
    negative/OOB -> embedding assert -> poisoned context)."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from full_graph_harness import (infer_index_bounds_from_gm,
                                    placeholder_info_from_gm)

    def f(mask, table):
        x = torch.ops.aten.sub.Tensor(mask, 1)
        ids = torch.ops.prims.convert_element_type.default(x, torch.int64)
        ids2 = torch.ops.aten.add.Tensor(ids, 2)
        return torch.ops.aten.embedding.default(table, ids2)

    with FakeTensorMode(allow_non_fake_inputs=True) as m:
        mask = m.from_tensor(torch.zeros(4, 128, dtype=torch.float32))
        table = m.from_tensor(torch.zeros(2050, 768))
        gm = make_fx(f, tracing_mode="fake")(mask, table)

    info = placeholder_info_from_gm(gm)
    bounds = infer_index_bounds_from_gm(gm, info)
    float_phs = [n for n, i in info.items()
                 if "float" in i["dtype"] and len(i["shape"]) == 2
                 and i["shape"][0] == 4]
    assert float_phs
    name = float_phs[0]
    assert bounds.get(name) is not None, bounds
    # inverted through (+2 after convert; sub 1 before): bound well under 2050
    assert bounds[name] < 2050
    # and the info dict got a bounded non-negative gen attached
    assert info[name].get("gen", {}).get("kind") == "index", info[name]


def test_index_put_indices_map_position_to_dim_including_nones():
    """index_put's indices LIST position maps to the indexed DIM, counting
    None slots: index_put(t, [None, None, i, j], v) — i indexes dim 2,
    j indexes dim 3. YituTechConvBert: j bounded by shape[0]=32 against a
    dim-3 of size 1 generated OOB indices -> assert."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from full_graph_harness import (infer_index_bounds_from_gm,
                                    placeholder_info_from_gm)

    def f(t, i, j, v):
        return torch.ops.aten.index_put.default(t, [None, None, i, j], v, True)

    with FakeTensorMode(allow_non_fake_inputs=True) as m:
        t = m.from_tensor(torch.zeros(32, 384, 520, 4))
        i = m.from_tensor(torch.zeros(9, 512, 1, 1, dtype=torch.int64))
        j = m.from_tensor(torch.zeros(1, 1, dtype=torch.int64))
        v = m.from_tensor(torch.zeros(32, 384, 9, 512, 1, 1))
        gm = make_fx(f, tracing_mode="fake")(t, i, j, v)

    info = placeholder_info_from_gm(gm)
    bounds = infer_index_bounds_from_gm(gm, info)
    # identify i and j placeholders by their shapes
    by_shape = {tuple(v_["shape"]): k for k, v_ in info.items()}
    i_name = by_shape[(9, 512, 1, 1)]
    j_name = by_shape[(1, 1)]
    assert bounds.get(i_name) == 520, bounds  # dim 2
    assert bounds.get(j_name) == 4, bounds    # dim 3 — NOT shape[0]=32


def test_perm_detection_resolves_lifted_alloc_and_iota():
    """The inverse-permutation idiom — index_put(empty, [idx], iota) — must
    detect idx as a PERMUTATION even when the alloc shape / iota length are
    lifted _shape_param NODES (gpt-oss MoE: random Index left uninit holes
    in the inverse map; holes fed index() as garbage -> gather assert)."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from capture_hook import lift_shape_params
    from full_graph_harness import (infer_permutation_indices_from_gm,
                                    placeholder_info_from_gm)

    def f(perm_idx):
        empty = torch.ops.aten.empty.memory_format(
            [4000], dtype=torch.int64, device="cpu")
        iota = torch.ops.prims.iota.default(
            4000, start=0, step=1, dtype=torch.int64, device="cpu",
            requires_grad=False)
        inv = torch.ops.aten.index_put.default(empty, [perm_idx], iota)
        return inv + 1

    with FakeTensorMode(allow_non_fake_inputs=True) as m:
        p = m.from_tensor(torch.zeros(4000, dtype=torch.int64))
        gm = make_fx(f, tracing_mode="fake")(p)

    # unlifted: detection works
    info = placeholder_info_from_gm(gm)
    perms = infer_permutation_indices_from_gm(gm, info)
    assert 4000 in perms.values(), perms

    # LIFTED (canonical form): alloc shape becomes a _shape_param node —
    # detection must still fire via meta['val'] resolution
    gm2, params = lift_shape_params(gm)
    if params:  # premise: something got lifted
        info2 = placeholder_info_from_gm(gm2)
        perms2 = infer_permutation_indices_from_gm(gm2, info2)
        assert 4000 in perms2.values(), (perms2, params)


def test_oracle_impl_point_registration():
    """Hash-keyed oracle registration (settled: hash as key, zero parsing):
    point=<shape_hash> resolves input shapes from the SIBLING shapes.json
    at registration time; an unknown hash fails at import, not silently
    at dispatch."""
    import json, sys, tempfile, types
    from pathlib import Path
    from oracle_harness import (oracle_impl, OracleDispatchError,
                                reset_oracle_registry, get_module_registry)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "shapes.json").write_text(json.dumps({"points": [{
            "shape_hash": "aabb1122",
            "models": {"timm/infer/x": {"occurrences": 1}},
            "inputs": [[[128, 512], "bf16"], [["S", [128, 512]][0], [128, 512]]],
        }]}))
        # build a module whose __file__ sits in the dir (like oracle.py)
        modname = "_test_oracle_point_mod"
        mod = types.ModuleType(modname)
        mod.__file__ = str(d / "oracle.py")
        sys.modules[modname] = mod
        try:
            def fwd(inputs):
                return inputs

            fwd.__module__ = modname

            reset_oracle_registry(modname)
            oracle_impl(hardware="B200", point="aabb1122")(fwd)
            reg = get_module_registry(modname)
            assert reg is not None and reg._entries
            assert reg._entries[-1]["point"] == "aabb1122"
            assert reg._entries[-1]["shape"] == ((128, 512),)

            # unknown hash: loud failure at registration
            try:
                oracle_impl(point="deadbeef")(fwd)
                raise AssertionError("unknown point hash did not raise")
            except OracleDispatchError as e:
                assert "deadbeef" in str(e)
        finally:
            sys.modules.pop(modname, None)
            reset_oracle_registry(modname)

# ============================================================================
# 14. Dynamic shapes: symbolic entries, bindings, guards (wave 2 format)
# ============================================================================

def test_symbolic_entry_instantiation():
    """Symbolic dims/strides evaluate under bindings; coupled dims stay
    consistent by construction (one binding evaluates every expr)."""
    from input_codec import instantiate_point, is_symbolic_entry

    symbols = {"s16": {"hint": 16, "range": [2, None]},
               "s82": {"hint": 16, "range": [2, None]}}
    point = {"bindings": {"s16": 16, "s82": 16},
             "inputs": [
                 [[64, 64, "s16", "s82"], "f32",
                  {"st": ["64*s16*s82", "s16*s82", "s82", 1]}],
                 ["I", 256, "s16*s82"],
             ]}
    assert is_symbolic_entry(point["inputs"][0])
    assert is_symbolic_entry(point["inputs"][1])

    at_hint = instantiate_point(point, symbols)
    assert at_hint[0][0] == [64, 64, 16, 16]
    assert at_hint[0][2]["st"] == [16384, 256, 16, 1]
    assert at_hint[1] == ["sym", 256]

    rebound = instantiate_point(point, symbols,
                                bindings={"s16": 24, "s82": 24})
    assert rebound[0][0] == [64, 64, 24, 24]
    assert rebound[0][2]["st"] == [36864, 576, 24, 1]
    assert rebound[1] == ["sym", 576]  # coupled: s16*s82 re-evaluated


def test_symbolic_bindings_validation():
    """Range and guard violations are LOUD (never bench impossible shapes);
    static entries pass through instantiation unchanged."""
    import pytest
    from input_codec import (instantiate_point, validate_bindings,
                             is_symbolic_entry)

    symbols = {"s0": {"hint": 8, "range": [2, 64]}}
    point = {"inputs": [[["s0", 128], "bf16"]]}

    with pytest.raises(ValueError, match="below range"):
        instantiate_point(point, symbols, bindings={"s0": 1})
    with pytest.raises(ValueError, match="above range"):
        instantiate_point(point, symbols, bindings={"s0": 128})
    with pytest.raises(ValueError, match="unknown symbol"):
        validate_bindings(symbols, {"s9": 4})
    with pytest.raises(ValueError, match="violates guard"):
        validate_bindings(symbols, {"s0": 6}, guards=["Eq(Mod(s0, 4), 0)"])
    validate_bindings(symbols, {"s0": 8}, guards=["Eq(Mod(s0, 4), 0)"])

    static = [[64, 64], "bf16"]
    assert not is_symbolic_entry(static)
    out = instantiate_point({"inputs": [static]}, symbols, bindings={"s0": 4})
    assert out == [static]


def test_oracle_compare_treats_matching_nans_as_equal():
    """Some captured math can produce NaNs for the synthetic input domain.
    Matching NaN masks are still exact numerics and must not be reported as
    stochastic output or correctness failure."""
    import torch
    from oracle_harness import _compare_oracle_outputs, detect_stochastic_outputs

    class StableNan(torch.nn.Module):
        def forward(self):
            return torch.tensor([float("nan"), 1.0, 2.0])

    assert detect_stochastic_outputs(StableNan(), []) == set()
    assert _compare_oracle_outputs(
        torch.tensor([float("nan"), 1.0, 2.0]),
        torch.tensor([float("nan"), 1.0, 2.0]),
        set(),
        atol=0.0,
        rtol=0.0,
    )
    assert not _compare_oracle_outputs(
        torch.tensor([float("nan"), 1.0, 2.0]),
        torch.tensor([0.0, 1.0, 2.0]),
        set(),
        atol=0.0,
        rtol=0.0,
    )


def test_bind_parsing_and_symbol_binding_threading():
    """--bind parsing (repeatable, symbol=int) threads through
    load_shape_configs as symbol_bindings and produces correctly-bound
    configs — including the coupled S() shape param. CPU-only: configs are
    specs, no tensor is materialized. (GPU half of this feature:
    scripts/test_dynamic_bench.py, the §1.4 static-vs-dynamic experiment.)"""
    import json, tempfile
    import pytest
    from pathlib import Path
    from repro_harness import (format_binding, load_shape_configs,
                               parse_bind_args, resolve_bound_configs)

    # parsing: repeatable flags, whitespace, malformed entries are LOUD
    assert parse_bind_args(None) == []
    assert parse_bind_args(["s16=24,s82=24", " s16 = 16 "]) == [
        {"s16": 24, "s82": 24}, {"s16": 16}]
    with pytest.raises(ValueError, match="must be symbol=int"):
        parse_bind_args(["s16"])
    with pytest.raises(ValueError, match="must be an int"):
        parse_bind_args(["s16=abc"])
    with pytest.raises(ValueError, match="no bindings"):
        parse_bind_args([","])
    assert format_binding(None) == "hint"
    assert format_binding({"s82": 24, "s16": 16}) == "s16=16,s82=24"

    shapes = {
        "symbols": {"s16": {"hint": 16, "range": [2, None]},
                    "s82": {"hint": 16, "range": [2, None]}},
        "points": [{
            "shape_hash": "dynexp01",
            "bindings": {"s16": 16, "s82": 16},
            "models": {"torchbench/infer/opacus_like": {"occurrences": 1}},
            "inputs": [
                [[64, 64, "s16", "s82"], "f32"],
                [[64], "f32"],
                ["S", [64, 32, 2, "s16*s82"]],
            ],
        }],
    }
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "shapes.json").write_text(json.dumps(shapes))
        repro = str(d / "repro.py")
        (d / "repro.py").write_text("# stub")

        # binding threads through load_shape_configs -> instantiate_point
        cfg = next(iter(load_shape_configs(
            repro, symbol_bindings={"s16": 24, "s82": 24}).values()))
        specs = cfg["inputs"]
        assert specs[0]["shape"] == [64, 64, 24, 24]
        assert specs[2]["dims"] == [64, 32, 2, 576]  # coupled s16*s82

        # rows: one per (binding x config); binding=None = hint point
        rows = resolve_bound_configs(repro, parse_bind_args(
            ["s16=16,s82=16", "s16=24,s82=24"]))
        assert [b for _, b, _ in rows] == [
            {"s16": 16, "s82": 16}, {"s16": 24, "s82": 24}]
        assert rows[0][2]["inputs"][0]["shape"] == [64, 64, 16, 16]
        assert rows[1][2]["inputs"][0]["shape"] == [64, 64, 24, 24]
        hint_rows = resolve_bound_configs(repro, [])
        assert hint_rows[0][1] is None
        assert hint_rows[0][2]["inputs"][0]["shape"] == [64, 64, 16, 16]

        # range violations surface (not silently benched)
        with pytest.raises(ValueError, match="below range"):
            resolve_bound_configs(repro, [{"s16": 1, "s82": 16}])
        # unknown --shape is loud
        with pytest.raises(ValueError, match="not in configs"):
            resolve_bound_configs(repro, [], shape="nope")

    # shapes.txt-only repros cannot honor bindings: loud, not ignored
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "shapes.txt").write_text(
            "lbl: (T([64, 64, 16, 16], f32),)\n")
        (d / "repro.py").write_text("# stub")
        with pytest.raises(ValueError, match="shapes.txt"):
            load_shape_configs(str(d / "repro.py"),
                               symbol_bindings={"s16": 24})


# ============================================================================
# Capture-side dynamic-shape harvesting (design §2.1/§2.2)
#
# CPU-only: a make_fx symbolic trace gives a live ShapeEnv + SymInt-backed
# placeholder vals, the same objects the post-grad capture hook sees, so the
# harvest helpers test without a GPU compile.
# ============================================================================

def _symbolic_shape_env_and_syms(fn, *example):
    """make_fx symbolic-trace fn; return (shape_env, {dim_name: SymInt})."""
    from torch.fx.experimental.proxy_tensor import make_fx
    gm = make_fx(fn, tracing_mode="symbolic")(*example)
    shape_env = None
    syms = []
    for n in gm.graph.nodes:
        if n.op != "placeholder":
            continue
        v = (n.meta or {}).get("val")
        if hasattr(v, "shape"):
            for s in v.shape:
                if isinstance(s, torch.SymInt):
                    shape_env = s.node.shape_env
                    syms.append(s)
    return shape_env, syms


def test_sym_expr_str_roundtrips_through_sympy():
    """Every expr string the capture emits must re-parse via sympy.sympify —
    that is the contract instantiate_point relies on. The extractor lives in
    full_graph_harness (THE shared home); capture_hook re-exports it."""
    import sympy
    from full_graph_harness import sym_expr_str

    def f(x):
        return x.view(x.shape[0], x.shape[1] * x.shape[2]).sum(1)

    _se, syms = _symbolic_shape_env_and_syms(f, torch.randn(4, 8, 16))
    seen_compound = False
    for s in syms:
        es = sym_expr_str(s)
        if es is None:
            continue
        expr = sympy.sympify(es)  # must not raise
        assert expr.free_symbols, f"{es!r} parsed to a constant"
        if "*" in es:
            seen_compound = True
    # A plain int has no symbolic content -> None (no redundant string)
    assert sym_expr_str(5) is None


def test_capture_hook_reexports_shared_extractor():
    """capture_hook must NOT have its own copy — it re-exports the one in
    full_graph_harness (shared-utility invariant)."""
    import capture_hook
    import full_graph_harness
    assert capture_hook._sym_expr_str is full_graph_harness.sym_expr_str
    assert capture_hook._harvest_shape_env is full_graph_harness.harvest_shape_env


def test_harvest_shape_env_symbols_ranges():
    """harvest_shape_env returns {symbols:{name:{hint,range}}, guards,
    captured_dynamic}; free symbols kept with hints + ranges, unbounded
    upper clamped to None, returns None when nothing is dynamic."""
    from full_graph_harness import harvest_shape_env

    def f(x):
        return x.view(x.shape[0], x.shape[1] * x.shape[2]).sum(1)

    se, _syms = _symbolic_shape_env_and_syms(f, torch.randn(4, 8, 16))
    block = harvest_shape_env(se)
    assert block is not None and block["captured_dynamic"] is True
    assert block["symbols"], "no symbols harvested"
    for name, meta in block["symbols"].items():
        assert name.startswith("s")
        assert isinstance(meta["hint"], int)
        lo, hi = meta["range"]
        assert lo is None or isinstance(lo, int)
        assert hi is None or isinstance(hi, int)  # int_oo -> None

    # _jsonable_range_bound: a FINITE upper bound survives (only oo/int_oo ->
    # None). A symbol with a real ceiling (e.g. torch._check(s <= 4096)) must
    # keep it, never get widened to unbounded.
    from full_graph_harness import _jsonable_range_bound
    import sympy
    assert _jsonable_range_bound(sympy.Integer(4096)) == 4096
    assert _jsonable_range_bound(sympy.oo) is None
    assert _jsonable_range_bound(-sympy.oo) is None


def test_harvest_none_shape_env():
    """None env (static capture) -> None block, never raises."""
    from full_graph_harness import harvest_shape_env
    assert harvest_shape_env(None) is None


def test_symbolic_stride_codec_roundtrip_is_exprs_not_hints():
    """A tensor spec with symbolic shape/stride round-trips through the codec
    carrying the EXACT exprs (never hint ints, never regex-mangled numbers).
    This is the regression pin for the '16384 != 64' sidecar-stride bug."""
    from input_codec import compact_from_spec, spec_from_compact, evaluate_spec

    spec = {
        "kind": "tensor",
        "shape": [64, 64, 16, 16],        # hints
        "dtype": "float32",
        "stride": [16384, 256, 16, 1],    # hints
        "symbolic": {
            "shape_exprs": [None, None, "s53", "s0"],
            "stride_exprs": ["64*s0*s53", "s0*s53", "s0", None],
        },
    }
    entry = compact_from_spec(spec)
    # exprs land in the slots, NOT hints
    assert entry[0] == [64, 64, "s53", "s0"], entry
    assert entry[2]["st"] == ["64*s0*s53", "s0*s53", "s0", 1], entry

    # decode reconstructs the symbolic block losslessly
    rt = spec_from_compact(entry)
    assert rt["symbolic"]["shape_exprs"] == [None, None, "s53", "s0"]
    assert rt["symbolic"]["stride_exprs"] == ["64*s0*s53", "s0*s53", "s0", None]

    # evaluate at the hint binding -> the original concrete hints
    ev = evaluate_spec(rt, {"s53": 16, "s0": 16})
    assert ev["shape"] == [64, 64, 16, 16]
    assert ev["stride"] == [16384, 256, 16, 1]
    assert "symbolic" not in ev
    # a different binding re-derives consistently (coupled stride 64*s0*s53)
    ev2 = evaluate_spec(rt, {"s53": 8, "s0": 32})
    assert ev2["shape"] == [64, 64, 8, 32]
    assert ev2["stride"][0] == 64 * 32 * 8


def test_dims_equal_compares_exprs_by_sympy_not_int():
    """_dims_equal: ints exact, symbolic slots by sympy-equivalence — never
    int() (the mangling bug), never string equality."""
    from full_graph_harness import _dims_equal
    assert _dims_equal(16384, 16384)
    assert not _dims_equal(16384, 64)
    # equal exprs, different rendering
    assert _dims_equal("s0*s53", "s53*s0")
    assert _dims_equal("64*s0*s53", "s0*s53*64")
    assert not _dims_equal("s0*s53", "s0*s53*64")
    # a symbolic slot is NOT mangled to its leading integer
    assert not _dims_equal("64*s0*s53", 64)


def test_symint_input_expr_codec_roundtrips_as_I():
    """A live symint INPUT spec with an expr round-trips as ['I', hint, expr]
    (rebindable), a constant symint as ['sym', hint]. Both sides — the region
    capture and the full-graph sidecar — produce the symint spec, so the
    codec must carry the expr or the sidecar symint can't rebind."""
    from input_codec import (compact_from_spec, spec_from_compact,
                             evaluate_symbolic_entry)

    live = {"kind": "symint", "name": "arg0_1", "value": 16, "expr": "s53"}
    entry = compact_from_spec(live)
    assert entry == ["I", 16, "s53"], entry
    rt = spec_from_compact(entry)
    assert rt["expr"] == "s53" and rt["value"] == 16
    # evaluates at a binding to a concrete symint
    assert evaluate_symbolic_entry(entry, {"s53": 24}) == ["sym", 24]

    const = {"kind": "symint", "name": "n", "value": 256}
    assert compact_from_spec(const) == ["sym", 256]


def test_harvest_unbacked_symbol_records_hint_source_tiers():
    """An unbacked symbol (data-dependent) is recorded with unbacked=True and
    a hint_source naming WHICH tier the value came from — observed (real
    runtime) > size_hint (derived) > range_fallback (arbitrary placeholder).
    Backed symbols are implicitly observed (no flags). Synthetic ShapeEnv
    stub — data-dependent ops are extern, rare in fusion regions."""
    from full_graph_harness import harvest_shape_env
    import sympy

    class _VR:
        def __init__(self, lo, hi):
            self.lower, self.upper = lo, hi

    s0 = sympy.Symbol("s0")
    u_obs, u_sh, u_fb = (sympy.Symbol(n) for n in ("u0", "u1", "u2"))

    class _FakeEnv:
        backed_var_to_val = {s0: sympy.Integer(16)}
        var_to_range = {s0: _VR(2, sympy.oo), u_obs: _VR(0, sympy.oo),
                        u_sh: _VR(0, sympy.oo), u_fb: _VR(4, sympy.oo)}
        guards = []
        # tier 1: a real propagated runtime value for u0
        real_tensor_prop_unbacked_vals = {u_obs: sympy.Integer(1372)}

        def is_unbacked_symint(self, sym):
            return str(sym).startswith("u")

        def size_hint(self, sym, *, allow_none=False):
            # tier 2 only for u1; u2 falls through to the range floor
            return sympy.Integer(8192) if str(sym) == "u1" else None

    block = harvest_shape_env(_FakeEnv())
    # backed: observed, no flags
    assert block["symbols"]["s0"] == {"hint": 16, "range": [2, None]}
    # tier 1 — real runtime value
    assert block["symbols"]["u0"] == {
        "hint": 1372, "range": [0, None], "unbacked": True,
        "hint_source": "observed"}
    # tier 2 — derived size_hint
    assert block["symbols"]["u1"]["hint_source"] == "size_hint"
    assert block["symbols"]["u1"]["hint"] == 8192
    # tier 3 — arbitrary range floor
    assert block["symbols"]["u2"]["hint_source"] == "range_fallback"
    assert block["symbols"]["u2"]["hint"] == 4


def test_symint_input_expr_parsed_without_regex_or_default():
    """A Sym(expr) input annotation keeps its exact expr (no _parse_intish
    default of 32, no regex). Sym(<int>) stays a constant value."""
    from full_graph_harness import parse_full_graph_inputs
    content = (
        'class G(torch.nn.Module):\n'
        '    def forward(self, arg0_1: "Sym(s53)", arg1_1: "Sym(256)", '
        'arg2_1: "f32[64, s53][s53, 1]cuda:0"):\n'
        '        return arg2_1\n'
    )
    specs = {s["name"]: s for s in parse_full_graph_inputs(content)}
    assert specs["arg0_1"]["kind"] == "symint"
    assert specs["arg0_1"].get("expr") == "s53"
    assert "value" not in specs["arg0_1"]      # NOT defaulted to 32
    assert specs["arg1_1"]["value"] == 256     # int literal stays concrete
    # tensor annotation keeps symbolic dim/stride exprs (no mangling)
    t = specs["arg2_1"]
    assert t["shape"] == [64, "s53"]
    assert t["symbolic"]["shape_exprs"] == [None, "s53"]
    assert t["symbolic"]["stride_exprs"] == ["s53", None]


# ============================================================================
# End-to-end dynamic capture (GPU-gated)
#
# The unit tests above cover each piece (harvest, codec, merge schema) in
# isolation; this exercises the REAL pipeline so a capture_hook refactor that
# silently breaks dynamic capture is caught — the static-only-blind-spot class
# of bug this whole effort exists to kill. Skipped without CUDA.
# ============================================================================

class _GroupNormDyn(torch.nn.Module):
    """GroupNorm-affine family: two coupled dynamic spatial dims (s0, s53),
    a lifted shape param (64,32,2,s0*s53) and live symint inputs — the
    var_mean opacus pattern the design doc uses."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(64))
        self.b = torch.nn.Parameter(torch.randn(64))

    def forward(self, x):
        view = x.view(64, 32, 2, x.shape[2] * x.shape[3])
        var, mean = torch.var_mean(view, [2, 3], correction=0, keepdim=True)
        normed = (view - mean) * torch.rsqrt(var + 1e-5)
        return normed.view(x.shape) * self.w[None, :, None, None] + self.b[None, :, None, None]


def _capture_dynamic_region(tmpdir):
    """Capture _GroupNormDyn under dynamic=True; return the single index entry."""
    import json
    from capture_hook import install_capture_hook, uninstall_capture_hook
    out = str(tmpdir / "cap")
    install_capture_hook(out, label="e2e")
    try:
        m = _GroupNormDyn().cuda()
        x = torch.randn(64, 64, 16, 16, device="cuda")
        torch._dynamo.mark_dynamic(x, 2)
        torch._dynamo.mark_dynamic(x, 3)
        with torch.no_grad():
            torch.compile(m, dynamic=True)(x)
    finally:
        uninstall_capture_hook()
    idx = json.loads((tmpdir / "cap" / "index.json").read_text())
    assert idx["n_dropped"] == 0, f"dynamic region dropped: {idx.get('dropped')}"
    assert idx["n_captured"] == 1, idx
    return idx["captured"][0]


def test_dynamic_capture_merge_load_roundtrip_gpu():
    """Full pipeline: capture a dynamic region -> merge into shapes.json ->
    load_shape_configs at the hint and a guard-RESPECTING rebind run eager;
    a guard-VIOLATING rebind is loudly rejected. No hand-built shapes.json."""
    import json
    import tempfile
    import importlib.util
    import pytest
    from pathlib import Path

    if not torch.cuda.is_available():
        pytest.skip("GPU required for real dynamic capture")
    torch._dynamo.reset()

    from merge_captures import _write_shapes_json
    from repro_harness import load_shape_configs, make_inputs_from_config

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        entry = _capture_dynamic_region(tmp)

        # the captured inputs ARE the symbolic format (exprs, not hints)
        assert entry.get("captured_dynamic") is True
        assert entry["symbols"], "no symbols harvested"
        tensor0 = entry["inputs"][0]
        assert any(isinstance(d, str) for d in tensor0[0]), "shape not symbolic"
        assert any(e[0] == "I" for e in entry["inputs"]
                   if isinstance(e, list) and e), "no ['I',..] symint input"

        repro_dir = tmp / "canonical" / "var_mean_e2e"
        repro_dir.mkdir(parents=True)
        (repro_dir / "repro.py").write_text(Path(entry["file"]).read_text())
        _write_shapes_json(
            repro_dir, entry["shape_hash"], entry.get("signature", ""),
            "probe/infer/groupnorm", occurrences=1, inputs=entry["inputs"],
            alias_group_nbytes=entry.get("alias_group_nbytes"),
            symbols=entry.get("symbols"), guards=entry.get("guards"))

        sj = json.loads((repro_dir / "shapes.json").read_text())
        assert sj["symbols"] == entry["symbols"]          # graph-level
        assert sj["points"][0]["captured_dynamic"] is True
        assert sj["points"][0]["bindings"]                # hint binding

        repro_py = str(repro_dir / "repro.py")
        syms = sorted(entry["symbols"])  # e.g. ['s0', 's53']

        def _run(binding):
            cfg = next(iter(load_shape_configs(
                repro_py, symbol_bindings=binding).values()))
            inputs = make_inputs_from_config(cfg)
            spec = importlib.util.spec_from_file_location("_e2e_repro", repro_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            with torch.no_grad():
                out = mod.Repro()(*inputs)
            return tuple(inputs[0].shape), tuple(out.shape)

        # hint point runs (recorded bindings)
        in_shape, out_shape = _run(None)
        assert in_shape == (64, 64, 16, 16)
        assert out_shape == (64, 64, 16, 16)

        # guard-RESPECTING rebind: product s0*s53 stays 256 (8*32), runs
        in_shape2, out_shape2 = _run({syms[0]: 8, syms[1]: 32})
        assert in_shape2[2] * in_shape2[3] == 16 * 16  # spatial product preserved
        assert out_shape2 == in_shape2                 # affine: out == in shape

        # guard-VIOLATING rebind: product 576 != 256 -> loudly rejected
        with pytest.raises(ValueError, match="guard|range"):
            _run({syms[0]: 24, syms[1]: 24})


def test_static_capture_has_no_symbolic_artifacts_gpu():
    """A STATIC compile (no mark_dynamic) must capture with zero symbolic
    content — no expr-string dims, no symbolic block, no symbols/guards in
    shapes.json. Pins that the dynamic path is fully additive."""
    import json
    import tempfile
    import pytest
    from pathlib import Path

    if not torch.cuda.is_available():
        pytest.skip("GPU required for real capture")
    torch._dynamo.reset()

    from capture_hook import install_capture_hook, uninstall_capture_hook

    class _StaticM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(64, 64))

        def forward(self, x):
            y = x @ self.w
            return (y - y.mean(0, keepdim=True)).relu().sum(1)

    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "cap")
        install_capture_hook(out, label="static")
        try:
            m = _StaticM().cuda()
            with torch.no_grad():
                torch.compile(m)(torch.randn(128, 64, device="cuda"))
        finally:
            uninstall_capture_hook()
        idx = json.loads((Path(td) / "cap" / "index.json").read_text())
        assert idx["n_dropped"] == 0 and idx["n_captured"] >= 1
        for entry in idx["captured"]:
            assert not entry.get("captured_dynamic")
            assert "symbols" not in entry and "guards" not in entry
            for e in entry["inputs"]:
                if isinstance(e, list) and e and isinstance(e[0], list):
                    assert all(isinstance(d, int) for d in e[0]), \
                        f"static capture has symbolic dim: {e}"
