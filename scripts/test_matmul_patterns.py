"""Alternate partitioning views: matmul epilogues & mm->pointwise->mm chains.

WHAT THIS DEMONSTRATES (without touching the default partitioner)
-----------------------------------------------------------------
The default capture partitioner (capture_hook.get_fusion_partitions) treats
every matmul (mm/addmm/bmm/_grouped_mm) as a NON-fusible extern op. So the
elementwise tail after a GEMM, and the elementwise bridge between two GEMMs,
always land in SEPARATE fusible partitions. scripts/matmul_pattern_analysis.py
re-reads that same partition structure to surface two reattach-able patterns:

  1. matmul + strictly-pointwise epilogue   (GEMM + bias/act/cast fusion target)
  2. matmul -> pointwise -> matmul           (MLP-style back-to-back GEMM target)

This file is both a regression test for the detector AND a worked example: each
test builds a tiny FX graph by hand, runs the REAL detector on it, and asserts
the hits. ``main()`` prints the detector output on each example so you can read
exactly what gets found.

Run (CPU only -- pure graph structure, no GPU / no timing):
    python scripts/test_matmul_patterns.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from scripts.matmul_pattern_analysis import analyze_graph


def _trace(fn, *example_inputs):
    """make_fx in fake mode -> GraphModule with shape meta.

    Applies Inductor's decomposition table so composite ops (gelu, relu,
    layer_norm, ...) lower to the same aten primitives the post-grad graphs in
    repros/models/ contain -- otherwise gelu stays a single non-fusible node
    and never forms a pointwise partition. This mirrors the real capture path
    (Inductor decomposes before the post_grad partitioner runs)."""
    from torch._inductor.decomposition import decompositions as inductor_decomps

    return make_fx(
        fn, decomposition_table=inductor_decomps, tracing_mode="fake"
    )(*example_inputs)


# ---------------------------------------------------------------------------
# Pattern 1: matmul + strictly-pointwise epilogue
# ---------------------------------------------------------------------------

def test_addmm_bias_gelu_epilogue_is_clean():
    """addmm (with its bias add) followed by a tanh-GELU tail = one clean,
    strictly-pointwise epilogue on a single matmul."""
    def f(x, w, b):
        y = torch.addmm(b, x, w)               # GEMM + bias
        return torch.nn.functional.gelu(y)     # pure pointwise tail

    gm = _trace(
        f,
        torch.randn(32, 64),
        torch.randn(64, 128),
        torch.randn(128),
    )
    res = analyze_graph(gm, "addmm_gelu")
    assert res["n_matmuls"] == 1, res["n_matmuls"]
    assert len(res["epilogues"]) == 1, res["epilogues"]
    e = res["epilogues"][0]
    assert e["matmul_op"] == "addmm", e
    assert e["clean"] is True, e          # the ONLY consumer is pointwise
    # gelu decomposes to erf/mul/add/... -- all pointwise. Assert no reduction
    # leaked in and that the activation core is present.
    assert all(op not in ("sum", "mean", "_softmax", "var") for op in e["epilogue_ops"]), e
    assert any(op in ("erf", "tanh", "mul") for op in e["epilogue_ops"]), e
    print("PASS addmm+gelu is a clean pointwise epilogue:", e["epilogue_ops"])


def test_mm_reduction_tail_is_not_pointwise_epilogue():
    """A reduction (softmax/sum) after the matmul must NOT be reported as a
    strictly-pointwise epilogue -- it's a different, harder fusion target."""
    def f(x, w):
        y = torch.mm(x, w)
        return torch.softmax(y, dim=-1)        # reduction -> disqualifies

    gm = _trace(f, torch.randn(16, 32), torch.randn(32, 48))
    res = analyze_graph(gm, "mm_softmax")
    assert res["n_matmuls"] == 1, res
    # The consuming partition contains a reduction, so it is not strictly
    # pointwise -> no epilogue hit (and certainly not "clean").
    assert all(not e["clean"] for e in res["epilogues"]), res["epilogues"]
    # Either zero epilogue hits, or a hit that is explicitly not clean.
    if res["epilogues"]:
        assert res["epilogues"][0]["clean"] is False
    print("PASS mm+softmax is NOT a clean pointwise epilogue")


def test_matmul_feeding_matmul_has_no_pointwise_epilogue():
    """mm whose output feeds straight into another mm (extern consumer) is not
    a pointwise epilogue for the first mm."""
    def f(a, b, c):
        return torch.mm(torch.mm(a, b), c)

    gm = _trace(f, torch.randn(8, 16), torch.randn(16, 24), torch.randn(24, 12))
    res = analyze_graph(gm, "mm_mm")
    # The first mm's only real consumer is another mm -> no pointwise epilogue.
    assert res["n_matmuls"] == 2, res
    first_mm_epis = [e for e in res["epilogues"]]
    assert len(first_mm_epis) == 0, first_mm_epis
    print("PASS mm->mm produces no pointwise epilogue")


# ---------------------------------------------------------------------------
# Pattern 2: matmul -> pointwise -> matmul
# ---------------------------------------------------------------------------

def test_mlp_mm_gelu_mm_chain():
    """Classic MLP block: linear -> activation -> linear, i.e.
    mm -> (strictly pointwise) -> mm. The detector must report exactly one
    chain with a non-empty pointwise bridge."""
    def f(x, w1, w2):
        h = torch.mm(x, w1)
        h = torch.nn.functional.relu(h)        # pointwise bridge
        return torch.mm(h, w2)

    gm = _trace(f, torch.randn(32, 64), torch.randn(64, 128), torch.randn(128, 32))
    res = analyze_graph(gm, "mlp_relu")
    pw_chains = [c for c in res["chains"] if c["n_bridge_pointwise"] >= 1]
    assert len(pw_chains) == 1, res["chains"]
    c = pw_chains[0]
    assert c["mm1_op"] == "mm" and c["mm2_op"] == "mm", c
    assert "relu" in c["bridge_pointwise_ops"], c
    print("PASS mm->relu->mm chain detected, bridge:", c["bridge_pointwise_ops"])


def test_mm_then_mm_no_pointwise_is_separate_bucket():
    """Back-to-back matmul with only a view between counts as a chain, but with
    an EMPTY pointwise bridge (the 'no pointwise between' bucket)."""
    def f(a, b, c):
        y = torch.mm(a, b)            # [8, 24]
        y = y.reshape(8, 24)          # transparent view only
        return torch.mm(y, c)

    gm = _trace(f, torch.randn(8, 16), torch.randn(16, 24), torch.randn(24, 12))
    res = analyze_graph(gm, "mm_view_mm")
    no_pw = [c for c in res["chains"] if c["n_bridge_pointwise"] == 0]
    pw = [c for c in res["chains"] if c["n_bridge_pointwise"] >= 1]
    assert len(no_pw) == 1 and len(pw) == 0, res["chains"]
    print("PASS mm->view->mm is a no-pointwise chain")


def test_reduction_bridge_breaks_the_chain():
    """A reduction (layernorm/sum) between two matmuls must NOT count as a
    pointwise bridge -- the chain is only reported as pointwise when the bridge
    partition is strictly elementwise."""
    def f(x, w1, w2):
        h = torch.mm(x, w1)                          # [32, 64]
        h = h - h.mean(dim=-1, keepdim=True)         # reduction in the bridge
        return torch.mm(h, w2)

    gm = _trace(f, torch.randn(32, 48), torch.randn(48, 64), torch.randn(64, 16))
    res = analyze_graph(gm, "mm_meanbridge_mm")
    pw = [c for c in res["chains"] if c["n_bridge_pointwise"] >= 1]
    assert len(pw) == 0, ("reduction bridge wrongly counted as pointwise", res["chains"])
    print("PASS reduction bridge is not a pointwise chain")


# ---------------------------------------------------------------------------
# Pattern 3: fan-in -- two matmuls reduce to one output
# ---------------------------------------------------------------------------

def test_two_matmuls_add_to_one_output():
    """mm(a,b) + mm(c,d): two INDEPENDENT matmuls combined by a pointwise add
    into a single output. The classic parallel-projection / residual fan-in."""
    def f(a, b, c, d):
        return torch.mm(a, b) + torch.mm(c, d)     # both -> [32, 48]

    gm = _trace(
        f,
        torch.randn(32, 64), torch.randn(64, 48),
        torch.randn(32, 16), torch.randn(16, 48),
    )
    res = analyze_graph(gm, "mm_add_mm")
    fan = [f for f in res["fanins"] if f["strictly_pointwise"]]
    assert len(fan) == 1, res["fanins"]
    h = fan[0]
    assert h["n_matmuls"] == 2, h
    assert h["combine_op"] == "add", h
    assert all(op == "mm" for op in h["matmul_ops"]), h
    # Both matmul outputs feed ONLY the add -> both eliminable from DRAM.
    assert h["all_exclusive"] is True, h
    assert h["matmul_exclusive"] == [True, True], h
    assert h["eliminable_read_bytes"] == 2 * (32 * 48 * 4), h  # 2x f32 [32,48]
    # NOT reported as a sequential chain (the matmuls are siblings, not stacked)
    assert all(c["n_bridge_pointwise"] == 0 for c in res["chains"]) or not res["chains"], \
        ("fan-in should not masquerade as a sequential chain", res["chains"])
    print("PASS mm + mm -> one output (fan-in), exclusive, "
          f"elim={h['eliminable_read_bytes']}B")


def test_gated_mlp_silu_mul_fanin():
    """Gated MLP: silu(mm(x,Wg)) * mm(x,Wu). Two matmuls converge through a
    SiLU+multiply bridge into one output -- the SwiGLU shape."""
    def f(x, wg, wu):
        gate = torch.nn.functional.silu(torch.mm(x, wg))
        up = torch.mm(x, wu)
        return gate * up

    gm = _trace(f, torch.randn(32, 64), torch.randn(64, 128), torch.randn(64, 128))
    res = analyze_graph(gm, "swiglu")
    fan = [f for f in res["fanins"] if f["strictly_pointwise"]]
    assert len(fan) == 1, res["fanins"]
    h = fan[0]
    assert h["n_matmuls"] == 2, h
    assert h["combine_op"] == "mul", h            # gate * up
    # silu decomposes to the sigmoid form (neg/exp/add/div) * x -- accept either
    # a literal sigmoid/silu node or its decomposed exp-based shape.
    assert (any(op in ("sigmoid", "silu") for op in h["bridge_ops"])
            or "exp" in h["bridge_ops"]), h
    assert h["all_exclusive"] is True, h    # gate & up feed only the silu*mul
    print("PASS gated MLP silu(mm)*mm fan-in detected, combine:", h["combine_op"])


def test_reused_matmul_output_loses_memory_win():
    """If a contributing matmul's output is ALSO consumed outside the bridge, it
    must be materialized to DRAM regardless -- so it is NOT exclusive and the
    fan-in's memory win is partial (all_exclusive=False)."""
    def f(a, b, c, d):
        m1 = torch.mm(a, b)              # [32, 48]
        m2 = torch.mm(c, d)              # [32, 48]
        combined = m1 + m2               # the fan-in
        # m1 is ALSO used elsewhere -> must be written to DRAM anyway.
        side = torch.relu(m1)
        return combined, side

    gm = _trace(
        f,
        torch.randn(32, 64), torch.randn(64, 48),
        torch.randn(32, 16), torch.randn(16, 48),
    )
    res = analyze_graph(gm, "mm_reused")
    fan = [f for f in res["fanins"] if f["strictly_pointwise"]]
    assert len(fan) == 1, res["fanins"]
    h = fan[0]
    assert h["n_matmuls"] == 2, h
    # exactly ONE of the two matmuls is exclusive (m2); m1 is reused.
    assert h["all_exclusive"] is False, h
    assert sum(1 for e in h["matmul_exclusive"] if e) == 1, h
    # only the exclusive matmul's bytes are eliminable (one f32 [32,48]).
    assert h["eliminable_read_bytes"] == 32 * 48 * 4, h
    print("PASS reused matmul output is not exclusive (partial mem win):",
          h["matmul_exclusive"])


def test_single_matmul_is_not_fanin():
    """One matmul + pointwise tail must NOT be reported as a fan-in (needs
    >=2 converging matmuls)."""
    def f(x, w):
        return torch.relu(torch.mm(x, w))

    gm = _trace(f, torch.randn(32, 64), torch.randn(64, 48))
    res = analyze_graph(gm, "single_mm")
    assert res["fanins"] == [], res["fanins"]
    print("PASS single matmul is not a fan-in")


# ---------------------------------------------------------------------------
# Worked example printout
# ---------------------------------------------------------------------------

def _demo():
    """Print the detector's output on a small MLP-with-bias-and-activation so a
    reader sees concretely what both patterns look like."""
    import json

    def gated_mlp(x, wg, wu, wd):
        # SwiGLU: two parallel projections (gate, up) FAN IN through silu*mul,
        # then a down projection. Shows all three patterns at once.
        gate = torch.nn.functional.silu(torch.mm(x, wg))   # GEMM_g + act
        up = torch.mm(x, wu)                               # GEMM_u
        h = gate * up                                      # <- fan-in (2 -> 1)
        return torch.mm(h, wd)                             # GEMM_d (chain tail)

    gm = _trace(
        gated_mlp,
        torch.randn(32, 64),
        torch.randn(64, 128), torch.randn(64, 128), torch.randn(128, 64),
    )
    res = analyze_graph(gm, "demo_swiglu")
    print("\n" + "=" * 70)
    print("WORKED EXAMPLE: gated MLP (SwiGLU)  silu(mm)*mm -> mm")
    print("=" * 70)
    print(f"matmuls={res['n_matmuls']}  fusible_partitions={res['n_partitions']}")
    print("\n-- matmuls with strictly-pointwise epilogue --")
    for e in res["epilogues"]:
        print(f"  {e['matmul_op']} {e['output_shape']} -> "
              f"epilogue [{'+'.join(e['epilogue_ops'])}]  clean={e['clean']}")
    print("\n-- mm -> pointwise -> mm chains (SEQUENTIAL) --")
    for c in res["chains"]:
        bridge = "+".join(c["bridge_pointwise_ops"]) or "(views only)"
        print(f"  {c['mm1_op']} {c['mm1_out_shape']} -> [{bridge}] -> "
              f"{c['mm2_op']} {c['mm2_out_shape']}")
    print("\n-- matmul FAN-IN (>=2 matmuls -> 1 output) --")
    for f in res["fanins"]:
        print(f"  {f['n_matmuls']}x [{'+'.join(f['matmul_ops'])}] "
              f"{f['matmul_shapes']} --{f['combine_op']}--> "
              f"[{'+'.join(f['bridge_ops'])}] -> {f['output_shape']}  "
              f"strictly_pw={f['strictly_pointwise']}  "
              f"all_exclusive={f['all_exclusive']}  "
              f"elim_read={f['eliminable_read_bytes']}B")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    tests = [
        test_addmm_bias_gelu_epilogue_is_clean,
        test_mm_reduction_tail_is_not_pointwise_epilogue,
        test_matmul_feeding_matmul_has_no_pointwise_epilogue,
        test_mlp_mm_gelu_mm_chain,
        test_mm_then_mm_no_pointwise_is_separate_bucket,
        test_reduction_bridge_breaks_the_chain,
        test_two_matmuls_add_to_one_output,
        test_gated_mlp_silu_mul_fanin,
        test_reused_matmul_output_loses_memory_win,
        test_single_matmul_is_not_fanin,
    ]
    for t in tests:
        t()
    _demo()
    print(f"All {len(tests)} matmul-pattern tests passed.")
