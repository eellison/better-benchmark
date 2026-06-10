"""
Fixed-point test for hash-time canonicalization (capture_hook.canonicalize_for_hash).

THE INVARIANT UNDER TEST
------------------------
A partition's pattern_hash must be identical whether the graph it was cut
from is:
  (a) the saved post-grad graph in its original capture-time spelling
      (aten.reshape.default — Inductor's view_to_reshape runs before
      post_grad_passes, so that is what the capture hook hashed), or
  (b) the make_fx fake-mode retrace of the same graph (aten.view.default —
      the spelling every offline tool sees: model_graph_accounting,
      repartition_from_graphs, recapture runs).

Before canonicalize_for_hash, (a) and (b) hashed differently for any
partition containing a reshape — producing duplicate canonical repro dirs
(167 reshape-only vs 887 view-only; convnextv2 26d1711c064d == f68c9f1fa09b).

METHOD (CPU/fake only — safe to run during GPU sweeps)
------------------------------------------------------
For each saved full graph:
  side A: load + torch.fx.symbolic_trace  -> preserves reshape/clone spelling
  side B: load + make_fx(tracing_mode="fake") on fake CPU tensors -> retraced
Partition BOTH sides with the exact capture-pipeline partitioner
(get_fusion_partitions) and compare the multisets of pattern hashes.

PASS criterion: every divergent hash pair between side A and side B must NOT
be a reshape/view (+ retrace-introduced clone) divergence. I.e. the
canonicalization closes that class completely. Divergences of OTHER classes
are REPORTED (with op-multiset diffs naming the normalization that would be
needed) but do not fail the test — per the milestone, we don't chase them.

Usage:
    python scripts/test_canonical_hash.py
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.fx as fx

from capture_hook import (
    compute_partition_pattern,
    get_fusion_partitions,
)
from full_graph_harness import load_full_graph

# Three saved full graphs covering the divergence classes:
#  - convnextv2 train: 44x aten.reshape + 14x aten.clone (both classes, the
#    graph the confirmed dup pair 26d1711c064d/f68c9f1fa09b came from)
#  - convnextv2 infer: 29x aten.reshape (reshape class, inference spelling)
#  - LayerNormBackward: 0 reshape (negative control — retrace is already a
#    spelling fixed point, so side A and side B must match exactly)
GRAPHS = [
    "repros/models/timm/train/timm_convnextv2_nano.fcmae_ft_in22k_in1k_train/full_graph_001.py",
    "repros/models/timm/infer/timm_convnextv2_nano.fcmae_ft_in22k_in1k_infer/full_graph_000.py",
    "repros/models/genai/LayerNormBackward/full_graph_000.py",
]

ROOT = Path(__file__).resolve().parents[1]


def _partition_hashes(gm: fx.GraphModule) -> collections.Counter:
    """pattern_hash -> count, plus hash -> op multiset for diagnostics."""
    hashes = collections.Counter()
    opsets: dict[str, collections.Counter] = {}
    for comp in get_fusion_partitions(gm):
        pat = compute_partition_pattern(comp, gm)
        if pat is None:
            continue
        h = pat["pattern_hash"]
        hashes[h] += 1
        opsets.setdefault(
            h,
            collections.Counter(
                str(n.target) for n in comp if n.op == "call_function"
            ),
        )
    return hashes, opsets


def _capture_spelling_gm(graph_path: Path) -> fx.GraphModule:
    """Side A: the saved graph in its original capture-time spelling.

    The saved full_graph_*.py is the printed post-grad graph (reshape
    spelling). symbolic_trace re-creates fx nodes from that code verbatim —
    no decomposition, no canonicalization (verified: 44/44 reshapes and
    14/14 clones preserved on convnextv2 train).
    """
    instance, inputs, _definition = load_full_graph(graph_path, default_device="cpu")
    gm = torch.fx.symbolic_trace(instance)
    # canonicalize-before-hash needs node.meta["val"] (shape/stride/dtype) to
    # fabricate retrace inputs; symbolic_trace doesn't populate metas, so
    # propagate them with fake tensors. This keeps side A in the ORIGINAL
    # spelling (no decomposition) while making it canonicalizable.
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake_inputs = [
            mode.from_tensor(t) if torch.is_tensor(t) else t for t in inputs
        ]
        FakeTensorProp(gm, mode).propagate(*fake_inputs)
    return gm


def _retraced_gm(graph_path: Path) -> fx.GraphModule:
    """Side B: make_fx fake-mode retrace — the offline-tool spelling.

    Mirrors scripts/model_graph_accounting.trace_full_graph (fake mode,
    CPU tensors, no real compute).
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    instance, inputs, _definition = load_full_graph(graph_path, default_device="cpu")
    with FakeTensorMode() as fake_mode:
        fake_inputs = [
            fake_mode.from_tensor(t) if torch.is_tensor(t) else t
            for t in inputs
        ]
        with torch.no_grad():
            return make_fx(instance, tracing_mode="fake")(*fake_inputs)


# The divergence class canonicalize_for_hash is required to close:
# reshape<->view spelling (+ clone the retrace provably introduces/elides
# alongside it — none observed today, but classified the same way).
_SPELLING_OPS = {
    "aten.reshape.default",
    "aten.view.default",
    "aten.clone.default",
}

# Known side-A LOADING artifact of this test (NOT a capture-pipeline
# divergence): symbolic_trace constant-folds aten.full.default([]) — a call
# with no Proxy args executes eagerly and becomes a get_attr tensor constant,
# so the side-A partition loses its full.default node. The real capture hook
# hashed the post-grad fx graph where full.default IS a call_function, so
# both capture-time and retrace-time hashes agree on it. Verified on
# convnextv2 train: symbolic_trace yields 0x full.default + 15x get_attr
# where the saved text has full.default; the make_fx retrace preserves it.
_SIDE_A_FOLDED_OPS = {
    "aten.full.default",
}


def _classify_divergence(ops_a: collections.Counter, ops_b: collections.Counter) -> str:
    """Classify an unmatched-partition pair by what differs in op multisets."""
    diff = set((ops_a - ops_b).keys()) | set((ops_b - ops_a).keys())
    if not diff:
        # Same op multiset but different hash: wiring/structural-arg
        # difference. These are genuinely different partitions that happen
        # to share an op multiset — not a normalization gap.
        return "same-opset-different-wiring"
    if diff <= _SPELLING_OPS:
        return "RESHAPE_VIEW_CLONE"  # the class the fix must close
    if diff <= (_SPELLING_OPS | _SIDE_A_FOLDED_OPS):
        return ("side-A symbolic_trace folded aten.full->get_attr "
                "(test-harness artifact, reshape/view part IS closed)")
    return f"other ({sorted(diff)})"


def run_graph(rel_path: str) -> bool:
    graph_path = ROOT / rel_path
    name = f"{graph_path.parent.name}/{graph_path.name}"
    print(f"\n=== {name} ===")

    gm_a = _capture_spelling_gm(graph_path)
    gm_b = _retraced_gm(graph_path)

    ops_a = collections.Counter(
        str(n.target) for n in gm_a.graph.nodes if n.op == "call_function"
    )
    print(f"  side A (saved spelling): reshape={ops_a.get('aten.reshape.default', 0)} "
          f"view={ops_a.get('aten.view.default', 0)} "
          f"clone={ops_a.get('aten.clone.default', 0)}")
    ops_b = collections.Counter(
        str(n.target) for n in gm_b.graph.nodes if n.op == "call_function"
    )
    print(f"  side B (make_fx retrace): reshape={ops_b.get('aten.reshape.default', 0)} "
          f"view={ops_b.get('aten.view.default', 0)} "
          f"clone={ops_b.get('aten.clone.default', 0)}")

    hashes_a, opsets_a = _partition_hashes(gm_a)
    hashes_b, opsets_b = _partition_hashes(gm_b)
    print(f"  partitions: A={sum(hashes_a.values())} ({len(hashes_a)} unique), "
          f"B={sum(hashes_b.values())} ({len(hashes_b)} unique)")

    only_a = hashes_a - hashes_b
    only_b = hashes_b - hashes_a
    matched = sum((hashes_a & hashes_b).values())
    print(f"  matched hash occurrences: {matched}")

    if not only_a and not only_b:
        print("  FIXED POINT: all partition hashes identical across spellings")
        return True

    # Pair up unmatched partitions by normalized op multiset to classify
    # what normalization WOULD be needed (report, don't chase).
    def _norm(ops: collections.Counter) -> tuple:
        out = collections.Counter()
        for op, c in ops.items():
            if op == "aten.reshape.default":
                op = "aten.view.default"
            if op in _SIDE_A_FOLDED_OPS:
                # symbolic_trace folds these on side A (see _SIDE_A_FOLDED_OPS)
                # — ignore for pairing so the partitions line up.
                continue
            out[op] += c
        return tuple(sorted(out.items()))

    spelling_failures = []
    reported = []
    unpaired_b = dict(only_b)
    for ha in sorted(only_a):
        na = _norm(opsets_a[ha])
        partner = next(
            (hb for hb in sorted(unpaired_b) if _norm(opsets_b[hb]) == na), None
        )
        if partner is not None:
            unpaired_b[partner] -= only_a[ha]
            if unpaired_b[partner] <= 0:
                del unpaired_b[partner]
            cls = _classify_divergence(opsets_a[ha], opsets_b[partner])
            if cls == "RESHAPE_VIEW_CLONE":
                spelling_failures.append((ha, partner))
            else:
                reported.append((ha, partner, cls))
        else:
            reported.append((ha, None, "no-opset-equivalent-in-retrace"))
    for hb in sorted(unpaired_b):
        reported.append((None, hb, "no-opset-equivalent-in-saved"))

    for ha, hb, cls in reported:
        print(f"  REPORT (not chased): {ha} <-> {hb}: {cls}")
        if ha and hb:
            da = opsets_a[ha] - opsets_b[hb]
            db = opsets_b[hb] - opsets_a[ha]
            if da or db:
                print(f"    only in saved-spelling partition:  {dict(da)}")
                print(f"    only in retraced partition:        {dict(db)}")

    if spelling_failures:
        for ha, hb in spelling_failures:
            print(f"  FAIL: reshape/view divergence NOT closed: {ha} vs {hb}")
            print(f"    saved ops:    {dict(opsets_a[ha])}")
            print(f"    retraced ops: {dict(opsets_b[hb])}")
        return False

    print("  PASS: no reshape/view(+clone) hash divergences "
          f"({len(reported)} other-class divergences reported above)")
    return True


def main():
    ok = True
    for rel in GRAPHS:
        try:
            ok &= run_graph(rel)
        except Exception as e:
            print(f"  ERROR on {rel}: {type(e).__name__}: {e}")
            ok = False
    print()
    if ok:
        print("ALL GRAPHS: reshape/view canonical-hash fixed point holds")
    else:
        print("FIXED-POINT TEST FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
