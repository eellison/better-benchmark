#!/usr/bin/env python3
"""Alternate kernel-partitioning views: matmul epilogues and mm->pw->mm chains.

This is an ANALYSIS layer on top of the existing capture-pipeline partitioner.
It does NOT change the default partitioner: it calls the very same
``get_fusion_partitions`` (CapabilityBasedPartitioner + is_fusible_node) that
``capture_hook`` / ``model_graph_accounting`` use, then re-reads the resulting
partition structure to surface two patterns that the default cut deliberately
keeps separate (matmuls are non-fusible "extern" ops, so they never land in a
fusible partition with their pointwise neighbors):

  1. MATMUL + STRICTLY-POINTWISE EPILOGUE
     A matmul (mm/addmm/bmm/_grouped_mm/...) whose output flows -- through
     transparent view ops only -- into a fusible partition whose every compute
     node is pointwise. These are the clean "GEMM + epilogue fusion" targets:
     the elementwise tail (bias/activation/cast/residual) could be folded into
     the GEMM kernel's epilogue.

  2. MATMUL -> POINTWISE -> MATMUL
     A matmul M1 whose output feeds a strictly-pointwise partition P whose
     output feeds another matmul M2. The classic case is an MLP block
     (linear -> activation -> linear) or any back-to-back GEMM with an
     elementwise bridge -- candidates for a fused GEMM-epilogue-GEMM /
     persistent kernel. We also record the degenerate "no pointwise between"
     case (M1 -> view -> M2) separately.

"Strictly pointwise" = the partition has >=1 compute node and EVERY compute
(non view/metadata) node is pointwise. Reductions (sum/mean/softmax/layernorm)
disqualify it -- those are a different, harder fusion target.

Usage:
    # One model
    python scripts/matmul_pattern_analysis.py --model BertForMaskedLM

    # Whole corpus -> JSON + markdown summary
    python scripts/matmul_pattern_analysis.py --all \
        --output-dir investigation_results/matmul_patterns

    # Quick smoke on N models
    python scripts/matmul_pattern_analysis.py --all --max-models 5
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = ROOT / "repros" / "canonical"
sys.path.insert(0, str(ROOT))

import torch

from capture_hook import (
    TRANSPARENT_OPS,
    compute_partition_pattern,
    get_fusion_partitions,
)

# ---------------------------------------------------------------------------
# Op classification
# ---------------------------------------------------------------------------

# Every matmul-family extern op. These are non-fusible (flop-counter / BLAS)
# ops, so the default partitioner never folds them into a fusible partition --
# which is exactly why their pointwise neighbors sit in a *separate* partition
# that we can re-attach here. Spellings confirmed present in the corpus:
# mm/addmm/bmm/_grouped_mm; the rest are included defensively.
MATMUL_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.baddbmm.default,
    torch.ops.aten._grouped_mm.default,
    torch.ops.aten.mv.default,
    torch.ops.aten.dot.default,
}
# _scaled_mm may not exist on every build; add if present.
for _opt in ("_scaled_mm",):
    _ol = getattr(torch.ops.aten, _opt, None)
    if _ol is not None:
        for _o in _ol.overloads():
            MATMUL_OPS.add(getattr(_ol, _o))

# Pointwise ops that the torch.Tag.pointwise tag MISSES. The tag is reliable
# for activations and the tensor overloads of arithmetic (add.Tensor,
# mul.Tensor, where.self, prims.convert_element_type), but aten._to_copy (the
# dtype cast that bf16 GEMM epilogues lean on) is untagged, so we add it.
POINTWISE_EXTRA = {
    torch.ops.aten._to_copy.default,
}


def is_matmul(node) -> bool:
    return node.op == "call_function" and node.target in MATMUL_OPS


def is_transparent(node) -> bool:
    return node.op == "call_function" and node.target in TRANSPARENT_OPS


def is_pointwise(node) -> bool:
    """True if the node is an elementwise op (tag-based + the cast escapes)."""
    if node.op != "call_function":
        return False
    if node.target in POINTWISE_EXTRA:
        return True
    if isinstance(node.target, torch._ops.OpOverload):
        return torch.Tag.pointwise in node.target.tags
    return False


def short_op(target) -> str:
    """gelu.default -> 'gelu'; prims.convert_element_type.default -> 'convert_element_type'."""
    s = str(target)
    parts = s.split(".")
    # drop trailing overload token (default / Tensor / int / ...) if present
    if len(parts) >= 2:
        return parts[-2]
    return s


def compute_nodes(comp) -> list:
    """The real compute nodes of a partition (call_function, non-transparent)."""
    return [
        n for n in comp
        if n.op == "call_function" and n.target not in TRANSPARENT_OPS
    ]


def partition_is_strictly_pointwise(comp) -> bool:
    cn = compute_nodes(comp)
    return bool(cn) and all(is_pointwise(n) for n in cn)


def _safe_shape(val):
    if val is None:
        return None
    if torch.is_tensor(val):
        out = []
        for d in val.shape:
            try:
                out.append(int(d.node.hint) if hasattr(d, "node") and hasattr(d.node, "hint") else int(d))
            except (TypeError, ValueError):
                out.append(-1)
        return out
    if isinstance(val, (list, tuple)):
        for item in val:
            s = _safe_shape(item)
            if s is not None:
                return s
    return None


def _node_nbytes(node) -> int | None:
    """Size in bytes of a node's output tensor (numel * dtype itemsize)."""
    val = node.meta.get("val")
    if not torch.is_tensor(val):
        return None
    shape = _safe_shape(val)
    if shape is None or any(d < 0 for d in shape):
        return None
    numel = 1
    for d in shape:
        numel *= d
    try:
        return numel * val.dtype.itemsize
    except (AttributeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Graph traversal helpers (transparent-op aware)
# ---------------------------------------------------------------------------

def forward_real_targets(node) -> tuple[list, bool]:
    """Follow ``node``'s users through transparent view ops only.

    Returns (real_target_nodes, feeds_graph_output): the first non-transparent
    consumer reached along each path, plus whether any path reaches the graph
    output. A matmul output that goes mm -> view -> reshape -> gelu yields the
    gelu node (the view/reshape are skipped).
    """
    targets = []
    feeds_output = False
    seen = set()
    stack = list(node.users)
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        if u.op == "output":
            feeds_output = True
            continue
        if is_transparent(u):
            stack.extend(u.users)
        else:
            targets.append(u)
    return targets, feeds_output


def partition_escaping_nodes(comp, comp_set) -> list:
    """Nodes in a partition with at least one user outside the partition
    (including the graph output) -- i.e. the partition's outputs."""
    out = []
    for n in comp:
        for u in n.users:
            if u not in comp_set or u.op == "output":
                out.append(n)
                break
    return out


def _input_nodes(n) -> list:
    """The fx.Node arguments of n (args + kwargs), in no particular order."""
    out = []
    torch.fx.map_arg(
        (n.args, n.kwargs),
        lambda x: out.append(x) if isinstance(x, torch.fx.Node) else None,
    )
    return out


def matmuls_feeding_output(out_node, comp_set) -> list:
    """Matmuls that converge INTO ``out_node`` (a partition output).

    Walks backward from out_node through the partition's own compute nodes and
    transparent view ops, collecting every matmul reached. This is the fan-in
    side: if >=2 distinct matmuls are returned, their results are combined --
    through the pointwise partition -- into this single output. (mm -> pw -> mm
    is the SEQUENTIAL chain handled elsewhere; here the two matmuls are
    independent siblings that MERGE.)
    """
    found = []
    seen = set()
    stack = [out_node]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for inp in _input_nodes(n):
            if is_matmul(inp):
                if inp not in found:
                    found.append(inp)
                # do NOT recurse past a matmul -- its own inputs are upstream.
            elif is_transparent(inp) or inp in comp_set:
                # stay inside the bridge partition / view chain feeding out_node
                stack.append(inp)
    return found


def matmul_consumed_only_by(mm_node, comp_set) -> bool:
    """True if mm_node's output is consumed ONLY inside comp_set (the bridge),
    reaching it through transparent views and never escaping.

    This is the memory-elimination precondition: if the matmul output also feeds
    a node outside the bridge OR the graph output, it must be written to DRAM
    regardless, so fusing the combine into the GEMM epilogue saves no read. We
    walk FORWARD from the matmul through transparent ops; every terminal
    (non-transparent) consumer must be a member of comp_set, and no path may hit
    the graph output.
    """
    seen = set()
    stack = list(mm_node.users)
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        if u.op == "output":
            return False                       # escapes as a graph output
        if is_transparent(u):
            # A view is not itself a materializing consumer; what matters is
            # where its data finally lands. Recurse through to the real users.
            stack.extend(u.users)
        elif u not in comp_set:
            return False                       # a real consumer other than the bridge
    return True


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

@dataclass
class PartitionInfo:
    comp_index: int
    strictly_pointwise: bool
    n_compute: int
    compute_ops: list           # short op names, in graph order
    pattern_hash: str | None
    in_corpus: bool             # pattern_hash maps to a repros/canonical dir


@dataclass
class EpilogueHit:
    graph_name: str
    matmul: str                 # node name
    matmul_op: str              # short op name
    output_shape: list | None
    clean: bool                 # ALL compute-consumers are strictly-pw partitions
    feeds_graph_output: bool
    epilogue_ops: list          # union of compute ops across the pw epilogue partition(s)
    epilogue_partitions: list   # [PartitionInfo-as-dict, ...] (strictly-pw only)
    n_consumers: int            # distinct real consumer partitions / extern ops


@dataclass
class ChainHit:
    graph_name: str
    mm1: str
    mm1_op: str
    mm2: str
    mm2_op: str
    bridge_pointwise_ops: list  # compute ops of the bridge partition ([] if none)
    n_bridge_pointwise: int
    bridge_pattern_hash: str | None
    bridge_in_corpus: bool
    mm1_out_shape: list | None
    mm2_out_shape: list | None


@dataclass
class FanInHit:
    """Two-or-more matmuls that CONVERGE into one output via pointwise ops.

    The combining ('reduce to one output') pattern: independent sibling
    matmuls (mm(a,b) + mm(c,d); or a gated MLP silu(mm(x,Wg)) * mm(x,Wu))
    merged through a strictly-pointwise bridge partition into a SINGLE result.
    Distinct from ChainHit, where one matmul feeds the next in sequence.

    The MEMORY win lives in `all_exclusive`: a matmul output that ALSO feeds
    something outside this bridge (another consumer, or a graph output) must be
    materialized to DRAM regardless, so fusing the combine into its epilogue
    saves nothing. Only when EVERY contributing matmul's sole consumer is this
    bridge does the intermediate never touch global memory -- that's the
    eliminable read. `eliminable_read_bytes` sums those intermediates.
    """
    graph_name: str
    matmuls: list               # node names of the converging matmuls
    matmul_ops: list            # their short op names
    n_matmuls: int
    matmul_shapes: list         # per-matmul output shape
    combine_op: str             # the pointwise op that joins them (add/mul/...) if identifiable
    strictly_pointwise: bool    # the bridge partition is strictly pointwise
    bridge_ops: list            # compute ops of the bridge partition
    output_node: str
    output_shape: list | None
    bridge_pattern_hash: str | None
    bridge_in_corpus: bool
    matmul_exclusive: list      # per-matmul: True if its ONLY consumer is this bridge
    all_exclusive: bool         # every contributing matmul is exclusive (full mem win)
    eliminable_read_bytes: int | None  # sum bytes of exclusive matmul outputs (never hit DRAM)


def _corpus_pattern_hashes() -> set[str]:
    """Pattern hashes present in repros/canonical (dir name suffix after '_')."""
    hashes = set()
    if CANONICAL_DIR.exists():
        for d in CANONICAL_DIR.iterdir():
            if d.is_dir() and "_" in d.name:
                hashes.add(d.name.rsplit("_", 1)[-1])
    return hashes


_CORPUS_HASHES: set[str] | None = None


def analyze_graph(gm, graph_name: str) -> dict:
    """Find matmul epilogues and mm->pw->mm chains in ONE traced full graph.

    Reuses get_fusion_partitions (the DEFAULT partitioner) verbatim.
    """
    global _CORPUS_HASHES
    if _CORPUS_HASHES is None:
        _CORPUS_HASHES = _corpus_pattern_hashes()

    comps = get_fusion_partitions(gm)
    comp_sets = [set(c) for c in comps]
    node2comp = {}
    for i, c in enumerate(comps):
        for n in c:
            node2comp[n] = i

    # Cache cheap per-partition info. The pattern_hash is DEFERRED: it requires
    # a make_fx retrace per partition (compute_partition_pattern) which dominates
    # runtime on big graphs, and we only need it for the handful of chain bridges
    # we actually emit -- not for every matmul consumer scanned during epilogue
    # detection. pattern_hash_for(ci) fills it in lazily on demand.
    pinfo_cache: dict[int, PartitionInfo] = {}

    def pinfo(ci: int) -> PartitionInfo:
        if ci in pinfo_cache:
            return pinfo_cache[ci]
        comp = comps[ci]
        cn = compute_nodes(comp)
        info = PartitionInfo(
            comp_index=ci,
            strictly_pointwise=partition_is_strictly_pointwise(comp),
            n_compute=len(cn),
            compute_ops=[short_op(n.target) for n in cn],
            pattern_hash=None,        # filled lazily by pattern_hash_for
            in_corpus=False,
        )
        pinfo_cache[ci] = info
        return info

    _hashed: set[int] = set()

    def pattern_hash_for(ci: int) -> tuple[str | None, bool]:
        info = pinfo(ci)
        if ci not in _hashed:
            _hashed.add(ci)
            pat = compute_partition_pattern(comps[ci], gm)
            info.pattern_hash = pat["pattern_hash"] if pat else None
            info.in_corpus = (info.pattern_hash in _CORPUS_HASHES) if info.pattern_hash else False
        return info.pattern_hash, info.in_corpus

    matmuls = [n for n in gm.graph.nodes if is_matmul(n)]

    # ---- pattern 1: matmul + strictly-pointwise epilogue --------------------
    epilogues: list[EpilogueHit] = []
    for mm in matmuls:
        targets, feeds_out = forward_real_targets(mm)
        consumer_comps = []          # comp indices (may repeat-> dedup)
        all_targets_pw = bool(targets)
        for t in targets:
            ci = node2comp.get(t)
            if ci is None:
                # consumer is itself non-fusible (another matmul/conv/sdpa) or
                # an unpartitioned op -> not a pointwise epilogue.
                all_targets_pw = False
                continue
            consumer_comps.append(ci)
            if not pinfo(ci).strictly_pointwise:
                all_targets_pw = False
        consumer_comps = sorted(set(consumer_comps))
        pw_parts = [pinfo(ci) for ci in consumer_comps if pinfo(ci).strictly_pointwise]
        if not pw_parts:
            continue  # no pointwise epilogue at all
        epi_ops = []
        for p in pw_parts:
            epi_ops.extend(p.compute_ops)
        epilogues.append(EpilogueHit(
            graph_name=graph_name,
            matmul=mm.name,
            matmul_op=short_op(mm.target),
            output_shape=_safe_shape(mm.meta.get("val")),
            clean=all_targets_pw and not feeds_out,
            feeds_graph_output=feeds_out,
            epilogue_ops=epi_ops,
            epilogue_partitions=[asdict(p) for p in pw_parts],
            n_consumers=len(consumer_comps) + sum(
                1 for t in targets if node2comp.get(t) is None),
        ))

    # ---- pattern 2: matmul -> pointwise -> matmul ---------------------------
    chains: list[ChainHit] = []
    seen_triples: set[tuple] = set()
    for mm1 in matmuls:
        targets1, _ = forward_real_targets(mm1)
        for t in targets1:
            # case A: direct mm1 -> (transparent only) -> mm2, no partition.
            if is_matmul(t):
                key = (mm1.name, None, t.name)
                if key in seen_triples:
                    continue
                seen_triples.add(key)
                chains.append(ChainHit(
                    graph_name=graph_name, mm1=mm1.name, mm1_op=short_op(mm1.target),
                    mm2=t.name, mm2_op=short_op(t.target),
                    bridge_pointwise_ops=[], n_bridge_pointwise=0,
                    bridge_pattern_hash=None, bridge_in_corpus=False,
                    mm1_out_shape=_safe_shape(mm1.meta.get("val")),
                    mm2_out_shape=_safe_shape(t.meta.get("val")),
                ))
                continue
            # case B: mm1 -> strictly-pw partition P -> mm2
            ci = node2comp.get(t)
            if ci is None or not pinfo(ci).strictly_pointwise:
                continue
            # find matmuls downstream of partition P's escaping outputs
            for e in partition_escaping_nodes(comps[ci], comp_sets[ci]):
                targets2, _ = forward_real_targets(e)
                for t2 in targets2:
                    if not is_matmul(t2):
                        continue
                    key = (mm1.name, ci, t2.name)
                    if key in seen_triples:
                        continue
                    seen_triples.add(key)
                    p = pinfo(ci)
                    phash, in_corpus = pattern_hash_for(ci)  # deferred hash
                    chains.append(ChainHit(
                        graph_name=graph_name, mm1=mm1.name,
                        mm1_op=short_op(mm1.target),
                        mm2=t2.name, mm2_op=short_op(t2.target),
                        bridge_pointwise_ops=p.compute_ops,
                        n_bridge_pointwise=p.n_compute,
                        bridge_pattern_hash=phash,
                        bridge_in_corpus=in_corpus,
                        mm1_out_shape=_safe_shape(mm1.meta.get("val")),
                        mm2_out_shape=_safe_shape(t2.meta.get("val")),
                    ))

    # ---- pattern 3: >=2 matmuls CONVERGE into one output (fan-in) -----------
    # The "two matmuls reduce to one output" case: independent sibling matmuls
    # merged through a pointwise bridge partition into a single escaping value.
    # mm(a,b)+mm(c,d) (residual/parallel proj) or silu(mm(x,Wg))*mm(x,Wu)
    # (gated MLP). We scan each partition's escaping outputs and trace back
    # through the partition's compute + view ops to the matmuls that feed it.
    fanins: list[FanInHit] = []
    seen_fanin: set[tuple] = set()
    for ci, comp in enumerate(comps):
        cs = comp_sets[ci]
        for out_node in partition_escaping_nodes(comp, cs):
            mms = matmuls_feeding_output(out_node, cs)
            if len(mms) < 2:
                continue
            key = (ci, out_node.name, tuple(sorted(m.name for m in mms)))
            if key in seen_fanin:
                continue
            seen_fanin.add(key)
            p = pinfo(ci)
            phash, in_corpus = pattern_hash_for(ci)
            # The combine op = the bridge node where >=2 distinct matmul
            # lineages MERGE. For each node we compute its backward cone of
            # source matmuls (through the partition's own compute nodes and
            # transparent views, stopping at each matmul). The merge point is a
            # node with >=2 direct inputs that each carry a non-empty source set,
            # whose union is >=2 -- i.e. the silu(mm)*mm `mul`, or the mm+mm
            # `add`. (best-effort label: add / mul / where ...)
            def _source_matmuls(node, _cache={}):
                key = id(node)
                if key in _cache:
                    return _cache[key]
                if is_matmul(node):
                    _cache[key] = frozenset({node})
                    return _cache[key]
                if not (is_transparent(node) or node in cs):
                    _cache[key] = frozenset()
                    return _cache[key]
                acc = set()
                for inp in _input_nodes(node):
                    acc |= _source_matmuls(inp)
                _cache[key] = frozenset(acc)
                return _cache[key]

            combine = "?"
            for n in comp:
                if n.op != "call_function" or n.target in TRANSPARENT_OPS:
                    continue
                inp_sets = [_source_matmuls(inp) for inp in _input_nodes(n)]
                nonempty = [sset for sset in inp_sets if sset]
                union = set().union(*nonempty) if nonempty else set()
                if len(nonempty) >= 2 and len(union) >= 2:
                    combine = short_op(n.target)
                    break

            # Memory-elimination test: a contributing matmul's output is an
            # ELIMINABLE DRAM read only if the bridge is its sole consumer. If it
            # escapes (other consumer / graph output) it must be materialized
            # anyway and fusing the combine saves nothing.
            exclusive = [matmul_consumed_only_by(m, cs) for m in mms]
            elim_bytes = 0
            have_bytes = True
            for m, ex in zip(mms, exclusive):
                if not ex:
                    continue
                nb = _node_nbytes(m)
                if nb is None:
                    have_bytes = False
                else:
                    elim_bytes += nb

            fanins.append(FanInHit(
                graph_name=graph_name,
                matmuls=[m.name for m in mms],
                matmul_ops=[short_op(m.target) for m in mms],
                n_matmuls=len(mms),
                matmul_shapes=[_safe_shape(m.meta.get("val")) for m in mms],
                combine_op=combine,
                strictly_pointwise=p.strictly_pointwise,
                bridge_ops=p.compute_ops,
                output_node=out_node.name,
                output_shape=_safe_shape(out_node.meta.get("val")),
                bridge_pattern_hash=phash,
                bridge_in_corpus=in_corpus,
                matmul_exclusive=exclusive,
                all_exclusive=all(exclusive),
                eliminable_read_bytes=(elim_bytes if have_bytes else None),
            ))

    return {
        "graph_name": graph_name,
        "n_partitions": len(comps),
        "n_matmuls": len(matmuls),
        "epilogues": [asdict(e) for e in epilogues],
        "chains": [asdict(c) for c in chains],
        "fanins": [asdict(f) for f in fanins],
    }


# ---------------------------------------------------------------------------
# Corpus driver
# ---------------------------------------------------------------------------

def discover_model_dirs() -> list[Path]:
    models_dir = ROOT / "repros" / "models"
    if not models_dir.exists():
        return []
    dirs = set()
    for fg in models_dir.rglob("full_graph_*.py"):
        dirs.add(fg.parent)
    return sorted(dirs)


def analyze_full_graph_file(graph_path: Path, device: str = "cuda") -> dict | None:
    from scripts.model_graph_accounting import trace_full_graph
    try:
        gm = trace_full_graph(graph_path, device=device)
    except Exception as e:  # noqa: BLE001
        return {"graph_name": graph_path.stem, "error": f"{type(e).__name__}: {e}"}
    return analyze_graph(gm, graph_path.stem)


def run_corpus(model_dirs, device: str, max_graphs_per_model: int = 0) -> dict:
    per_graph = []
    failures = []
    for md in model_dirs:
        graphs = sorted(md.glob("full_graph_*.py"))
        if max_graphs_per_model:
            graphs = graphs[:max_graphs_per_model]
        for gp in graphs:
            rel = str(gp.relative_to(ROOT))
            print(f"  {rel}", file=sys.stderr, flush=True)
            res = analyze_full_graph_file(gp, device=device)
            if res is None:
                continue
            res["model"] = md.name
            res["path"] = rel
            if "error" in res:
                failures.append({"path": rel, "error": res["error"]})
                continue
            per_graph.append(res)
    return {"per_graph": per_graph, "failures": failures}


def summarize(corpus: dict) -> dict:
    per_graph = corpus["per_graph"]
    n_graphs = len(per_graph)
    n_matmuls = sum(g["n_matmuls"] for g in per_graph)

    epi_total = sum(len(g["epilogues"]) for g in per_graph)
    epi_clean = sum(1 for g in per_graph for e in g["epilogues"] if e["clean"])
    chains_all = [c for g in per_graph for c in g["chains"]]
    chains_pw = [c for c in chains_all if c["n_bridge_pointwise"] >= 1]

    epi_op_counter = Counter()
    for g in per_graph:
        for e in g["epilogues"]:
            # signature = sorted distinct epilogue ops
            epi_op_counter[tuple(sorted(set(e["epilogue_ops"])))] += 1
    bridge_op_counter = Counter()
    for c in chains_pw:
        bridge_op_counter[tuple(sorted(set(c["bridge_pointwise_ops"])))] += 1

    matmul_op_counter = Counter()
    for g in per_graph:
        for e in g["epilogues"]:
            matmul_op_counter[e["matmul_op"]] += 1

    # fan-in: >=2 matmuls converging to one output via a pointwise bridge.
    fanins_all = [f for g in per_graph for f in g.get("fanins", [])]
    fanins_pw = [f for f in fanins_all if f["strictly_pointwise"]]
    # The MEMORY-eliminating subset: every contributing matmul's sole consumer
    # is the bridge, so the intermediates never touch DRAM.
    fanins_elim = [f for f in fanins_pw if f.get("all_exclusive")]
    fanin_combine_counter = Counter(f["combine_op"] for f in fanins_pw)
    fanin_elim_combine_counter = Counter(f["combine_op"] for f in fanins_elim)
    fanin_bridge_counter = Counter()
    for f in fanins_elim:
        fanin_bridge_counter[tuple(sorted(set(f["bridge_ops"])))] += 1
    fanin_arity_counter = Counter(f["n_matmuls"] for f in fanins_pw)
    total_elim_bytes = sum(
        f["eliminable_read_bytes"] for f in fanins_elim
        if f.get("eliminable_read_bytes") is not None
    )

    return {
        "n_graphs_analyzed": n_graphs,
        "n_failures": len(corpus["failures"]),
        "total_matmuls": n_matmuls,
        "matmuls_with_pw_epilogue": epi_total,
        "matmuls_with_clean_pw_epilogue": epi_clean,
        "mm_pw_mm_chains_total": len(chains_all),
        "mm_pw_mm_chains_with_pointwise": len(chains_pw),
        "mm_then_mm_no_pointwise": len(chains_all) - len(chains_pw),
        "matmul_fanin_total": len(fanins_all),
        "matmul_fanin_strictly_pointwise": len(fanins_pw),
        "matmul_fanin_memory_eliminating": len(fanins_elim),
        "fanin_eliminable_read_bytes_total": total_elim_bytes,
        "fanin_elim_combine_op_breakdown": dict(fanin_elim_combine_counter.most_common()),
        "fanin_arity_breakdown": dict(sorted(fanin_arity_counter.items())),
        "fanin_combine_op_breakdown": dict(fanin_combine_counter.most_common()),
        "top_epilogue_op_signatures": [
            {"ops": list(k), "count": v}
            for k, v in epi_op_counter.most_common(25)
        ],
        "top_bridge_op_signatures": [
            {"ops": list(k), "count": v}
            for k, v in bridge_op_counter.most_common(25)
        ],
        "top_fanin_bridge_signatures": [
            {"ops": list(k), "count": v}
            for k, v in fanin_bridge_counter.most_common(25)
        ],
        "epilogue_matmul_op_breakdown": dict(matmul_op_counter.most_common()),
    }


def write_markdown(summary: dict, corpus: dict, out: Path) -> None:
    L = []
    L.append("# Matmul partitioning views\n")
    L.append("Built on the **existing** `get_fusion_partitions` (default "
             "partitioner, unchanged). Matmuls are non-fusible extern ops, so "
             "their pointwise neighbors sit in separate partitions; this view "
             "re-attaches them.\n")
    s = summary
    L.append("## Totals\n")
    L.append(f"- Graphs analyzed: **{s['n_graphs_analyzed']}** "
             f"(load failures: {s['n_failures']})")
    L.append(f"- Matmuls seen: **{s['total_matmuls']}**")
    L.append(f"- Matmuls with a strictly-pointwise epilogue: "
             f"**{s['matmuls_with_pw_epilogue']}** "
             f"(clean = every consumer is pointwise: "
             f"**{s['matmuls_with_clean_pw_epilogue']}**)")
    L.append(f"- `mm -> pointwise -> mm` chains (SEQUENTIAL, >=1 pointwise op "
             f"between): **{s['mm_pw_mm_chains_with_pointwise']}**")
    L.append(f"- `mm -> mm` with only views between (no pointwise): "
             f"**{s['mm_then_mm_no_pointwise']}**")
    L.append(f"- **>=2 matmuls converging to ONE output** (fan-in / reduce: "
             f"`mm(a,b) (+|*) mm(c,d)`): **{s.get('matmul_fanin_total', 0)}** "
             f"(through a strictly-pointwise bridge: "
             f"**{s.get('matmul_fanin_strictly_pointwise', 0)}**)")
    elim_mb = s.get('fanin_eliminable_read_bytes_total', 0) / 1e6
    L.append(f"  - of which **memory-eliminating** (every contributing matmul's "
             f"ONLY consumer is the bridge -> intermediate never hits DRAM): "
             f"**{s.get('matmul_fanin_memory_eliminating', 0)}**, "
             f"eliminating **{elim_mb:.1f} MB** of matmul-output reads\n")

    L.append("## Matmul op breakdown (epilogue hits)\n")
    for op, c in s["epilogue_matmul_op_breakdown"].items():
        L.append(f"- `{op}`: {c}")
    L.append("")

    L.append("## Top epilogue op-signatures\n")
    L.append("| count | epilogue ops (distinct, sorted) |")
    L.append("|------:|---------------------------------|")
    for row in s["top_epilogue_op_signatures"]:
        L.append(f"| {row['count']} | `{', '.join(row['ops'])}` |")
    L.append("")

    L.append("## Top `mm -> pointwise -> mm` bridge signatures\n")
    L.append("| count | bridge pointwise ops |")
    L.append("|------:|----------------------|")
    for row in s["top_bridge_op_signatures"]:
        L.append(f"| {row['count']} | `{', '.join(row['ops'])}` |")
    L.append("")

    # --- fan-in (convergent) tables --------------------------------------
    L.append("## Matmul fan-in (>=2 matmuls -> 1 output)\n")
    L.append("The 'two matmuls reduce to one output' pattern: independent "
             "sibling matmuls combined through a pointwise bridge into a single "
             "value (`mm(a,b)+mm(c,d)` residual/parallel-proj; "
             "`silu(mm(x,Wg))*mm(x,Wu)` gated MLP). Distinct from the "
             "sequential chain above.\n")
    L.append("**The memory win requires EXCLUSIVITY.** Fusing the combine into "
             "the GEMM epilogue only eliminates a DRAM read when each "
             "contributing matmul's output feeds *nothing but* the bridge. If a "
             "matmul output is reused elsewhere (or escapes to a graph output) "
             "it must be materialized regardless, so that matmul is not counted "
             "as eliminable. `all_exclusive` rows are the clean targets.\n")
    L.append("**Combine op** (the pointwise op joining the matmuls) -- "
             "all strict-pw vs memory-eliminating subset:\n")
    L.append("| combine | strict-pw | memory-eliminating |")
    L.append("|---------|----------:|-------------------:|")
    allc = s.get("fanin_combine_op_breakdown", {})
    elimc = s.get("fanin_elim_combine_op_breakdown", {})
    for op in allc:
        L.append(f"| `{op}` | {allc[op]} | {elimc.get(op, 0)} |")
    L.append("")
    L.append("**Arity** (how many matmuls converge, strict-pw):\n")
    for k, v in s.get("fanin_arity_breakdown", {}).items():
        L.append(f"- {k} matmuls: {v}")
    L.append("")
    L.append("**Top memory-eliminating fan-in bridge signatures:**\n")
    L.append("| count | bridge ops |")
    L.append("|------:|------------|")
    for row in s.get("top_fanin_bridge_signatures", []):
        L.append(f"| {row['count']} | `{', '.join(row['ops'])}` |")
    L.append("")

    # Concrete examples -- spread across DISTINCT models, one row per model,
    # so the list shows corpus breadth rather than 15 rows from one graph.
    L.append("## Example clean epilogues (one per model)\n")
    seen_models: set[str] = set()
    for g in corpus["per_graph"]:
        if g["model"] in seen_models:
            continue
        for e in g["epilogues"]:
            if not e["clean"]:
                continue
            L.append(f"- `{g['model']}` / {g['graph_name']}: "
                     f"`{e['matmul_op']}` {e['output_shape']} -> "
                     f"epilogue `{'+'.join(e['epilogue_ops'])}`")
            seen_models.add(g["model"])
            break
        if len(seen_models) >= 25:
            break
    L.append("")

    L.append("## Example mm -> pointwise -> mm chains (one per model)\n")
    seen_models = set()
    for g in corpus["per_graph"]:
        if g["model"] in seen_models:
            continue
        for c in g["chains"]:
            if c["n_bridge_pointwise"] < 1:
                continue
            L.append(f"- `{g['model']}` / {g['graph_name']}: "
                     f"`{c['mm1_op']}` {c['mm1_out_shape']} -> "
                     f"`{'+'.join(c['bridge_pointwise_ops'])}` -> "
                     f"`{c['mm2_op']}` {c['mm2_out_shape']}"
                     + (f"  (bridge in corpus: {c['bridge_pattern_hash']})"
                        if c["bridge_in_corpus"] else ""))
            seen_models.add(g["model"])
            break
        if len(seen_models) >= 25:
            break
    L.append("")

    L.append("## Example memory-eliminating fan-in (one per model)\n")
    L.append("(`all_exclusive` -- every contributing matmul feeds only the "
             "bridge, so its output never hits DRAM.)\n")
    seen_models = set()
    for g in corpus["per_graph"]:
        if g["model"] in seen_models:
            continue
        for f in g.get("fanins", []):
            if not (f["strictly_pointwise"] and f.get("all_exclusive")):
                continue
            shapes = " , ".join(str(sh) for sh in f["matmul_shapes"])
            eb = f.get("eliminable_read_bytes")
            eb_str = f", elim={eb/1e3:.1f}KB" if eb else ""
            L.append(f"- `{g['model']}` / {g['graph_name']}: "
                     f"{f['n_matmuls']}x `{'+'.join(f['matmul_ops'])}` "
                     f"({shapes}) `{f['combine_op']}`-combined via "
                     f"`{'+'.join(f['bridge_ops'])}` -> {f['output_shape']}"
                     f"{eb_str}"
                     + (f"  (bridge in corpus: {f['bridge_pattern_hash']})"
                        if f["bridge_in_corpus"] else ""))
            seen_models.add(g["model"])
            break
        if len(seen_models) >= 25:
            break
    L.append("")
    out.write_text("\n".join(L))


def _flat_hits(corpus: dict, pattern_key: str) -> list[dict]:
    """Every hit of one pattern type, with model/graph/path stamped on each."""
    rows = []
    for g in corpus["per_graph"]:
        for hit in g.get(pattern_key, []):
            rows.append({
                "model": g["model"],
                "graph": g["graph_name"],
                "path": g["path"],
                **hit,
            })
    return rows


def _dedupe(rows: list[dict], key_fn, signature_fn, example_fn,
            sum_field: str | None = None) -> list[dict]:
    """Collapse rows sharing a STRUCTURAL signature into one entry.

    Same philosophy as the corpus pattern_hash: the key ignores shapes/node
    names and keys only on op structure, so the 16 identical gated-MLP fan-ins
    in MT5 become ONE row with n_occurrences=16 instead of 16 near-duplicates.
    Each entry keeps a representative `example` (with its concrete shapes) and
    an `occurrences_by_model` count so you can still see where it lands.
    """
    groups: dict = {}
    order: list = []
    for r in rows:
        k = key_fn(r)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)

    out = []
    for k in order:
        grp = groups[k]
        by_model: Counter = Counter(r["model"] for r in grp)
        entry = dict(signature_fn(grp[0]))
        entry["n_occurrences"] = len(grp)
        entry["n_models"] = len(by_model)
        entry["occurrences_by_model"] = dict(by_model.most_common())
        if sum_field:
            entry[f"total_{sum_field}"] = sum(
                (r.get(sum_field) or 0) for r in grp)
        entry["example"] = example_fn(grp[0])
        out.append(entry)
    return out


def write_split_json(summary: dict, corpus: dict, od: Path) -> list[Path]:
    """Serialize each pattern type into its OWN DEDUPED JSON file under od/.

    The single combined matmul_patterns.json nests every graph's hits and runs
    to ~15 MB -- unreadable. Instead emit one file per pattern, each a list of
    UNIQUE structural patterns (shapes ignored, same as the corpus pattern_hash)
    with an occurrence count + per-model breakdown + one concrete example:

      summary.json    -- the corpus rollup (counts, breakdowns)
      epilogues.json  -- unique matmul+epilogue signatures
      chains.json     -- unique mm -> pointwise -> mm signatures
      fanins.json     -- unique >=2-matmul fan-in signatures (memory-eliminating
                         first, then by total eliminable_read_bytes desc)
      failures.json   -- graphs that failed to load
      index.json      -- file manifest + unique/total counts

    Returns the list of paths written.
    """
    written: list[Path] = []

    def _dump(name: str, obj) -> None:
        p = od / name
        p.write_text(json.dumps(obj, indent=2))
        written.append(p)

    # ---- epilogues: key on (matmul op, clean, ordered epilogue ops) ----------
    epi_rows = _flat_hits(corpus, "epilogues")
    epilogues = _dedupe(
        epi_rows,
        key_fn=lambda r: (r["matmul_op"], r["clean"], tuple(r["epilogue_ops"])),
        signature_fn=lambda r: {
            "matmul_op": r["matmul_op"],
            "clean": r["clean"],
            "epilogue_ops": r["epilogue_ops"],
        },
        example_fn=lambda r: {
            "model": r["model"], "graph": r["graph"],
            "matmul": r["matmul"], "output_shape": r["output_shape"],
        },
    )
    epilogues.sort(key=lambda e: -e["n_occurrences"])

    # ---- chains: key on (mm1 op, ordered bridge ops, mm2 op) -----------------
    chain_rows = _flat_hits(corpus, "chains")
    chains = _dedupe(
        chain_rows,
        key_fn=lambda r: (r["mm1_op"], tuple(r["bridge_pointwise_ops"]), r["mm2_op"]),
        signature_fn=lambda r: {
            "mm1_op": r["mm1_op"],
            "bridge_pointwise_ops": r["bridge_pointwise_ops"],
            "mm2_op": r["mm2_op"],
            "bridge_pattern_hash": r["bridge_pattern_hash"],
            "bridge_in_corpus": r["bridge_in_corpus"],
        },
        example_fn=lambda r: {
            "model": r["model"], "graph": r["graph"],
            "mm1": r["mm1"], "mm1_out_shape": r["mm1_out_shape"],
            "mm2": r["mm2"], "mm2_out_shape": r["mm2_out_shape"],
        },
    )
    chains.sort(key=lambda e: -e["n_occurrences"])

    # ---- fanins: key on (sorted matmul ops, combine, bridge ops, exclusive) --
    fanin_rows = _flat_hits(corpus, "fanins")
    fanins = _dedupe(
        fanin_rows,
        key_fn=lambda r: (
            tuple(sorted(r["matmul_ops"])), r["combine_op"],
            tuple(r["bridge_ops"]), r["all_exclusive"], r["strictly_pointwise"],
        ),
        signature_fn=lambda r: {
            "matmul_ops": r["matmul_ops"],
            "n_matmuls": r["n_matmuls"],
            "combine_op": r["combine_op"],
            "bridge_ops": r["bridge_ops"],
            "strictly_pointwise": r["strictly_pointwise"],
            "all_exclusive": r["all_exclusive"],
            "bridge_pattern_hash": r["bridge_pattern_hash"],
            "bridge_in_corpus": r["bridge_in_corpus"],
        },
        example_fn=lambda r: {
            "model": r["model"], "graph": r["graph"],
            "matmul_shapes": r["matmul_shapes"],
            "output_shape": r["output_shape"],
            "eliminable_read_bytes": r["eliminable_read_bytes"],
        },
        sum_field="eliminable_read_bytes",
    )
    # Most actionable first: the clean fusion targets -- strictly-pointwise
    # bridge AND every contributing matmul exclusive (this is the "198
    # memory-eliminating" set) -- then by total DRAM eliminated across all
    # occurrences. Non-strict (e.g. softmax-bridge attention) and non-exclusive
    # signatures sort below.
    def _fanin_rank(e):
        clean_elim = e.get("strictly_pointwise") and e.get("all_exclusive")
        return (
            not clean_elim,
            not e.get("all_exclusive", False),
            -(e.get("total_eliminable_read_bytes") or 0),
        )
    fanins.sort(key=_fanin_rank)

    _dump("summary.json", summary)
    _dump("epilogues.json", epilogues)
    _dump("chains.json", chains)
    _dump("fanins.json", fanins)
    _dump("failures.json", corpus["failures"])

    _dump("index.json", {
        "files": {
            "summary.json": "corpus rollup (counts + breakdowns)",
            "epilogues.json": "unique matmul + strictly-pointwise epilogue signatures",
            "chains.json": "unique mm -> pointwise -> mm sequential chain signatures",
            "fanins.json": "unique >=2 matmuls -> 1 output signatures (memory-eliminating first)",
            "failures.json": "graphs that failed to load",
        },
        "note": "each pattern file is DEDUPED by op-structure (shapes ignored, "
                "like the corpus pattern_hash); n_occurrences is the raw count.",
        "unique_patterns": {
            "epilogues": len(epilogues),
            "chains": len(chains),
            "fanins": len(fanins),
        },
        "total_occurrences": {
            "epilogues": len(epi_rows),
            "chains": len(chain_rows),
            "fanins": len(fanin_rows),
            "failures": len(corpus["failures"]),
        },
    })
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", "-m", default=None,
                    help="Model name filter (partial match)")
    ap.add_argument("--all", "-a", action="store_true", help="All models")
    ap.add_argument("--max-models", type=int, default=0, help="Cap model count")
    ap.add_argument("--max-graphs", type=int, default=0,
                    help="Cap graphs per model (0=all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default=None,
                    help="Write split per-pattern JSON files + SUMMARY.md here")
    ap.add_argument("--combined-json", action="store_true",
                    help="Also write the single nested matmul_patterns.json "
                         "(off by default; the split files supersede it)")
    args = ap.parse_args()

    if not args.model and not args.all:
        ap.error("specify --model <name> or --all")

    model_dirs = discover_model_dirs()
    if args.model:
        model_dirs = [d for d in model_dirs if args.model.lower() in d.name.lower()]
        if not model_dirs:
            print(f"no models matching '{args.model}'", file=sys.stderr)
            sys.exit(1)
    if args.max_models:
        model_dirs = model_dirs[:args.max_models]

    print(f"Analyzing {len(model_dirs)} model dir(s)...", file=sys.stderr)
    corpus = run_corpus(model_dirs, args.device, args.max_graphs)
    summary = summarize(corpus)

    print(json.dumps(summary, indent=2))

    if args.output_dir:
        od = Path(args.output_dir)
        if not od.is_absolute():
            od = ROOT / od
        od.mkdir(parents=True, exist_ok=True)
        written = write_split_json(summary, corpus, od)
        write_markdown(summary, corpus, od / "SUMMARY.md")
        if args.combined_json:
            (od / "matmul_patterns.json").write_text(
                json.dumps({"summary": summary, **corpus}, indent=2))
            written.append(od / "matmul_patterns.json")
        print(f"\nWrote SUMMARY.md + {len(written)} JSON files to {od}:",
              file=sys.stderr)
        for p in written:
            print(f"  {p.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
