#!/usr/bin/env python3
"""Model-level graph accounting via the capture pipeline's OWN partitioner.

Loads saved full_graph_*.py artifacts (via full_graph_harness.load_full_graph),
traces them with make_fx to get shape metadata on every node, then partitions
the graph with EXACTLY the same CapabilityBasedPartitioner + is_fusible_node +
create_op_support setup the capture pipeline uses (shared helpers in
capture_hook.py: get_fusion_partitions / compute_partition_pattern). This
guarantees the accounting partitions are identical to how the canonical repros
in repros/canonical/ were originally cut, and that each partition's
pattern_hash maps directly to a repros/canonical/<family>_<hash> directory.

For each model, produces:
  - Table of pattern_hash -> canonical repro dir, OCCURRENCE COUNT (how many
    times the pattern appears across the model's graphs -- the piece that was
    missing from earlier dedup-only projections), representative shapes
  - Non-fusible ops (BLAS/cuDNN/fallback) with actual shapes from
    node.meta['val']
  - UNMATCHED partitions: pattern hashes with no canonical repro dir
    (corpus coverage gaps)
  - Optional time projection: SUM of (per-partition oracle_us x occurrence
    count) -- never averaged ratios

Usage:
    python scripts/model_graph_accounting.py --model convnextv2
    python scripts/model_graph_accounting.py --model convnextv2 --json
    python scripts/model_graph_accounting.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "repros" / "models"
CANONICAL_DIR = ROOT / "repros" / "canonical"
DEFAULT_TIMINGS = ROOT / "results" / "all_oracle_timings_b200_v2.json"

sys.path.insert(0, str(ROOT))


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class PartitionOccurrence:
    """One concrete occurrence of a fusible partition in one graph."""
    graph_name: str
    pattern_hash: str
    shape_hash: str
    n_ops: int
    op_counts: dict[str, int]
    origin_ops: list[str]         # str(node.target), capture-pipeline format
    input_shapes: list[str]       # "shape:dtype" for each partition input
    max_output_shape: list[int] | None
    has_reduction: bool


@dataclass
class NonFusibleOp:
    """A single non-fusible op instance in the graph."""
    graph_name: str
    target: str
    output_shape: list[int] | None
    output_dtype: str | None
    node_name: str


@dataclass
class GraphAccounting:
    """Accounting for one full_graph_*.py file."""
    graph_path: str
    graph_name: str
    total_nodes: int = 0
    call_function_count: int = 0
    occurrences: list[PartitionOccurrence] = field(default_factory=list)
    non_fusible_ops: list[NonFusibleOp] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ModelAccounting:
    """Aggregate accounting for all graphs in a model directory."""
    model_name: str
    model_dir: str
    graphs: list[GraphAccounting] = field(default_factory=list)
    manifest_patterns: list[str] = field(default_factory=list)

    @property
    def all_occurrences(self) -> list[PartitionOccurrence]:
        return [o for g in self.graphs for o in g.occurrences]

    @property
    def all_non_fusible(self) -> list[NonFusibleOp]:
        return [op for g in self.graphs for op in g.non_fusible_ops]

    def pattern_counts(self) -> Counter:
        """pattern_hash -> occurrence count across ALL the model's graphs."""
        return Counter(o.pattern_hash for o in self.all_occurrences)


# ============================================================================
# Graph loading and tracing
# ============================================================================

def trace_full_graph(graph_path: Path, device: str = "cuda"):
    """Load a full_graph_*.py via full_graph_harness and trace it with make_fx.

    Returns a GraphModule with shape metadata (node.meta['val']) on all nodes.
    Tries fake-mode tracing first (no real compute), then real mode.
    """
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    from full_graph_harness import load_full_graph

    instance, inputs, _definition = load_full_graph(graph_path, default_device=device)

    # Fake mode first: shape propagation without executing convolutions etc.
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode() as fake_mode:
            fake_inputs = [
                fake_mode.from_tensor(t) if torch.is_tensor(t) else t
                for t in inputs
            ]
            with torch.no_grad():
                return make_fx(instance, tracing_mode="fake")(*fake_inputs)
    except Exception:
        pass

    # Fallback: real-mode tracing (executes the graph once).
    with torch.no_grad():
        return make_fx(instance, tracing_mode="real")(*inputs)


# ============================================================================
# Core analysis: REUSES the capture pipeline's partitioner + hashing
# ============================================================================

def _safe_shape(val) -> list[int] | None:
    """Extract a concrete shape list from a meta 'val' tensor."""
    import torch

    if val is None:
        return None
    if torch.is_tensor(val):
        out = []
        for d in val.shape:
            try:
                if hasattr(d, "node") and hasattr(d.node, "hint"):
                    out.append(int(d.node.hint))
                else:
                    out.append(int(d))
            except (TypeError, ValueError):
                out.append(-1)
        return out
    if isinstance(val, (list, tuple)):
        for item in val:
            shape = _safe_shape(item)
            if shape is not None:
                return shape
    return None


def _target_name(target) -> str:
    if hasattr(target, "_schema"):
        name = target._schema.name
        overload = target._schema.overload_name
        return f"{name}.{overload}" if overload else f"{name}.default"
    return str(target)


def analyze_graph(gm, graph_path: str, graph_name: str) -> GraphAccounting:
    """Partition a traced GraphModule with the capture pipeline's partitioner
    and account for every fusible partition and non-fusible op."""
    import torch

    from capture_hook import (
        compute_partition_pattern,
        get_fusion_partitions,
        partition_has_reduction,
        partition_node_is_supported,
    )

    acc = GraphAccounting(graph_path=graph_path, graph_name=graph_name)

    partitioned_nodes: set = set()
    components = get_fusion_partitions(gm)

    for comp in components:
        partitioned_nodes.update(comp)
        pattern = compute_partition_pattern(comp, gm)
        if pattern is None:
            acc.errors.append(
                f"subgraph extraction failed for partition with "
                f"{len(comp)} nodes"
            )
            continue

        op_counts = Counter(
            _target_name(n.target) for n in comp if n.op == "call_function"
        )
        input_shapes = sorted(
            f"{info.get('shape', '?')}:{info.get('dtype', '?')}"
            for info in pattern["placeholder_info"].values()
        )

        max_numel = -1
        max_shape = None
        for n in comp:
            shape = _safe_shape(n.meta.get("val"))
            if shape is None:
                continue
            numel = 1
            for d in shape:
                numel *= max(d, 1)
            if numel > max_numel:
                max_numel = numel
                max_shape = shape

        acc.occurrences.append(PartitionOccurrence(
            graph_name=graph_name,
            pattern_hash=pattern["pattern_hash"],
            shape_hash=pattern["shape_hash"],
            n_ops=len(comp),
            op_counts=dict(op_counts),
            origin_ops=sorted(
                str(n.target) for n in comp if n.op == "call_function"
            ),
            input_shapes=input_shapes,
            max_output_shape=max_shape,
            has_reduction=partition_has_reduction(comp),
        ))

    for node in gm.graph.nodes:
        acc.total_nodes += 1
        if node.op != "call_function":
            continue
        acc.call_function_count += 1
        if node in partitioned_nodes:
            continue
        if partition_node_is_supported(node):
            # Fusible/transparent node the partitioner chose not to place in
            # any compute partition (e.g. dangling view) -- not a kernel.
            continue
        val = node.meta.get("val")
        dtype = None
        if torch.is_tensor(val):
            dtype = str(val.dtype)
        elif isinstance(val, (list, tuple)):
            for item in val:
                if torch.is_tensor(item):
                    dtype = str(item.dtype)
                    break
        acc.non_fusible_ops.append(NonFusibleOp(
            graph_name=graph_name,
            target=_target_name(node.target),
            output_shape=_safe_shape(val),
            output_dtype=dtype,
            node_name=node.name,
        ))

    return acc


def _collect_manifest_patterns(model_dir: Path) -> list[str]:
    """Collect manifest pattern hashes for a model directory.

    Two layouts exist:
      - new: <model_dir>/manifest.json with a "patterns" list
      - old: sibling dirs named <suite>_<model>_<mode>_<NNN>/manifest.json,
        one per captured graph (e.g. torchbench_BERT_pytorch_train_000)
    """
    patterns: set[str] = set()

    manifest_path = model_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            patterns.update(manifest.get("patterns", []))
        except (json.JSONDecodeError, OSError):
            pass

    if not patterns:
        # Old layout: per-graph label dirs next to the clean model dir.
        parent = model_dir.parent          # e.g. .../torchbench/train
        mode = parent.name                 # train / infer / model name
        suite = parent.parent.name         # torchbench / hf / timm
        candidates = [f"{suite}_{model_dir.name}_{mode}_*"]
        if suite not in ("hf", "timm", "torchbench"):
            # vllm/genai layout: <suite>/<model>/...
            suite = parent.name
            candidates = [f"{suite}_{model_dir.name}_*"]
        for pattern in candidates:
            for sib in parent.glob(pattern):
                mf = sib / "manifest.json"
                if not mf.exists():
                    continue
                try:
                    manifest = json.loads(mf.read_text())
                    patterns.update(manifest.get("patterns", []))
                except (json.JSONDecodeError, OSError):
                    continue

    return sorted(patterns)


def process_model_directory(model_dir: Path, device: str = "cuda",
                            max_graphs: int = 0) -> ModelAccounting:
    """Process all full_graph_*.py files in a model directory."""
    acc = ModelAccounting(model_name=model_dir.name, model_dir=str(model_dir))

    acc.manifest_patterns = _collect_manifest_patterns(model_dir)

    graph_files = sorted(model_dir.glob("full_graph_*.py"))
    if max_graphs > 0:
        graph_files = graph_files[:max_graphs]

    for gf in graph_files:
        print(f"  Processing {gf.name}...", file=sys.stderr)
        try:
            gm = trace_full_graph(gf, device=device)
            ga = analyze_graph(gm, str(gf), gf.name)
        except Exception as e:
            ga = GraphAccounting(graph_path=str(gf), graph_name=gf.name)
            ga.errors.append(f"{type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)
        acc.graphs.append(ga)

    return acc


# ============================================================================
# Canonical repro mapping + timings
# ============================================================================

def build_hash_to_repro() -> dict[str, str]:
    """Map pattern_hash -> canonical repro dir name (authoritative: meta.json)."""
    mapping: dict[str, str] = {}
    if not CANONICAL_DIR.exists():
        return mapping
    for meta_path in sorted(CANONICAL_DIR.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        ph = meta.get("pattern_hash")
        if ph:
            mapping.setdefault(ph, meta_path.parent.name)
    return mapping


def _normalize_origin_ops(origin_ops: list[str]) -> tuple:
    """Normalize an origin-op list so trace-equivalent variants compare equal.

    make_fx retracing of saved full graphs canonicalizes some view ops
    (aten.reshape -> aten.view, drops no-op clones), which changes the exact
    DAG-signature hash without changing the kernel. Used ONLY to annotate
    UNMATCHED partitions with their likely canonical variant -- the exact
    pattern_hash remains the primary key.
    """
    normalized = []
    for op in origin_ops:
        if op == "aten.reshape.default":
            op = "aten.view.default"
        if op == "aten.clone.default":
            continue
        normalized.append(op)
    return tuple(sorted(normalized))


def build_opset_to_repros() -> dict[tuple, list[str]]:
    """Map normalized origin-op multiset -> canonical repro dir names."""
    mapping: dict[tuple, list[str]] = defaultdict(list)
    if not CANONICAL_DIR.exists():
        return mapping
    for meta_path in sorted(CANONICAL_DIR.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        origin_ops = meta.get("origin_ops")
        if origin_ops:
            mapping[_normalize_origin_ops(origin_ops)].append(meta_path.parent.name)
    return mapping


def load_timings(timings_path: Path) -> dict[str, dict]:
    """Map canonical repro dir name -> timing entry (oracle_us, compile_us)."""
    if not timings_path.exists():
        return {}
    try:
        data = json.loads(timings_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ============================================================================
# Reporting
# ============================================================================

def _rollup_occurrence_timings(
    timing: dict | None,
    occs: list["PartitionOccurrence"],
) -> dict:
    """Sum oracle/compile time over occurrences, matching each to ITS shape.

    The old roll-up booked every occurrence at one per-dir number (the min
    across all of a dir's shape points), undercounting by up to ~144x when a
    dir's points span a wide shape range. The correct roll-up looks up the
    timing for each occurrence's own ``shape_hash`` and sums.

    Uses ``timing['points_by_shape']`` (shape_hash -> {oracle_us, compile_us}).
    Occurrences whose shape_hash has no timed point fall back to the dir's
    representative ``oracle_us``/``compile_us`` (a MEDIAN, never the min). If
    ``points_by_shape`` is absent (older timings file) we warn and fall back to
    the legacy per-dir-number behaviour.

    Returns: {oracle_us_total, compile_us_total, matched_occ, total_occ, mode}
    where ``matched_occ`` is how many occurrences hit a per-shape point.
    """
    total_occ = len(occs)
    if not timing:
        return {
            "oracle_us_total": None,
            "compile_us_total": None,
            "matched_occ": 0,
            "total_occ": total_occ,
            "mode": "untimed",
        }

    rep_oracle = timing.get("oracle_us")
    rep_compile = timing.get("compile_us")
    by_shape = timing.get("points_by_shape")

    if not isinstance(by_shape, dict):
        # Backward compat: legacy timings file without per-shape data.
        if not _rollup_occurrence_timings._warned_legacy:
            print(
                "  [warn] timings lack 'points_by_shape'; falling back to "
                "legacy per-dir roll-up (oracle_us * occurrences) which can "
                "undercount wide-shape dirs. Re-run bench_parallel --oracles "
                "to regenerate.",
                file=sys.stderr,
            )
            _rollup_occurrence_timings._warned_legacy = True
        return {
            "oracle_us_total": (
                round(rep_oracle * total_occ, 2)
                if isinstance(rep_oracle, (int, float)) else None
            ),
            "compile_us_total": (
                round(rep_compile * total_occ, 2)
                if isinstance(rep_compile, (int, float)) else None
            ),
            "matched_occ": 0,
            "total_occ": total_occ,
            "mode": "legacy",
        }

    oracle_sum = 0.0
    compile_sum = 0.0
    have_oracle = False
    have_compile = False
    matched = 0
    for occ in occs:
        point = by_shape.get(occ.shape_hash)
        o_us = point.get("oracle_us") if isinstance(point, dict) else None
        c_us = point.get("compile_us") if isinstance(point, dict) else None
        if isinstance(o_us, (int, float)):
            matched += 1
        else:
            # No per-shape timing for this occurrence: use the dir median.
            o_us = rep_oracle if isinstance(rep_oracle, (int, float)) else None
        if not isinstance(c_us, (int, float)):
            c_us = rep_compile if isinstance(rep_compile, (int, float)) else None
        if isinstance(o_us, (int, float)):
            oracle_sum += o_us
            have_oracle = True
        if isinstance(c_us, (int, float)):
            compile_sum += c_us
            have_compile = True

    mode = "per_shape" if matched == total_occ else (
        "partial" if matched else "fallback_only"
    )
    return {
        "oracle_us_total": round(oracle_sum, 2) if have_oracle else None,
        "compile_us_total": round(compile_sum, 2) if have_compile else None,
        "matched_occ": matched,
        "total_occ": total_occ,
        "mode": mode,
    }


_rollup_occurrence_timings._warned_legacy = False


def _pattern_rows(acc: ModelAccounting, hash_to_repro: dict[str, str],
                  timings: dict[str, dict],
                  opset_to_repros: dict[tuple, list[str]]) -> list[dict]:
    """One row per unique pattern hash, with occurrence counts and shapes."""
    by_hash: dict[str, list[PartitionOccurrence]] = defaultdict(list)
    for occ in acc.all_occurrences:
        by_hash[occ.pattern_hash].append(occ)

    rows = []
    for phash, occs in by_hash.items():
        repro_dir = hash_to_repro.get(phash)
        rep = max(occs, key=lambda o: o.n_ops)
        # For UNMATCHED partitions, annotate likely trace-equivalent canonical
        # variants (reshape->view / clone-elision differences from retracing).
        variant_candidates = []
        if repro_dir is None:
            variant_candidates = [
                d for d in opset_to_repros.get(
                    _normalize_origin_ops(rep.origin_ops), []
                )
                if not d.endswith(f"_{phash}")
            ]
        timing = timings.get(repro_dir) if repro_dir else None
        oracle_us = None
        compile_us = None
        if timing:
            oracle_us = timing.get("oracle_us")
            compile_us = timing.get("compile_us")
        roll = _rollup_occurrence_timings(timing, occs)
        rows.append({
            "pattern_hash": phash,
            "repro_dir": repro_dir,
            "trace_equivalent_repro_dirs": variant_candidates,
            "occurrences": len(occs),
            "unique_shape_hashes": len({o.shape_hash for o in occs}),
            "n_ops": rep.n_ops,
            "kind": "reduction" if rep.has_reduction else "pointwise",
            "op_summary": dict(Counter(
                {op: c for o in [rep] for op, c in o.op_counts.items()}
            )),
            "representative_input_shapes": rep.input_shapes,
            "max_output_shape": rep.max_output_shape,
            "oracle_us": oracle_us,
            "compile_us": compile_us,
            "oracle_us_total": roll["oracle_us_total"],
            "compile_us_total": roll["compile_us_total"],
            "matched_occ": roll["matched_occ"],
            "total_occ": roll["total_occ"],
            "rollup_mode": roll["mode"],
        })
    rows.sort(key=lambda r: -r["occurrences"])
    return rows


def _non_fusible_rows(acc: ModelAccounting) -> list[dict]:
    """Group non-fusible ops by (target, shape) with counts."""
    groups: dict[tuple, list[NonFusibleOp]] = defaultdict(list)
    for op in acc.all_non_fusible:
        key = (op.target, tuple(op.output_shape) if op.output_shape else None)
        groups[key].append(op)
    rows = []
    for (target, shape), ops in groups.items():
        rows.append({
            "target": target,
            "output_shape": list(shape) if shape else None,
            "output_dtype": ops[0].output_dtype,
            "count": len(ops),
        })
    rows.sort(key=lambda r: (-r["count"], r["target"]))
    return rows


def build_model_report(acc: ModelAccounting, hash_to_repro: dict[str, str],
                       timings: dict[str, dict],
                       opset_to_repros: dict[tuple, list[str]]) -> dict:
    pattern_rows = _pattern_rows(acc, hash_to_repro, timings, opset_to_repros)
    nf_rows = _non_fusible_rows(acc)

    found_hashes = {r["pattern_hash"] for r in pattern_rows}
    manifest_set = set(acc.manifest_patterns)
    unmatched = [r for r in pattern_rows if r["repro_dir"] is None]
    unmatched_no_variant = [
        r for r in unmatched if not r["trace_equivalent_repro_dirs"]
    ]

    timed = [r for r in pattern_rows if r["oracle_us_total"] is not None]
    total_oracle_us = round(sum(r["oracle_us_total"] for r in timed), 2)
    total_compile_us = round(sum(
        r["compile_us_total"] for r in timed if r["compile_us_total"] is not None
    ), 2)
    # Per-shape coverage: how many occurrences (in timed patterns) were priced
    # at their OWN shape vs fell back to the dir median.
    matched_occ_total = sum(r.get("matched_occ", 0) for r in timed)
    timed_occ_total = sum(r.get("total_occ", r["occurrences"]) for r in timed)

    # For manifest patterns we did NOT re-find, check whether a re-found
    # partition is trace-equivalent (same ops modulo reshape->view / clone
    # canonicalization that make_fx retracing applies to saved graphs).
    opset_by_found_hash = {}
    for occ in acc.all_occurrences:
        opset_by_found_hash.setdefault(
            occ.pattern_hash, _normalize_origin_ops(occ.origin_ops)
        )
    manifest_unexplained = []
    manifest_explained = {}
    for mh in sorted(manifest_set - found_hashes):
        canon_meta = CANONICAL_DIR.glob(f"*_{mh}/meta.json")
        manifest_opset = None
        for mp in canon_meta:
            try:
                manifest_opset = _normalize_origin_ops(
                    json.loads(mp.read_text()).get("origin_ops", [])
                )
            except (json.JSONDecodeError, OSError):
                pass
            break
        equivalents = [
            fh for fh, fo in opset_by_found_hash.items()
            if manifest_opset is not None and fo == manifest_opset
        ]
        if equivalents:
            manifest_explained[mh] = equivalents
        else:
            manifest_unexplained.append(mh)

    return {
        "model_name": acc.model_name,
        "model_dir": acc.model_dir,
        "num_graphs": len(acc.graphs),
        "errors": [e for g in acc.graphs for e in g.errors],
        "total_partition_occurrences": len(acc.all_occurrences),
        "unique_patterns": len(found_hashes),
        "unmatched_patterns": len(unmatched),
        "unmatched_no_trace_equivalent": len(unmatched_no_variant),
        "total_non_fusible_ops": len(acc.all_non_fusible),
        # Aggregate time: SUM of (per-partition us x occurrence count).
        # Covers only patterns with canonical timings; never an averaged ratio.
        "fusible_oracle_us_total": total_oracle_us,
        "fusible_compile_us_total": total_compile_us,
        "timed_patterns": len(timed),
        "shape_matched_occurrences": matched_occ_total,
        "timed_occurrences": timed_occ_total,
        "manifest": {
            "patterns": sorted(manifest_set),
            "in_manifest_not_found": sorted(manifest_set - found_hashes),
            "in_manifest_not_found_trace_equivalent": manifest_explained,
            "in_manifest_not_found_unexplained": manifest_unexplained,
            "found_not_in_manifest": sorted(found_hashes - manifest_set),
        },
        "patterns": pattern_rows,
        "non_fusible_ops": nf_rows,
    }


def format_text_report(report: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"MODEL: {report['model_name']}")
    lines.append(f"DIR:   {report['model_dir']}")
    lines.append("=" * 78)
    lines.append(
        f"Graphs: {report['num_graphs']} | "
        f"partition occurrences: {report['total_partition_occurrences']} | "
        f"unique patterns: {report['unique_patterns']} | "
        f"UNMATCHED: {report['unmatched_patterns']} "
        f"(of which {report['unmatched_no_trace_equivalent']} have no "
        f"trace-equivalent canonical repro) | "
        f"non-fusible ops: {report['total_non_fusible_ops']}"
    )
    if report["timed_patterns"]:
        lines.append(
            f"Fusible time projection (SUM of per-shape us over "
            f"{report['timed_patterns']} timed patterns): "
            f"oracle={report['fusible_oracle_us_total']}us "
            f"compile={report['fusible_compile_us_total']}us"
        )
        lines.append(
            f"  Per-shape coverage: {report['shape_matched_occurrences']}/"
            f"{report['timed_occurrences']} occurrences priced at own shape "
            f"(rest fell back to dir median)"
        )
    if report["errors"]:
        lines.append(f"ERRORS: {report['errors']}")
    lines.append("")

    lines.append("Fusible partitions (capture-pipeline partitioner, exact hashes):")
    lines.append(
        f"  {'pattern_hash':14s} {'occur':>5s} {'shapes':>6s} "
        f"{'kind':9s} {'oracle_us':>9s} {'sum_us':>10s}  canonical repro dir"
    )
    for r in report["patterns"]:
        oracle = f"{r['oracle_us']:.1f}" if r["oracle_us"] is not None else "-"
        total = f"{r['oracle_us_total']:.1f}" if r["oracle_us_total"] is not None else "-"
        repro = r["repro_dir"] or "*** UNMATCHED ***"
        lines.append(
            f"  {r['pattern_hash']:14s} {r['occurrences']:5d} "
            f"{r['unique_shape_hashes']:6d} {r['kind']:9s} "
            f"{oracle:>9s} {total:>10s}  {repro}"
        )
        if r["repro_dir"] is None and r["trace_equivalent_repro_dirs"]:
            lines.append(
                f"  {'':14s} trace-equivalent variant(s): "
                f"{', '.join(r['trace_equivalent_repro_dirs'])}"
            )
        shapes = ", ".join(r["representative_input_shapes"][:4])
        if len(r["representative_input_shapes"]) > 4:
            shapes += f", ... +{len(r['representative_input_shapes']) - 4} more"
        lines.append(f"  {'':14s} inputs: {shapes}")
    lines.append("")

    if report["manifest"]["patterns"]:
        m = report["manifest"]
        lines.append(
            f"Manifest cross-check: {len(m['patterns'])} manifest patterns"
        )
        if m["in_manifest_not_found"]:
            lines.append(
                f"  in manifest but NOT re-found: {m['in_manifest_not_found']}"
            )
            for mh, eq in m["in_manifest_not_found_trace_equivalent"].items():
                lines.append(
                    f"    {mh} is trace-equivalent to re-found {eq} "
                    f"(reshape->view / clone canonicalization on retrace)"
                )
            if m["in_manifest_not_found_unexplained"]:
                lines.append(
                    f"    UNEXPLAINED (true discrepancy): "
                    f"{m['in_manifest_not_found_unexplained']}"
                )
        if m["found_not_in_manifest"]:
            lines.append(
                f"  found but NOT in manifest:    {m['found_not_in_manifest']}"
            )
        if not m["in_manifest_not_found"] and not m["found_not_in_manifest"]:
            lines.append("  EXACT MATCH with manifest")
        lines.append("")

    lines.append("Non-fusible ops (with shapes from node.meta['val']):")
    for r in report["non_fusible_ops"]:
        lines.append(
            f"  {r['target']:55s} -> {r['output_shape']} "
            f"{r['output_dtype'] or ''} x{r['count']}"
        )
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Discovery
# ============================================================================

def discover_model_dirs() -> list[Path]:
    """Find all model directories that contain full_graph_*.py files."""
    if not MODELS_DIR.exists():
        return []
    dirs = set()
    for fg in MODELS_DIR.rglob("full_graph_*.py"):
        dirs.add(fg.parent)
    return sorted(dirs)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model-level graph accounting via the capture pipeline's partitioner"
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Model name filter (partial match). E.g. 'convnextv2' or 'BERT_pytorch'",
    )
    parser.add_argument(
        "--all", "-a", action="store_true",
        help="Process all models (can be slow)",
    )
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of text report")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file path (default: stdout)")
    parser.add_argument(
        "--device", default="cuda",
        help="Device for input construction (cuda or cpu); tracing is fake-mode",
    )
    parser.add_argument("--max-graphs", type=int, default=0,
                        help="Max graphs to process per model (0 = all)")
    parser.add_argument(
        "--timings", default=str(DEFAULT_TIMINGS),
        help="Per-oracle timings JSON (repro dir name -> oracle_us) for "
             "SUM(us x occurrences) projection",
    )
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Specify --model <name> or --all")

    model_dirs = discover_model_dirs()
    print(f"Found {len(model_dirs)} model directories with full graphs",
          file=sys.stderr)

    if args.model:
        model_dirs = [
            d for d in model_dirs if args.model.lower() in d.name.lower()
        ]
        if not model_dirs:
            print(f"No models matching '{args.model}'", file=sys.stderr)
            sys.exit(1)

    print(f"Processing {len(model_dirs)} model(s)...", file=sys.stderr)

    hash_to_repro = build_hash_to_repro()
    print(f"Loaded {len(hash_to_repro)} canonical repro mappings", file=sys.stderr)
    opset_to_repros = build_opset_to_repros()
    timings = load_timings(Path(args.timings))
    print(f"Loaded {len(timings)} oracle timings", file=sys.stderr)

    reports = []
    for model_dir in model_dirs:
        print(f"\nProcessing: {model_dir.name}", file=sys.stderr)
        acc = process_model_directory(
            model_dir, device=args.device, max_graphs=args.max_graphs
        )
        reports.append(
            build_model_report(acc, hash_to_repro, timings, opset_to_repros)
        )

    if args.json:
        output_text = json.dumps(
            reports[0] if len(reports) == 1 else reports, indent=2
        )
    else:
        output_text = "\n\n".join(format_text_report(r) for r in reports)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_text + "\n")
        print(f"\nOutput written to {args.output}", file=sys.stderr)
    else:
        print(output_text)

    print(f"\n{'=' * 70}", file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)
    for r in reports:
        err_str = f" ({len(r['errors'])} errors)" if r["errors"] else ""
        print(
            f"  {r['model_name']}: {r['num_graphs']} graphs, "
            f"{r['total_partition_occurrences']} partition occurrences, "
            f"{r['unique_patterns']} unique patterns "
            f"({r['unmatched_patterns']} unmatched), "
            f"{r['total_non_fusible_ops']} non-fusible ops"
            f"{err_str}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
