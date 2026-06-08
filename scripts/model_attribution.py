#!/usr/bin/env python3
"""Per-model performance attribution tool.

For model X, answers: "How much time is spent in each partition, and what's
the achievable floor?"

TWO APPROACHES:

Approach A (Static accounting, bottom-up):
  - Map model graphs -> canonical repros via manifest pattern hashes
  - Look up measured compile_us and oracle_us for each partition
  - Sum: predicted_model_time = sum(partition_time * count) + non_fusible_time
  - Sum: achievable_model_time = sum(min(oracle_us, compile_us) * count) + non_fusible

Approach B (End-to-end, top-down):
  - Load full_graph_*.py, torch.compile, time the whole graph
  - Compare vs sum-of-partitions from Approach A
  - Gap reveals cross-graph overhead (dispatch, memory pressure, partition boundaries)

LIMITATIONS:
  - We do NOT have per-shape oracles. Each repro was captured at ONE specific shape.
    If the model runs at a different shape (batch size, seq length), our oracle
    measurement doesn't directly apply.
  - Non-fusible ops (matmul, conv, embedding lookups) are NOT in the repro corpus.
    They appear as "non-captured" overhead.
  - Approach B requires GPU hardware and pytorch installation with CUDA.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Project root
ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = ROOT / "repros" / "canonical"
MODELS_DIR = ROOT / "repros" / "models"
RESULTS_DIR = ROOT / "investigation_results"


@dataclass
class PartitionInfo:
    """Info about one partition (pattern) in a model graph."""
    pattern_hash: str
    repro_id: str  # e.g. "var_mean_0990c69ae9bb"
    compile_us: float | None = None
    oracle_us: float | None = None
    best_compile_us: float | None = None
    classification: str = "unknown"
    kind: str = "unknown"  # pointwise, reduction, etc.
    n_ops: int = 0
    captured: bool = True  # False if pattern hash has no matching canonical repro


@dataclass
class ModelGraphInfo:
    """Info about one model graph (one manifest)."""
    graph_name: str  # e.g. "torchbench_hf_Bert_infer_000"
    partitions: list[PartitionInfo] = field(default_factory=list)
    has_full_graph: bool = False
    full_graph_path: str | None = None


@dataclass
class ModelAttribution:
    """Full attribution for a model."""
    model_name: str  # e.g. "torchbench_hf_Bert_infer"
    graphs: list[ModelGraphInfo] = field(default_factory=list)

    @property
    def total_partitions(self) -> int:
        return sum(len(g.partitions) for g in self.graphs)

    @property
    def captured_partitions(self) -> int:
        return sum(1 for g in self.graphs for p in g.partitions if p.captured)

    @property
    def has_timing(self) -> int:
        return sum(1 for g in self.graphs for p in g.partitions
                   if p.captured and p.best_compile_us is not None)

    @property
    def has_oracle(self) -> int:
        return sum(1 for g in self.graphs for p in g.partitions
                   if p.captured and p.oracle_us is not None)

    @property
    def sum_compile_us(self) -> float:
        total = 0.0
        for g in self.graphs:
            for p in g.partitions:
                if p.best_compile_us is not None:
                    total += p.best_compile_us
        return total

    @property
    def sum_achievable_us(self) -> float:
        """Best of oracle_us and best_compile_us for each partition."""
        total = 0.0
        for g in self.graphs:
            for p in g.partitions:
                if p.best_compile_us is not None:
                    if p.oracle_us is not None and p.oracle_us < p.best_compile_us:
                        total += p.oracle_us
                    else:
                        total += p.best_compile_us
        return total

    @property
    def sum_gap_us(self) -> float:
        return self.sum_compile_us - self.sum_achievable_us

    @property
    def speedup(self) -> float:
        achievable = self.sum_achievable_us
        if achievable <= 0:
            return 1.0
        return self.sum_compile_us / achievable


def build_hash_to_repro_id() -> dict[str, str]:
    """Map pattern_hash -> repro_id from canonical directory names."""
    mapping = {}
    if not CANONICAL_DIR.exists():
        return mapping
    for d in CANONICAL_DIR.iterdir():
        if d.is_dir():
            parts = d.name.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) == 12:
                mapping[parts[1]] = d.name
    return mapping


def load_baseline_results() -> dict[str, dict[str, Any]]:
    """Load baseline_results.csv into dict keyed by repro_id."""
    csv_path = RESULTS_DIR / "baseline_results.csv"
    if not csv_path.exists():
        return {}
    results = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["repro_id"]] = row
    return results


def load_oracle_floors() -> dict[str, float]:
    """Load measured_oracle_floors.csv -> repro_id: oracle_us."""
    csv_path = RESULTS_DIR / "measured_oracle_floors.csv"
    if not csv_path.exists():
        return {}
    floors = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            oracle_us = row.get("oracle_us", "")
            if oracle_us:
                try:
                    floors[row["repro_id"]] = float(oracle_us)
                except ValueError:
                    pass
    return floors


def load_oracle_tracker() -> dict[str, dict[str, Any]]:
    """Load oracle_vs_compile_tracker.csv for classification."""
    csv_path = RESULTS_DIR / "oracle_vs_compile_tracker.csv"
    if not csv_path.exists():
        return {}
    tracker = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tracker[row["repro_id"]] = row
    return tracker


def load_repro_meta(repro_id: str) -> dict[str, Any]:
    """Load meta.json for a canonical repro."""
    meta_path = CANONICAL_DIR / repro_id / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def discover_model_manifests() -> dict[str, list[tuple[str, list[str]]]]:
    """Discover all model manifests.

    Returns: {model_base_name: [(graph_name, [pattern_hashes]), ...]}
    """
    if not MODELS_DIR.exists():
        return {}

    raw_graphs: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)

    for manifest_path in sorted(MODELS_DIR.rglob("manifest.json")):
        try:
            data = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        graph_name = manifest_path.parent.name
        patterns = data.get("patterns", [])
        # Derive base model name by stripping trailing _NNN
        base = re.sub(r"_\d{3}$", "", graph_name)
        raw_graphs[base].append((graph_name, patterns))

    return dict(raw_graphs)


def discover_full_graphs() -> dict[str, list[Path]]:
    """Find full_graph_*.py files grouped by model directory."""
    full_graphs: dict[str, list[Path]] = defaultdict(list)
    if not MODELS_DIR.exists():
        return full_graphs
    for fg in sorted(MODELS_DIR.rglob("full_graph_*.py")):
        model_dir = fg.parent.name
        full_graphs[model_dir].append(fg)
    return dict(full_graphs)


def build_attribution(
    model_name: str,
    graphs: list[tuple[str, list[str]]],
    hash_to_repro: dict[str, str],
    baseline: dict[str, dict[str, Any]],
    oracle_floors: dict[str, float],
    oracle_tracker: dict[str, dict[str, Any]],
    full_graphs: dict[str, list[Path]],
) -> ModelAttribution:
    """Build full attribution for one model."""
    attr = ModelAttribution(model_name=model_name)

    for graph_name, pattern_hashes in graphs:
        ginfo = ModelGraphInfo(graph_name=graph_name)

        # Check for full graph files
        fg_list = full_graphs.get(graph_name, [])
        if fg_list:
            ginfo.has_full_graph = True
            ginfo.full_graph_path = str(fg_list[0])

        for phash in pattern_hashes:
            repro_id = hash_to_repro.get(phash)
            if repro_id is None:
                # Pattern hash not found in canonical repros
                pinfo = PartitionInfo(
                    pattern_hash=phash,
                    repro_id=f"UNKNOWN_{phash}",
                    captured=False,
                )
                ginfo.partitions.append(pinfo)
                continue

            pinfo = PartitionInfo(
                pattern_hash=phash,
                repro_id=repro_id,
                captured=True,
            )

            # Load meta for classification
            meta = load_repro_meta(repro_id)
            pinfo.kind = meta.get("kind", "unknown")
            pinfo.n_ops = meta.get("n_ops", 0)

            # Load timing from baseline
            brow = baseline.get(repro_id)
            if brow:
                try:
                    pinfo.compile_us = float(brow["compiled_us"])
                except (ValueError, KeyError):
                    pass
                try:
                    pinfo.best_compile_us = float(brow["best_compile_us"])
                except (ValueError, KeyError):
                    pass

            # Load oracle floor
            if repro_id in oracle_floors:
                pinfo.oracle_us = oracle_floors[repro_id]

            # Load classification from tracker
            trow = oracle_tracker.get(repro_id)
            if trow:
                pinfo.classification = trow.get("family", "unknown")
            elif brow:
                # Fallback: use pattern_prefix as classification
                pinfo.classification = brow.get("pattern_prefix", "unknown")

            ginfo.partitions.append(pinfo)

        attr.graphs.append(ginfo)

    return attr


def format_attribution_report(attr: ModelAttribution, verbose: bool = False) -> str:
    """Format a human-readable attribution report."""
    lines = []
    lines.append(f"# Model Attribution: {attr.model_name}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Graphs: {len(attr.graphs)}")
    lines.append(f"- Total partitions: {attr.total_partitions}")
    lines.append(f"- Captured in corpus: {attr.captured_partitions}")
    lines.append(f"- With timing data: {attr.has_timing}")
    lines.append(f"- With oracle floor: {attr.has_oracle}")
    lines.append("")

    if attr.has_timing > 0:
        lines.append("## Approach A: Static Accounting (Bottom-Up)")
        lines.append("")
        lines.append(f"- Sum compile time (best config): {attr.sum_compile_us:.1f} us")
        lines.append(f"- Achievable floor (min of oracle/compile): {attr.sum_achievable_us:.1f} us")
        lines.append(f"- Gap (opportunity): {attr.sum_gap_us:.1f} us")
        lines.append(f"- Potential speedup: {attr.speedup:.3f}x")
        lines.append("")
        lines.append("### Limitations")
        lines.append("")
        lines.append("- Oracles measured at ONE specific shape only; model may run at different shapes")
        lines.append("- Non-fusible ops (matmul, conv, etc.) not captured in repro corpus")
        lines.append("- No cross-partition overhead accounted for (dispatch, memory pressure)")
        non_captured = attr.total_partitions - attr.captured_partitions
        if non_captured > 0:
            lines.append(f"- {non_captured} partition(s) not found in canonical corpus")
        lines.append("")

    # Per-partition detail table
    lines.append("## Per-Partition Breakdown")
    lines.append("")
    lines.append("| # | Graph | Repro ID | Kind | Compile (us) | Oracle (us) | Gap (us) | Classification |")
    lines.append("|---|-------|----------|------|-------------|-------------|----------|----------------|")

    idx = 0
    for g in attr.graphs:
        for p in g.partitions:
            idx += 1
            compile_str = f"{p.best_compile_us:.1f}" if p.best_compile_us else "N/A"
            oracle_str = f"{p.oracle_us:.1f}" if p.oracle_us else "N/A"
            if p.best_compile_us and p.oracle_us:
                gap = p.best_compile_us - min(p.oracle_us, p.best_compile_us)
                gap_str = f"{gap:.1f}"
            else:
                gap_str = "N/A"
            captured_mark = "" if p.captured else " [NOT CAPTURED]"
            lines.append(
                f"| {idx} | {g.graph_name} | {p.repro_id}{captured_mark} | "
                f"{p.kind} | {compile_str} | {oracle_str} | {gap_str} | "
                f"{p.classification} |"
            )

    lines.append("")

    # Classification summary
    class_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"compile": 0.0, "achievable": 0.0, "count": 0}
    )
    for g in attr.graphs:
        for p in g.partitions:
            if p.best_compile_us is not None:
                cls = p.classification
                class_totals[cls]["compile"] += p.best_compile_us
                if p.oracle_us is not None and p.oracle_us < p.best_compile_us:
                    class_totals[cls]["achievable"] += p.oracle_us
                else:
                    class_totals[cls]["achievable"] += p.best_compile_us
                class_totals[cls]["count"] += 1

    if class_totals:
        lines.append("## Time by Classification")
        lines.append("")
        lines.append("| Classification | Count | Compile (us) | Achievable (us) | Gap (us) |")
        lines.append("|---------------|-------|-------------|-----------------|----------|")
        for cls, totals in sorted(class_totals.items(), key=lambda x: -(x[1]["compile"] - x[1]["achievable"])):
            gap = totals["compile"] - totals["achievable"]
            lines.append(
                f"| {cls} | {int(totals['count'])} | "
                f"{totals['compile']:.1f} | {totals['achievable']:.1f} | {gap:.1f} |"
            )
        lines.append("")

    # Approach B note
    has_full = any(g.has_full_graph for g in attr.graphs)
    if has_full:
        lines.append("## Approach B: End-to-End (Top-Down)")
        lines.append("")
        lines.append("Full graph files available for end-to-end benchmarking:")
        for g in attr.graphs:
            if g.has_full_graph:
                lines.append(f"  - {g.full_graph_path}")
        lines.append("")
        lines.append("Run with: `python scripts/model_attribution.py --model <name> --bench`")
        lines.append("")
        lines.append("NOTE: Approach B requires GPU. Without per-shape oracles, the comparison")
        lines.append("between A and B reveals cross-graph overhead but the 'achievable' number")
        lines.append("from A may not match the actual achievable with correct-shape oracles.")
    else:
        lines.append("## Approach B: End-to-End (Top-Down)")
        lines.append("")
        lines.append("No full_graph_*.py files found for this model.")
        lines.append("End-to-end timing not available.")

    lines.append("")
    return "\n".join(lines)


def run_approach_b(attr: ModelAttribution) -> dict[str, Any]:
    """Run Approach B: compile and time full graphs.

    Returns timing results dict. Requires GPU.
    """
    try:
        import torch
        import torch._inductor.config as cfg
    except ImportError:
        return {"error": "PyTorch not available"}

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    sys.path.insert(0, str(ROOT))
    from full_graph_harness import load_full_graph

    results = {}
    for g in attr.graphs:
        if not g.has_full_graph:
            continue

        try:
            instance, inputs, definition = load_full_graph(g.full_graph_path)
            instance = instance.cuda().eval()

            # Compile
            cfg.coordinate_descent_tuning = True
            cfg.combo_kernels = True
            compiled = torch.compile(instance)

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    compiled(*inputs)
            torch.cuda.synchronize()

            # Time
            import time
            torch.cuda.synchronize()
            start = time.perf_counter()
            n_rep = 10
            with torch.no_grad():
                for _ in range(n_rep):
                    compiled(*inputs)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / n_rep * 1e6  # us

            results[g.graph_name] = {
                "full_graph_us": elapsed,
                "path": g.full_graph_path,
            }
        except Exception as e:
            results[g.graph_name] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Per-model performance attribution tool"
    )
    parser.add_argument(
        "--model", "-m",
        default="all",
        help="Model name (or 'all' for all models). Partial matches supported.",
    )
    parser.add_argument(
        "--bench", "-b",
        action="store_true",
        help="Run Approach B (end-to-end GPU benchmarking). Requires CUDA.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output CSV format instead of markdown report.",
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=0,
        help="Show only top N models by gap (0 = all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include per-graph detail",
    )
    args = parser.parse_args()

    # Load data
    print("Loading data...", file=sys.stderr)
    hash_to_repro = build_hash_to_repro_id()
    baseline = load_baseline_results()
    oracle_floors = load_oracle_floors()
    oracle_tracker = load_oracle_tracker()
    model_manifests = discover_model_manifests()
    full_graphs = discover_full_graphs()

    print(f"  Canonical repros: {len(hash_to_repro)}", file=sys.stderr)
    print(f"  Baseline results: {len(baseline)}", file=sys.stderr)
    print(f"  Oracle floors: {len(oracle_floors)}", file=sys.stderr)
    print(f"  Model manifests: {len(model_manifests)}", file=sys.stderr)
    print(f"  Full graph dirs: {len(full_graphs)}", file=sys.stderr)

    # Filter models
    if args.model == "all":
        selected_models = list(model_manifests.keys())
    else:
        # Partial match
        selected_models = [
            m for m in model_manifests.keys()
            if args.model.lower() in m.lower()
        ]
        if not selected_models:
            print(f"No models matching '{args.model}'. Available:", file=sys.stderr)
            for m in sorted(model_manifests.keys())[:20]:
                print(f"  {m}", file=sys.stderr)
            sys.exit(1)

    # Build attributions
    attributions = []
    for model_name in sorted(selected_models):
        graphs = model_manifests[model_name]
        attr = build_attribution(
            model_name, graphs, hash_to_repro, baseline,
            oracle_floors, oracle_tracker, full_graphs,
        )
        attributions.append(attr)

    # Sort by gap (descending)
    attributions.sort(key=lambda a: -a.sum_gap_us)

    if args.top > 0:
        attributions = attributions[:args.top]

    # Output
    if args.csv:
        output_csv(attributions, args.output)
    else:
        output_report(attributions, args.output, bench=args.bench, verbose=args.verbose)


def output_csv(attributions: list[ModelAttribution], output_path: str | None):
    """Output CSV summary."""
    out = open(output_path, "w", newline="") if output_path else sys.stdout
    writer = csv.writer(out)
    writer.writerow([
        "model", "n_graphs", "n_partitions", "n_captured", "n_with_timing",
        "n_with_oracle", "sum_compile_us", "sum_achievable_us", "gap_us", "speedup",
    ])
    for attr in attributions:
        writer.writerow([
            attr.model_name,
            len(attr.graphs),
            attr.total_partitions,
            attr.captured_partitions,
            attr.has_timing,
            attr.has_oracle,
            f"{attr.sum_compile_us:.1f}",
            f"{attr.sum_achievable_us:.1f}",
            f"{attr.sum_gap_us:.1f}",
            f"{attr.speedup:.3f}",
        ])
    if output_path:
        out.close()


def output_report(
    attributions: list[ModelAttribution],
    output_path: str | None,
    bench: bool = False,
    verbose: bool = False,
):
    """Output markdown report."""
    lines = []
    lines.append("# Model Performance Attribution Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"Models analyzed: {len(attributions)}")
    total_compile = sum(a.sum_compile_us for a in attributions)
    total_achievable = sum(a.sum_achievable_us for a in attributions)
    total_gap = total_compile - total_achievable
    lines.append(f"Total compile time across all models: {total_compile:.1f} us")
    lines.append(f"Total achievable time: {total_achievable:.1f} us")
    lines.append(f"Total gap (opportunity): {total_gap:.1f} us")
    if total_achievable > 0:
        lines.append(f"Overall potential speedup: {total_compile/total_achievable:.3f}x")
    lines.append("")
    lines.append("### Key Limitations")
    lines.append("")
    lines.append("1. **No per-shape oracles**: Each repro measured at ONE specific shape.")
    lines.append("   Model may run at different batch/seq shapes; oracle doesn't directly apply.")
    lines.append("2. **Non-fusible ops not captured**: matmul, conv, embedding are not in corpus.")
    lines.append("3. **No cross-partition overhead**: dispatch, memory pressure, boundaries not modeled.")
    lines.append("4. **Approach B can be totally off** without per-shape measurements,")
    lines.append("   since the full graph runs at the model's actual shape while partition")
    lines.append("   timings come from potentially different shapes.")
    lines.append("")

    # Summary table
    lines.append("## Summary Table (sorted by gap)")
    lines.append("")
    lines.append("| Model | Graphs | Partitions | Compile (us) | Achievable (us) | Gap (us) | Speedup |")
    lines.append("|-------|--------|-----------|-------------|-----------------|----------|---------|")
    for attr in attributions:
        if attr.sum_compile_us > 0:
            lines.append(
                f"| {attr.model_name} | {len(attr.graphs)} | "
                f"{attr.total_partitions} | {attr.sum_compile_us:.1f} | "
                f"{attr.sum_achievable_us:.1f} | {attr.sum_gap_us:.1f} | "
                f"{attr.speedup:.3f}x |"
            )
    lines.append("")

    # Detailed per-model reports
    if verbose or len(attributions) <= 5:
        lines.append("## Detailed Per-Model Reports")
        lines.append("")
        for attr in attributions:
            if attr.sum_compile_us > 0:
                lines.append(format_attribution_report(attr, verbose=verbose))
                lines.append("---")
                lines.append("")

    # Approach B results
    if bench:
        lines.append("## Approach B: End-to-End Results")
        lines.append("")
        lines.append("| Model | Graph | Full Graph (us) | Sum of Parts (us) | Overhead |")
        lines.append("|-------|-------|-----------------|-------------------|----------|")
        for attr in attributions:
            b_results = run_approach_b(attr)
            for gname, result in b_results.items():
                if "error" in result:
                    lines.append(f"| {attr.model_name} | {gname} | ERROR: {result['error']} | - | - |")
                else:
                    fg_us = result["full_graph_us"]
                    # Sum partitions for this graph
                    sum_parts = 0.0
                    for g in attr.graphs:
                        if g.graph_name == gname:
                            for p in g.partitions:
                                if p.best_compile_us:
                                    sum_parts += p.best_compile_us
                    overhead = fg_us - sum_parts if sum_parts > 0 else 0
                    lines.append(
                        f"| {attr.model_name} | {gname} | {fg_us:.1f} | "
                        f"{sum_parts:.1f} | {overhead:.1f} |"
                    )
        lines.append("")
        lines.append("NOTE: Without per-shape oracles, this comparison has limited validity.")
        lines.append("The gap between full-graph time and sum-of-partitions reveals cross-graph")
        lines.append("overhead but shape mismatches can make the numbers incomparable.")

    output = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(output + "\n")
        print(f"Report written to {output_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
