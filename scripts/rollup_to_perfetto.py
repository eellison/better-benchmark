#!/usr/bin/env python3
"""Export a model's per-kernel rollup as a Perfetto / Chrome-trace JSON, in the
model's REAL GRAPH-EXECUTION ORDER (a reconstructed invocation timeline).

A model's kernel rollup is, in effect, a flame profile: a sequence of kernels
whose per-kernel times sum to an end-to-end estimate. This tool turns that into
a Chrome Trace Event Format JSON so you can SEE it in Perfetto / chrome://tracing
-- and, crucially, visually compare the ORACLE reference-kernel ceiling against
the CURRENT torch.compile timing on the SAME kernels, in the order they run. The
gap between the two track lengths IS the headroom.

WHY EXECUTION ORDER (not an aggregated histogram):
A Perfetto trace should look like the model actually ran -- conv -> bn -> ... ->
matmul -> softmax, each kernel timed in the order it executes -- not "pattern A
x22 laid end-to-end, then pattern B x5". So this tool does NOT consume the
AGGREGATED ``model_graph_accounting --json`` pattern_rows (those are sorted by
occurrence count, which destroys order). Instead it IMPORTS the accounting
module and re-runs its partition pass, then walks each graph's nodes in
TOPOLOGICAL (execution) order, emitting one slice per item -- a fused partition
= one slice, a non-fusible op = one extern slice -- interleaved exactly as the
graph executes.

----------------------------------------------------------------------------
TRACE LAYOUT (Chrome Trace Event Format -- the format Perfetto UI ingests)
----------------------------------------------------------------------------
One process per model. Within a model, three tracks (thread rows):

  tid 1  "oracle (reference ceiling)"   -- the execution-ordered kernel
                                           timeline, each fused-partition slice
                                           dur = its oracle_us. Track length =
                                           summed oracle time.
  tid 2  "compile (current torch.compile)" -- the SAME ordered timeline, each
                                           slice dur = its compile_us (current
                                           coord-descent perf). Track length =
                                           summed compile time. compile_total >
                                           oracle_total => headroom.
  tid 3  "extern (NOT an oracle target)"   -- non-fusible ops (conv/GEMM/SDPA/
                                           backward) at their execution position.
                                           These have NO oracle/compile timing in
                                           the corpus (they are the
                                           inductor-invariant denominator), so
                                           they are drawn as fixed-width markers,
                                           labeled "extern, untimed", honestly
                                           distinct from the timed tracks.

Slices on the oracle/compile tracks are laid end-to-end in graph-node order
(ts advances by each slice's dur). Multi-graph models are laid sequentially
(graph 0, then graph 1, ...) with a graph-boundary marker between graphs, so the
whole timeline reads as one reconstructed invocation.

ts / dur are in MICROSECONDS (the corpus us values used directly);
displayTimeUnit "ns" is only a display hint -- relative lengths (the gap) are
what matter.

----------------------------------------------------------------------------
SOL TRACK -- OUT OF SCOPE FOR NOW (deliberate)
----------------------------------------------------------------------------
A fourth track "SOL (bandwidth floor)" would slot in as tid 4 using each
kernel's memcopy_sol_us (speed-of-light memory-bandwidth floor). That datum is
NOT in the per-oracle timings file used here; it lives in the per-(repro,shape)
STATE JSONs under results/pytorch_landing/A1_commit_walk/states/*.json (field
memcopy_sol_us). See _sol_track_hook(). NOT implemented on purpose -- a marked
hook only.

----------------------------------------------------------------------------
WHICH TIMINGS INPUT TO USE
----------------------------------------------------------------------------
A --timings JSON keyed by canonical repro-dir name with oracle_us / compile_us
per dir. Use:

    results/all_oracle_timings_b200_v2.json

This PRICES models like mobilenetv2_100 and repvgg_a2 (verified). A kernel whose
canonical dir is absent from the timings file is UNPRICED: it still appears on
the timeline as a 0-dur marker labeled "unpriced" so the order is honest, but it
adds nothing to the track totals. If a whole model traces to empty timed tracks,
you picked a timings file that does not price it.

----------------------------------------------------------------------------
HOW TO VIEW
----------------------------------------------------------------------------
Drag the emitted .json into https://ui.perfetto.dev  (or open chrome://tracing
and Load the file). The "oracle" and "compile" rows under the model process show
the two execution-ordered timelines; the length difference is the headroom.

----------------------------------------------------------------------------
USAGE
----------------------------------------------------------------------------
  python scripts/rollup_to_perfetto.py \
      --model mobilenetv2_100 \
      --timings results/all_oracle_timings_b200_v2.json \
      --out /tmp/scratch_space/mobilenetv2_100.perfetto.json

  # all models -> one trace, one process per model
  python scripts/rollup_to_perfetto.py --all \
      --timings results/all_oracle_timings_b200_v2.json \
      --out /tmp/scratch_space/all_models.perfetto.json

This re-runs model_graph_accounting's partition pass (needs the same env: the
graphs trace fake-mode but input construction defaults to --device cuda).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model_graph_accounting as mga  # noqa: E402  (REUSE its partition pass)

DEFAULT_TIMINGS = ROOT / "results" / "all_oracle_timings_b200_v2.json"

# Track (thread) ids within each model process.
TID_ORACLE = 1
TID_COMPILE = 2
TID_EXTERN = 3
# TID_SOL = 4  # reserved -- see _sol_track_hook(); NOT implemented.

TRACK_ORACLE = "oracle (reference ceiling)"
TRACK_COMPILE = "compile (current torch.compile)"
TRACK_EXTERN = "extern (NOT an oracle target)"

# Fixed visual width (us) for extern marker slices -- they have no real timing
# in the corpus, so we draw them as uniform markers, not as a summed timeline.
EXTERN_MARKER_US = 10.0


# ===========================================================================
# Ordered execution stream (REUSES the accounting module's partition pass)
# ===========================================================================

class OrderedItem:
    """One item in a graph's execution-ordered stream.

    kind == "kernel": a fused partition (one Perfetto slice on oracle+compile).
    kind == "extern": a non-fusible op (one slice on the extern track).
    """

    __slots__ = ("kind", "name", "pattern_hash", "shape_hash", "repro_dir",
                 "n_ops", "is_reduction", "op_summary", "input_shapes",
                 "output_shape", "output_dtype", "target")

    def __init__(self, kind: str, **kw: Any) -> None:
        self.kind = kind
        for s in self.__slots__:
            if s != "kind":
                setattr(self, s, kw.get(s))


def _ordered_stream_for_graph(gm, hash_to_repro: dict[str, str]) -> list[OrderedItem]:
    """Walk one traced graph in TOPOLOGICAL (execution) node order and produce
    an interleaved stream of fused-partition kernels and non-fusible externs.

    Reuses capture_hook's partitioner (via the same helpers the accounting
    module uses). A fused partition is emitted ONCE, at the graph position of
    its LAST (output-most) node, so its slice sits where the fused kernel would
    actually run relative to surrounding externs.
    """
    from capture_hook import (
        compute_partition_pattern,
        get_fusion_partitions,
        partition_has_reduction,
        partition_node_is_supported,
    )

    # Stable graph-order index for every node.
    node_order = {n: i for i, n in enumerate(gm.graph.nodes)}

    # Build node -> partition payload, and find each partition's anchor (the
    # member node with the greatest graph-order index = where it "completes").
    partitioned: dict = {}        # node -> partition_id
    anchor_to_payload: dict = {}  # anchor_node -> OrderedItem
    components = get_fusion_partitions(gm)
    for pid, comp in enumerate(components):
        pattern = compute_partition_pattern(comp, gm)
        if pattern is None:
            continue
        for n in comp:
            partitioned[n] = pid
        anchor = max(comp, key=lambda n: node_order[n])
        op_counts = Counter(
            mga._target_name(n.target) for n in comp if n.op == "call_function"
        )
        input_shapes = sorted(
            f"{info.get('shape', '?')}:{info.get('dtype', '?')}"
            for info in pattern["placeholder_info"].values()
        )
        max_numel, max_shape = -1, None
        for n in comp:
            shape = mga._safe_shape(n.meta.get("val"))
            if shape is None:
                continue
            numel = 1
            for d in shape:
                numel *= max(d, 1)
            if numel > max_numel:
                max_numel, max_shape = numel, shape
        phash = pattern["pattern_hash"]
        anchor_to_payload[anchor] = OrderedItem(
            "kernel",
            name=hash_to_repro.get(phash) or phash,
            pattern_hash=phash,
            shape_hash=pattern["shape_hash"],
            repro_dir=hash_to_repro.get(phash),
            n_ops=len(comp),
            is_reduction=partition_has_reduction(comp),
            op_summary=dict(op_counts),
            input_shapes=input_shapes,
            output_shape=max_shape,
        )

    # Walk the graph once in order. Emit a kernel at its anchor; emit an extern
    # for every non-fusible call_function node in place. Skip transparent/
    # fusible nodes that the partitioner did not place in a compute partition.
    import torch
    stream: list[OrderedItem] = []
    for node in gm.graph.nodes:
        if node in anchor_to_payload:
            stream.append(anchor_to_payload[node])
            continue
        if node in partitioned:
            continue  # non-anchor member of an already-emitted partition
        if node.op != "call_function":
            continue
        if partition_node_is_supported(node):
            continue  # transparent/fusible node not in any compute partition
        val = node.meta.get("val")
        dtype = None
        if torch.is_tensor(val):
            dtype = str(val.dtype)
        elif isinstance(val, (list, tuple)):
            for item in val:
                if torch.is_tensor(item):
                    dtype = str(item.dtype)
                    break
        stream.append(OrderedItem(
            "extern",
            name=mga._target_name(node.target),
            target=mga._target_name(node.target),
            output_shape=mga._safe_shape(val),
            output_dtype=dtype,
        ))
    return stream


def build_model_streams(model_dir: Path, device: str,
                        hash_to_repro: dict[str, str]
                        ) -> list[tuple[str, list[OrderedItem]]]:
    """Return [(graph_name, ordered_stream), ...] for every graph in a model,
    graphs themselves in filename (capture) order."""
    out: list[tuple[str, list[OrderedItem]]] = []
    graph_files = sorted(model_dir.glob("full_graph_*.py"))
    for gf in graph_files:
        print(f"  tracing {gf.name}...", file=sys.stderr)
        try:
            gm = mga.trace_full_graph(gf, device=device)
            stream = _ordered_stream_for_graph(gm, hash_to_repro)
        except Exception as e:  # noqa: BLE001 -- keep going, report per graph
            print(f"    [warn] {gf.name}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            stream = []
        out.append((gf.name, stream))
    return out


# ===========================================================================
# Timings lookup (REUSES the same per-oracle timings file as the accounting)
# ===========================================================================

def _kernel_us(item: OrderedItem, timings: dict[str, dict]) -> tuple[float | None,
                                                                     float | None]:
    """oracle_us, compile_us for a fused-partition kernel via its canonical dir.

    Mirrors model_graph_accounting's per-dir lookup. The per-oracle timings file
    prices by canonical repro dir (oracle_us / compile_us). A kernel with no dir
    or an untimed dir is UNPRICED (None, None)."""
    if not item.repro_dir:
        return None, None
    t = timings.get(item.repro_dir)
    if not t:
        return None, None
    o = t.get("oracle_us")
    c = t.get("compile_us")
    return (o if isinstance(o, (int, float)) else None,
            c if isinstance(c, (int, float)) else None)


# ===========================================================================
# Trace event construction
# ===========================================================================

def _process_metadata(pid: int, model_name: str) -> list[dict]:
    return [
        {"name": "process_name", "ph": "M", "pid": pid, "tid": 0,
         "args": {"name": model_name}},
        {"name": "thread_name", "ph": "M", "pid": pid, "tid": TID_ORACLE,
         "args": {"name": TRACK_ORACLE}},
        {"name": "thread_name", "ph": "M", "pid": pid, "tid": TID_COMPILE,
         "args": {"name": TRACK_COMPILE}},
        {"name": "thread_name", "ph": "M", "pid": pid, "tid": TID_EXTERN,
         "args": {"name": TRACK_EXTERN}},
    ]


def _kernel_slice(item: OrderedItem, us: float, which: str, priced: bool,
                  graph_name: str, occ_index: int) -> dict:
    label = item.repro_dir or item.pattern_hash
    suffix = "" if priced else " [unpriced]"
    return {
        "name": f"{label} ({which}){suffix}",
        "ph": "X",
        "dur": round(us, 3),
        "args": {
            "pattern_hash": item.pattern_hash,
            "shape_hash": item.shape_hash,
            "canonical_dir": item.repro_dir,
            "graph": graph_name,
            "occurrence_index": occ_index,  # which time this pattern is seen
            "kind": "reduction" if item.is_reduction else "pointwise",
            "n_ops": item.n_ops,
            "op_summary": item.op_summary,
            "input_shapes": item.input_shapes,
            "output_shape": item.output_shape,
            "track": which,
            "us": round(us, 3),
            "priced": priced,
        },
    }


def _extern_slice(item: OrderedItem, graph_name: str) -> dict:
    return {
        "name": f"{item.target} (extern, untimed)",
        "ph": "X",
        "dur": EXTERN_MARKER_US,
        "args": {
            "target": item.target,
            "graph": graph_name,
            "output_shape": item.output_shape,
            "output_dtype": item.output_dtype,
            "note": "non-fusible extern (conv/GEMM/SDPA/backward): NOT an oracle "
                    "target; no oracle/compile timing in the corpus. Marker "
                    "width is fixed, not a measured duration.",
        },
    }


def _graph_boundary(pid: int, tid: int, ts: float, graph_name: str) -> dict:
    """A zero-width instant marking the start of a new graph on a track."""
    return {
        "name": f"--- graph: {graph_name} ---",
        "ph": "i", "s": "t",
        "ts": round(ts, 3),
        "pid": pid, "tid": tid,
        "args": {"graph": graph_name},
    }


def build_trace_for_model(pid: int, model_name: str,
                          graph_streams: list[tuple[str, list[OrderedItem]]],
                          timings: dict[str, dict]
                          ) -> tuple[list[dict], dict]:
    """Build all events for one model, laying graphs sequentially and slices in
    execution order. Returns (events, per-model summary)."""
    events: list[dict] = _process_metadata(pid, model_name)

    ts_oracle = 0.0
    ts_compile = 0.0
    ts_extern = 0.0
    oracle_slices = compile_slices = extern_slices = 0
    unpriced_kernels = 0
    occ_counter: Counter = Counter()  # pattern_hash -> times seen (for args)
    first_slice_names: list[str] = []  # for the ordered-proof report

    for graph_name, stream in graph_streams:
        if len(graph_streams) > 1:
            events.append(_graph_boundary(pid, TID_ORACLE, ts_oracle, graph_name))
            events.append(_graph_boundary(pid, TID_COMPILE, ts_compile, graph_name))
            events.append(_graph_boundary(pid, TID_EXTERN, ts_extern, graph_name))

        for item in stream:
            if item.kind == "extern":
                ev = _extern_slice(item, graph_name)
                ev.update(ts=round(ts_extern, 3), pid=pid, tid=TID_EXTERN)
                events.append(ev)
                ts_extern += EXTERN_MARKER_US
                extern_slices += 1
                if len(first_slice_names) < 12:
                    first_slice_names.append(f"[extern] {item.target}")
                continue

            # kernel
            occ_counter[item.pattern_hash] += 1
            occ_index = occ_counter[item.pattern_hash]
            o_us, c_us = _kernel_us(item, timings)
            priced = o_us is not None or c_us is not None
            if not priced:
                unpriced_kernels += 1
            if len(first_slice_names) < 12:
                first_slice_names.append(
                    f"[kernel] {item.repro_dir or item.pattern_hash}")

            # oracle track slice
            o_draw = o_us if isinstance(o_us, (int, float)) else 0.0
            ev = _kernel_slice(item, o_draw, "oracle", priced, graph_name, occ_index)
            ev.update(ts=round(ts_oracle, 3), pid=pid, tid=TID_ORACLE)
            events.append(ev)
            ts_oracle += o_draw
            oracle_slices += 1

            # compile track slice (same kernel, same position semantics)
            c_draw = c_us if isinstance(c_us, (int, float)) else 0.0
            ev = _kernel_slice(item, c_draw, "compile", priced, graph_name, occ_index)
            ev.update(ts=round(ts_compile, 3), pid=pid, tid=TID_COMPILE)
            events.append(ev)
            ts_compile += c_draw
            compile_slices += 1

    # SOL track intentionally omitted (see _sol_track_hook).
    ratio = (ts_compile / ts_oracle) if ts_oracle else None
    summary = {
        "pid": pid,
        "oracle_slices": oracle_slices,
        "compile_slices": compile_slices,
        "extern_slices": extern_slices,
        "unpriced_kernel_slices": unpriced_kernels,
        "oracle_track_total_us": round(ts_oracle, 2),
        "compile_track_total_us": round(ts_compile, 2),
        "compile_over_oracle_ratio": round(ratio, 3) if ratio else None,
        "first_slices_in_execution_order": first_slice_names,
    }
    return events, summary


def _sol_track_hook(graph_streams, pid: int) -> list[dict]:
    """HOOK ONLY -- the SOL (bandwidth floor) track is intentionally NOT built.

    When implemented it would emit a fourth track (tid 4, "SOL (bandwidth
    floor)") with one slice per kernel whose dur is memcopy_sol_us (the
    speed-of-light memory-bandwidth floor), in the same execution order. That
    datum is NOT in the per-oracle timings file used here; it lives in the
    per-(repro,shape) STATE JSONs:
        results/pytorch_landing/A1_commit_walk/states/<commit>.json
    keyed by "repros/canonical/<dir>/repro.py" -> {model_label: {memcopy_sol_us}}.
    Joining it onto each kernel (by canonical dir + shape) is the remaining work.
    Deliberately deferred."""
    return []  # not implemented


def emit(events: list[dict]) -> dict:
    """Wrap events in the Chrome-trace object Perfetto ingests directly."""
    return {"traceEvents": events, "displayTimeUnit": "ns"}


# ===========================================================================
# Model discovery / selection
# ===========================================================================

def select_model_dirs(model_filter: str | None, all_models: bool) -> list[Path]:
    dirs = mga.discover_model_dirs()
    if all_models:
        return dirs
    matched = [d for d in dirs if model_filter.lower() in d.name.lower()]
    return matched


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Export a model's kernel rollup as an execution-ordered "
                    "Perfetto/Chrome-trace JSON (oracle ceiling vs current "
                    "torch.compile)."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", "-m",
                     help="Model name filter (partial match)")
    src.add_argument("--all", "-a", action="store_true",
                     help="All models (one process per model in one trace)")
    p.add_argument("--timings", default=str(DEFAULT_TIMINGS),
                   help="Per-oracle timings JSON keyed by canonical dir "
                        "(default: results/all_oracle_timings_b200_v2.json -- "
                        "the file that PRICES models).")
    p.add_argument("--device", default="cuda",
                   help="Device for input construction (tracing is fake-mode); "
                        "default cuda")
    p.add_argument("--out", "-o", required=True,
                   help="Output Chrome-trace JSON path")
    args = p.parse_args()

    model_dirs = select_model_dirs(args.model, args.all)
    if not model_dirs:
        raise SystemExit(f"No models matching '{args.model}'")

    hash_to_repro = mga.build_hash_to_repro()
    timings = mga.load_timings(Path(args.timings))
    print(f"[rollup_to_perfetto] {len(hash_to_repro)} canonical mappings, "
          f"{len(timings)} priced dirs, {len(model_dirs)} model(s)",
          file=sys.stderr)

    all_events: list[dict] = []
    summary: dict[str, Any] = {}
    pid = 0
    for model_dir in model_dirs:
        pid += 1
        print(f"\n[rollup_to_perfetto] model: {model_dir.name}", file=sys.stderr)
        graph_streams = build_model_streams(model_dir, args.device, hash_to_repro)
        events, s = build_trace_for_model(
            pid, model_dir.name, graph_streams, timings
        )
        all_events += events
        # Disambiguate identical model names across suites by appending pid.
        key = model_dir.name if model_dir.name not in summary else \
            f"{model_dir.name}#{pid}"
        summary[key] = s

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(emit(all_events), indent=2))

    # ---- report -----------------------------------------------------------
    print(f"\n[rollup_to_perfetto] wrote {out} "
          f"({len(all_events)} trace events across {len(model_dirs)} model(s))",
          file=sys.stderr)
    any_timed = False
    for name, s in summary.items():
        ratio = s["compile_over_oracle_ratio"]
        ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
        timed = s["oracle_track_total_us"] > 0 and s["compile_track_total_us"] > 0
        any_timed = any_timed or timed
        flag = "" if timed else "  [WARN: no priced kernels -> empty timed tracks]"
        print(
            f"  {name}: oracle_slices={s['oracle_slices']} "
            f"compile_slices={s['compile_slices']} "
            f"extern_slices={s['extern_slices']} "
            f"(unpriced_kernels={s['unpriced_kernel_slices']}) | "
            f"oracle_total={s['oracle_track_total_us']}us "
            f"compile_total={s['compile_track_total_us']}us "
            f"ratio(compile/oracle)={ratio_str}{flag}",
            file=sys.stderr,
        )
        if s["first_slices_in_execution_order"]:
            print("    first slices (execution order): "
                  + " -> ".join(s["first_slices_in_execution_order"][:10]),
                  file=sys.stderr)
    if not any_timed:
        print(
            "  [WARN] No model had priced kernels -> empty timed tracks. Pick a "
            "--timings file that prices your model "
            "(e.g. results/all_oracle_timings_b200_v2.json).",
            file=sys.stderr,
        )
    print("\n  View: drag the JSON into https://ui.perfetto.dev "
          "(or chrome://tracing -> Load).", file=sys.stderr)


if __name__ == "__main__":
    main()
