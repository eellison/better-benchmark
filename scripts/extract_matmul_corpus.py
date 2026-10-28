"""Extract all saved BB graphs with the MM policy, without GPU compilation."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POLICY_ID = hashlib.sha256(
    (ROOT / "matmul_partitioner.py").read_bytes()
    + (ROOT / "capture_hook.py").read_bytes()
).hexdigest()[:16]

VISIBLE_MM = re.compile(
    r"torch\.ops\.aten\.(?:mm|bmm|matmul|addmm|baddbmm|mv|addmv|_scaled_mm|_int_mm)\."
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def source_id(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes())
    sidecar = path.with_suffix(".meta.json")
    if sidecar.exists():
        digest.update(sidecar.read_bytes())
    return digest.hexdigest()


def describe_region(region):
    import torch

    from matmul_partitioner import (
        INDEX_NAMES,
        is_computation,
        is_matmul,
        op_name,
        tensor_shape,
        value_inputs,
    )

    ancestors, math_path = {}, {}
    core = set(region)
    mm = [n for n in region if is_matmul(n)]
    mm_shapes = []
    reductions = []
    traits = set()
    for n in region:
        inputs = [a for a in value_inputs(n) if a in core]
        ancestors[n] = set().union(*(ancestors[a] | {a} for a in inputs))
        math_path[n] = is_computation(n) or any(math_path[a] for a in inputs)
        if is_matmul(n):
            offset = (
                1
                if op_name(n) in {"aten::addmm", "aten::baddbmm", "aten::addmv"}
                else 0
            )
            left, right = n.args[offset : offset + 2]
            parents = [p for p in mm if p in ancestors[n]]
            mm_shapes.append(
                {
                    "node": n.name,
                    "op": op_name(n),
                    "lhs": tensor_shape(left),
                    "rhs": tensor_shape(right),
                    "out": tensor_shape(n),
                    "parents": [p.name for p in parents],
                }
            )
            if parents:
                traits.add("vertical")
            if any(is_computation(p) for p in ancestors[n]):
                traits.add("prologue")
            if op_name(n) in {"aten::_scaled_mm", "aten::_int_mm"}:
                traits.add("native_scaled_or_integer_mm")
        elif is_computation(n) and any(p in ancestors[n] for p in mm):
            traits.add("epilogue")
        if op_name(n) in INDEX_NAMES:
            traits.add(
                "scatter"
                if any(s in op_name(n) for s in ("scatter", "index_put", "index_copy"))
                else "gather"
            )
        if torch.Tag.reduction in getattr(n.target, "tags", ()):
            bindings = {
                arg.name: value
                for arg, value in zip(n.target._schema.arguments, n.args)
            }
            bindings.update(n.kwargs)
            dims = bindings.get("dim", bindings.get("dims"))
            source = n.all_input_nodes[0]
            source_shape = tensor_shape(source)
            dims = [dims] if isinstance(dims, int) else dims
            dims = (
                list(range(len(source_shape)))
                if dims is None or dims == []
                else list(dims)
            )
            dims = [d % len(source_shape) for d in dims]
            after_mm = any(p in ancestors[n] for p in mm)
            reductions.append(
                {
                    "op": op_name(n),
                    "source_shape": source_shape,
                    "dims": dims,
                    "after_mm": after_mm,
                }
            )
            if after_mm:
                traits.add("mm_reduction")
                if len(dims) == 1 and dims[0] == len(source_shape) - 1:
                    traits.add("row_reduction")
                if len(dims) > 1 or any(
                    op_name(a) in {"aten::view", "aten::reshape", "aten::_unsafe_view"}
                    and len(tensor_shape(a) or ()) > 2
                    for a in ancestors[n]
                ):
                    traits.add("block_or_reshaped_reduction")
    for i, a in enumerate(mm):
        for b in mm[i + 1 :]:
            if a not in ancestors[b] and b not in ancestors[a]:
                traits.add("horizontal")
                if any(
                    a in ancestors[n] and b in ancestors[n]
                    for n in region
                    if not is_matmul(n)
                ):
                    traits.add("horizontal_then_combine")
    if reductions and any("rsqrt" in op_name(n) for n in region):
        traits.add("normalization_math")
    if "gather" in traits and "scatter" in traits:
        traits.add("gather_mm_scatter")
    bare = [n.name for n in region.outputs if not math_path[n]]
    return {
        "metrics": region.metrics,
        "traits": sorted(traits),
        "mm_shapes": mm_shapes,
        "reductions": reductions,
        "outputs": [n.name for n in region.outputs],
        "bare_mm_outputs": bare,
        "allowed_bare_horizontal": bool(
            bare
            and region.metrics["non_mm_compute"] == 0
            and region.metrics["mm"] >= 2
            and region.metrics["mm_depth"] == 1
        ),
    }


def extract_one(task):
    index, source, work_dir = task
    import torch

    from capture_hook import _CaptureState, get_fusion_partitions
    from full_graph_harness import load_full_graph_definition
    from matmul_partitioner import is_matmul
    from roundtrip_validation import trace_full_graph_fake

    torch.set_num_threads(1)
    capture_dir = Path(work_dir) / f"{index:04d}"
    capture_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "source": source,
        "capture_dir": str(capture_dir),
        "regions": [],
        "policy_id": POLICY_ID,
    }
    with (
        (capture_dir / "worker.log").open("w") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            t0 = time.perf_counter()
            row["source_id"] = source_id(source)
            if not VISIBLE_MM.search(Path(source).read_text()):
                row.update(
                    status="no_visible_mm",
                    original_mm=0,
                    selected_mm=0,
                    trace_seconds=0,
                    partition_seconds=0,
                    capture_seconds=0,
                )
                write_json(capture_dir / "result.json", row)
                return row
            definition = load_full_graph_definition(source)
            gm = trace_full_graph_fake(definition)
            t1 = time.perf_counter()
            diagnostics = {}
            components = get_fusion_partitions(
                gm, only_include_mm=True, diagnostics=diagnostics
            )
            t2 = time.perf_counter()
            state = _CaptureState(
                str(capture_dir), label=source, only_include_mm=True, validate=False
            )
            observations = state.process_graph(gm, components=components)
            state.finalize()
            if state.dropped or len(observations) != len(components):
                raise RuntimeError(
                    f"capture mismatch: {len(components)=}, {len(observations)=}, {state.dropped=}"
                )
            lookup = {tuple(o["source_nodes"]): o for o in observations}
            for region in components:
                description = describe_region(region)
                description.update(lookup[tuple(n.name for n in region)])
                description["n_aten_ops"] = sum(
                    op.startswith("aten.") for op in description["canonical_ops"]
                )
                description["n_mm_ops"] = region.metrics["mm"]
                row["regions"].append(description)
            row.update(
                status="ok",
                original_mm=sum(is_matmul(n) for n in gm.graph.nodes),
                selected_mm=len({n for r in components for n in r if is_matmul(n)}),
                graph_nodes=len(tuple(gm.graph.nodes)),
                diagnostics=diagnostics,
                trace_seconds=round(t1 - t0, 4),
                partition_seconds=round(t2 - t1, 4),
                capture_seconds=round(time.perf_counter() - t2, 4),
            )
        except Exception:  # noqa: BLE001 -- record failures and fail the final gate
            row.update(status="failed", error=traceback.format_exc())
            print(row["error"], flush=True)
    write_json(capture_dir / "result.json", row)
    return row


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}

    def percentile(q):
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return round(
            values[lower] + (values[upper] - values[lower]) * (position - lower), 4
        )

    return {
        "count": len(values),
        "min": values[0],
        "p25": percentile(0.25),
        "p50": percentile(0.5),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "p100": values[-1],
        "mean": round(sum(values) / len(values), 4),
    }


def fusion_profile(traits):
    prologue, epilogue = "prologue" in traits, "epilogue" in traits
    if prologue and epilogue:
        return "prologue_and_epilogue"
    if epilogue:
        return "epilogue_only"
    if prologue:
        return "prologue_only"
    return "horizontal_mm"


def merge_and_report(rows, output, wall_seconds):
    from matmul_partitioner import MatmulLimits
    from merge_captures import merge_one_capture

    points = defaultdict(list)
    failed = [r for r in rows if r["status"] == "failed"]
    for row in rows:
        if row["status"] != "ok":
            continue
        source = Path(row["source"]).relative_to(ROOT / "repros" / "models")
        suite, mode = source.parts[:2]
        model = "/".join(source.parts[2:-1])
        with contextlib.redirect_stdout(sys.stderr):
            merge_one_capture(
                Path(row["capture_dir"]), output, model, suite=suite, mode=mode
            )
        for region in row["regions"]:
            points[(region["pattern_hash"], region["shape_hash"])].append(
                {"source": row["source"], **region}
            )
    canonical = output / "canonical"
    repros = sorted(canonical.glob("*/repro.py"))
    inventory = []
    mm_operator_kinds = Counter()
    for repro in repros:
        meta_path = repro.with_name("meta.json")
        meta = json.loads(meta_path.read_text())
        pattern = meta["pattern_hash"]
        matching = {
            shape: occurrences
            for (pat, shape), occurrences in points.items()
            if pat == pattern
        }
        traits = sorted(
            {
                t
                for occurrences in matching.values()
                for r in occurrences
                for t in r["traits"]
            }
        )
        ops = re.findall(r"torch\.ops\.([\w.]+)\(", repro.read_text())
        aten = sum(op.startswith("aten.") for op in ops)
        mm = sum(VISIBLE_MM.search("torch.ops." + op + ".") is not None for op in ops)
        mm_operator_kinds.update(
            op for op in ops if VISIBLE_MM.search("torch.ops." + op + ".")
        )
        entry = {
            "pattern_hash": pattern,
            "repro": str(repro.relative_to(output)),
            "n_aten_ops": aten,
            "n_mm_ops": mm,
            "shape_points": len(matching),
            "traits": traits,
        }
        meta.update(entry)
        write_json(meta_path, meta)
        shapes_path = repro.with_name("shapes.json")
        shapes = json.loads(shapes_path.read_text())
        for point in shapes["points"]:
            occurrences = matching.get(point["shape_hash"], [])
            if occurrences:
                first = occurrences[0]
                point.update(
                    partition_metrics=first["metrics"],
                    mm_shapes=first["mm_shapes"],
                    n_aten_ops=first["n_aten_ops"],
                    n_mm_ops=first["n_mm_ops"],
                    traits=sorted({t for r in occurrences for t in r["traits"]}),
                )
        write_json(shapes_path, shapes)
        inventory.append(entry)
    occurrences = [r for row in rows for r in row["regions"]]
    bad_outputs = [
        r
        for r in occurrences
        if r["bare_mm_outputs"] and not r["allowed_bare_horizontal"]
    ]
    traits = Counter(t for r in occurrences for t in r["traits"])
    report = {
        "policy": asdict(MatmulLimits()),
        "sources": len(rows),
        "sources_traced": sum(r["status"] == "ok" for r in rows),
        "sources_without_visible_mm": sum(r["status"] == "no_visible_mm" for r in rows),
        "failures": [{"source": r["source"], "error": r["error"]} for r in failed],
        "unique_repros": len(repros),
        "shape_points": len(points),
        "region_occurrences": len(occurrences),
        "original_mm": sum(r.get("original_mm", 0) for r in rows),
        "selected_mm_union": sum(r.get("selected_mm", 0) for r in rows),
        "shapes_per_repro": distribution([r["shape_points"] for r in inventory]),
        "aten_ops_per_repro": distribution([r["n_aten_ops"] for r in inventory]),
        "aten_ops_per_shape_point": distribution(
            [rs[0]["n_aten_ops"] for rs in points.values()]
        ),
        "mm_ops_per_repro": distribution([r["n_mm_ops"] for r in inventory]),
        "mm_ops_per_shape_point": distribution(
            [rs[0]["n_mm_ops"] for rs in points.values()]
        ),
        "mm_ops_histogram_repro": dict(Counter(r["n_mm_ops"] for r in inventory)),
        "mm_ops_histogram_occurrence": dict(
            Counter(r["n_mm_ops"] for r in occurrences)
        ),
        "traits_occurrences": dict(traits),
        "traits_repros": dict(Counter(t for r in inventory for t in r["traits"])),
        "fusion_profiles_repros": dict(
            Counter(fusion_profile(r["traits"]) for r in inventory)
        ),
        "fusion_profiles_occurrences": dict(
            Counter(fusion_profile(r["traits"]) for r in occurrences)
        ),
        "mm_operator_kinds_repro": dict(mm_operator_kinds),
        "unexpected_bare_output_regions": len(bad_outputs),
        "pure_horizontal_occurrences": sum(
            r["allowed_bare_horizontal"] for r in occurrences
        ),
        "timing": {
            "wall_seconds": round(wall_seconds, 3),
            **{
                key: round(sum(r.get(key, 0) for r in rows), 3)
                for key in ("trace_seconds", "partition_seconds", "capture_seconds")
            },
        },
    }
    write_json(output / "inventory.json", inventory)
    public_rows = [{k: v for k, v in row.items() if k != "capture_dir"} for row in rows]
    for row in public_rows:
        row["source"] = str(Path(row["source"]).relative_to(ROOT))
        row["regions"] = [
            {
                key: region[key]
                for key in ("pattern_hash", "shape_hash", "source_nodes", "outputs")
            }
            for region in row["regions"]
        ]
    write_json(output / "occurrences.json", public_rows)
    write_json(output / "report.json", report)
    if failed or bad_outputs:
        raise RuntimeError(
            f"{len(failed)} failed sources, {len(bad_outputs)} unexpected bare output regions; see report.json"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "mm_repro")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--source", action="append", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep intermediate captures for debugging/resume",
    )
    args = parser.parse_args()
    sources = args.source or [
        ROOT / p
        for p in subprocess.check_output(
            ["rg", "--files", "repros/models", "-g", "full_graph_*.py"],
            cwd=ROOT,
            text=True,
        ).splitlines()
    ]
    sources = sorted(p.resolve() for p in sources)
    args.output.mkdir(parents=True, exist_ok=True)
    work = args.output / ".work"
    start = time.perf_counter()
    rows, pending = [], []
    for i, source in enumerate(sources):
        result = work / f"{i:04d}" / "result.json"
        if args.resume and result.exists():
            row = json.loads(result.read_text())
            if (
                row["source"] == str(source)
                and row["status"] != "failed"
                and row.get("policy_id") == POLICY_ID
                and row.get("source_id") == source_id(source)
            ):
                rows.append(row)
                continue
        pending.append((i, str(source), str(work)))
    print(
        f"Start: {len(sources)} sources, {len(pending)} pending, {args.workers} CPU workers",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(extract_one, task): task for task in pending}
        last_report = 0
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            now = time.perf_counter()
            if (
                len(rows) % 10 == 0
                or now - last_report > 15
                or row["status"] == "failed"
            ):
                progress = {
                    "completed": len(rows),
                    "total": len(sources),
                    "regions": sum(len(r["regions"]) for r in rows),
                    "failures": sum(r["status"] == "failed" for r in rows),
                    "seconds": round(now - start, 1),
                    "last": row["source"],
                }
                write_json(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
                last_report = now
    rows.sort(key=lambda r: r["source"])
    report = merge_and_report(rows, args.output, time.perf_counter() - start)
    report["policy_id"] = POLICY_ID
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not args.keep_cache:
        shutil.rmtree(work)
        (args.output / "progress.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
