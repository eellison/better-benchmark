"""Regenerate the MM pattern corpus (mm_repro/) from the saved model graphs.

    python scripts/extract_matmul_corpus.py --workers 32
    python scripts/validate_matmul_repros.py     # FakeTensor-run every shape

Everything runs on CPU with FakeTensors; nothing is compiled.

1. Capture: trace every repros/models/**/full_graph_*.py, select its MM
   pattern regions (matmul_partitioner.get_matmul_regions, one region per
   category) and capture them with the regular capture/merge code into a
   temporary "raw" corpus.
2. Curate: scripts/curate_mm_corpus.py normalizes the raw repros, applies the
   size limits, merges equivalent DAGs, keeps one repro per pattern family
   and writes canonical/<category>/<mm ops>_<hash>/.
3. Publish: the finished tree atomically replaces mm_repro/, so files from an
   older run can never mix with the new ones.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Fingerprint of the code that decides which regions are captured and how
# their inputs are generated. Recorded in report.json.
POLICY_ID = hashlib.sha256(
    (ROOT / "matmul_partitioner.py").read_bytes()
    + (ROOT / "mm_region_quality.py").read_bytes()
    + (ROOT / "capture_hook.py").read_bytes()
    + (ROOT / "mm_input_constraints.py").read_bytes()
    + (ROOT / "input_codec.py").read_bytes()
    + (ROOT / "repro_harness.py").read_bytes()
).hexdigest()[:16]

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def extract_one(task):
    """Capture the MM regions of one source graph (runs in a worker process)."""
    index, source, work_dir = task
    import torch

    from capture_hook import _CaptureState, partition_node_is_supported
    from full_graph_harness import load_full_graph_definition
    from matmul_partitioner import get_matmul_regions, is_matmul
    from roundtrip_validation import trace_full_graph_fake

    torch.set_num_threads(1)
    capture_dir = Path(work_dir) / f"{index:04d}"
    capture_dir.mkdir(parents=True, exist_ok=True)
    row = {"source": source, "capture_dir": str(capture_dir), "regions": []}
    # Capture code prints a lot; keep the console for progress lines.
    with (
        (capture_dir / "worker.log").open("w") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            gm = trace_full_graph_fake(load_full_graph_definition(source))
            original_mm = sum(is_matmul(n) for n in gm.graph.nodes)
            if not original_mm:
                row.update(status="no_mm", original_mm=0)
                return row
            components = get_matmul_regions(gm, partition_node_is_supported)
            state = _CaptureState(
                str(capture_dir), label=source, only_include_mm=True, validate=False
            )
            observations = state.process_graph(gm, components=components)
            state.finalize()
            if state.dropped or len(observations) != len(components):
                raise RuntimeError(
                    f"capture mismatch: {len(components)=}, {len(observations)=}, {state.dropped=}"
                )
            for region, observed in zip(components, observations, strict=True):
                row["regions"].append(
                    {
                        "pattern_hash": observed["pattern_hash"],
                        "shape_hash": observed["shape_hash"],
                        "metrics": region.metrics,
                    }
                )
            row.update(
                status="ok",
                original_mm=original_mm,
            )
        except Exception:  # noqa: BLE001 -- reported by merge_raw_captures
            row.update(status="failed", error=traceback.format_exc())
            print(row["error"], flush=True)
    return row


def merge_raw_captures(rows, output):
    """Merge per-source captures into the raw corpus that curation reads.

    Besides canonical/<dir>/{repro.py,shapes.json,meta.json}, curation needs
    each shape point's partitioner metrics, the occurrence rows, and the
    search policy (report.json).
    """
    from matmul_partitioner import MatmulLimits
    from merge_captures import merge_one_capture

    failed = [r for r in rows if r["status"] == "failed"]
    if failed:
        raise RuntimeError(
            f"{len(failed)} source graphs failed; first: {failed[0]['source']}\n"
            + failed[0]["error"]
        )
    metrics = {}
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
            key = region["pattern_hash"], region["shape_hash"]
            metrics.setdefault(key, region.pop("metrics"))
    for shapes_path in sorted((output / "canonical").glob("*/shapes.json")):
        pattern = json.loads(shapes_path.with_name("meta.json").read_text())[
            "pattern_hash"
        ]
        shapes = json.loads(shapes_path.read_text())
        for point in shapes["points"]:
            point["partition_metrics"] = metrics[(pattern, point["shape_hash"])]
        write_json(shapes_path, shapes)
    public_rows = []
    for row in rows:
        row = {k: v for k, v in row.items() if k != "capture_dir"}
        row["source"] = str(Path(row["source"]).relative_to(ROOT))
        public_rows.append(row)
    write_json(output / "occurrences.json", public_rows)
    write_json(
        output / "report.json",
        {"policy": asdict(MatmulLimits()), "policy_id": POLICY_ID},
    )


def extract_raw_corpus(output: Path, sources=None, workers: int = 2):
    """Capture every source graph in parallel and merge the raw corpus."""
    sources = sources or [
        ROOT / p
        for p in subprocess.check_output(
            ["rg", "--files", "repros/models", "-g", "full_graph_*.py"],
            cwd=ROOT,
            text=True,
        ).splitlines()
    ]
    sources = sorted(Path(p).resolve() for p in sources)
    work = output / ".work"
    work.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print(f"Capturing {len(sources)} source graphs, {workers} CPU workers", flush=True)
    rows = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(extract_one, (i, str(source), str(work)))
            for i, source in enumerate(sources)
        ]
        last_report = 0
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            now = time.perf_counter()
            if len(rows) % 10 == 0 or now - last_report > 15 or row["status"] == "failed":
                progress = {
                    "completed": len(rows),
                    "total": len(sources),
                    "regions": sum(len(r["regions"]) for r in rows),
                    "failures": sum(r["status"] == "failed" for r in rows),
                    "seconds": round(now - start, 1),
                    "last": row["source"],
                }
                print(json.dumps(progress), flush=True)
                last_report = now
    rows.sort(key=lambda r: r["source"])
    merge_raw_captures(rows, output)


def generate_corpus(output: Path, sources=None, workers: int = 2):
    """Regenerate the corpus and atomically replace ``output``."""
    from scripts.curate_mm_corpus import curate_corpus

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not (output / "report.json").is_file():
        raise ValueError(f"Refusing to replace a non-corpus directory: {output}")
    with tempfile.TemporaryDirectory(
        prefix=".mm_repro_build_", dir=output.parent
    ) as tmp:
        staging = Path(tmp)
        extract_raw_corpus(staging / "raw", sources, workers)
        report = curate_corpus(staging / "raw", staging / "core")
        backup = staging / "previous"
        if output.exists():
            output.rename(backup)
        try:
            (staging / "core").rename(output)
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "mm_repro")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--source", action="append", type=Path, help="only these graphs (repeatable)"
    )
    args = parser.parse_args()
    report = generate_corpus(args.output, args.source, args.workers)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "unique_repros": report["unique_repros"],
                "shape_points": report["shape_points"],
                "pattern_categories": report["pattern_categories"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
