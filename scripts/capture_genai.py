#!/usr/bin/env python3
"""Recapture the genai microbenchmark graphs through the new pipeline.

The genai suite is saved full graphs (repros/models/genai/<name>/
full_graph_*.py + sidecars), not runner-loadable models. STATIC recapture:
load each graph at its existing recorded shapes (the same shapes the old
repro coverage used), run it through torch.compile with the capture hook
installed, merge into the new corpus. Dynamic-shape capture of these
families stays wave 2 (this gets their static patterns onto new hashes /
occurrence counts / compact format now).

Usage:
    python scripts/capture_genai.py --corpus-root /tmp/recapture_corpus/repros \
        [--models LayerNormBackward,SoftmaxForward] [--gpu 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

GENAI_DIR = ROOT / "repros" / "models" / "genai"


def capture_one(name: str, corpus_root: Path, timeout_s: int) -> dict:
    import torch
    from full_graph_harness import load_full_graph
    from capture_hook import install_capture_hook, uninstall_capture_hook
    from run_recapture import write_run_log_entry, _count_roundtrips
    from merge_captures import temporary_capture_for_merge

    t0 = time.time()
    entry = {
        "model_key": f"genai/static/{name}",
        "suite": "genai",
        "model": name,
        "mode": "static",
        "status": "failed",
        "wall_time_s": 0.0,
        "region_count": 0,
        "roundtrip_summary": {"ok": 0, "failed": 0},
        "error_tail": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    src_dir = GENAI_DIR / name
    graphs = sorted(src_dir.glob("full_graph_*.py"))
    if not graphs:
        entry["error_tail"] = f"no full graphs in {src_dir}"
        return entry

    model_dir = corpus_root / "models" / "genai" / "static" / name
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        with temporary_capture_for_merge(
                corpus_root, name, suite="genai", mode="static") as capture:
            install_capture_hook(
                output_dir=str(capture.capture_dir),
                label=f"genai_{name}_static",
                graph_dir=str(model_dir),
                validate=True,
            )
            try:
                for g in graphs:
                    # Existing recorded shapes = the same shapes the old
                    # repro coverage used (sidecar inputs).
                    instance, inputs, _ = load_full_graph(
                        g, default_device="cuda")
                    inputs = [t.cuda() if torch.is_tensor(t) else t
                              for t in inputs]
                    torch._dynamo.reset()
                    compiled = torch.compile(instance)
                    with torch.no_grad():
                        compiled(*inputs)
                    torch.cuda.synchronize()
            finally:
                uninstall_capture_hook()

            # fail-hard on drops, same as run_recapture
            index_path = capture.capture_dir / "index.json"
            if index_path.exists():
                idx = json.loads(index_path.read_text())
                dropped = (idx.get("dropped", [])
                           if isinstance(idx, dict) else [])
                if dropped:
                    raise RuntimeError(
                        f"{len(dropped)} region(s) DROPPED: "
                        f"{dropped[0].get('reason', '?')[:200]}")

            n = capture.merge()

        rt_ok, rt_failed = _count_roundtrips(model_dir)
        entry["status"] = "ok"
        entry["region_count"] = n
        entry["roundtrip_summary"] = {"ok": rt_ok, "failed": rt_failed}
    except Exception as e:  # noqa: BLE001 -- per-model isolation
        import traceback
        entry["error_tail"] = traceback.format_exc()[-1500:]
    entry["wall_time_s"] = time.time() - t0
    write_run_log_entry(corpus_root, entry)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--models", default="all")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    if args.models == "all":
        names = sorted(d.name for d in GENAI_DIR.iterdir()
                       if d.is_dir() and list(d.glob("full_graph_*.py")))
    else:
        names = [m for m in args.models.split(",") if m]

    print(f"genai static recapture: {len(names)} graph dirs")
    ok = failed = 0
    for name in names:
        e = capture_one(name, args.corpus_root, args.timeout)
        ok += e["status"] == "ok"
        failed += e["status"] != "ok"
        print(f"  {name}: {e['status']} ({e['region_count']} regions, "
              f"{e['wall_time_s']:.0f}s)"
              + (f" — {e['error_tail'][-120:]}" if e["error_tail"] else ""))
    print(f"genai: {ok} ok, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
