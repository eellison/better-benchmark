"""Collect helion_vertical_result.json files into one markdown table.

    python scripts/vertical_fusion_summary.py > /tmp/vertical_fusion_table.md

Rows: every kernel under benchmarks/vertical_helion/new (regions of the
current corpus, or regions it dropped) and benchmarks/vertical_helion/old
(regions of the previous largest-window corpus) that has a result file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def chain(directory: Path) -> str:
    points = json.loads((directory / "shapes.json").read_text())["points"]
    vertical = points[0].get("partition_metrics", {}).get("vertical")
    if not vertical:
        return ""
    return " -> ".join(f'{s["op"]}(R{s["R"]},F{s["F"]})' for s in vertical["stages"])


def row(corpus: str, path: Path) -> str:
    r = json.loads(path.read_text())
    fused_ok = r.get("fused_check", {}).get("ok")
    status = (
        "error" if "fused_error" in r
        else f'{r.get("fused_kernels")} kernel' + ("" if fused_ok else ", WRONG")
    )

    def us(key):
        return f'{r[key]:.1f}' if key in r else "-"

    variants = [
        f'{k[:-3]} {r[k]:.1f}'
        for k in r
        if k.endswith("_us") and k[:-3] not in ("eager", "inductor", "unfused", "fused")
    ]
    speedup = r.get("fused_speedup_vs_unfused")
    return (
        f'| {corpus} | `{r["repro"]}` | {chain(path.parent)} | {status} | '
        f'{us("fused_us")} | {us("unfused_us")} ({r.get("unfused_kernels", "-")}) | '
        f'{us("inductor_us")} ({r.get("inductor_kernels", "-")}) | '
        f'{speedup if speedup is not None else "-"} | {", ".join(variants)} |'
    )


def main():
    print(
        "| corpus | repro | stages (new-spec R/F) | fused | fused us | "
        "unfused us (kernels) | inductor us (kernels) | fused/unfused | variants |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    in_corpus = {p.parent.name for p in ROOT.glob("mm_repro/canonical/vertical/*/repro.py")}
    for corpus, pattern in (
        ("new", "benchmarks/vertical_helion/new/*/helion_vertical_result.json"),
        ("old", "benchmarks/vertical_helion/old/*/helion_vertical_result.json"),
    ):
        for path in sorted(ROOT.glob(pattern)):
            label = corpus
            if corpus == "new" and path.parent.name not in in_corpus:
                label = "new (dropped)"
            print(row(label, path))

if __name__ == "__main__":
    sys.exit(main())
