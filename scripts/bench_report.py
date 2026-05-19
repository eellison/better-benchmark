"""
Generate benchmark comparison reports and CI-compatible output.

Outputs:
  1. v3 OSS benchmark JSON (for upload to PyTorch performance dashboard)
  2. Markdown comparison table (for PR comments)
  3. Regression/improvement summary

Usage:
    # Run sweep and output report:
    python scripts/bench_report.py --tag my_fix --output-json results.json

    # Compare two existing result files:
    python scripts/bench_report.py --compare baseline.json head.json

    # Output CI-compatible JSON:
    python scripts/bench_report.py --compare baseline.json head.json --ci-json test/test-reports/kernel_benchmark.json
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REGRESSION_THRESHOLD = 0.05  # 5%


@dataclass
class KernelDelta:
    name: str
    base_us: float
    head_us: float
    delta_pct: float
    kind: str = ""


def load_results(path: Path) -> dict[str, dict]:
    """Load a sweep results JSON. Returns {kernel_name: {compiled_us, memcopy_sol_us, ...}}."""
    data = json.loads(path.read_text())
    results = {}
    for key, val in data.items():
        if isinstance(val, dict) and "default" in val:
            name = key.split("/")[-2] if "/" in key else key
            results[name] = val["default"]
    return results


def compute_deltas(base: dict, head: dict) -> list[KernelDelta]:
    """Compute per-kernel deltas between base and head results."""
    deltas = []
    common_keys = set(base.keys()) & set(head.keys())
    for name in sorted(common_keys):
        base_us = base[name].get("compiled_us", 0)
        head_us = head[name].get("compiled_us", 0)
        if base_us <= 0:
            continue
        delta_pct = (head_us - base_us) / base_us
        deltas.append(KernelDelta(
            name=name,
            base_us=base_us,
            head_us=head_us,
            delta_pct=delta_pct,
        ))
    return deltas


def render_markdown(deltas: list[KernelDelta], title: str = "Kernel Benchmark Results") -> str:
    """Render a markdown comparison report."""
    regressions = [d for d in deltas if d.delta_pct > REGRESSION_THRESHOLD]
    improvements = [d for d in deltas if d.delta_pct < -REGRESSION_THRESHOLD]
    neutral = [d for d in deltas if abs(d.delta_pct) <= REGRESSION_THRESHOLD]

    regressions.sort(key=lambda d: -d.delta_pct)
    improvements.sort(key=lambda d: d.delta_pct)

    lines = [f"## {title}"]
    lines.append(f"**{len(improvements)} improved, {len(regressions)} regressed, {len(neutral)} neutral** (threshold: {REGRESSION_THRESHOLD*100:.0f}%)")
    lines.append("")

    if regressions:
        lines.append("### Regressions")
        lines.append("| Kernel | Base (us) | Head (us) | Delta |")
        lines.append("|--------|-----------|-----------|-------|")
        for d in regressions[:20]:
            lines.append(f"| {d.name} | {d.base_us:.1f} | {d.head_us:.1f} | +{d.delta_pct*100:.1f}% |")
        if len(regressions) > 20:
            lines.append(f"| ... and {len(regressions)-20} more | | | |")
        lines.append("")

    if improvements:
        lines.append("### Improvements")
        lines.append("| Kernel | Base (us) | Head (us) | Delta |")
        lines.append("|--------|-----------|-----------|-------|")
        for d in improvements[:20]:
            lines.append(f"| {d.name} | {d.base_us:.1f} | {d.head_us:.1f} | {d.delta_pct*100:.1f}% |")
        if len(improvements) > 20:
            lines.append(f"| ... and {len(improvements)-20} more | | | |")
        lines.append("")

    # Summary stats
    if deltas:
        all_deltas = [d.delta_pct for d in deltas]
        geomean = 1.0
        for d in all_deltas:
            geomean *= (1 + d)
        geomean = geomean ** (1 / len(all_deltas)) - 1

        lines.append("### Summary")
        lines.append(f"- Kernels compared: {len(deltas)}")
        lines.append(f"- Geometric mean speedup: {-geomean*100:.2f}%")
        lines.append(f"- Median delta: {sorted(all_deltas)[len(all_deltas)//2]*100:.2f}%")

    return "\n".join(lines)


def write_ci_json(deltas: list[KernelDelta], head_results: dict, output_path: Path,
                  device: str = "cuda", arch: str = ""):
    """Write v3 OSS benchmark database JSON for CI upload."""
    if not arch:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True, timeout=5
            ).strip().split("\n")[0]
            arch = out
        except Exception:
            arch = "unknown"

    records = []
    for name, result in head_results.items():
        compiled_us = result.get("compiled_us", 0)
        sol_us = result.get("memcopy_sol_us", 0)
        gap = result.get("gap_default", 0)

        records.append({
            "benchmark": {
                "name": "inductor-kernel-benchmark",
                "mode": "inference",
                "dtype": "float32",
                "extra_info": {"device": device, "arch": arch},
            },
            "model": {
                "name": name,
                "type": "micro-benchmark",
                "origins": ["pytorch"],
            },
            "metric": {
                "name": "latency_us",
                "benchmark_values": [compiled_us],
            },
        })
        if gap:
            records.append({
                "benchmark": {
                    "name": "inductor-kernel-benchmark",
                    "mode": "inference",
                    "dtype": "float32",
                    "extra_info": {"device": device, "arch": arch},
                },
                "model": {
                    "name": name,
                    "type": "micro-benchmark",
                    "origins": ["pytorch"],
                },
                "metric": {
                    "name": "gap_vs_sol",
                    "benchmark_values": [gap],
                },
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Kernel benchmark report generator")
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "HEAD"),
                        help="Compare two result JSON files")
    parser.add_argument("--ci-json", type=Path, default=None,
                        help="Write CI-compatible v3 JSON to this path")
    parser.add_argument("--output-md", type=Path, default=None,
                        help="Write markdown report to file")
    parser.add_argument("--title", default="Kernel Benchmark Results")
    args = parser.parse_args()

    if args.compare:
        base_path, head_path = Path(args.compare[0]), Path(args.compare[1])
        base = load_results(base_path)
        head = load_results(head_path)

        deltas = compute_deltas(base, head)
        md = render_markdown(deltas, title=args.title)
        print(md)

        if args.output_md:
            args.output_md.write_text(md)
            print(f"\nMarkdown written to {args.output_md}")

        if args.ci_json:
            write_ci_json(deltas, head, args.ci_json)

        # Exit code: 1 if regressions found
        regressions = [d for d in deltas if d.delta_pct > REGRESSION_THRESHOLD]
        if regressions:
            print(f"\n⚠ {len(regressions)} regressions detected (>{REGRESSION_THRESHOLD*100:.0f}%)")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
