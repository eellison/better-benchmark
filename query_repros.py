"""
Query the canonical repro manifest.

Usage:
    python query_repros.py shared          # Patterns shared by most models
    python query_repros.py models PATTERN  # Which models have this pattern
    python query_repros.py patterns MODEL  # Which patterns this model uses
    python query_repros.py summary         # Overall stats
"""
import argparse
import json
from collections import Counter
from pathlib import Path


MANIFEST_PATH = Path("repros/manifest.json")
CANONICAL_DIR = Path("repros/canonical")


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def cmd_shared(args):
    manifest = load_manifest()
    pattern_models = Counter()
    for model, data in manifest["models"].items():
        seen = set()
        for r in data["repros"]:
            ph = r["pattern_hash"]
            if ph not in seen:
                pattern_models[ph] += 1
                seen.add(ph)

    print(f"{'Pattern Hash':<20} {'Models':>6}  {'Dir'}")
    print("-" * 70)
    for ph, count in pattern_models.most_common(args.top):
        # Find the canonical dir name
        dir_name = "?"
        for d in CANONICAL_DIR.iterdir():
            if d.is_dir() and ph in d.name:
                dir_name = d.name
                break
        print(f"{ph:<20} {count:>6}  {dir_name}")


def cmd_models(args):
    manifest = load_manifest()
    pattern = args.pattern
    print(f"Models using pattern {pattern}:\n")
    for model, data in sorted(manifest["models"].items()):
        for r in data["repros"]:
            if r["pattern_hash"] == pattern:
                print(f"  {model}  (shape: {r['shape_config']})")
                break


def cmd_patterns(args):
    manifest = load_manifest()
    model = args.model
    if model not in manifest["models"]:
        # Try partial match
        matches = [m for m in manifest["models"] if model.lower() in m.lower()]
        if len(matches) == 1:
            model = matches[0]
        elif matches:
            print(f"Ambiguous model. Matches: {matches}")
            return
        else:
            print(f"Model not found: {model}")
            return

    data = manifest["models"][model]
    print(f"Patterns in {model} ({len(data['repros'])} regions):\n")
    for r in data["repros"]:
        ph = r["pattern_hash"]
        meta_path = None
        for d in CANONICAL_DIR.iterdir():
            if d.is_dir() and ph in d.name:
                meta_path = d / "meta.json"
                break
        n_models = "?"
        if meta_path and meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            n_models = meta.get("n_models", "?")
        print(f"  {ph}  shape={r['shape_config']:<40s}  shared_by={n_models} models")


def cmd_summary(args):
    manifest = load_manifest()
    n_models = len(manifest["models"])
    all_patterns = set()
    total_repros = 0
    for data in manifest["models"].values():
        for r in data["repros"]:
            all_patterns.add(r["pattern_hash"])
            total_repros += 1

    print(f"Models:             {n_models}")
    print(f"Unique patterns:    {len(all_patterns)}")
    print(f"Total repro refs:   {total_repros}")
    print(f"Avg patterns/model: {total_repros / n_models:.1f}")
    print(f"Avg models/pattern: {total_repros / len(all_patterns):.1f}")

    # Kind breakdown
    kinds = Counter()
    for d in CANONICAL_DIR.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            kinds[meta.get("kind", "unknown")] += 1
    print(f"\nBy kind:")
    for kind, count in kinds.most_common():
        print(f"  {kind}: {count}")


def cmd_perf(args):
    """Show perf results across canonical repros for a given hardware."""
    hardware = args.hardware
    rows = []
    for d in CANONICAL_DIR.iterdir():
        if not d.is_dir():
            continue
        perf_path = d / "perf.json"
        if not perf_path.exists():
            continue
        with open(perf_path) as f:
            perf = json.load(f)
        if hardware not in perf:
            continue
        for shape, result in perf[hardware].items():
            rows.append({
                "pattern": d.name,
                "shape": shape,
                "gap_default": result.get("gap_default"),
                "gap_cd": result.get("gap_cd"),
                "compiled_us": result.get("compiled_us"),
                "sol_us": result.get("memcopy_sol_us"),
                "n_kernels": result.get("n_kernels"),
                "timestamp": result.get("timestamp", ""),
            })

    if not rows:
        print(f"No perf data for hardware={hardware}")
        return

    sort_key = args.sort
    rows.sort(key=lambda r: -(r.get(sort_key) or 0))

    print(f"{'Pattern':<40} {'Shape':<30} {'Gap':>6} {'CD Gap':>7} {'us':>8} {'SOL':>8} {'K':>2}")
    print("-" * 105)
    for r in rows[:args.top]:
        gap = f"{r['gap_default']:.2f}x" if r['gap_default'] else "?"
        cd_gap = f"{r['gap_cd']:.2f}x" if r['gap_cd'] else "?"
        us = f"{r['compiled_us']:.1f}" if r['compiled_us'] else "?"
        sol = f"{r['sol_us']:.1f}" if r['sol_us'] else "?"
        print(f"{r['pattern']:<40} {r['shape']:<30} {gap:>6} {cd_gap:>7} {us:>8} {sol:>8} {r['n_kernels'] or '?':>2}")


def cmd_gaps(args):
    """Show patterns with worst SOL gap on given hardware (optimization targets)."""
    hardware = args.hardware
    pattern_worst = {}
    for d in CANONICAL_DIR.iterdir():
        if not d.is_dir():
            continue
        perf_path = d / "perf.json"
        if not perf_path.exists():
            continue
        with open(perf_path) as f:
            perf = json.load(f)
        if hardware not in perf:
            continue
        for shape, result in perf[hardware].items():
            gap = result.get("gap_cd") or result.get("gap_default")
            if gap and (d.name not in pattern_worst or gap > pattern_worst[d.name]["gap"]):
                pattern_worst[d.name] = {
                    "gap": gap,
                    "shape": shape,
                    "compiled_us": result.get("compiled_us"),
                    "n_kernels": result.get("n_kernels"),
                }

    if not pattern_worst:
        print(f"No perf data for hardware={hardware}")
        return

    sorted_patterns = sorted(pattern_worst.items(), key=lambda x: -x[1]["gap"])
    print(f"Worst SOL gaps on {hardware} (top {args.top}):\n")
    print(f"{'Pattern':<45} {'Gap':>7} {'us':>8} {'K':>2} {'Shape'}")
    print("-" * 100)
    for name, data in sorted_patterns[:args.top]:
        print(f"{name:<45} {data['gap']:.2f}x {data['compiled_us']:>8.1f} {data['n_kernels'] or '?':>2} {data['shape']}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_shared = sub.add_parser("shared", help="Top patterns by model count")
    p_shared.add_argument("--top", type=int, default=20)

    p_models = sub.add_parser("models", help="Which models have a pattern")
    p_models.add_argument("pattern", help="Pattern hash")

    p_patterns = sub.add_parser("patterns", help="Which patterns a model uses")
    p_patterns.add_argument("model", help="Model name (partial match ok)")

    sub.add_parser("summary", help="Overall stats")

    p_perf = sub.add_parser("perf", help="Show perf results for a hardware")
    p_perf.add_argument("hardware", help="Hardware label (e.g., H100, B200)")
    p_perf.add_argument("--top", type=int, default=30)
    p_perf.add_argument("--sort", choices=["gap_default", "gap_cd", "compiled_us"], default="gap_default")

    p_gaps = sub.add_parser("gaps", help="Worst SOL gaps (optimization targets)")
    p_gaps.add_argument("hardware", help="Hardware label")
    p_gaps.add_argument("--top", type=int, default=20)

    args = parser.parse_args()
    if args.cmd == "shared":
        cmd_shared(args)
    elif args.cmd == "models":
        cmd_models(args)
    elif args.cmd == "patterns":
        cmd_patterns(args)
    elif args.cmd == "summary":
        cmd_summary(args)
    elif args.cmd == "perf":
        cmd_perf(args)
    elif args.cmd == "gaps":
        cmd_gaps(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
