#!/usr/bin/env python3
"""Compare two sweep JSON files (baseline vs with-stash) and report summary."""
import json
import math
import sys
from pathlib import Path

def compare(baseline_path, stash_path):
    baseline = json.load(open(baseline_path))
    with_stash = json.load(open(stash_path))

    stash_repros = {k for k in with_stash if k != '_metadata'}
    base_repros = {k for k in baseline if k != '_metadata'}
    common = stash_repros & base_repros

    print(f"Baseline repros: {len(base_repros)}")
    print(f"With-stash repros: {len(stash_repros)}")
    print(f"Common repros: {len(common)}")

    if not common:
        print("No common repros to compare")
        return

    # Best-of-configs comparison
    speedups = []
    repro_details = []

    for repro in sorted(common):
        base_configs = baseline[repro].get('configs', {})
        stash_configs = with_stash[repro].get('configs', {})
        if not base_configs or not stash_configs:
            continue
        base_best = min(c['us'] for c in base_configs.values())
        stash_best = min(c['us'] for c in stash_configs.values())
        if base_best > 0 and stash_best > 0:
            speedup = base_best / stash_best
            speedups.append(speedup)
            repro_details.append((speedup, repro, base_best, stash_best))

    if not speedups:
        print("No valid comparisons")
        return

    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    improved = sum(1 for s in speedups if s > 1.02)
    regressed = sum(1 for s in speedups if s < 0.98)
    neutral = len(speedups) - improved - regressed

    print(f"\n=== BEST-OF-CONFIGS COMPARISON ===")
    print(f"Geomean speedup: {geomean:.4f}x")
    print(f"Improved (>2%): {improved}/{len(speedups)}")
    print(f"Regressed (<-2%): {regressed}/{len(speedups)}")
    print(f"Neutral: {neutral}/{len(speedups)}")

    # Per-config
    print(f"\n=== PER-CONFIG ===")
    all_config_names = set()
    for repro in common:
        all_config_names.update(baseline[repro].get('configs', {}).keys())
        all_config_names.update(with_stash[repro].get('configs', {}).keys())

    for cfg_name in sorted(all_config_names):
        cfg_speedups = []
        for repro in sorted(common):
            base_configs = baseline[repro].get('configs', {})
            stash_configs = with_stash[repro].get('configs', {})
            if cfg_name in base_configs and cfg_name in stash_configs:
                bt = base_configs[cfg_name]['us']
                st = stash_configs[cfg_name]['us']
                if bt > 0 and st > 0:
                    cfg_speedups.append(bt / st)
        if cfg_speedups:
            geo = math.exp(sum(math.log(s) for s in cfg_speedups) / len(cfg_speedups))
            imp = sum(1 for s in cfg_speedups if s > 1.02)
            reg = sum(1 for s in cfg_speedups if s < 0.98)
            print(f"  {cfg_name}: geomean={geo:.4f}x  improved={imp}  regressed={reg}  n={len(cfg_speedups)}")

    # Top improvements/regressions
    repro_details.sort(reverse=True)
    print(f"\nTop 10 improvements:")
    for sp, name, bt, st in repro_details[:10]:
        short = name.split('/')[-2]
        print(f"  {sp:.3f}x  {short}  ({bt:.1f}us -> {st:.1f}us)")

    print(f"\nTop 10 regressions:")
    for sp, name, bt, st in repro_details[-10:]:
        short = name.split('/')[-2]
        print(f"  {sp:.3f}x  {short}  ({bt:.1f}us -> {st:.1f}us)")


if __name__ == "__main__":
    baseline_path = sys.argv[1] if len(sys.argv) > 1 else "sweep_3config_interleaved.json"
    stash_path = sys.argv[2] if len(sys.argv) > 2 else "sweep_3config_with_stash.json"
    compare(baseline_path, stash_path)
