#!/usr/bin/env python
"""Compare base vs branch bench_parallel JSONs (per shape point + geomean)."""
import json, math, sys

def load(p):
    d = json.load(open(p))
    pts = {}
    for repro, shapes in d.items():
        if repro.startswith("_"):
            continue
        for sk, sv in shapes.items():
            if isinstance(sv, dict) and sv.get("compiled_us"):
                pts[(repro.split("/")[-2], sk)] = (sv["compiled_us"], sv.get("n_kernels"))
    return pts

def compare(basef, branchf, label):
    b = load(basef); h = load(branchf)
    keys = sorted(set(b) & set(h))
    ratios = []
    print(f"\n== {label} ({len(keys)} shape points) ==")
    print(f"{'repro':44s} {'shape':28s} {'base_us':>9s} {'branch_us':>9s} {'speedup':>8s} {'nk b->h'}")
    for k in keys:
        bu, bnk = b[k]; hu, hnk = h[k]
        r = bu / hu
        ratios.append(r)
        print(f"{k[0]:44s} {k[1]:28s} {bu:9.1f} {hu:9.1f} {r:8.2f}x {bnk}->{hnk}")
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    print(f"GEOMEAN speedup (base/branch): {gm:.4f}x  min={min(ratios):.2f} max={max(ratios):.2f}")
    return gm

if __name__ == "__main__":
    P = "/tmp/scratch_space/pr56_verify/perf"
    compare(f"{P}/u10_set__base.json", f"{P}/u10_set__layout-sinking.json", "PR5 U10 layout-sinking vs base")
    compare(f"{P}/u10_set__layout-sinking.json", f"{P}/u10_set2__layout-sinking.json", "PR5 branch run1 vs run2 (noise)")
    compare(f"{P}/u30_set__base.json", f"{P}/u30_set__mor-finalize.json", "PR6 U30 mor-finalize vs base")
    compare(f"{P}/u30_set__mor-finalize.json", f"{P}/u30_set2__mor-finalize.json", "PR6 branch run1 vs run2 (noise)")
