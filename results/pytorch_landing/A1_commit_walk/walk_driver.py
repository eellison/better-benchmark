#!/usr/bin/env python3
"""A1 COMMIT WALK driver — cumulative per-commit model-perf deltas.

ISOLATION: benches each commit-K state from a DEDICATED worktree
(/tmp/scratch_space/pl-walk) via a PYTHONPATH shadow (worktree torch shadows the
install; build artifacts symlinked once). NEVER touches /tmp/pytorch-work HEAD or
its uncommitted WIP. Verify-before-bench: confirms `import torch._inductor` resolves
to the worktree at the expected commit.

NO new timing code: drives scripts/bench_parallel.py --all-shapes (the same path
that produced the endpoint kernels.json files) and reuses the EXACT rollup
arithmetic from results/perf_ab/rebaseline_5e2ab_2026-06-19/compute_rebaseline.py.

For internal consistency, ALL states (including A=base and B=HEAD) are RE-BENCHED
in-shadow here — per-commit deltas are tiny (noise floor ~0.82%), so they must be
computed across a self-consistent shadow sweep, not mixed with the install-benched
endpoints (which carry a ~5% setup offset). The install endpoints remain the
reconciliation reference.

Usage:
  python walk_driver.py <sha1> <sha2> ...        # bench these states (in order given)
  python walk_driver.py --from-file states.txt   # one sha per line
Env:
  PL_GPUS (default "0,1,2,3"), PL_WORKTREE, PL_OUT
"""
import json, os, subprocess, sys, time, statistics, tempfile, hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path("/tmp/scratch_space/better_benchmark")
WT = Path(os.environ.get("PL_WORKTREE", "/tmp/scratch_space/pl-walk"))
OUT = Path(os.environ.get("PL_OUT", str(ROOT / "results/pytorch_landing/A1_commit_walk")))
STATES_DIR = OUT / "states"          # per-state corpus kernels.json
GPUS = os.environ.get("PL_GPUS", "0,1,2,3")
OCCDIR = ROOT / "results/perf_ab/occurrences"
PYT = "/tmp/pytorch-work"

STATES_DIR.mkdir(parents=True, exist_ok=True)

# ---------- rollup arithmetic (verbatim from compute_rebaseline.py) ----------
def _phash(repro_path: str) -> str:
    return Path(repro_path).parent.name.rsplit("_", 1)[-1]

def load_arm(path: Path):
    data = json.loads(Path(path).read_text())
    exact, by_pat = {}, defaultdict(list)
    for key, val in data.items():
        if key.startswith("_") or not isinstance(val, dict):
            continue
        ph = _phash(key)
        for label, m in val.items():
            if not isinstance(m, dict):
                continue
            us = m.get("coord_descent_us")
            if us is None:
                us = m.get("compiled_us")
            if us is None:
                continue
            sh = label.rsplit("_", 1)[-1]
            exact[(ph, sh)] = float(us)
            by_pat[ph].append(float(us))
    rep = {ph: statistics.median(v) for ph, v in by_pat.items() if v}
    return exact, rep

def fusible_us(exact, rep, ph, sh):
    v = exact.get((ph, sh))
    if v is not None:
        return v, "exact"
    v = rep.get(ph)
    if v is not None:
        return v, "rep"
    return None, "unmatched"

def rollup_models(base_path: Path, br_path: Path):
    """Per-model pct_improvement of br vs base. EXACTLY compute_rebaseline.rollup."""
    be, brp = load_arm(base_path)
    bre, brrp = load_arm(br_path)
    models = {}
    for sc in sorted(OCCDIR.glob("*.json")):
        rec = json.loads(sc.read_text())
        suite, mode, model = rec["suite"], rec["mode"], rec["model"]
        fus_base = fus_branch = delta = ext_total = 0.0
        unmatched = []
        for ph, shapes in rec["fusible"].items():
            for sh, cnt in shapes.items():
                b, bk = fusible_us(be, brp, ph, sh)
                r, rk = fusible_us(bre, brrp, ph, sh)
                if b is None or r is None:
                    unmatched.append(f"{ph}/{sh}x{cnt}")
                    continue
                fus_base += b * cnt
                fus_branch += r * cnt
                delta += (b - r) * cnt
        for e in rec.get("extern", []):
            eu = e.get("baseline_us")
            if eu is None:
                continue
            ext_total += eu * e["count"]
        mb = fus_base + ext_total
        pct = (delta / mb * 100.0) if mb > 0 else None
        models[f"{suite}/{mode}/{model}"] = pct
    return models

def cut(models, genai_excl=True):
    pcts = [p for m, p in models.items()
            if p is not None and (("genai" not in m) if genai_excl else True)]
    if not pcts:
        return None
    ratios = [1.0/(1.0 - p/100.0) for p in pcts if (1.0 - p/100.0) > 0]
    geo = statistics.geometric_mean(ratios) if ratios else None
    return dict(
        n=len(pcts),
        median=round(statistics.median(pcts), 4),
        mean=round(statistics.mean(pcts), 4),
        geomean=round((geo-1.0)*100.0, 4) if geo is not None else None,
    )

# ---------- worktree shadow management ----------
def assert_tmpwork_untouched():
    head = subprocess.run(["git","-C",PYT,"rev-parse","HEAD"],
                          capture_output=True,text=True).stdout.strip()
    assert head == "daa79cd25ca9a80bfd65799394cf4255d6be75a6", f"tmp_work HEAD MOVED: {head}"

def checkout_state(sha: str):
    """Checkout commit SHA in the worktree (clean), verify it took."""
    r = subprocess.run(["git","-C",str(WT),"checkout","--quiet","--force",sha],
                       capture_output=True,text=True)
    if r.returncode != 0:
        raise RuntimeError(f"checkout {sha} failed: {r.stderr}")
    got = subprocess.run(["git","-C",str(WT),"rev-parse","HEAD"],
                        capture_output=True,text=True).stdout.strip()
    assert got.startswith(sha) or sha.startswith(got[:11]), f"checkout mismatch {got} != {sha}"
    return got

def verify_shadow_resolves(expect_sha: str):
    """Confirm `import torch._inductor` resolves to the worktree before benching."""
    code = (
        "import sys; sys.path.insert(0, %r);\n"
        "import torch, torch._inductor.config as c;\n"
        "import os;\n"
        "assert c.__file__.startswith(%r), c.__file__;\n"
        "print('OK', torch.__file__, c.__file__)\n"
    ) % (str(WT), str(WT))
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(WT)
    r = subprocess.run([sys.executable,"-c",code], capture_output=True, text=True,
                       env=env, cwd=str(ROOT))
    if r.returncode != 0 or "OK" not in r.stdout:
        raise RuntimeError(f"shadow verify FAILED for {expect_sha}: {r.stderr or r.stdout}")

def bench_state(sha: str) -> Path:
    """Full-corpus shadow sweep at commit SHA -> states/<sha>.json. Returns path."""
    out_json = STATES_DIR / f"{sha}.json"
    if out_json.exists():
        # idempotent resume: trust an existing complete sweep (1727 ok)
        try:
            d = json.loads(out_json.read_text())
            n = sum(1 for k in d if not k.startswith("_"))
            if n >= 1700 and d.get("__summary__",{}).get("failed",0) == 0:
                print(f"  [{sha}] reuse existing sweep ({n} repros)", flush=True)
                return out_json
        except Exception:
            pass
    assert_tmpwork_untouched()
    checkout_state(sha)
    verify_shadow_resolves(sha)
    cache = tempfile.mkdtemp(prefix=f"ind_{sha[:8]}_", dir="/tmp/scratch_space")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WT)
    env["TORCHINDUCTOR_CACHE_DIR"] = cache
    t0 = time.time()
    log = STATES_DIR / f"{sha}.log"
    with open(log, "w") as lf:
        p = subprocess.run(
            [sys.executable, "scripts/bench_parallel.py", "repros/canonical",
             "--all-shapes", "--gpus", GPUS, "--output", str(out_json)],
            env=env, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    # cleanup cache to save disk
    subprocess.run(["rm","-rf",cache])
    d = json.loads(out_json.read_text())
    n = sum(1 for k in d if not k.startswith("_"))
    meta_pyt = d.get("_metadata",{}).get("pytorch_commit","")
    assert meta_pyt.startswith(sha) or sha.startswith(meta_pyt[:11]), \
        f"BENCH METADATA pytorch_commit={meta_pyt} != state {sha}"
    print(f"  [{sha}] sweep done: {n} repros, {dt/60:.1f}min, "
          f"rc={p.returncode}, meta_pyt={meta_pyt[:11]}", flush=True)
    return out_json

# ---------- main ----------
def main():
    if "--from-file" in sys.argv:
        states = [l.strip() for l in
                  Path(sys.argv[sys.argv.index("--from-file")+1]).read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
    else:
        states = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not states:
        print("no states given"); return

    print(f"A1 walk driver: {len(states)} states, GPUs={GPUS}, WT={WT}")
    assert_tmpwork_untouched()

    # The BASE for rollup is the in-shadow base sweep (states/<BASE>.json).
    BASE_SHA = "5e2ab3055de1bc4bffe2e7feffe1fc7ff7af8b10"
    base_path = STATES_DIR / f"{BASE_SHA}.json"

    results_path = OUT / "walk_running.json"
    running = {}
    if results_path.exists():
        running = json.loads(results_path.read_text())

    for i, sha in enumerate(states):
        print(f"\n=== [{i+1}/{len(states)}] state {sha} ===", flush=True)
        sp = bench_state(sha)
        # rollup vs in-shadow base (if base benched yet)
        rec = {"sha": sha, "state_json": str(sp.name)}
        if base_path.exists() and sha != BASE_SHA[:len(sha)] and not BASE_SHA.startswith(sha):
            models = rollup_models(base_path, sp)  # base vs state -> state's improvement over base
            rec["genai_excl"] = cut(models, True)
            rec["genai_incl"] = cut(models, False)
            ge = rec["genai_excl"]
            print(f"  vs BASE: genai-excl geomean={ge['geomean']:+.3f}% "
                  f"median={ge['median']:+.3f}% (n={ge['n']})", flush=True)
        elif BASE_SHA.startswith(sha) or sha == BASE_SHA:
            rec["genai_excl"] = {"n": 0, "median": 0.0, "mean": 0.0, "geomean": 0.0,
                                 "note": "this IS the base"}
            print("  (base state — 0% by definition)", flush=True)
        running[sha] = rec
        results_path.write_text(json.dumps(running, indent=2))

    assert_tmpwork_untouched()
    print(f"\nDONE. results -> {results_path}")

if __name__ == "__main__":
    main()
