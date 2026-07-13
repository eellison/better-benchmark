#!/usr/bin/env python3
"""Canonical genai per-commit A/B delta tool (LOCKED measurement path).

================================================================================
NOISY-PATH LESSON (genai_stability finding, 2026-06-23)
================================================================================
genai single-kernel deltas MUST go through `scripts/bench_parallel.py
--full-graphs` with the per-GPU bench lock ON (INDUCTOR_GPU_BENCH_LOCK=1, the
default). Any bespoke or *concurrent* timing route is INVALID: the ±10-44%
run-to-run "instability" we once attributed to these micros was a
MEASUREMENT-PATH artifact from a route that skipped the GPU lock, NOT real
kernel instability and NOT a harness bug.

Through this locked path the proven cold run-to-run spread is 0.03-0.42%, the
generated Triton source is bit-identical across cold runs, and the autotune
best_config is identical across cold runs. The single-kernel DETECTION THRESHOLD
is therefore +-0.5% (measured floor 0.03-0.38%): a per-pair delta < 0.5% is "at
floor / no move"; > 1% is "real signal".

=> The ONLY supported way to get a genai per-commit delta is this tool. Do not
   write a timing loop. This module never times anything itself; it shells out
   to bench_parallel --full-graphs once per (commit, micro) and post-processes
   the JSON it writes.

================================================================================
MECHANISM (shadow checkout, no torch rebuild)
================================================================================
Every file that differs base<->HEAD on this perf branch is PURE-PYTHON inductor,
so the prebuilt torch .so from /tmp/pytorch-work (HEAD) is valid for ALL these
shadow checkouts. We do NOT rebuild torch per commit. Instead:

  - a THROWAWAY git worktree of the pytorch repo is checked out at each commit
    (default /tmp/scratch_space/genai_ab_wt; NEVER touches /tmp/pytorch-work or
    /tmp/scratch_space/pl-walk HEADs/WIP),
  - the worktree's build artifacts (*.so, *.a, torch/_C*.so, torch/{bin,include},
    torch/lib/{cmake,pkgconfig}) are SYMLINKED from /tmp/pytorch-work once,
  - PYTHONPATH shadows the worktree's python sources,
  - we VERIFY torch + torch._inductor.config resolve to the worktree and that the
    bench JSON's _metadata.pytorch_commit matches the requested commit BEFORE
    trusting any timing.

Per commit: a FRESH inductor cache (a shared inductor cache poisons cross-commit
deltas -- correctness). A single TRITON_CACHE_DIR MAY be shared across commits
(keys on generated source -> safe; saves PTX recompile).

================================================================================
USAGE
================================================================================
  # Run the 4 canonical mover A/B pairs (each mover vs its predecessor):
  python genai_ab.py --pairs-preset movers

  # Arbitrary A/B pairs (repeatable): A is predecessor, B is mover.
  python genai_ab.py --pair <A_sha> <B_sha> [--pair <A> <B> ...]

  # Bench an explicit commit list (no pairing; e.g. a fine walk):
  python genai_ab.py --commits <sha1> <sha2> ...

  # Restrict the micro set (default = all 8 fwd+bwd):
  python genai_ab.py --pairs-preset movers --only SoftmaxForward,CrossEntropyForward

Env:
  GENAI_AB_GPUS      GPU index list for bench_parallel --gpus (default "0")
  GENAI_AB_WT        throwaway worktree path (default /tmp/scratch_space/genai_ab_wt)
  GENAI_AB_OUT       output dir (default results/pytorch_landing/genai_stability/mover_ab)
  GENAI_AB_N         baseline trials per (commit,micro) (default 3)
  GENAI_AB_TRITON    shared TRITON_CACHE_DIR (default /tmp/scratch_space/genai_ab_triton)
  PYTORCH_SRC        prebuilt-.so source repo (default /tmp/pytorch-work)
"""
import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/tmp/scratch_space/better_benchmark")
PYT = Path(os.environ.get("PYTORCH_SRC", "/tmp/pytorch-work"))
WT = Path(os.environ.get("GENAI_AB_WT", "/tmp/scratch_space/genai_ab_wt"))
OUT = Path(os.environ.get(
    "GENAI_AB_OUT",
    str(ROOT / "results/pytorch_landing/genai_stability/mover_ab")))
GPUS = os.environ.get("GENAI_AB_GPUS", "0")
N_BASE = int(os.environ.get("GENAI_AB_N", "3"))
TRITON_CACHE = os.environ.get("GENAI_AB_TRITON", "/tmp/scratch_space/genai_ab_triton")

# Pin: the prebuilt .so we shadow with. assert_pytorch_untouched() enforces this.
PYT_HEAD = "daa79cd25ca9a80bfd65799394cf4255d6be75a6"

MICROS = {
    "SoftmaxForward":       "repros/models/genai/static/SoftmaxForward",
    "SoftmaxBackward":      "repros/models/genai/static/SoftmaxBackward",
    "CrossEntropyForward":  "repros/models/genai/static/CrossEntropyForward",
    "CrossEntropyBackward": "repros/models/genai/static/CrossEntropyBackward",
    "RMSNormForward":       "repros/models/genai/static/RMSNormForward",
    "RMSNormBackward":      "repros/models/genai/static/RMSNormBackward",
    "LayerNormForward":     "repros/models/genai/static/LayerNormForward",
    "LayerNormBackward":    "repros/models/genai/static/LayerNormBackward",
}

# The 4 canonical mover pairs (predecessor, mover). predecessor=A, mover=B.
MOVER_PAIRS = [
    ("mega-commit",    "5e2ab3055de", "97385fb3273"),
    ("scalar-acc",     "a85d79a900a", "9dde2c59a51"),
    ("online-softmax", "e9a67c98a8c", "a26fc2c8bf4"),
    ("rotate_half",    "52d4cadfac0", "1406552b9d3"),
]

# Detection thresholds (percent).
FLOOR = 0.5    # < this = "no-move / at floor"
SIGNAL = 1.0   # > this = "real signal"
SPREAD_BUMP = 1.0  # if a micro's 3-trial spread exceeds this %, re-run with N=5

OUT.mkdir(parents=True, exist_ok=True)
Path(TRITON_CACHE).mkdir(parents=True, exist_ok=True)


# ----------------------------- provenance guards ----------------------------
def assert_pytorch_untouched():
    head = subprocess.run(["git", "-C", str(PYT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert head == PYT_HEAD, f"PYTORCH_SRC HEAD MOVED: {head} != {PYT_HEAD}"


# ----------------------------- worktree shadow ------------------------------
def wire_build_artifacts():
    """Symlink prebuilt .so/.a build artifacts from PYT into the worktree.
    Idempotent. Must be called after any checkout (checkout leaves them absent).
    """
    src = PYT
    # torch/*.so compiled extensions (e.g. _C.cpython-*.so)
    for f in glob.glob(str(src / "torch" / "*.so")):
        b = os.path.basename(f)
        dst = WT / "torch" / b
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(f)
    # torch/lib/*.so + *.a
    libdst = WT / "torch" / "lib"
    libdst.mkdir(parents=True, exist_ok=True)
    for pat in ("*.so", "*.a", "*.so.*"):
        for f in glob.glob(str(src / "torch" / "lib" / pat)):
            b = os.path.basename(f)
            dst = libdst / b
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(f)
    for d in ("cmake", "pkgconfig"):
        s = src / "torch" / "lib" / d
        if not s.exists():
            continue
        dst = libdst / d
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink():
                dst.unlink()
            else:
                subprocess.run(["rm", "-rf", str(dst)])
        dst.symlink_to(s)
    for d in ("bin", "include"):
        s = src / "torch" / d
        if not s.exists():
            continue
        dst = WT / "torch" / d
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            subprocess.run(["rm", "-rf", str(dst)])
        dst.symlink_to(s)


def ensure_worktree():
    """Create the throwaway worktree if absent (detached at PYT_HEAD)."""
    wl = subprocess.run(["git", "-C", str(PYT), "worktree", "list"],
                        capture_output=True, text=True).stdout
    if str(WT) not in wl:
        subprocess.run(["git", "-C", str(PYT), "worktree", "add", "--detach",
                        str(WT), PYT_HEAD], check=True,
                       capture_output=True, text=True)
    wire_build_artifacts()


def checkout_state(sha: str) -> str:
    r = subprocess.run(["git", "-C", str(WT), "checkout", "--quiet", "--force", sha],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"checkout {sha} failed: {r.stderr}")
    wire_build_artifacts()  # checkout may overwrite torch/ tree
    got = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    assert got.startswith(sha) or sha.startswith(got[:11]), \
        f"checkout mismatch {got} != {sha}"
    return got


def verify_shadow_resolves(sha: str):
    code = (
        "import torch, torch._inductor.config as c\n"
        f"assert torch.__file__.startswith({str(WT)!r}), torch.__file__\n"
        f"assert c.__file__.startswith({str(WT)!r}), c.__file__\n"
        "print('OK', torch.__version__)\n"
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(WT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, cwd=str(ROOT))
    if r.returncode != 0 or "OK" not in r.stdout:
        raise RuntimeError(f"shadow verify FAILED for {sha}: {r.stderr or r.stdout}")


# ----------------------------- harvest helpers ------------------------------
def harvest_best_configs(cache_dir: str):
    out = []
    for f in sorted(glob.glob(os.path.join(cache_dir, "**", "*.best_config"),
                              recursive=True)):
        try:
            cfg = json.load(open(f))
            out.append({k: cfg.get(k) for k in
                        ("XBLOCK", "R0_BLOCK", "R1_BLOCK", "num_warps",
                         "num_stages", "found_by_coordesc")})
        except Exception as e:
            out.append({"_err": str(e), "_file": os.path.basename(f)})
    return out


def parse_timings(jpath: Path):
    """Return {graph_basename: {compiled_us, coord_descent_us, n_kernels}}."""
    d = json.load(open(jpath))
    meta = d.get("_metadata", {})
    res = {}
    for k, v in d.items():
        if k.startswith("_") or k.endswith("__") or not isinstance(v, dict):
            continue
        pt = v.get("default")
        if not pt:
            pt = next((vv for kk, vv in v.items()
                       if kk != "__graph__" and isinstance(vv, dict)
                       and "compiled_us" in vv), None)
        if pt:
            res[os.path.basename(k)] = {
                "compiled_us": pt.get("compiled_us"),
                "coord_descent_us": pt.get("coord_descent_us"),
                "n_kernels": pt.get("n_kernels"),
            }
    return res, meta.get("pytorch_commit", "")


def spread_pct(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    mn, mx = min(vals), max(vals)
    return (mx - mn) / mn * 100.0 if mn else 0.0


# ----------------------------- bench one (commit,micro) ---------------------
def bench_micro(sha: str, micro: str, path: str, n_trials: int, tag: str):
    """Run bench_parallel --full-graphs n_trials times, fresh inductor cache each
    trial, shared TRITON_CACHE_DIR. Returns a record with per-graph stats +
    autotune best_config (from the last trial's cache)."""
    trials = []  # each: {graph: {compiled_us, coord_descent_us, n_kernels}}
    last_cfgs = []
    rundir = OUT / "runs" / f"{sha[:11]}_{micro}"
    rundir.mkdir(parents=True, exist_ok=True)
    for i in range(n_trials):
        cache = tempfile.mkdtemp(prefix=f"ind_{sha[:8]}_{micro}_{i}_",
                                 dir="/tmp/scratch_space")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(WT)
        env["TORCHINDUCTOR_CACHE_DIR"] = cache       # FRESH per trial (correctness)
        env["TRITON_CACHE_DIR"] = TRITON_CACHE       # shared (safe; saves PTX)
        jpath = rundir / f"trial_{i}.json"
        log = rundir / f"trial_{i}.log"
        cmd = [sys.executable, "scripts/bench_parallel.py", path,
               "--full-graphs", "--gpus", GPUS, "--output", str(jpath)]
        with open(log, "w") as lf:
            subprocess.run(cmd, env=env, cwd=str(ROOT),
                           stdout=lf, stderr=subprocess.STDOUT)
        if not jpath.exists():
            raise RuntimeError(f"bench produced no JSON for {sha} {micro} trial {i} "
                               f"(see {log})")
        timings, meta_pyt = parse_timings(jpath)
        assert meta_pyt.startswith(sha) or sha.startswith(meta_pyt[:11]), \
            f"BENCH metadata pytorch_commit={meta_pyt} != requested {sha}"
        trials.append(timings)
        last_cfgs = harvest_best_configs(cache)
        subprocess.run(["rm", "-rf", cache])
    # aggregate per graph
    graphs = sorted({g for t in trials for g in t})
    per_graph = {}
    for g in graphs:
        comp = [t[g]["compiled_us"] for t in trials if g in t and t[g]["compiled_us"] is not None]
        cd = [t[g]["coord_descent_us"] for t in trials if g in t and t[g]["coord_descent_us"] is not None]
        per_graph[g] = {
            "compiled_us_median": round(statistics.median(comp), 2) if comp else None,
            "compiled_us_min": round(min(comp), 2) if comp else None,
            "compiled_us_spread_pct": round(spread_pct(comp), 3),
            "cd_us_median": round(statistics.median(cd), 2) if cd else None,
            "cd_us_min": round(min(cd), 2) if cd else None,
            "cd_us_spread_pct": round(spread_pct(cd), 3),
            "n_kernels": trials[0][g].get("n_kernels") if g in trials[0] else None,
            "n_trials": len(comp),
        }
    return {"sha": sha, "micro": micro, "n_trials": n_trials,
            "per_graph": per_graph, "best_configs": last_cfgs}


def bench_commit(sha: str, micros: dict, tag: str):
    """Bench all requested micros at one commit. Spread-guarded N."""
    assert_pytorch_untouched()
    print(f"\n=== checkout {sha} ({tag}) ===", flush=True)
    got = checkout_state(sha)
    verify_shadow_resolves(sha)
    print(f"  shadow verified @ {got[:11]}", flush=True)
    recs = {}
    for micro, path in micros.items():
        t0 = time.time()
        rec = bench_micro(sha, micro, path, N_BASE, tag)
        # spread guard: if any graph exceeds SPREAD_BUMP%, re-run at N=5
        worst = max((g["compiled_us_spread_pct"] or 0.0)
                    for g in rec["per_graph"].values()) if rec["per_graph"] else 0.0
        bumped = False
        if worst > SPREAD_BUMP:
            print(f"  [{micro}] spread {worst:.2f}% > {SPREAD_BUMP}% -> re-run N=5",
                  flush=True)
            rec = bench_micro(sha, micro, path, 5, tag)
            rec["spread_bumped"] = True
            bumped = True
        dt = time.time() - t0
        mins = ", ".join(
            f"{g.split('full_graph_')[-1].split('.')[0]}={v['compiled_us_min']}"
            for g, v in rec["per_graph"].items())
        print(f"  [{micro}] {dt:.0f}s worst_spread={worst:.2f}%"
              f"{' (bumped)' if bumped else ''}  min_us: {mins}", flush=True)
        recs[micro] = rec
    assert_pytorch_untouched()
    return recs


# ----------------------------- delta arithmetic -----------------------------
def verdict(delta_pct):
    a = abs(delta_pct)
    if a < FLOOR:
        return "no-move"
    if a > SIGNAL:
        return "real"
    return "marginal"


def compute_pair(name, a_sha, b_sha, recsA, recsB):
    """Per-micro, per-graph delta: improvement of B (mover) over A (predecessor).
    delta_pct > 0 means B is FASTER (us went down)."""
    rows = []
    for micro in sorted(set(recsA) & set(recsB)):
        gA, gB = recsA[micro]["per_graph"], recsB[micro]["per_graph"]
        for g in sorted(set(gA) & set(gB)):
            pa = gA[g].get("compiled_us_min")
            pb = gB[g].get("compiled_us_min")
            if pa is None or pb is None:
                continue
            d = (pa - pb) / pa * 100.0
            cfgA = recsA[micro]["best_configs"]
            cfgB = recsB[micro]["best_configs"]
            cfg_flip = cfgA != cfgB
            rows.append({
                "pair": name, "A": a_sha, "B": b_sha, "micro": micro,
                "graph": g,
                "predecessor_us": pa, "mover_us": pb,
                "delta_pct": round(d, 3),
                "verdict": verdict(d),
                "config_flip": cfg_flip,
                "best_config_A": cfgA, "best_config_B": cfgB,
            })
    return rows


# ----------------------------- main -----------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-preset", choices=["movers"],
                    help="run the 4 canonical mover A/B pairs")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A_SHA", "B_SHA"),
                    help="A/B pair (A=predecessor, B=mover); repeatable")
    ap.add_argument("--commits", nargs="+", help="explicit commit list (no pairing)")
    ap.add_argument("--only", help="comma list of micro names to restrict to")
    args = ap.parse_args()

    micros = dict(MICROS)
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        micros = {k: v for k, v in MICROS.items() if k in want}
        assert micros, f"--only matched no micros (have {list(MICROS)})"

    # Build pair list (name, A, B).
    pairs = []
    if args.pairs_preset == "movers":
        pairs = list(MOVER_PAIRS)
    if args.pair:
        pairs += [(f"pair{i}", a, b) for i, (a, b) in enumerate(args.pair)]

    assert_pytorch_untouched()
    ensure_worktree()

    # Unique commits to bench (dedup across pairs / --commits).
    commits = []
    for _, a, b in pairs:
        commits += [a, b]
    if args.commits:
        commits += args.commits
    seen, uniq = set(), []
    for c in commits:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    assert uniq, "nothing to bench: pass --pairs-preset, --pair, or --commits"

    print(f"genai_ab: {len(uniq)} commits x {len(micros)} micros, GPUs={GPUS}, "
          f"N={N_BASE} (bump to 5 if spread>{SPREAD_BUMP}%), WT={WT}")

    bench = {}  # sha -> {micro -> rec}
    for sha in uniq:
        bench[sha] = bench_commit(sha, micros, sha)
        json.dump(bench, open(OUT / "raw_per_commit.json", "w"), indent=1)

    # Per-pair deltas.
    all_rows = []
    for name, a, b in pairs:
        all_rows += compute_pair(name, a, b, bench[a], bench[b])
    json.dump(all_rows, open(OUT / "pair_deltas.json", "w"), indent=1)

    # Human table.
    print("\n" + "=" * 100)
    print(f"{'pair':<15}{'micro':<22}{'graph':<16}"
          f"{'pred_us':>10}{'mover_us':>10}{'delta%':>9}  {'verdict':<9} cfg_flip")
    print("-" * 100)
    for r in all_rows:
        g = r["graph"].split("full_graph_")[-1].split(".")[0]
        print(f"{r['pair']:<15}{r['micro']:<22}{g:<16}"
              f"{r['predecessor_us']:>10.1f}{r['mover_us']:>10.1f}"
              f"{r['delta_pct']:>+9.2f}  {r['verdict']:<9} {r['config_flip']}")
    print("=" * 100)
    print(f"wrote {OUT/'pair_deltas.json'} and {OUT/'raw_per_commit.json'}")
    assert_pytorch_untouched()


if __name__ == "__main__":
    main()
