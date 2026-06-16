"""Corpus reference-counting / GC — the migration transaction tool.

The corpus lifecycle problem: recapturing a model changes its manifest's
pattern set. Old patterns may lose their last referencing model (orphaned
repros carrying hand-written oracles); new patterns need repro dirs and
oracles. Doing that bookkeeping by hand per-model is error-prone, and the
user requires the corpus NEVER be half-migrated: old repros stay fully
functional until ONE atomic, revertible commit flips the corpus.

This tool prepares and applies that commit.

Commands
--------
  refcount                       health report: pattern -> referencing models
  orphans                        repros referenced by zero manifests
  diff <model_dir> <new.json>    what one recapture WOULD change
  prepare --plan <plan.json>     build + verify the atomic migration plan
  apply   --plan <plan.json>     execute a verified plan (quarantine, stamp)

Policy
------
- Orphans are QUARANTINED, never deleted: meta.json gains
  {"quarantined": {"date": ..., "reason": ...}}. Hand-written oracles are
  expensive and content-addressed patterns recur; deleting burns work.
- `prepare` HARD-FAILS unless every superseded pattern has a successor that
  is benchmarked AND oracle-covered (or is explicitly waived in the plan).
  This is the gate that keeps the migration atomic: you cannot produce an
  applicable plan from a half-finished recapture.
- `apply` touches files but does NOT git-commit; it prints the exact file
  list so the caller makes the single atomic commit (and can revert it).

Plan file schema (input to prepare; prepare adds "verified"/"actions")
----------------------------------------------------------------------
{
  "supersedes": {"<old_hash>": "<new_hash>" | null, ...},
      # null = pattern dropped with no successor (needs waiver)
  "waivers":   {"<old_hash>": "<reason>", ...},
  "benchmark_results": "results/<sweep>.json",      # must contain new hashes
  "oracle_status":     "results/<oracle_timings>.json"  # ditto
}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "repros/canonical"
MODELS = ROOT / "repros/models"


# -- shared corpus views -----------------------------------------------------

def load_refcounts():
    """pattern_hash -> [model dirs referencing it]"""
    refs = defaultdict(list)
    for manifest in sorted(MODELS.rglob("manifest.json")):
        model = str(manifest.parent.relative_to(MODELS))
        for h in json.load(open(manifest)).get("patterns", []):
            refs[h].append(model)
    return refs


def canonical_by_hash():
    """pattern_hash -> canonical repro dir"""
    by_hash = {}
    for d in sorted(CANONICAL.iterdir()):
        if d.is_dir():
            by_hash[d.name.rsplit("_", 1)[-1]] = d
    return by_hash


def repro_has_oracle(d: Path) -> bool:
    return any(d.glob("oracle_*.py"))


def repro_quarantined(d: Path) -> bool:
    meta = d / "meta.json"
    if not meta.exists():
        return False
    return "quarantined" in json.load(open(meta))


# -- read-only commands -------------------------------------------------------

def cmd_refcount():
    refs = load_refcounts()
    repros = canonical_by_hash()
    print(f"Patterns referenced by manifests: {len(refs)}")
    print(f"Canonical repro dirs: {len(repros)}")
    quarantined = sum(1 for d in repros.values() if repro_quarantined(d))
    if quarantined:
        print(f"Quarantined repros: {quarantined}")
    missing = [h for h in refs if h not in repros]
    print(f"Manifest references with NO canonical repro: {len(missing)}")
    for h in missing[:10]:
        print(f"  {h} <- {refs[h][:3]}")
    return 1 if missing else 0


def cmd_orphans():
    refs = load_refcounts()
    repros = canonical_by_hash()
    orphans = [(h, d) for h, d in sorted(repros.items())
               if h not in refs and not repro_quarantined(d)]
    print(f"Active repros referenced by ZERO manifests: {len(orphans)}")
    with_oracle = sum(1 for _, d in orphans if repro_has_oracle(d))
    print(f"  ...with hand-written oracles: {with_oracle}")
    for h, d in orphans:
        print(f"  {d.name} ({'oracle' if repro_has_oracle(d) else 'no-oracle'})")
    return 0


def cmd_diff(model_dir, new_patterns_file):
    refs = load_refcounts()
    repros = canonical_by_hash()
    manifest = MODELS / model_dir / "manifest.json"
    old = set(json.load(open(manifest)).get("patterns", []))
    new = set(json.load(open(new_patterns_file)))
    added, dropped, kept = new - old, old - new, old & new
    print(f"Recapture diff for {model_dir}: {len(kept)} unchanged")
    print(f"  + {len(added)} new patterns:")
    for h in sorted(added):
        status = "repro EXISTS (pattern recurred)" if h in repros else "NEW — needs repro + oracle"
        print(f"      {h} [{status}]")
    print(f"  - {len(dropped)} dropped patterns:")
    for h in sorted(dropped):
        others = [m for m in refs.get(h, []) if m != model_dir]
        verdict = f"still used by {len(others)} models — keep" if others \
                  else "would ORPHAN -> quarantine candidate"
        print(f"      {h} [{verdict}]")
    return 0


# -- migration transaction ------------------------------------------------------

def cmd_prepare(plan_path):
    """Verify a migration plan: every superseded pattern must have a
    benchmarked + oracle-covered successor, or an explicit waiver.
    Writes the verified plan (with concrete actions) back to the plan file.
    HARD-FAILS (exit 1) on any unverified entry — this is the atomicity gate.
    """
    plan = json.load(open(plan_path))
    supersedes = plan.get("supersedes", {})
    waivers = plan.get("waivers", {})
    repros = canonical_by_hash()
    refs = load_refcounts()

    bench = {}
    if plan.get("benchmark_results"):
        raw = json.load(open(ROOT / plan["benchmark_results"]))
        bench = {k: v for k, v in raw.items() if not k.startswith("_")}
    oracle_res = {}
    if plan.get("oracle_status"):
        raw = json.load(open(ROOT / plan["oracle_status"]))
        oracle_res = {k: v for k, v in raw.items() if not k.startswith("_")}

    def hash_in_results(h, results):
        # results are keyed by repro_id (<family>_<hash>) or paths containing it
        return any(h in k for k in results)

    failures = []
    actions = []
    for old_hash, new_hash in sorted(supersedes.items()):
        old_dir = repros.get(old_hash)
        if old_dir is None:
            failures.append(f"{old_hash}: superseded pattern has no repro dir")
            continue
        if new_hash is None:
            if old_hash not in waivers:
                failures.append(
                    f"{old_hash}: dropped with no successor and no waiver")
                continue
            actions.append({"op": "quarantine", "dir": old_dir.name,
                            "reason": f"dropped in migration: {waivers[old_hash]}"})
            continue
        new_dir = repros.get(new_hash)
        if new_dir is None:
            failures.append(f"{old_hash} -> {new_hash}: successor repro dir missing")
            continue
        if not repro_has_oracle(new_dir) and old_hash not in waivers:
            failures.append(f"{old_hash} -> {new_hash}: successor has NO oracle")
            continue
        if bench and not hash_in_results(new_hash, bench) and old_hash not in waivers:
            failures.append(f"{old_hash} -> {new_hash}: successor NOT in benchmark results")
            continue
        if oracle_res and not hash_in_results(new_hash, oracle_res) and old_hash not in waivers:
            failures.append(f"{old_hash} -> {new_hash}: successor NOT in oracle results")
            continue
        still_used = [m for m in refs.get(old_hash, [])]
        actions.append({"op": "quarantine", "dir": old_dir.name,
                        "reason": f"superseded by {new_hash}",
                        "blocked_by_refs": still_used})

    # any superseded pattern still referenced by a manifest the plan doesn't
    # update is a half-migration — fail
    for a in actions:
        if a.get("blocked_by_refs"):
            failures.append(
                f"{a['dir']}: still referenced by manifests {a['blocked_by_refs'][:3]} — "
                f"recapture must update those manifests before this plan can apply")

    print(f"prepare: {len(actions)} actions, {len(failures)} failures")
    for f in failures[:25]:
        print(f"  FAIL {f}")
    if failures:
        print("\nPlan NOT verified — fix the above (finish benchmarking/oracles, "
              "add waivers, or update manifests via recapture) and re-run.")
        return 1

    plan["verified"] = {"date": str(date.today()), "actions": actions}
    Path(plan_path).write_text(json.dumps(plan, indent=2) + "\n")
    print(f"\nPlan VERIFIED: {len(actions)} quarantine actions written to {plan_path}.")
    print("Run `apply --plan` then make ONE commit of the printed file list.")
    return 0


def cmd_apply(plan_path):
    """Execute a verified plan. Stamps quarantine into meta.json files.
    Prints the exact changed-file list; the CALLER makes the single atomic
    git commit (and owns the revert)."""
    plan = json.load(open(plan_path))
    verified = plan.get("verified")
    if not verified:
        print("Plan is not verified — run `prepare --plan` first.")
        return 1
    repros = canonical_by_hash()
    changed = []
    for action in verified["actions"]:
        if action["op"] != "quarantine":
            continue
        d = CANONICAL / action["dir"]
        meta_path = d / "meta.json"
        meta = json.load(open(meta_path))
        if "quarantined" in meta:
            continue  # idempotent
        meta["quarantined"] = {"date": verified["date"],
                               "reason": action["reason"]}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        changed.append(str(meta_path.relative_to(ROOT)))
    print(f"apply: stamped {len(changed)} meta.json files")
    for c in changed[:20]:
        print(f"  {c}")
    print("\nNow make the atomic commit:")
    print(f"  git add {plan_path} " + " ".join(changed[:3]) + (" ..." if len(changed) > 3 else ""))
    print('  git commit -m "Corpus migration: quarantine superseded repros (atomic)"')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refcount")
    sub.add_parser("orphans")
    p = sub.add_parser("diff")
    p.add_argument("model_dir")
    p.add_argument("new_patterns_file")
    for name in ("prepare", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True)
    args = ap.parse_args()

    if args.cmd == "refcount":
        return cmd_refcount()
    if args.cmd == "orphans":
        return cmd_orphans()
    if args.cmd == "diff":
        return cmd_diff(args.model_dir, args.new_patterns_file)
    if args.cmd == "prepare":
        return cmd_prepare(args.plan)
    if args.cmd == "apply":
        return cmd_apply(args.plan)


if __name__ == "__main__":
    sys.exit(main())
