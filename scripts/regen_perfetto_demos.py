#!/usr/bin/env python3
"""One-command regeneration of the Perfetto demo-trace set.

This is a thin ORCHESTRATOR -- it contains NO timing code of its own. Per
model it shells out to the two existing tools, in order:

  1. GPU (locked, ~4 min/model):
       python scripts/with_gpu_lock.py --gpu <N> -- \
         python scripts/model_attribution.py --corpus-root repros \
           --suite <suite> --mode <mode> --models <model> --collect-order \
           -o <out-dir>/attr_<model>.json
     model_attribution benches every fusible partition + extern once under
     the per-GPU flock and emits the execution-ordered per_node timeline.

  2. CPU (seconds, pure JSON join -- no benching):
       python scripts/rollup_to_perfetto.py --source attribution \
         --attribution <attr.json> --model <model> \
         --oracle-timings <timings> --out <out-dir>/<Model>_oracle_overlay.perfetto.json
     joins the oracle-ceiling track onto the true-to-e2e timeline.

Models are run SEQUENTIALLY: the GPU step holds the GPU lock anyway, so
there is nothing to parallelize.

After generating each trace the script VALIDATES it (valid Chrome-trace
JSON; exactly ONE model/pid in the trace, named exactly the requested model
-- guarding against rollup_to_perfetto's substring --model match pulling a
sibling model like MobileBertForMaskedLM in for BertForMaskedLM; tid 1/2
slice counts equal to the attr JSON's own fusible/extern per_node counts
for the exact model key, with each tid required only when the model
actually has entries of that kind) and prints the per-trace summary line
(fusible compile total vs oracle total, ratio, oracle hit/miss counts). A
trace where ALL fusible oracle slices are misses is honest-but-unpriced: it
is named <Model>_UNPRICED_example.perfetto.json instead of
<Model>_oracle_overlay.perfetto.json, matching the existing convention.

A MANIFEST.json is written/refreshed in the out-dir recording, per trace:
the model, the PyTorch commit the attribution was benched on
(torch.version.git_version, stamped into the attr JSON right after the GPU
step), the timings file + its priced-family count, the fusible
compile/oracle totals + ratio, and hit/miss counts. This is what makes
staleness VISIBLE -- a demo trace generated on an old PyTorch commit or an
old timings file now says so.

USAGE (from the REPO ROOT, with the GPU lock available):

  # full regen of the demo set (GPU: ~4 min/model, sequential)
  python scripts/regen_perfetto_demos.py

  # only the CPU overlay join, reusing attr JSONs already in the out-dir
  # (for when only the timings/oracle side changed)
  python scripts/regen_perfetto_demos.py --skip-gpu

  # a subset / a scratch dir
  python scripts/regen_perfetto_demos.py --models alexnet --out-dir /tmp/scratch_space/regen_check
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = "AllenaiLongformerBase,alexnet,mobilenetv2_100"
DEFAULT_OUT_DIR = ROOT / "results" / "pytorch_landing" / "perfetto_demos"
DEFAULT_TIMINGS = ROOT / "results" / "all_oracle_timings_b200_v2.json"

# Track ids in rollup_to_perfetto's attribution mode (with --oracle-timings).
TID_FUSIBLE = 1
TID_EXTERN = 2
TID_ORACLE = 3


def fail(msg: str) -> None:
    print(f"[regen_perfetto_demos] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Model discovery: map a model name to its (suite, mode) corpus location
# ---------------------------------------------------------------------------

def find_suite_mode(model: str, corpus_root: Path) -> tuple[str, str]:
    """Locate repros/models/<suite>/<mode>/<model>; prefer infer over train."""
    hits: list[tuple[str, str]] = []
    models_root = corpus_root / "models"
    for suite_dir in sorted(models_root.iterdir()):
        if not suite_dir.is_dir():
            continue
        for mode_dir in sorted(suite_dir.iterdir()):
            if (mode_dir / model).is_dir():
                hits.append((suite_dir.name, mode_dir.name))
    if not hits:
        fail(f"model '{model}' not found under {models_root}/<suite>/<mode>/")
    # Prefer infer (the demo set is inference), then static, then anything.
    for preferred in ("infer", "static"):
        for suite, mode in hits:
            if mode == preferred:
                return suite, mode
    return hits[0]


# ---------------------------------------------------------------------------
# Attr JSON handling
# ---------------------------------------------------------------------------

def attr_path_for(model: str, out_dir: Path) -> Path:
    return out_dir / f"attr_{model}.json"


def find_reusable_attr(model: str, out_dir: Path) -> Path | None:
    """Find an existing attribution JSON in out_dir usable for `model`.

    Prefers the canonical attr_<model>.json; otherwise scans every
    attr_*.json for one whose models dict has a per_node entry for `model`
    (covers legacy names like attr_longformer.json /
    attr_mobilenetv2_collect_order.json). The lookup is an EXACT key match
    into the models dict; if the chosen file also contains OTHER models, warn
    loudly -- rollup_to_perfetto's --model filter is a case-insensitive
    SUBSTRING match, so a sibling model whose name contains `model` (e.g.
    MobileBertForMaskedLM when asking for BertForMaskedLM) would be baked
    into the trace too. validate_and_summarize catches that and aborts."""
    canonical = attr_path_for(model, out_dir)
    candidates = [canonical] + sorted(
        p for p in out_dir.glob("attr_*.json") if p != canonical)
    for p in candidates:
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        models = data.get("models") or {}
        m = models.get(model)  # EXACT key match, never substring
        if isinstance(m, dict) and m.get("per_node"):
            extras = sorted(n for n in models if n != model)
            if extras:
                print(f"[regen_perfetto_demos] WARNING: {p.name} contains "
                      f"additional models besides {model}: {extras}. "
                      f"rollup_to_perfetto matches --model by SUBSTRING, so "
                      f"any of these whose name contains "
                      f"'{model}' would pollute the trace (the post-generate "
                      f"validation will abort if that happens).",
                      file=sys.stderr, flush=True)
            return p
    return None


_GIT_VERSION_CACHE: list[str | None] = []


def torch_git_version() -> str | None:
    """torch.version.git_version of the current env (the env the attribution
    subprocess just ran in). Queried once, in a subprocess, so this
    orchestrator never imports torch itself."""
    if not _GIT_VERSION_CACHE:
        try:
            out = subprocess.run(
                [sys.executable, "-c",
                 "import torch; print(torch.version.git_version)"],
                capture_output=True, text=True, timeout=120, cwd=ROOT)
            _GIT_VERSION_CACHE.append(
                out.stdout.strip() if out.returncode == 0 else None)
        except (OSError, subprocess.TimeoutExpired):
            _GIT_VERSION_CACHE.append(None)
    return _GIT_VERSION_CACHE[0]


def stamp_attr_provenance(attr_path: Path) -> None:
    """Record the PyTorch commit + timestamp INSIDE the attr JSON, right
    after the attribution subprocess wrote it, so a reused attr JSON
    (--skip-gpu) still knows what commit it was benched on."""
    data = json.loads(attr_path.read_text())
    gv = torch_git_version()
    if gv is None:
        # Stamp an explicit marker, not null: a null could later be confused
        # with "never stamped" (legacy attr JSON).
        gv = "UNAVAILABLE (torch import failed at stamp time)"
    data["_pytorch_git_version"] = gv
    data["_generated_at"] = (
        datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    attr_path.write_text(json.dumps(data, indent=2) + "\n")


def attr_provenance(attr_path: Path) -> tuple[str | None, str | None]:
    """(pytorch_git_version, generated_at) from an attr JSON, if stamped.

    Returns git_version=None ONLY when the key is absent (a legacy,
    never-stamped attr JSON). A file stamped while torch was unimportable
    carries an explicit "UNAVAILABLE (...)" string instead (older stamped
    files may carry a literal null; that is mapped to the same string here)."""
    try:
        data = json.loads(attr_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None, None
    generated_at = data.get("_generated_at")
    if "_pytorch_git_version" not in data:
        return None, generated_at  # key absent: pre-provenance legacy file
    gv = data["_pytorch_git_version"]
    if gv is None:  # stamped, but torch import failed at stamp time
        gv = "UNAVAILABLE (torch import failed at stamp time)"
    return gv, generated_at


# ---------------------------------------------------------------------------
# The two subprocess steps (NO new timing code -- reuse the existing tools)
# ---------------------------------------------------------------------------

def run_attribution(model: str, suite: str, mode: str, attr_path: Path,
                    gpu: str) -> None:
    cmd = [
        sys.executable, "scripts/with_gpu_lock.py", "--gpu", gpu, "--",
        sys.executable, "scripts/model_attribution.py",
        "--corpus-root", "repros",
        "--suite", suite, "--mode", mode,
        "--models", model, "--collect-order",
        "-o", str(attr_path),
    ]
    print(f"[regen_perfetto_demos] GPU step (locked, ~4 min): {' '.join(cmd)}",
          flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        fail(f"model_attribution failed for {model} (exit {r.returncode})")
    if not attr_path.is_file():
        fail(f"model_attribution reported success but wrote no {attr_path}")
    stamp_attr_provenance(attr_path)


def run_overlay(model: str, attr_path: Path, timings: Path,
                trace_path: Path) -> None:
    cmd = [
        sys.executable, "scripts/rollup_to_perfetto.py",
        "--source", "attribution",
        "--attribution", str(attr_path),
        "--model", model,
        "--oracle-timings", str(timings),
        "--out", str(trace_path),
    ]
    print(f"[regen_perfetto_demos] CPU overlay step: {' '.join(cmd)}",
          flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        fail(f"rollup_to_perfetto failed for {model} (exit {r.returncode})")
    if not trace_path.is_file():
        fail(f"rollup_to_perfetto reported success but wrote no {trace_path}")


# ---------------------------------------------------------------------------
# Post-generate validation + summary (fail loudly, never write junk silently)
# ---------------------------------------------------------------------------

def attr_expected_counts(model: str, attr_path: Path) -> tuple[int, int]:
    """(n_fusible, n_extern) per_node entries for EXACTLY `model` (exact dict
    key, never a substring match) in the attribution JSON."""
    try:
        data = json.loads(attr_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        fail(f"cannot re-read {attr_path} for validation: {e}")
    m = (data.get("models") or {}).get(model)  # EXACT key
    if not isinstance(m, dict) or not m.get("per_node"):
        fail(f"{attr_path} has no per_node entry under the EXACT model key "
             f"'{model}' -- refusing to validate a trace against a "
             f"substring-matched sibling model")
    per_node = m["per_node"]
    n_fusible = sum(1 for it in per_node if it.get("kind") == "fusible")
    return n_fusible, len(per_node) - n_fusible


def validate_and_summarize(model: str, trace_path: Path,
                           attr_path: Path) -> dict:
    """Validate the trace and compute the per-trace summary.

    Checks: valid JSON; non-empty traceEvents; EXACTLY ONE process (pid) in
    the trace, whose process_name is exactly `model` -- rollup_to_perfetto's
    --model filter is a case-insensitive SUBSTRING match, so a multi-model
    attr JSON with a sibling model whose name contains `model` (e.g.
    MobileBertForMaskedLM vs BertForMaskedLM, adv_inception_v3 vs
    inception_v3) would otherwise bake the sibling in as an extra pid,
    doubling the stats and possibly flipping the UNPRICED classification.
    Slice counts on tid 1 (fusible) / tid 2 (extern) must equal the attr
    JSON's own per_node counts for the EXACT model key; a tid is required to
    have slices only if the attr says the model actually has entries of that
    kind (a pure-fusible model has, and needs, no extern slices). Returns the
    summary dict used for the printed line + the MANIFEST entry."""
    try:
        trace = json.loads(trace_path.read_text())
    except json.JSONDecodeError as e:
        fail(f"{trace_path} is not valid JSON: {e}")
    events = trace.get("traceEvents")
    if not isinstance(events, list) or not events:
        fail(f"{trace_path}: no traceEvents")

    # ---- exact-model guard (see docstring) --------------------------------
    x_pids = sorted({e.get("pid") for e in events if e.get("ph") == "X"})
    pid_names = {e.get("pid"): (e.get("args") or {}).get("name")
                 for e in events
                 if e.get("ph") == "M" and e.get("name") == "process_name"}
    if len(x_pids) != 1:
        fail(f"{trace_path}: expected exactly ONE model (pid) in the trace "
             f"but found {len(x_pids)}: "
             f"{[(pid, pid_names.get(pid)) for pid in x_pids]}. "
             f"rollup_to_perfetto --model matches by SUBSTRING, so the attr "
             f"JSON {attr_path.name} contains sibling model(s) whose name "
             f"contains '{model}' and they were baked into the trace too. "
             f"Use an attr JSON containing only '{model}' (rerun without "
             f"--skip-gpu, or point --out-dir at a clean dir).")
    got_name = pid_names.get(x_pids[0])
    if got_name != model:
        fail(f"{trace_path}: the single model in the trace is "
             f"'{got_name}', not the requested '{model}'. "
             f"rollup_to_perfetto --model matches by SUBSTRING; the attr "
             f"JSON {attr_path.name} matched a sibling model instead of the "
             f"exact one requested.")

    exp_fusible, exp_extern = attr_expected_counts(model, attr_path)
    slices = {tid: [e for e in events
                    if e.get("ph") == "X" and e.get("tid") == tid]
              for tid in (TID_FUSIBLE, TID_EXTERN, TID_ORACLE)}
    if len(slices[TID_FUSIBLE]) != exp_fusible:
        fail(f"{trace_path}: {len(slices[TID_FUSIBLE])} fusible slices on "
             f"tid {TID_FUSIBLE} but the attr JSON has {exp_fusible} fusible "
             f"per_node entries for '{model}' (exact key). A substring-"
             f"matched sibling model in {attr_path.name} is the usual cause "
             f"-- refusing to keep a junk trace")
    if len(slices[TID_EXTERN]) != exp_extern:
        fail(f"{trace_path}: {len(slices[TID_EXTERN])} extern slices on "
             f"tid {TID_EXTERN} but the attr JSON has {exp_extern} extern "
             f"per_node entries for '{model}' (exact key) -- refusing to "
             f"keep a junk trace")
    # tid requirements are CONDITIONAL on what the model actually has:
    # a pure-fusible model legitimately has 0 extern slices, and vice versa.
    required = []
    if exp_fusible:
        required += [(TID_FUSIBLE, "fusible compile"),
                     (TID_ORACLE, "oracle ceiling")]
    if exp_extern:
        required.append((TID_EXTERN, "extern"))
    if not required:
        fail(f"{attr_path}: per_node for '{model}' has neither fusible nor "
             f"extern entries -- nothing to trace")
    for tid, name in required:
        if not slices[tid]:
            fail(f"{trace_path}: 0 slices on tid {tid} ({name}) -- "
                 f"refusing to keep a junk trace")

    # Recompute the fusible compile-vs-oracle totals from the slice args
    # (same semantics as rollup_to_perfetto: a MISS contributes the node's
    # own real compile us to the oracle-track length so it never collapses).
    oracle_fusible = [e for e in slices[TID_ORACLE]
                      if e.get("args", {}).get("kind") == "fusible"]
    n_hit = n_miss = 0
    miss_reasons: dict[str, int] = {}
    compile_fusible_us = sum(float(e.get("dur", 0.0))
                             for e in slices[TID_FUSIBLE])
    oracle_fusible_us = 0.0
    for e in oracle_fusible:
        a = e.get("args", {})
        if a.get("has_oracle"):
            n_hit += 1
            oracle_fusible_us += float(a.get("oracle_us") or 0.0)
        else:
            n_miss += 1
            reason = a.get("miss_reason") or "?"
            miss_reasons[reason] = miss_reasons.get(reason, 0) + 1
            oracle_fusible_us += float(a.get("node_real_compile_us") or 0.0)

    unpriced = n_hit == 0
    # When UNPRICED every oracle slice fell back to the node's real compile
    # us, so ratio would read a meaningless 1.0 ("perfect") and headroom 0.
    # Report null instead of a number that invites misreading.
    ratio = (round(compile_fusible_us / oracle_fusible_us, 4)
             if oracle_fusible_us and not unpriced else None)
    headroom = (round(compile_fusible_us - oracle_fusible_us, 2)
                if not unpriced else None)
    summary = {
        "model": model,
        "n_fusible_slices": len(slices[TID_FUSIBLE]),
        "n_extern_slices": len(slices[TID_EXTERN]),
        "n_oracle_hit": n_hit,
        "n_oracle_miss": n_miss,
        "oracle_miss_reasons": miss_reasons,
        "compile_fusible_us": round(compile_fusible_us, 2),
        "oracle_fusible_us": round(oracle_fusible_us, 2),
        "fusible_compile_over_oracle_ratio": ratio,
        "fusible_headroom_us": headroom,
        "unpriced": unpriced,
    }
    print(
        f"[regen_perfetto_demos] {model}: "
        f"fusible compile={summary['compile_fusible_us']}us "
        f"vs oracle={summary['oracle_fusible_us']}us "
        f"ratio={ratio} "
        f"oracle hit/miss={n_hit}/{n_miss}"
        + (f" {miss_reasons}" if miss_reasons else "")
        + (" [UNPRICED: all fusible oracle slices are misses]"
           if summary["unpriced"] else ""),
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------

def display_path(p: Path) -> str:
    """Repo-root-relative when possible (stable across checkouts/worktrees)."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def timings_family_count(timings: Path) -> int | None:
    try:
        t = json.loads(timings.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return sum(1 for k in t if not k.startswith("_"))


def write_manifest(out_dir: Path, entries: dict[str, dict], timings: Path,
                   n_families: int | None) -> Path:
    """Merge this run's entries into out_dir/MANIFEST.json (keyed by model),
    so a subset regen refreshes only the models it touched. Called once per
    model, right after its trace is committed, so a mid-run failure never
    leaves a fresh trace undescribed. The write is tmp-then-rename atomic; a
    corrupt existing manifest is backed up (never silently reset)."""
    manifest_path = out_dir / "MANIFEST.json"
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            ts = datetime.datetime.now(datetime.timezone.utc) \
                .strftime("%Y%m%dT%H%M%SZ")
            backup = manifest_path.with_name(f"MANIFEST.json.corrupt-{ts}")
            manifest_path.replace(backup)
            print(f"[regen_perfetto_demos] WARNING: existing "
                  f"{manifest_path} is not valid JSON ({e}); backed it up to "
                  f"{backup.name} and starting a fresh manifest. Entries for "
                  f"models not in this run are LOST from the new manifest -- "
                  f"recover them from the backup if needed.",
                  file=sys.stderr, flush=True)
            manifest = {}
    traces = manifest.get("traces") if isinstance(manifest.get("traces"), dict) \
        else {}
    traces.update(entries)
    manifest = {
        "_what": "Provenance manifest for the perfetto demo traces in this "
                 "dir. Regenerate with scripts/regen_perfetto_demos.py.",
        "last_regen_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "timings_file": display_path(timings),
        "timings_priced_families": n_families,
        "traces": traces,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(manifest_path)  # atomic: never a half-written MANIFEST
    return manifest_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Regenerate the Perfetto demo-trace set end-to-end: "
                    "per model, the locked GPU attribution run "
                    "(model_attribution.py --collect-order, ~4 min/model) "
                    "then the CPU oracle-overlay join "
                    "(rollup_to_perfetto.py --source attribution). "
                    "Run from the repo root with the GPU lock available.")
    p.add_argument("--models", default=DEFAULT_MODELS,
                   help=f"Comma-separated model names "
                        f"(default: the demo set {DEFAULT_MODELS})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output dir for traces + attr JSONs + MANIFEST.json "
                        f"(default: {DEFAULT_OUT_DIR})")
    p.add_argument("--timings", type=Path, default=DEFAULT_TIMINGS,
                   help=f"Per-oracle timings JSON for the overlay join "
                        f"(default: {DEFAULT_TIMINGS})")
    p.add_argument("--skip-gpu", action="store_true",
                   help="Reuse existing attribution JSONs in the out-dir and "
                        "only redo the CPU overlay join (for when only the "
                        "timings/oracle side changed). Errors if a model has "
                        "no reusable attr JSON.")
    p.add_argument("--gpu", default="0",
                   help="GPU id to lock for the attribution step (default 0)")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        fail("--models is empty")
    out_dir: Path = args.out_dir if args.out_dir.is_absolute() \
        else (ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: Path = args.timings if args.timings.is_absolute() \
        else (ROOT / args.timings)
    if not timings.is_file():
        fail(f"timings file not found: {timings}")
    n_families = timings_family_count(timings)
    print(f"[regen_perfetto_demos] models={models} out_dir={out_dir} "
          f"timings={timings} ({n_families} priced families) "
          f"skip_gpu={args.skip_gpu}", flush=True)

    entries: dict[str, dict] = {}
    for model in models:
        print(f"\n[regen_perfetto_demos] === {model} ===", flush=True)

        # --- attribution JSON: run the GPU step, or reuse ------------------
        if args.skip_gpu:
            attr_path = find_reusable_attr(model, out_dir)
            if attr_path is None:
                fail(f"--skip-gpu: no reusable attribution JSON for {model} "
                     f"in {out_dir} (expected attr_{model}.json or a legacy "
                     f"attr_*.json containing it). Run without --skip-gpu.")
            print(f"[regen_perfetto_demos] reusing {attr_path.name} "
                  f"(GPU step skipped)", flush=True)
        else:
            suite, mode = find_suite_mode(model, ROOT / "repros")
            attr_path = attr_path_for(model, out_dir)
            run_attribution(model, suite, mode, attr_path, args.gpu)

        # --- CPU overlay join ----------------------------------------------
        tmp_trace = out_dir / f"{model}.perfetto.json.tmp"
        run_overlay(model, attr_path, timings, tmp_trace)

        # --- validate, then commit to the convention-correct name ----------
        # fail() raises SystemExit: the finally block guarantees a trace
        # that flunks validation is deleted, never left lying around.
        try:
            summary = validate_and_summarize(model, tmp_trace, attr_path)
            overlay_name = out_dir / f"{model}_oracle_overlay.perfetto.json"
            unpriced_name = out_dir / f"{model}_UNPRICED_example.perfetto.json"
            final = unpriced_name if summary["unpriced"] else overlay_name
            stale = overlay_name if summary["unpriced"] else unpriced_name
            tmp_trace.replace(final)
        finally:
            if tmp_trace.is_file():
                tmp_trace.unlink()
        if stale.is_file():  # a previous regen classified this model the
            stale.unlink()   # other way; keep exactly one trace per model
            print(f"[regen_perfetto_demos] removed stale {stale.name} "
                  f"(classification changed)", flush=True)
        print(f"[regen_perfetto_demos] wrote {final}", flush=True)

        git_version, attr_generated_at = attr_provenance(attr_path)
        if git_version is None:
            # Legacy attr JSON that predates provenance stamping: say so
            # explicitly -- an unknown bench commit is exactly the staleness
            # this manifest exists to expose.
            git_version = ("UNKNOWN (legacy attr JSON, benched before "
                           "provenance stamping; rerun without --skip-gpu "
                           "to refresh)")
        entries[model] = {
            **summary,
            "trace_file": final.name,
            "attr_file": attr_path.name,
            "pytorch_git_version": git_version,
            "attr_generated_at": attr_generated_at,
            "trace_generated_at": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"),
            "timings_file": display_path(timings),
            "timings_priced_families": n_families,
        }
        # Merge this model's entry into the MANIFEST NOW (write_manifest
        # merges by model key), so a failure on a later model can't leave
        # this fresh trace undescribed.
        manifest_path = write_manifest(
            out_dir, {model: entries[model]}, timings, n_families)

    print(f"\n[regen_perfetto_demos] MANIFEST refreshed: {manifest_path}",
          flush=True)
    print(f"[regen_perfetto_demos] done: {len(entries)} trace(s) regenerated. "
          f"View: drag a .perfetto.json into https://ui.perfetto.dev",
          flush=True)


if __name__ == "__main__":
    main()
