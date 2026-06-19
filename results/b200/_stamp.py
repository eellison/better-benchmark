#!/usr/bin/env python3
"""One-shot stamper: copy source JSONs into the per-commit layout and add a
single-pytorch-commit `_metadata` block to each. NOT a benchmark runner — pure
file copy + JSON metadata stamping. Run once; verifies round-trip (non-metadata
content byte-identical to source) before writing.

Layout produced (see results/b200/README.md):
  results/b200/<pytorch_commit_short>/{kernels,oracle_floors,projections}.json

Each result's `_metadata` carries exactly ONE `pytorch_commit` (the
distinguishing axis). The A/B is NOT stored — it is derived by compute_ab.py.
"""
import json
import shutil
from pathlib import Path

ROOT = Path("/tmp/scratch_space/better_benchmark")
B200 = ROOT / "results" / "b200"

BRANCH_SHA = "daa79cd25ca9a80bfd65799394cf4255d6be75a6"
BASELINE_SHA = "244fdb379d11d5925da5610b22e1466222c4afb9"
# The git ref each SHA was HEAD of at measurement time. A SHA alone is ambiguous
# for a MOVING branch (daa79cd25ca is tmp_work HEAD today; once tmp_work advances
# it becomes an orphan hash), so the branch name is recorded as structured
# provenance (`pytorch_ref`), not only in the harness_caveats prose.
BRANCH_REF = "tmp_work"
BASELINE_REF = "trunk"
BRANCH_DIR = B200 / "daa79cd25ca"
BASELINE_DIR = B200 / "244fdb379d11"

CORPUS_SIZE = {"patterns": 1727, "points": 4977, "models_non_genai": 158, "genai": 8}
HW = "4x NVIDIA B200"
DATE = "2026-06-18"
CAVEATS = ("dynamo-reset-per-shape fix 4ca6d532b applied; WIP pytorch branch "
           "tmp_work not merged; genai excluded from headline aggregates")

REV = ROOT / "results" / "perf_ab" / "revalidate_2026-06-18"
ORACLE = ROOT / "results" / "oracle_floors_2026-06-18_v4"
PROJ = ORACLE / "model_projections"


def base_meta(pytorch_sha, bb_commit, sweep_type, pytorch_ref):
    return {
        "pytorch_commit": pytorch_sha,
        "pytorch_ref": pytorch_ref,
        "bb_commit": bb_commit,
        "hardware": HW,
        "date": DATE,
        "sweep_type": sweep_type,
        "corpus_size": dict(CORPUS_SIZE),
        "harness_caveats": CAVEATS,
    }


def stamp_kernels(src: Path, dst: Path, pytorch_sha: str, sweep_type: str,
                  pytorch_ref: str):
    """Kernels JSON: existing `_metadata.commit` is the BETTER-BENCHMARK commit.
    Rename it -> `bb_commit` (preserve value), ADD `pytorch_commit` + the
    `pytorch_ref` branch label. The kernel payload (every non-`_` key) is
    untouched."""
    src_text = src.read_text()
    data = json.loads(src_text)
    old_meta = data.get("_metadata", {})
    bb_commit = old_meta.get("commit")  # better-benchmark commit, preserve value

    new_meta = base_meta(pytorch_sha, bb_commit, sweep_type, pytorch_ref)
    # carry forward the original bench-tool fields for provenance, dropping the
    # ambiguous bare `commit` (now expressed as bb_commit + pytorch_commit).
    for k, v in old_meta.items():
        if k == "commit":
            continue
        new_meta.setdefault(k, v)
    data["_metadata"] = new_meta

    dst.write_text(json.dumps(data, indent=2) + "\n")
    _verify_roundtrip_dict(src_text, dst, label=dst.name)


def stamp_oracle(src: Path, dst: Path, pytorch_sha: str, sweep_type: str,
                 bb_commit: str, pytorch_ref: str):
    """Oracle-floor JSON: a flat dict keyed by pattern dir, NO existing
    `_metadata`. Add one; leave every pattern entry untouched."""
    src_text = src.read_text()
    data = json.loads(src_text)
    assert "_metadata" not in data, f"{src} unexpectedly already has _metadata"
    data = {"_metadata": base_meta(pytorch_sha, bb_commit, sweep_type, pytorch_ref), **data}
    dst.write_text(json.dumps(data, indent=2) + "\n")
    _verify_roundtrip_dict(src_text, dst, label=dst.name)


def stamp_projection(src: Path, dst: Path, pytorch_sha: str, bb_commit: str,
                     pytorch_ref: str):
    """Projection JSON is a LIST of model rows. Wrap as
    {"_metadata": {...}, "models": [...]} (noted in README)."""
    src_text = src.read_text()
    data = json.loads(src_text)
    assert isinstance(data, list), f"{src} expected a list, got {type(data)}"
    meta = base_meta(pytorch_sha, bb_commit, "projection", pytorch_ref)
    meta["wrapping_note"] = ("source was a bare JSON list of model rows; wrapped "
                             "as {_metadata, models:[...]} for stamping")
    out = {"_metadata": meta, "models": data}
    dst.write_text(json.dumps(out, indent=2) + "\n")
    # round-trip: the `models` list must equal the source list exactly.
    reloaded = json.loads(dst.read_text())["models"]
    assert reloaded == data, f"ROUND-TRIP FAIL (projection): {dst}"
    print(f"  [roundtrip OK] {dst.name}: models list == source ({len(data)} rows)")


def _verify_roundtrip_dict(src_text: str, dst: Path, label: str):
    """Confirm every non-`_metadata` key/value round-trips unchanged."""
    src_data = json.loads(src_text)
    dst_data = json.loads(dst.read_text())
    src_payload = {k: v for k, v in src_data.items() if k != "_metadata"}
    dst_payload = {k: v for k, v in dst_data.items() if k != "_metadata"}
    assert src_payload == dst_payload, f"ROUND-TRIP FAIL (payload): {dst}"
    n = len(dst_payload)
    print(f"  [roundtrip OK] {label}: {n} payload keys identical to source")


def main():
    print("Stamping BRANCH arm (pytorch daa79cd25ca):")
    # bb_commit for the oracle/projection sweeps = the branch-run bb commit
    # recorded in the branch kernels file (a7f8b0af...), since they were swept
    # against the same branch arm.
    branch_kernels_src = REV / "branch_kernels.json"
    branch_bb = json.loads(branch_kernels_src.read_text())["_metadata"]["commit"]

    stamp_kernels(branch_kernels_src, BRANCH_DIR / "kernels.json",
                  BRANCH_SHA, "kernels", BRANCH_REF)
    stamp_oracle(ORACLE / "all_oracle_timings_v4.json",
                 BRANCH_DIR / "oracle_floors.json", BRANCH_SHA, "oracle_floor",
                 branch_bb, BRANCH_REF)
    stamp_oracle(ORACLE / "all_oracle_timings_v4_minfloor.json",
                 BRANCH_DIR / "oracle_floors_minfloor.json", BRANCH_SHA,
                 "oracle_floor", branch_bb, BRANCH_REF)
    stamp_projection(PROJ / "projections_v4_coveragegated.json",
                     BRANCH_DIR / "projections.json", BRANCH_SHA, branch_bb,
                     BRANCH_REF)
    stamp_projection(PROJ / "projections_v4_minfloor.json",
                     BRANCH_DIR / "projections_minfloor.json", BRANCH_SHA,
                     branch_bb, BRANCH_REF)

    print("\nStamping BASELINE arm (pytorch 244fdb379d11):")
    stamp_kernels(REV / "baseline_kernels.json",
                  BASELINE_DIR / "kernels.json", BASELINE_SHA, "kernels",
                  BASELINE_REF)

    print("\nCopying prose deliverables to results/b200/ top level:")
    prose = [
        (REV / "REVALIDATION_SUMMARY.md", B200 / "REVALIDATION_SUMMARY.md"),
        (ORACLE / "DELIVERABLE2_v4_ranking.txt", B200 / "DELIVERABLE2_v4_ranking.txt"),
        (ORACLE / "TWO_FLOOR_README.md", B200 / "TWO_FLOOR_README.md"),
        (ROOT / "results" / "perf_ab" / "CANONICAL_HEADLINE.md", B200 / "CANONICAL_HEADLINE.md"),
        (ROOT / "results" / "longformer_amax_sum" / "HANDOFF.md", B200 / "longformer_HANDOFF.md"),
    ]
    for s, d in prose:
        shutil.copy2(s, d)
        print(f"  copied {s.name} -> {d.name}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
