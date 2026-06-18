"""
Verify manifest integrity: every pattern in every model manifest
must resolve to a directory in repros/canonical/.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "repros" / "models"
CANONICAL = ROOT / "repros" / "canonical"


def test_all_manifests_have_schema_version():
    errors = []
    for mf in sorted(MODELS.rglob("manifest.json")):
        data = json.loads(mf.read_text())
        if data.get("schema_version") != 1:
            errors.append(f"{mf.relative_to(ROOT)}: missing or wrong schema_version")
    assert not errors, "\n".join(errors)


def test_all_patterns_resolve_to_canonical():
    missing = []
    for mf in sorted(MODELS.rglob("manifest.json")):
        data = json.loads(mf.read_text())
        for pattern in data.get("patterns", []):
            matches = list(CANONICAL.glob(f"*{pattern}*"))
            if not matches:
                missing.append(f"{mf.relative_to(ROOT)}: {pattern}")
    assert not missing, f"{len(missing)} unresolved patterns:\n" + "\n".join(missing[:20])


def test_all_graph_files_exist():
    missing = []
    for mf in sorted(MODELS.rglob("manifest.json")):
        data = json.loads(mf.read_text())
        model_dir = mf.parent
        for graph in data.get("graphs", []):
            if not (model_dir / graph).exists():
                missing.append(f"{mf.relative_to(ROOT)}: {graph}")
    assert not missing, f"{len(missing)} missing graph files:\n" + "\n".join(missing[:20])


def test_no_empty_manifests_with_graphs():
    """Models with full_graph files should eventually have patterns."""
    warnings = []
    for mf in sorted(MODELS.rglob("manifest.json")):
        data = json.loads(mf.read_text())
        if data.get("graphs") and not data.get("patterns"):
            warnings.append(f"{mf.relative_to(ROOT)}: has graphs but 0 patterns (needs recapture)")
    if warnings:
        print(f"[WARN] {len(warnings)} models have graphs but no patterns:")
        for w in warnings[:10]:
            print(f"  {w}")


def test_canonical_dirs_have_repro_py():
    """Every dir in canonical/ should have a repro.py."""
    missing = []
    for d in sorted(CANONICAL.iterdir()):
        if d.is_dir() and not (d / "repro.py").exists():
            missing.append(d.name)
    assert not missing, f"{len(missing)} canonical dirs missing repro.py:\n" + "\n".join(missing[:20])


if __name__ == "__main__":
    print("Manifest integrity checks:")
    test_all_manifests_have_schema_version()
    print("  PASS: all manifests have schema_version 1")
    test_all_patterns_resolve_to_canonical()
    print("  PASS: all patterns resolve to canonical/")
    test_all_graph_files_exist()
    print("  PASS: all listed graph files exist")
    test_no_empty_manifests_with_graphs()
    test_canonical_dirs_have_repro_py()
    print("  PASS: all canonical dirs have repro.py")
    print("\nAll integrity checks passed!")
