"""
Migrate repros/models/ to clean per-model layout.

Target structure:
  repros/models/<suite>/<mode>/<model_name>/
    manifest.json      # schema_version 1, patterns, source metadata
    full_graph_*.py    # saved post-grad graphs for recapture

Changes:
  - Consolidate duplicate model dirs (vllm_X_000, vllm_X_001 → X)
  - Move full_graph files into the canonical model dir
  - Upgrade manifest.json to schema_version 1
  - Delete stale capture artifacts (capture_summary_*.json, repros/captures/)
  - Verify all manifest patterns exist in repros/canonical/

Idempotent: safe to re-run.
"""
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPROS = ROOT / "repros"
MODELS = REPROS / "models"
CANONICAL = REPROS / "canonical"


def _clean_model_name(dirname: str) -> str:
    """Strip vllm_/hf_/timm_ prefix, _NNN suffix, and _infer/_train suffix."""
    name = dirname
    # Strip numbered suffix like _000, _001
    name = re.sub(r'_\d{3}$', '', name)
    # Strip suite prefix if it duplicates the parent dir context
    for prefix in ('vllm_', 'hf_', 'timm_', 'torchbench_'):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Strip mode suffixes (already encoded in parent dir)
    for suffix in ('_infer', '_inference', '_train', '_training'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def _infer_mode_from_path(path: Path) -> str | None:
    """Infer mode from path components."""
    parts = [p.name for p in path.parents]
    if 'infer' in parts:
        return 'infer'
    if 'train' in parts:
        return 'train'
    return None


def _consolidate_suite(suite_dir: Path, suite_name: str):
    """Consolidate all model dirs under a suite into clean layout."""
    if not suite_dir.exists():
        return

    # Group dirs by clean model name
    model_groups: dict[str, list[Path]] = {}

    for child in sorted(suite_dir.iterdir()):
        if not child.is_dir():
            continue
        # Handle mode subdirs (infer/, train/)
        if child.name in ('infer', 'train'):
            for model_dir in sorted(child.iterdir()):
                if not model_dir.is_dir():
                    continue
                clean = _clean_model_name(model_dir.name)
                key = f"{child.name}/{clean}"
                model_groups.setdefault(key, []).append(model_dir)
        else:
            clean = _clean_model_name(child.name)
            mode = _infer_mode_from_path(child)
            key = clean if mode is None else f"{mode}/{clean}"
            model_groups.setdefault(key, []).append(child)

    for key, dirs in model_groups.items():
        # Determine target dir
        if '/' in key:
            mode, model_name = key.split('/', 1)
            target = suite_dir / mode / model_name
        else:
            target = suite_dir / key

        target.mkdir(parents=True, exist_ok=True)

        # Merge manifests
        all_patterns = set()
        for d in dirs:
            manifest = d / "manifest.json"
            if manifest.exists():
                try:
                    data = json.loads(manifest.read_text())
                    all_patterns.update(data.get("patterns", []))
                except Exception:
                    pass

        # Collect full_graph files
        graph_files = []
        for d in dirs:
            graph_files.extend(sorted(d.glob("full_graph_*.py")))

        # Move full_graph files to target (renumber if needed)
        existing_graphs = set(f.name for f in target.glob("full_graph_*.py"))
        next_idx = len(existing_graphs)
        for gf in graph_files:
            if gf.parent == target:
                continue  # already in target
            dest_name = gf.name
            if dest_name in existing_graphs:
                dest_name = f"full_graph_{next_idx:03d}.py"
                next_idx += 1
            dest = target / dest_name
            if not dest.exists():
                shutil.copy2(gf, dest)
                existing_graphs.add(dest_name)

        # Write upgraded manifest
        manifest_path = target / "manifest.json"
        # Resolve actual model name for source field
        model_display = key.split('/')[-1] if '/' in key else key

        # List actual graph files in target dir
        actual_graphs = sorted(f.name for f in target.glob("full_graph_*.py"))

        manifest_data = {
            "schema_version": 1,
            "source": suite_name,
            "model": model_display,
            "mode": key.split('/')[0] if '/' in key else None,
            "patterns": sorted(all_patterns),
            "graphs": actual_graphs,
        }
        # Don't include mode if None
        if manifest_data["mode"] is None:
            del manifest_data["mode"]

        manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n")

        # Remove old dirs that aren't the target
        for d in dirs:
            if d.resolve() != target.resolve() and d.exists():
                shutil.rmtree(d)
                print(f"  Removed {d.relative_to(REPROS)}")

        print(f"  {suite_name}/{key}: {len(all_patterns)} patterns, "
              f"{len(existing_graphs)} graphs -> {target.relative_to(REPROS)}")


def _verify_integrity():
    """Check that every pattern in every manifest exists in canonical."""
    errors = []
    for manifest_path in sorted(MODELS.rglob("manifest.json")):
        try:
            data = json.loads(manifest_path.read_text())
        except Exception as e:
            errors.append(f"{manifest_path}: invalid JSON: {e}")
            continue

        if data.get("schema_version") != 1:
            errors.append(f"{manifest_path}: missing or wrong schema_version")
            continue

        for pattern in data.get("patterns", []):
            # Pattern hash might be just the hash or full dir name
            matches = list(CANONICAL.glob(f"*{pattern}*"))
            if not matches:
                errors.append(f"{manifest_path}: pattern {pattern} not in canonical/")

    return errors


def main():
    print("Migrating repros/models/ layout...")
    print()

    # Consolidate each suite
    for suite in sorted(MODELS.iterdir()):
        if not suite.is_dir():
            continue
        print(f"Suite: {suite.name}")
        _consolidate_suite(suite, suite.name)
        print()

    # Delete stale artifacts
    stale = [
        REPROS / "captures",
        REPROS / "capture_summary_hf.json",
        REPROS / "capture_summary_timm.json",
        REPROS / "capture_summary_torchbench.json",
    ]
    for path in stale:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"Deleted stale: {path.relative_to(REPROS)}")

    print()
    print("Verifying manifest integrity...")
    errors = _verify_integrity()
    if errors:
        print(f"\n{len(errors)} errors found:")
        for e in errors[:20]:
            print(f"  {e}")
        return 1
    else:
        print("  All manifests valid. Every pattern resolves to canonical/.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
