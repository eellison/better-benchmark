"""
Shapes integrity tests — validates shapes.txt/shapes.json data for all canonical repros.

Checks:
1. All shape configs are parseable by parse_shapes_config / _parse_shapes_txt
2. shapes.txt/shapes.json pattern_hash matches the directory name
3. Warns if canonical dirs with >1 model in meta.json are missing shapes data

Usage:
    python scripts/test_shapes_integrity.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repro_harness import parse_shapes_config, _parse_shapes_txt

CANONICAL = Path("repros/canonical")
ERRORS = []
WARNINGS = []
PASS_COUNT = 0


def error(msg):
    ERRORS.append(msg)
    print(f"  ERROR: {msg}")


def warn(msg):
    WARNINGS.append(msg)


def passed():
    global PASS_COUNT
    PASS_COUNT += 1


def test_shapes_parseable():
    """For every pattern with shapes.txt, verify configs are parseable."""
    print("--- Test: shapes configs are parseable ---")
    total_configs = 0

    for shapes_txt in sorted(CANONICAL.rglob("shapes.txt")):
        pattern_dir = shapes_txt.parent
        try:
            configs = _parse_shapes_txt(shapes_txt)
        except Exception as e:
            error(f"{pattern_dir.name}: failed to parse shapes.txt: {e}")
            continue

        if not configs:
            error(f"{pattern_dir.name}: shapes.txt parsed but produced no configs")
            continue

        # Validate each config by running parse_shapes_config on the raw lines
        for line in shapes_txt.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            colon = line.find(":")
            if colon < 0:
                continue
            label = line[:colon].strip()
            expr = line[colon + 1:].strip()
            total_configs += 1
            try:
                inputs = parse_shapes_config(expr)
                if not inputs:
                    error(f"{pattern_dir.name}/{label}: parse_shapes_config returned empty")
                else:
                    passed()
            except Exception as e:
                error(f"{pattern_dir.name}/{label}: parse_shapes_config failed: {e}")

    # Also check legacy shapes.json
    for shapes_json in sorted(CANONICAL.rglob("shapes.json")):
        pattern_dir = shapes_json.parent
        try:
            data = json.loads(shapes_json.read_text())
        except Exception as e:
            error(f"{pattern_dir.name}: failed to parse shapes.json: {e}")
            continue

        configs = data.get("configs", {})
        if not configs:
            error(f"{pattern_dir.name}: shapes.json has no configs")
            continue

        for name, config_str in configs.items():
            total_configs += 1
            if isinstance(config_str, str):
                try:
                    inputs = parse_shapes_config(config_str)
                    if not inputs:
                        error(f"{pattern_dir.name}/{name}: empty config from shapes.json")
                    else:
                        passed()
                except Exception as e:
                    error(f"{pattern_dir.name}/{name}: {e}")
            else:
                # Old format: config_str is a dict with 'inputs' key
                passed()

    print(f"  Checked {total_configs} configs across shapes files")


def test_pattern_hash_matches_dir():
    """Check that shapes.txt/shapes.json lives in a dir whose hash matches meta.json."""
    print("--- Test: pattern_hash matches directory name ---")

    shapes_files = list(CANONICAL.rglob("shapes.txt")) + list(CANONICAL.rglob("shapes.json"))

    for shapes_file in sorted(shapes_files):
        pattern_dir = shapes_file.parent
        dir_name = pattern_dir.name

        # Extract hash from directory name (last 12 chars after final underscore)
        parts = dir_name.rsplit("_", 1)
        if len(parts) != 2 or len(parts[1]) != 12:
            error(f"{dir_name}: directory name does not follow <kind>_<hash> format")
            continue

        dir_hash = parts[1]

        # Check meta.json if it exists
        meta_path = pattern_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                meta_hash = meta.get("pattern_hash", "")
                if meta_hash and meta_hash != dir_hash:
                    error(
                        f"{dir_name}: meta.json pattern_hash '{meta_hash}' "
                        f"does not match directory hash '{dir_hash}'"
                    )
                else:
                    passed()
            except Exception as e:
                error(f"{dir_name}: failed to read meta.json: {e}")
        else:
            # No meta.json - just check format is valid
            passed()


def test_multi_model_patterns_have_shapes():
    """Warn if canonical dirs with >1 model are missing shapes data entirely."""
    print("--- Test: multi-model patterns have shapes data (warnings only) ---")

    missing_count = 0
    total_multi = 0

    for meta_path in sorted(CANONICAL.rglob("meta.json")):
        pattern_dir = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue

        n_models = meta.get("n_models", 0)
        if n_models <= 1:
            continue

        total_multi += 1
        has_shapes = (
            (pattern_dir / "shapes.txt").exists()
            or (pattern_dir / "shapes.json").exists()
        )

        if not has_shapes:
            missing_count += 1
            warn(
                f"{pattern_dir.name}: {n_models} models in meta.json but no shapes data"
            )
        else:
            passed()

    print(f"  {total_multi} multi-model patterns, {missing_count} missing shapes data")
    if missing_count > 0:
        print(f"  (These are warnings, not errors)")


def main():
    if not CANONICAL.exists():
        print(f"ERROR: {CANONICAL} does not exist")
        sys.exit(1)

    test_shapes_parseable()
    test_pattern_hash_matches_dir()
    test_multi_model_patterns_have_shapes()

    print()
    print(f"Results: {PASS_COUNT} passed, {len(ERRORS)} errors, {len(WARNINGS)} warnings")

    if ERRORS:
        print(f"\nERRORS ({len(ERRORS)}):")
        for e in ERRORS:
            print(f"  {e}")
        sys.exit(1)

    if WARNINGS:
        print(f"\nWARNINGS ({len(WARNINGS)}):")
        for w in WARNINGS[:20]:
            print(f"  {w}")
        if len(WARNINGS) > 20:
            print(f"  ... and {len(WARNINGS) - 20} more")

    print("\nAll integrity checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
