"""
Shapes integrity tests — validates shapes.txt/shapes.json data for all canonical repros.

Checks:
1. All shape configs are parseable by parse_shapes_config / _parse_shapes_txt
2. shapes.txt/shapes.json pattern_hash matches the directory name
3. Warns if canonical dirs with >1 model in meta.json are missing shapes data
4. Stride preservation for channels-last tensors

Usage:
    python scripts/test_shapes_integrity.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

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


def test_channels_last_stride_preserved():
    """Verify that parse_shapes_config correctly produces non-contiguous tensors with strides."""
    print("--- Test: channels-last stride preservation ---")

    # channels-last 4D tensor: NCHW stored as NHWC
    config = "(T([128, 3, 299, 299], f32, stride=(268203, 1, 897, 3)))"
    inputs = parse_shapes_config(config)
    assert len(inputs) == 1, f"Expected 1 input, got {len(inputs)}"
    t = inputs[0]
    assert t.shape == (128, 3, 299, 299), f"Wrong shape: {t.shape}"
    assert t.stride() == (268203, 1, 897, 3), f"Wrong stride: {t.stride()}"
    assert t.is_contiguous(memory_format=torch.channels_last), \
        "Tensor should be channels_last contiguous"
    assert not t.is_contiguous(), "Tensor should NOT be row-major contiguous"
    passed()
    print("  channels_last 4D: OK")

    # Multiple tensors, mix of contiguous and non-contiguous
    config2 = "(T([32, 3, 3, 3], f32, stride=(27, 1, 9, 3)), T([32], f32))"
    inputs2 = parse_shapes_config(config2)
    assert len(inputs2) == 2, f"Expected 2 inputs, got {len(inputs2)}"
    assert inputs2[0].stride() == (27, 1, 9, 3), f"Wrong stride: {inputs2[0].stride()}"
    assert inputs2[1].is_contiguous(), "1D tensor should be contiguous"
    passed()
    print("  mixed contiguous/non-contiguous: OK")

    # Contiguous tensor should not have stride in config (clean output)
    config3 = "(T([128, 3, 299, 299], f32))"
    inputs3 = parse_shapes_config(config3)
    assert inputs3[0].is_contiguous(), "Tensor without stride should be contiguous"
    passed()
    print("  contiguous without stride: OK")


def test_stride_extraction_from_annotations():
    """Verify _parse_input_shapes_from_annotations correctly extracts strides."""
    print("--- Test: stride extraction from annotation strings ---")

    from scripts.repartition_from_graphs import (
        _parse_input_shapes_from_annotations,
        _is_contiguous_stride,
    )

    # channels-last annotation
    content = '''class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 3, 3, 3][27, 1, 9, 3]cuda:0", arg1_1: "f32[128, 3, 299, 299][268203, 1, 897, 3]cuda:0"):
        pass
'''
    result = _parse_input_shapes_from_annotations(content)
    assert len(result) == 2, f"Expected 2 inputs, got {len(result)}"

    shape0, dtype0, stride0 = result[0]
    assert shape0 == (32, 3, 3, 3), f"Wrong shape: {shape0}"
    assert stride0 == (27, 1, 9, 3), f"Wrong stride: {stride0}"
    passed()
    print("  non-contiguous stride parsed: OK")

    shape1, dtype1, stride1 = result[1]
    assert shape1 == (128, 3, 299, 299), f"Wrong shape: {shape1}"
    assert stride1 == (268203, 1, 897, 3), f"Wrong stride: {stride1}"
    passed()
    print("  channels-last stride parsed: OK")

    # Contiguous tensor should have stride=None
    content2 = '''class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[768, 768][768, 1]cuda:0"):
        pass
'''
    result2 = _parse_input_shapes_from_annotations(content2)
    shape, dtype, stride = result2[0]
    assert shape == (768, 768), f"Wrong shape: {shape}"
    assert stride is None, f"Contiguous stride should be None, got {stride}"
    passed()
    print("  contiguous stride omitted: OK")

    # Test _is_contiguous_stride helper
    # Contiguous stride for [128, 3, 299, 299] is (3*299*299, 299*299, 299, 1) = (268203, 89401, 299, 1)
    assert _is_contiguous_stride([128, 3, 299, 299], [268203, 89401, 299, 1]) is True
    assert _is_contiguous_stride([128, 3, 299, 299], [268203, 1, 897, 3]) is False
    assert _is_contiguous_stride([32], [1]) is True
    assert _is_contiguous_stride([], []) is True
    passed()
    print("  _is_contiguous_stride helper: OK")


def test_timm_graphs_have_stride_info():
    """Check that timm full_graph annotations with non-contiguous strides are parsed."""
    print("--- Test: timm full_graphs contain stride info ---")

    from scripts.repartition_from_graphs import _parse_input_shapes_from_annotations

    timm_dir = CANONICAL.parent / "models" / "timm"
    if not timm_dir.exists():
        print("  SKIP: no timm models directory")
        return

    graph_files = sorted(timm_dir.rglob("full_graph_*.py"))
    if not graph_files:
        print("  SKIP: no timm full_graph files")
        return

    n_checked = 0
    n_with_strides = 0

    for gf in graph_files[:20]:  # check up to 20
        content = gf.read_text()
        inputs = _parse_input_shapes_from_annotations(content)
        n_checked += 1
        has_noncontiguous = any(
            len(entry) == 3 and entry[2] is not None
            for entry in inputs
            if entry[0] != 'sym_int'
        )
        if has_noncontiguous:
            n_with_strides += 1

    print(f"  Checked {n_checked} timm graphs, {n_with_strides} have non-contiguous strides")
    if n_with_strides > 0:
        passed()
    else:
        warn("No timm graphs had non-contiguous stride info (may need re-capture)")


def test_extraction_validates_before_writing():
    """The extraction script must not write invalid shapes.

    Creates a synthetic repro that expects 2 args, then tries to validate
    a shape config with 3 args. Verifies the validator rejects it.
    """
    print("--- Test: extraction validates before writing ---")
    import tempfile

    from scripts.extract_shapes_from_graphs import _validate_shape_entry

    # Create a synthetic repro that expects exactly 2 tensor args
    with tempfile.TemporaryDirectory(prefix="test_extract_val_") as tmp:
        repro_py = Path(tmp) / "repro.py"
        repro_py.write_text('''
import torch
import math

device = torch.device
inf = math.inf
nan = math.nan

_shapes_config = "(T([4, 4], f32), T([4, 4], f32))"

class Repro(torch.nn.Module):
    def forward(self, a, b):
        return a + b
''')

        # Valid config (2 args) should pass
        good_config = "(T([4, 4], f32), T([4, 4], f32))"
        assert _validate_shape_entry(repro_py, good_config), \
            "Valid 2-arg config should pass validation"
        passed()
        print("  valid config accepted: OK")

        # Invalid config (3 args) should fail
        bad_config = "(T([4, 4], f32), T([4, 4], f32), T([4, 4], f32))"
        assert not _validate_shape_entry(repro_py, bad_config), \
            "Invalid 3-arg config should be rejected"
        passed()
        print("  invalid config rejected: OK")

        # Completely broken config should fail
        broken_config = "(NONSENSE)"
        assert not _validate_shape_entry(repro_py, broken_config), \
            "Broken config should be rejected"
        passed()
        print("  broken config rejected: OK")


def main():
    if not CANONICAL.exists():
        print(f"ERROR: {CANONICAL} does not exist")
        sys.exit(1)

    test_shapes_parseable()
    test_pattern_hash_matches_dir()
    test_multi_model_patterns_have_shapes()
    test_channels_last_stride_preserved()
    test_stride_extraction_from_annotations()
    test_timm_graphs_have_stride_info()
    test_extraction_validates_before_writing()

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
