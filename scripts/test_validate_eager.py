"""Focused tests for scripts/validate_eager.py path selection.

Usage:
    python scripts/test_validate_eager.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_eager


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# fake repro\n")
    return path


class ValidateEagerTests(unittest.TestCase):
    def test_find_repros_accepts_files_repro_dirs_and_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct_file = _write(root / "direct.py")
            repro_dir = root / "canonical" / "pointwise_a"
            nested_repro = _write(repro_dir / "repro.py")
            sibling_repro = _write(root / "canonical" / "pointwise_b" / "repro.py")

            found = validate_eager._find_repros(
                [direct_file, repro_dir, root / "canonical"]
            )

            self.assertEqual(found, sorted({direct_file, nested_repro, sibling_repro}))

    def test_load_benchmark_set_rejects_missing_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "canonical"
            _write(canonical / "present" / "repro.py")
            manifest = root / "v1.json"
            manifest.write_text(json.dumps({
                "patterns": [
                    {"name": "present"},
                    {"name": "missing"},
                ],
            }))

            with self.assertRaisesRegex(FileNotFoundError, "missing canonical repros"):
                validate_eager._load_benchmark_set(manifest, canonical)


if __name__ == "__main__":
    unittest.main()
