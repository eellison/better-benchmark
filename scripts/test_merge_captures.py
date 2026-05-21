"""Focused tests for merge_captures.py.

Usage:
    python scripts/test_merge_captures.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from merge_captures import merge_one_capture, temporary_capture_for_merge


def _write_capture(
    root: Path,
    name: str,
    *,
    pattern_hash: str,
    reduction_types: list[str],
) -> Path:
    capture_dir = root / name
    capture_dir.mkdir(parents=True, exist_ok=True)
    (capture_dir / "index.json").write_text(
        json.dumps(
            [
                {
                    "pattern_hash": pattern_hash,
                    "shape_hash": name,
                    "kind": "reduction",
                    "reduction_types": reduction_types,
                    "n_ops": 1,
                    "origin_ops": ["aten.sum.default"],
                    "file": str(root / f"missing_{name}.py"),
                }
            ]
        )
        + "\n"
    )
    return capture_dir


class MergeCapturesTests(unittest.TestCase):
    def test_same_pattern_hash_reuses_existing_canonical_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            pattern_hash = "abcdef123456"
            first = _write_capture(
                root,
                "first",
                pattern_hash=pattern_hash,
                reduction_types=["amax", "sum"],
            )
            second = _write_capture(
                root,
                "second",
                pattern_hash=pattern_hash,
                reduction_types=["sum", "amax"],
            )

            merge_one_capture(first, output, "ModelA", suite="hf", mode="train")
            merge_one_capture(second, output, "ModelB", suite="hf", mode="train")

            kept = output / "canonical" / f"amax_sum_{pattern_hash}"
            duplicate = output / "canonical" / f"sum_amax_{pattern_hash}"
            self.assertTrue(kept.exists())
            self.assertFalse(duplicate.exists())

            meta = json.loads((kept / "meta.json").read_text())
            self.assertEqual(meta["pattern_hash"], pattern_hash)
            self.assertEqual(meta["models"], ["ModelA", "ModelB"])

    def test_temporary_capture_for_merge_removes_raw_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            pattern_hash = "deadbeef1234"

            with temporary_capture_for_merge(
                output,
                "ModelA",
                suite="hf",
                mode="infer",
                prefix="test_capture_",
            ) as capture:
                raw_dir = capture.capture_dir
                _write_capture(
                    raw_dir.parent,
                    raw_dir.name,
                    pattern_hash=pattern_hash,
                    reduction_types=["sum"],
                )

                self.assertEqual(capture.merge(), 1)
                self.assertTrue(raw_dir.exists())

            repro_dir = output / "canonical" / f"sum_{pattern_hash}"
            manifest = output / "models" / "hf" / "infer" / "ModelA" / "manifest.json"

            self.assertFalse(raw_dir.exists())
            self.assertTrue(repro_dir.exists())
            self.assertTrue(manifest.exists())
            self.assertFalse((output / "captures").exists())

    def test_temporary_capture_for_merge_requires_explicit_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"

            with temporary_capture_for_merge(
                output,
                "ModelA",
                suite="hf",
                mode="infer",
                prefix="test_capture_",
            ) as capture:
                raw_dir = capture.capture_dir
                _write_capture(
                    raw_dir.parent,
                    raw_dir.name,
                    pattern_hash="cafebabe1234",
                    reduction_types=["sum"],
                )

            self.assertFalse(raw_dir.exists())
            self.assertFalse((output / "canonical").exists())


if __name__ == "__main__":
    unittest.main()
