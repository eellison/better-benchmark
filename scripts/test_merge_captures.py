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

from merge_captures import merge_one_capture


def _write_capture(
    root: Path,
    name: str,
    *,
    pattern_hash: str,
    reduction_types: list[str],
) -> Path:
    capture_dir = root / name
    capture_dir.mkdir(parents=True)
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


if __name__ == "__main__":
    unittest.main()
