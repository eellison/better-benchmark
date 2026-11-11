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
    masked_reason: str | None = None,
) -> Path:
    capture_dir = root / name
    capture_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "pattern_hash": pattern_hash,
        "shape_hash": name,
        "kind": "reduction",
        "reduction_types": reduction_types,
        "n_ops": 1,
        "origin_ops": ["aten.sum.default"],
        "file": str(root / f"missing_{name}.py"),
    }
    if masked_reason is not None:
        entry["masked_reason"] = masked_reason
    (capture_dir / "index.json").write_text(json.dumps([entry]) + "\n")
    return capture_dir


class MergeCapturesTests(unittest.TestCase):
    def test_repro_is_masked_if_any_model_masks_it(self):
        for masked_first in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "repros"
                masked = _write_capture(
                    root, "masked", pattern_hash="abcdef123456",
                    reduction_types=["sum"], masked_reason="approx_index",
                )
                unmasked = _write_capture(
                    root, "unmasked", pattern_hash="abcdef123456",
                    reduction_types=["sum"],
                )
                order = [(masked, "ModelA"), (unmasked, "ModelB")]
                if not masked_first:
                    order.reverse()
                for capture, model in order:
                    merge_one_capture(capture, output, model, suite="hf", mode="train")

                meta = json.loads(
                    (output / "canonical" / "sum_abcdef123456" / "meta.json").read_text()
                )
                self.assertEqual(meta["masked_reason"], "approx_index", masked_first)

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
            # Model keys are fully qualified as suite/mode/model (matches the
            # shapes.json "models" keying); the bare-name form is retired.
            self.assertEqual(meta["models"], ["hf/train/ModelA", "hf/train/ModelB"])

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

    def test_graph_only_capture_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            capture_dir = root / "empty_capture"
            capture_dir.mkdir()
            model_dir = output / "models" / "hf" / "infer" / "ModelA"
            model_dir.mkdir(parents=True)
            (model_dir / "full_graph_000.py").write_text("# graph\n")
            (model_dir / "full_graph_000.meta.json").write_text("{}\n")

            merged = merge_one_capture(capture_dir, output, "ModelA", suite="hf", mode="infer")

            manifest = json.loads((model_dir / "manifest.json").read_text())
            self.assertEqual(merged, 0)
            self.assertEqual(manifest["patterns"], [])
            self.assertEqual(manifest["graphs"], ["full_graph_000.py"])
            self.assertEqual(
                manifest["graph_metadata"],
                {"full_graph_000.py": "full_graph_000.meta.json"},
            )

    def test_explicit_suite_preserves_prefixed_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            capture_dir = root / "empty_capture"
            capture_dir.mkdir()
            model_dir = output / "models" / "hf" / "train" / "hf_Foo_train"
            model_dir.mkdir(parents=True)
            (model_dir / "full_graph_000.py").write_text("# graph\n")

            merge_one_capture(
                capture_dir,
                output,
                "hf_Foo_train",
                suite="hf",
                mode="train",
            )

            self.assertTrue((model_dir / "manifest.json").exists())
            self.assertFalse(
                (output / "models" / "hf" / "train" / "Foo" / "manifest.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
