"""Focused tests for recapturing from saved full_graph artifacts.

Usage:
    python scripts/test_recapture_full_graphs.py
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import recapture_full_graphs as recapture


FAKE_FULL_GRAPH = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]"):
        return (torch.ops.aten.relu.default(x),)
"""


def _write(path: Path, source: str = FAKE_FULL_GRAPH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


class RecaptureFullGraphsTests(unittest.TestCase):
    def test_infer_target_preserves_suite_mode_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(
                root / "hf" / "train" / "BertForMaskedLM" / "full_graph_000.py"
            )

            target = recapture.infer_target(graph, models_root=root)

            self.assertEqual(target.suite, "hf")
            self.assertEqual(target.mode, "train")
            self.assertEqual(target.model, "BertForMaskedLM")
            self.assertEqual(target.graph_name, "full_graph_000")

    def test_infer_target_preserves_modeless_suite_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(root / "vllm" / "facebook_opt-125m" / "full_graph_001.py")

            target = recapture.infer_target(graph, models_root=root)

            self.assertEqual(target.suite, "vllm")
            self.assertIsNone(target.mode)
            self.assertEqual(target.model, "facebook_opt-125m")

    def test_find_full_graphs_accepts_files_and_directories_with_dedupe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            model_dir = root / "torchbench" / "infer" / "alexnet"
            graph0 = _write(model_dir / "full_graph_000.py")
            graph1 = _write(model_dir / "full_graph_001.py")
            _write(model_dir / "region_000_graph.py")

            found = recapture.find_full_graphs([graph0, model_dir], models_root=root)

            self.assertEqual(found, [graph0, graph1])

    def test_dry_run_does_not_load_or_process_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(root / "timm" / "infer" / "resnet18" / "full_graph_000.py")
            targets = recapture.discover_targets([graph], models_root=root)

            def fail_load(_path):
                raise AssertionError("dry run should not load")

            def fail_process(_gm, _target, _canonical_root):
                raise AssertionError("dry run should not process")

            results = recapture.recapture_targets(
                targets,
                Path(tmp) / "out",
                dry_run=True,
                load_fn=fail_load,
                process_fn=fail_process,
            )

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].ok)
            self.assertEqual(results[0].regions, 0)

    def test_recapture_uses_loader_processor_and_path_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(
                root / "torchbench" / "train" / "densenet121" / "full_graph_000.py"
            )
            targets = recapture.discover_targets([graph], models_root=root)
            calls = []

            fake_gm = object()

            def load_fn(path):
                calls.append(("load", path))
                return fake_gm

            def process_fn(gm, target, canonical_root):
                calls.append(
                    (
                        "process",
                        gm,
                        target.suite,
                        target.mode,
                        target.model,
                        canonical_root,
                    )
                )
                return 7

            canonical_root = Path(tmp) / "canonical_out"
            results = recapture.recapture_targets(
                targets,
                canonical_root,
                load_fn=load_fn,
                process_fn=process_fn,
            )

            self.assertEqual(results[0].regions, 7)
            self.assertEqual(
                calls,
                [
                    ("load", graph),
                    (
                        "process",
                        fake_gm,
                        "torchbench",
                        "train",
                        "densenet121",
                        canonical_root,
                    ),
                ],
            )

    def test_recapture_validates_by_default_and_can_disable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(root / "vllm" / "facebook_opt-125m" / "full_graph_000.py")
            targets = recapture.discover_targets([graph], models_root=root)
            calls = []
            original = recapture.process_graph_for_target

            def load_fn(_path):
                return object()

            def fake_process_graph_for_target(_gm, _target, _canonical_root, *, validate):
                calls.append(validate)
                return 1

            recapture.process_graph_for_target = fake_process_graph_for_target
            try:
                recapture.recapture_targets(
                    targets,
                    Path(tmp) / "out",
                    load_fn=load_fn,
                )
                recapture.recapture_targets(
                    targets,
                    Path(tmp) / "out",
                    validate=False,
                    load_fn=load_fn,
                )
            finally:
                recapture.process_graph_for_target = original

            self.assertEqual(calls, [True, False])

    def test_recapture_fail_fast_returns_only_attempted_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph0 = _write(root / "genai" / "KernelA" / "full_graph_000.py")
            graph1 = _write(root / "genai" / "KernelB" / "full_graph_000.py")
            targets = recapture.discover_targets([graph0, graph1], models_root=root)

            def load_fn(_path):
                raise RuntimeError("load failed")

            results = recapture.recapture_targets(
                targets,
                Path(tmp) / "out",
                fail_fast=True,
                load_fn=load_fn,
            )

            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].ok)
            self.assertIn("load failed", results[0].error)


if __name__ == "__main__":
    unittest.main()
