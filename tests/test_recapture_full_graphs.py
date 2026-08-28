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
from ingest_tlparse import load_graph_module


FAKE_FULL_GRAPH = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]"):
        return (torch.ops.aten.relu.default(x),)
"""


FULL_GRAPH_WITH_TENSOR_CONSTANT = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]"):
        _tensor_constant0: "f32[]" = self._tensor_constant0
        lift_fresh_copy: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(x, lift_fresh_copy);  x = lift_fresh_copy = None
        return (add,)
"""


FULL_GRAPH_WITH_PRIMS_FMA = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]", y: "f32[2, 2]", z: "f32[2, 2]"):
        fma: "f32[2, 2]" = torch.ops.prims.fma.default(x, y, z);  x = y = z = None
        return (fma,)
"""


FULL_GRAPH_WITH_STRIDED_INPUT = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 3][1, 2]cpu"):
        add: "f32[2, 3][1, 2]cpu" = torch.ops.aten.add.Tensor(x, 1.0);  x = None
        return (add,)
"""


FULL_GRAPH_WITH_NO_INPUTS = """
class GraphModule(torch.nn.Module):
    def forward(self):
        iota: "i64[4][1]cpu" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        add: "i64[4][1]cpu" = torch.ops.aten.add.Tensor(iota, 1);  iota = None
        return (add,)
"""


FULL_GRAPH_WITH_SYM_INPUT = """
class GraphModule(torch.nn.Module):
    def forward(self, s0: "Sym(s0)", x: "f32[2, s0]cpu"):
        view: "f32[2, s0]cpu" = torch.ops.aten.reshape.default(x, [2, s0]);  x = s0 = None
        return (view,)
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

    def test_isolated_recapture_reports_graph_load_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(root / "genai" / "Broken" / "full_graph_000.py", "x = 1\n")
            target = recapture.infer_target(graph, models_root=root)

            result = recapture.recapture_target_isolated(
                target,
                Path(tmp) / "out",
                validate=False,
            )

            self.assertFalse(result.ok)
            self.assertIn("graph load returned None", result.error)

    def test_loader_recreates_saved_tensor_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_TENSOR_CONSTANT,
            )

            gm = load_graph_module(graph)

            self.assertIsNotNone(gm)
            self.assertTrue(hasattr(gm, "graph"))

    def test_loader_registers_inductor_prims_for_fma(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_PRIMS_FMA,
            )

            gm = load_graph_module(graph)

            self.assertIsNotNone(gm)
            self.assertTrue(hasattr(gm, "graph"))

    def test_loader_recreates_printed_input_strides(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_STRIDED_INPUT,
            )

            gm = load_graph_module(graph)

            self.assertIsNotNone(gm)
            placeholder = next(n for n in gm.graph.nodes if n.op == "placeholder")
            self.assertEqual(tuple(placeholder.meta["val"].stride()), (1, 2))

    def test_loader_traces_zero_input_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_NO_INPUTS,
            )

            gm = load_graph_module(graph)

            self.assertIsNotNone(gm)
            self.assertFalse(any(n.op == "placeholder" for n in gm.graph.nodes))

    def test_loader_synthesizes_symint_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_SYM_INPUT,
            )

            gm = load_graph_module(graph)

            self.assertIsNotNone(gm)
            self.assertTrue(hasattr(gm, "graph"))


if __name__ == "__main__":
    unittest.main()
