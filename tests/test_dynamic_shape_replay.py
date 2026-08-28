"""Native ShapesSpec replay tests.

Capture records ranges, guards, and symbolic input relationships. These tests
prove that replay installs the same contract in Dynamo's ShapeEnv, including
runtime guard enforcement, raw-SymInt correspondence, composite dimensions,
and symbolic tensor metadata.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from dynamic_shape_replay import (  # noqa: E402
    _backed_args,
    _backed_replay_plan_for_repro,
    _backed_repro,
    _eval_shape_env_expr,
    _shape_env_repro,
    _shape_env_spec_for_repro,
)


class TestShapeEnvPlayback(unittest.TestCase):
    def _write_guarded(self, d: Path) -> str:
        (d / "shapes.json").write_text(json.dumps({
            "symbols": {"s0": {"hint": 8, "range": [2, None]}},
            "guards": ["Eq(Mod(s0, 4), 0)"],
            "points": [{
                "shape_hash": "guarded",
                "captured_dynamic": True,
                "bindings": {"s0": 8},
                "models": {"m": {"occurrences": 1}},
                "inputs": [[["s0", 3], "f32"], ["I", 8, "s0"]],
            }],
        }))
        repro = d / "repro.py"
        repro.write_text("# stub")
        return str(repro)

    def test_guard_and_tensor_symint_correspondence_run_in_shape_env(self):
        class UsesInt(torch.nn.Module):
            def forward(self, x, n):
                return x + n

        with tempfile.TemporaryDirectory() as td:
            repro = self._write_guarded(Path(td))
            (spec, live_names, input_count,
             metadata_checks, frozen) = _shape_env_spec_for_repro(repro)
            self.assertEqual(live_names, ("s0",))
            self.assertEqual(input_count, 2)
            # Core's config/metrics serializer must see data, not live SymInts.
            json.dumps(spec.to_jsonable())

            graphs = []

            def backend(gm, _example_inputs):
                graphs.append(gm)
                return gm.forward

            torch._dynamo.reset()
            compiled = torch.compile(
                _shape_env_repro(
                    UsesInt(), input_count, spec, live_names,
                    metadata_checks, frozen),
                backend=backend,
            )
            compiled(torch.zeros(8, 3), 8, 8)
            compiled(torch.zeros(12, 3), 12, 12)
            self.assertEqual(len(graphs), 1)
            self.assertTrue(any(
                node.op == "call_function"
                and "_assert_scalar" in str(node.target)
                for node in graphs[0].graph.nodes
            ))

            # Bypass the harness's external binding validation: these failures
            # prove the compiled artifact itself enforces both invariants.
            with self.assertRaises((AssertionError, RuntimeError)):
                compiled(torch.zeros(8, 3), 7, 8)
            with self.assertRaises((AssertionError, RuntimeError)):
                compiled(torch.zeros(6, 3), 6, 6)
            torch._dynamo.reset()

    def test_backed_replay_derives_symint_and_keeps_guard_out_of_graph(self):
        class UsesInt(torch.nn.Module):
            def forward(self, x, n):
                return x + n

        with tempfile.TemporaryDirectory() as td:
            repro = self._write_guarded(Path(td))
            plan = _backed_replay_plan_for_repro(repro)
            self.assertIsNotNone(plan)
            self.assertEqual(plan["kept"], (0,))

            graphs = []

            def backend(gm, _example_inputs):
                graphs.append(gm)
                return gm.forward

            compiled = torch.compile(
                _backed_repro(UsesInt(), plan), backend=backend)

            def invoke(size, raw_int):
                inputs = [torch.zeros(size, 3), raw_int]
                return compiled(*_backed_args(inputs, plan))

            invoke(8, 8)
            invoke(12, 12)
            self.assertEqual(len(graphs), 1)
            self.assertFalse(any(
                node.op == "call_function"
                and "_assert_scalar" in str(node.target)
                for node in graphs[0].graph.nodes
            ))
            # The lifted raw int is not a compiled argument: the callable
            # re-derives it from x.size(0), preserving the model relationship.
            self.assertTrue(torch.equal(
                invoke(8, 999), torch.full((8, 3), 8.0)))
            with self.assertRaises(RuntimeError):
                invoke(6, 6)  # violates Eq(Mod(s0, 4), 0)
            torch._dynamo.reset()

    def test_unbacked_metadata_prevents_backed_replay(self):
        """A tensor occurrence is a possible source, not symbol provenance."""
        with tempfile.TemporaryDirectory() as td:
            repro = self._write_guarded(Path(td))
            shapes_path = Path(repro).parent / "shapes.json"
            data = json.loads(shapes_path.read_text())
            data["symbols"]["s0"]["observed_value"] = (
                data["symbols"]["s0"].pop("hint"))
            data["symbols"]["s0"]["unbacked"] = True
            shapes_path.write_text(json.dumps(data))
            self.assertIsNone(_backed_replay_plan_for_repro(repro))

    def test_unbacked_optimization_hint_is_not_a_point(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "repro.py").write_text("# stub")
            (d / "shapes.json").write_text(json.dumps({
                "symbols": {
                    "s0": {
                        "range": [0, 64],
                        "unbacked": True,
                        "optimization_hint": 16,
                    },
                },
                "guards": [],
                "points": [{
                    "shape_hash": "template",
                    "captured_dynamic": True,
                    "bindings": {},
                    "requires_binding": ["s0"],
                    "models": {"m": {"occurrences": 1}},
                    "inputs": [[["s0", 3], "f32"], ["I", None, "s0"]],
                }],
            }))
            spec, names, *_ = _shape_env_spec_for_repro(
                str(d / "repro.py"))
            self.assertEqual(names, ("s0",))
            serialized = spec.to_jsonable()
            self.assertIn(
                '"optimization_hint": 16',
                json.dumps(serialized, sort_keys=True),
            )

    def test_shape_env_expression_parser_reuses_safe_boundary(self):
        from torch.fx.experimental.dynamic_spec import IntVar

        s0 = IntVar("s0", min=0)
        with self.assertRaises(ValueError):
            _eval_shape_env_expr("__import__(1)", {"s0": s0})

    def test_symbolic_stride_and_offset_are_runtime_invariants(self):
        class Scale(torch.nn.Module):
            def forward(self, x):
                return x * 2

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "repro.py").write_text("# stub")
            (d / "shapes.json").write_text(json.dumps({
                "symbols": {
                    "s0": {"hint": 4, "range": [0, None]},
                    "s1": {"hint": 4, "range": [2, None]},
                },
                "guards": [],
                "points": [{
                    "shape_hash": "metadata",
                    "captured_dynamic": True,
                    "bindings": {"s0": 4, "s1": 4},
                    "models": {"m": {"occurrences": 1}},
                    "inputs": [[
                        ["s1", 4], "f32",
                        {"st": ["2*s0", 1], "off": "s0"},
                    ]],
                }],
            }))
            (spec, names, input_count,
             checks, frozen) = _shape_env_spec_for_repro(str(d / "repro.py"))
            self.assertEqual(names, ("s0", "s1"))
            self.assertEqual([c[1] for c in checks],
                             ["stride", "storage_offset"])

            graphs = []

            def backend(gm, _example_inputs):
                graphs.append(gm)
                return gm.forward

            compiled = torch.compile(
                _shape_env_repro(
                    Scale(), input_count, spec, names, checks, frozen),
                backend=backend,
            )
            storage = torch.empty(4096)

            def invoke(s0, s1, stride=None, offset=None):
                stride = 2 * s0 if stride is None else stride
                offset = s0 if offset is None else offset
                tensor = storage.as_strided((s1, 4), (stride, 1), offset)
                return compiled(tensor, s0, s1)

            invoke(4, 4)
            invoke(6, 8)
            self.assertEqual(len(graphs), 1)
            with self.assertRaises((AssertionError, RuntimeError)):
                invoke(6, 8, stride=13)
            with self.assertRaises((AssertionError, RuntimeError)):
                invoke(6, 8, offset=7)
            torch._dynamo.reset()

    def test_stride_only_symbol_on_static_shape_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "repro.py").write_text("# stub")
            (d / "shapes.json").write_text(json.dumps({
                "symbols": {"s0": {"hint": 8, "range": [2, None]}},
                "guards": [],
                "points": [{
                    "shape_hash": "stride-only",
                    "captured_dynamic": True,
                    "bindings": {"s0": 8},
                    "models": {"m": {"occurrences": 1}},
                    "inputs": [[[4, 4], "f32", {"st": ["s0", 1]}]],
                }],
            }))
            with self.assertRaisesRegex(
                    ValueError, "cannot make stride/offset metadata dynamic"):
                _shape_env_spec_for_repro(str(d / "repro.py"))

    def test_partial_freeze_binds_composite_only_live_root(self):
        class Sum(torch.nn.Module):
            def forward(self, x):
                return x.sum()

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "repro.py").write_text("# stub")
            (d / "shapes.json").write_text(json.dumps({
                "symbols": {
                    "s0": {"hint": 4, "range": [2, None]},
                    "s1": {"hint": 8, "range": [2, None]},
                },
                "guards": [],
                "points": [{
                    "shape_hash": "composite",
                    "captured_dynamic": True,
                    "bindings": {"s0": 4, "s1": 8},
                    "models": {"m": {"occurrences": 1}},
                    "inputs": [[["s0*s1", 3], "f32"]],
                }],
            }))
            (spec, names, input_count,
             checks, frozen) = _shape_env_spec_for_repro(
                 str(d / "repro.py"), frozen={"s0": 4})
            self.assertIsNone(_backed_replay_plan_for_repro(
                str(d / "repro.py"), frozen={"s0": 4}))
            self.assertEqual(names, ("s1",))
            graphs = []

            def backend(gm, _example_inputs):
                graphs.append(gm)
                return gm.forward

            compiled = torch.compile(
                _shape_env_repro(
                    Sum(), input_count, spec, names, checks, frozen),
                backend=backend,
            )
            compiled(torch.empty(32, 3), 8)
            compiled(torch.empty(36, 3), 9)
            self.assertEqual(len(graphs), 1)
            with self.assertRaises(AssertionError):
                compiled(torch.empty(34, 3), 9)
            torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
