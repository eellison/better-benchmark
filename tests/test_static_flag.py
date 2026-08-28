"""Tests for --static: the static-at-hints artifact.

`python repro.py --static` takes the ONE dynamic family, binds every symbol
to `symbols[name].hint`, materializes all symbolic shapes/strides/offsets/
shape-params/symint values at that binding, and measures ONE fully
specialized artifact/row — never `mark_dynamic` (count_kernels compiles
separately before the timed artifact, per the established methodology).
The row records the ACTUAL hint binding (`label::s0=4::static`), not a
null placeholder.

Deliberately distinct from:
  - `--all-shapes`: statically bench every recorded point at ITS bindings;
  - `--dynamic`: every point through one shared dynamic artifact.

Loud-conflict contracts:
  - --static --dynamic / --static --bind / --static --all-shapes: SystemExit 2
  - --static --prewarm: caught by the existing dynamic-path prewarm guard
  - --static on a STATIC repro: ValueError (nothing to materialize)
  - --static with a hint-less symbol: ValueError (cannot build the binding)

Usage:
    python tests/test_static_flag.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import repro_harness as rh  # noqa: E402
from repro_harness import benchmark_repro  # noqa: E402


class _Scale(torch.nn.Module):
    def forward(self, x):
        return x * 2


def _mk(shape_config=None):
    return [torch.zeros(4, 3)]


def _write_two_point_family(d: Path, hint=4) -> str:
    """Table hint s0=4; recorded points at s0=4 AND s0=6 (the fixture:
    --static must compile ONE artifact at the table hint, not sweep points)."""
    (d / "shapes.json").write_text(json.dumps({
        "symbols": {"s0": {"hint": hint, "range": [2, None]}},
        "guards": [],
        "points": [
            {"shape_hash": "aaaa1111", "captured_dynamic": True,
             "bindings": {"s0": 4},
             "models": {"probe/infer/m": {"occurrences": 1}},
             "inputs": [[["s0", 3], "f32"]]},
            {"shape_hash": "bbbb2222", "captured_dynamic": True,
             "bindings": {"s0": 6},
             "models": {"probe/infer/m": {"occurrences": 1}},
             "inputs": [[["s0", 3], "f32"]]},
        ],
    }))
    repro = d / "repro.py"
    repro.write_text("# stub")
    return str(repro)


class TestStaticAtHints(unittest.TestCase):
    def test_static_compiles_one_artifact_at_table_hints(self):
        with tempfile.TemporaryDirectory() as td:
            repro_file = _write_two_point_family(Path(td))
            compile_calls = []

            def fake_compile(m, **kw):
                compile_calls.append(kw)
                return m

            with mock.patch.object(rh, "count_kernels",
                                   return_value=(1, ["k"])) as ck, \
                    mock.patch.object(rh, "timed_min_us", return_value=1.0), \
                    mock.patch.object(torch, "compile", fake_compile), \
                    mock.patch.object(torch._dynamo, "mark_dynamic") as md:
                results = benchmark_repro(
                    repro_file, _Scale, _mk,
                    args=["--static", "--no-gpu-lock"])

            # ONE artifact at the TABLE hint — not a per-point sweep.
            self.assertEqual(len(results), 1)
            (key, row), = results.items()
            self.assertTrue(key.endswith("::s0=4::static"), key)
            self.assertEqual(row["binding"], {"s0": 4})
            self.assertEqual(row["mode"], "static")

            # One measured non-dynamic artifact/row; never mark_dynamic.
            # (count_kernels is mocked here; its real implementation compiles
            # separately before the timed artifact.)
            self.assertEqual(len(compile_calls), 1)
            self.assertNotIn("dynamic", compile_calls[0])
            md.assert_not_called()

            # The materialized inputs ARE the hint binding: s0=4 -> [4, 3].
            ck.assert_called_once()
            (mod_arg, inputs_arg) = ck.call_args[0][:2]
            self.assertEqual(tuple(inputs_arg[0].shape), (4, 3))

    def test_static_dynamic_conflict_errors(self):
        with self.assertRaises(SystemExit) as cm:
            benchmark_repro("x.py", _Scale, _mk,
                            args=["--static", "--dynamic", "--no-gpu-lock"])
        self.assertEqual(cm.exception.code, 2)

    def test_static_run_at_gives_fresh_artifact_per_binding(self):
        # `--static --run-at A --run-at B` = independent fully static
        # artifacts at A and B (the old --static/--bind conflict is gone —
        # explicit static bindings are now a feature, not an ambiguity).
        with tempfile.TemporaryDirectory() as td:
            repro_file = _write_two_point_family(Path(td))
            fake_compile = lambda m, **kw: m  # noqa: E731
            with mock.patch.object(rh, "count_kernels",
                                   return_value=(1, ["k"])), \
                    mock.patch.object(rh, "timed_min_us", return_value=1.0), \
                    mock.patch.object(torch, "compile", fake_compile):
                results = benchmark_repro(
                    repro_file, _Scale, _mk,
                    args=["--static", "--run-at", "s0=8",
                          "--run-at", "s0=16", "--no-gpu-lock"])
            keys = sorted(results)
            self.assertEqual(len(keys), 2, keys)
            self.assertTrue(keys[0].endswith("::s0=16::static"), keys)
            self.assertTrue(keys[1].endswith("::s0=8::static"), keys)
            bindings = sorted(r["binding"]["s0"] for r in results.values())
            self.assertEqual(bindings, [8, 16])

    def test_static_all_shapes_conflict_errors(self):
        with self.assertRaises(SystemExit) as cm:
            benchmark_repro("x.py", _Scale, _mk,
                            args=["--static", "--all-shapes",
                                  "--no-gpu-lock"])
        self.assertEqual(cm.exception.code, 2)

    def test_static_prewarm_errors_via_dynamic_path_guard(self):
        with self.assertRaises(SystemExit) as cm:
            benchmark_repro("x.py", _Scale, _mk,
                            args=["--static", "--prewarm", "s0=4",
                                  "--no-gpu-lock"])
        self.assertEqual(cm.exception.code, 2)

    def test_static_on_static_repro_errors(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "shapes.json").write_text(json.dumps({"points": [{
                "shape_hash": "aa", "inputs": [[[4, 3], "f32"]],
                "models": {"m": {"occurrences": 1}}}]}))
            repro = d / "repro.py"
            repro.write_text("# stub")
            with self.assertRaises(ValueError) as cm:
                benchmark_repro(str(repro), _Scale, _mk,
                                args=["--static", "--no-gpu-lock"])
            self.assertIn("already static", str(cm.exception))

    def test_static_hintless_symbol_errors(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "shapes.json").write_text(json.dumps({
                "symbols": {"s0": {"range": [2, None]}},  # no hint recorded
                "guards": [],
                "points": [{"shape_hash": "aaaa1111",
                            "captured_dynamic": True, "bindings": {"s0": 4},
                            "models": {"m": {"occurrences": 1}},
                            "inputs": [[["s0", 3], "f32"]]}]}))
            repro = d / "repro.py"
            repro.write_text("# stub")
            with self.assertRaises(ValueError) as cm:
                benchmark_repro(str(repro), _Scale, _mk,
                                args=["--static", "--no-gpu-lock"])
            self.assertIn("no int hint", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
