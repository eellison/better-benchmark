"""Acceptance tests for the dynamic execution API.

Contract:
  - a repro with a symbols table runs DYNAMIC by default (no flags);
  - a genuinely static repro keeps its historical default path;
  - `--static [--run-at ...]` = fresh fully specialized artifacts;
  - `--freeze s0[=N]` = partial specialization (frozen symbols become
    compile-time constants; ShapesSpec keeps the remaining symbols dynamic,
    including symint-only and composite-dim symbols);
  - `--compile-at A --run-at B` = the "compile at A, run at B" experiment
    (exact warm order, no injected auto shapes);
  - every dynamic row records `compile_bindings` (ordered),
    `compile_bindings_source` ("auto"/"explicit"), and `frozen_symbols`;
  - old spellings --bind/--prewarm are hidden aliases; mixing spellings for
    one role errors loudly.

All compile/timing is mocked (CPU-only): these tests pin routing, binding
materialization, specialization, and the row contract — not kernel perf.

Usage:
    python tests/test_execution_api.py
"""
from __future__ import annotations

import contextlib
import io
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
    def forward(self, x, *unused):
        return x * 2


def _mk(shape_config=None):
    return [torch.zeros(4, 3)]


def _one_symbol_family(d: Path) -> str:
    """s0 over dim 0; two recorded points (s0=4, s0=6); table hint 4."""
    (d / "shapes.json").write_text(json.dumps({
        "symbols": {"s0": {"hint": 4, "range": [2, None]}},
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


def _two_symbol_family(d: Path) -> str:
    """s0 x s1 tensor; one recorded point at (4, 8)."""
    (d / "shapes.json").write_text(json.dumps({
        "symbols": {"s0": {"hint": 4, "range": [2, None]},
                    "s1": {"hint": 8, "range": [2, None]}},
        "guards": [],
        "points": [
            {"shape_hash": "aaaa1111", "captured_dynamic": True,
             "bindings": {"s0": 4, "s1": 8},
             "models": {"probe/infer/m": {"occurrences": 1}},
             "inputs": [[["s0", "s1"], "f32"]]},
        ],
    }))
    repro = d / "repro.py"
    repro.write_text("# stub")
    return str(repro)


def _unobserved_unbacked_family(d: Path) -> str:
    """No default point: range + optimization hint, explicit run required."""
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
            "models": {"probe/infer/m": {"occurrences": 1}},
            "inputs": [[["s0", 3], "f32"]],
        }],
    }))
    repro = d / "repro.py"
    repro.write_text("# stub")
    return str(repro)


def _mocked(extra_patches=()):
    fake_compile = lambda m, **kw: m  # noqa: E731 - eager stand-in
    patches = [
        mock.patch.object(rh, "count_kernels", return_value=(1, ["k"])),
        mock.patch.object(rh, "timed_min_us", return_value=1.0),
        mock.patch.object(torch, "compile", fake_compile),
    ]
    patches.extend(extra_patches)
    return patches


def _run(repro_file, args, extra_patches=()):
    ctx = contextlib.ExitStack()
    with ctx:
        for p in _mocked(extra_patches):
            ctx.enter_context(p)
        return benchmark_repro(repro_file, _Scale, _mk,
                               args=args + ["--no-gpu-lock"])


class TestDynamicByDefault(unittest.TestCase):
    def test_dynamic_family_runs_dynamic_with_no_flags(self):
        with tempfile.TemporaryDirectory() as td:
            results = _run(_one_symbol_family(Path(td)), [])
            self.assertEqual(len(results), 2)  # both recorded points
            self.assertEqual(
                {tuple(row["binding"].items()) for row in results.values()},
                {(("s0", 4),), (("s0", 6),)},
            )
            for key, row in results.items():
                self.assertTrue(key.endswith("::dynamic"), key)
                self.assertEqual(row["mode"], "dynamic")
                # Dynamic row contract.
                self.assertEqual(row["compile_bindings_source"], "auto")
                self.assertGreaterEqual(len(row["compile_bindings"]), 2)
                self.assertEqual(row["frozen_symbols"], {})

    def test_run_at_on_dynamic_family_measures_dynamic_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            results = _run(_one_symbol_family(Path(td)),
                           ["--run-at", "s0=16"])
            (key, row), = results.items()
            self.assertTrue(key.endswith("::s0=16::dynamic"), key)
            self.assertEqual(row["mode"], "dynamic")

    def test_unobserved_unbacked_requires_run_at(self):
        with tempfile.TemporaryDirectory() as td:
            repro = _unobserved_unbacked_family(Path(td))
            with self.assertRaisesRegex(
                    ValueError, "no complete observed point"):
                _run(repro, [])
            results = _run(repro, ["--run-at", "s0=7"])
            (_key, row), = results.items()
            self.assertEqual(row["binding"], {"s0": 7})
            self.assertEqual(
                row["compile_bindings"][0], {"s0": 7})

    def test_static_repro_keeps_historical_default(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "shapes.json").write_text(json.dumps({"points": [{
                "shape_hash": "aa", "inputs": [[[4, 3], "f32"]],
                "models": {"m": {"occurrences": 1}}}]}))
            repro = d / "repro.py"
            repro.write_text("# stub")
            results = _run(str(repro), [])
            self.assertTrue(results)
            for key, row in results.items():
                # Legacy per-shape rows: no bound-bench key suffix, none of
                # the dynamic-row contract fields.
                self.assertNotIn("::dynamic", key)
                self.assertNotIn("::static", key)
                self.assertNotIn("compile_bindings", row)

    def test_dynamic_flag_on_static_repro_keeps_blanket_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "shapes.json").write_text(json.dumps({"points": [{
                "shape_hash": "aa", "inputs": [[[4, 3], "f32"]],
                "models": {"m": {"occurrences": 1}}}]}))
            repro = d / "repro.py"
            repro.write_text("# stub")
            results = _run(str(repro), ["--dynamic"])
            self.assertTrue(results)
            for key, row in results.items():
                self.assertTrue(key.endswith("::dynamic"), key)
                self.assertEqual(row["mode"], "dynamic")

    def test_count_kernels_only_does_not_time_dynamic_family(self):
        with tempfile.TemporaryDirectory() as td:
            no_timing = mock.patch.object(
                rh, "timed_min_us",
                side_effect=AssertionError("timing must not run"),
            )
            results = _run(
                _one_symbol_family(Path(td)),
                ["--count-kernels-only"],
                extra_patches=(no_timing,),
            )
            self.assertEqual(len(results), 2)
            for row in results.values():
                self.assertEqual(row["n_kernels"], 1)
                self.assertNotIn("compiled_us", row)
                self.assertIsNone(row["recompiled"])

    def test_dynamic_measurement_matches_no_grad_compile_history(self):
        with tempfile.TemporaryDirectory() as td:
            grad_modes = []

            def time_once(fn, **_kwargs):
                grad_modes.append(torch.is_grad_enabled())
                fn()
                return 1.0

            results = _run(
                _one_symbol_family(Path(td)),
                [],
                extra_patches=(
                    mock.patch.object(
                        rh, "timed_min_us", side_effect=time_once),
                ),
            )
            self.assertTrue(results)
            self.assertEqual(grad_modes, [False, False])


class TestCompileAtRunAt(unittest.TestCase):
    def test_compile_at_a_run_at_b_exact_history(self):
        with tempfile.TemporaryDirectory() as td:
            results = _run(_one_symbol_family(Path(td)),
                           ["--compile-at", "s0=8", "--run-at", "s0=16"])
            (key, row), = results.items()
            self.assertTrue(key.endswith("::s0=16::dynamic"))
            # Exactly A invoked pre-measure, user-chosen, no injected shapes.
            self.assertEqual(row["compile_bindings"], [{"s0": 8}])
            self.assertEqual(row["compile_bindings_source"], "explicit")
            self.assertIn("recompiled", row)  # generalization is visible

    def test_repeated_compile_at_preserves_order(self):
        with tempfile.TemporaryDirectory() as td:
            results = _run(_one_symbol_family(Path(td)),
                           ["--compile-at", "s0=8", "--compile-at", "s0=32",
                            "--run-at", "s0=16"])
            (_key, row), = results.items()
            self.assertEqual(row["compile_bindings"],
                             [{"s0": 8}, {"s0": 32}])
            self.assertEqual(row["compile_bindings_source"], "explicit")

    def test_mixed_alias_spellings_error(self):
        err = io.StringIO()
        with self.assertRaises(SystemExit) as cm, \
                contextlib.redirect_stderr(err):
            benchmark_repro("x.py", _Scale, _mk,
                            args=["--run-at", "s0=8", "--bind", "s0=16",
                                  "--no-gpu-lock"])
        self.assertEqual(cm.exception.code, 2)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
            benchmark_repro("x.py", _Scale, _mk,
                            args=["--compile-at", "s0=8", "--prewarm",
                                  "s0=16", "--dynamic", "--no-gpu-lock"])


class TestFreeze(unittest.TestCase):
    def test_freeze_one_of_two_symbols_keeps_only_the_other_dynamic(self):
        with tempfile.TemporaryDirectory() as td:
            repro_file = _two_symbol_family(Path(td))
            results = _run(repro_file, ["--freeze", "s0=8"])
            (_key, row), = results.items()
            self.assertEqual(row["frozen_symbols"], {"s0": 8})
            # Recorded point s0=4 overridden by the freeze; s1 kept.
            self.assertEqual(row["binding"], {"s0": 8, "s1": 8})
            # Auto warm bindings all carry the frozen value.
            for wb in row["compile_bindings"]:
                self.assertEqual(wb["s0"], 8)

    def test_freeze_bare_name_uses_table_hint(self):
        with tempfile.TemporaryDirectory() as td:
            results = _run(_two_symbol_family(Path(td)), ["--freeze", "s0"])
            (_key, row), = results.items()
            self.assertEqual(row["frozen_symbols"], {"s0": 4})

    def test_freeze_conflicting_run_at_errors(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as cm:
                _run(_two_symbol_family(Path(td)),
                     ["--freeze", "s0=8", "--run-at", "s0=9,s1=16"])
            self.assertIn("contradicts --freeze", str(cm.exception))

    def test_freeze_all_symbols_errors_use_static(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as cm:
                _run(_two_symbol_family(Path(td)),
                     ["--freeze", "s0=8,s1=16"])
            self.assertIn("use --static", str(cm.exception))

    def test_freeze_supports_a_symint_only_live_symbol(self):
        # Native ShapeEnv replay gives a symint-only symbol a private root
        # input, preserving it even when no tensor dimension names it.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "shapes.json").write_text(json.dumps({
                "symbols": {"s0": {"hint": 4, "range": [2, None]},
                            "s1": {"hint": 8, "range": [2, None]}},
                "guards": [],
                "points": [
                    {"shape_hash": "aaaa1111", "captured_dynamic": True,
                     "bindings": {"s0": 4, "s1": 8},
                     "models": {"m": {"occurrences": 1}},
                     "inputs": [[["s0", 3], "f32"], ["I", 8, "s1"]]},
                ]}))
            repro = d / "repro.py"
            repro.write_text("# stub")
            results = _run(str(repro), ["--freeze", "s0=8"])
            (_key, row), = results.items()
            self.assertEqual(row["frozen_symbols"], {"s0": 8})

    def test_freeze_with_static_errors(self):
        with self.assertRaises(SystemExit) as cm:
            benchmark_repro("x.py", _Scale, _mk,
                            args=["--static", "--freeze", "s0=8",
                                  "--no-gpu-lock"])
        self.assertEqual(cm.exception.code, 2)

    def test_mixed_composite_dim_supports_partial_freeze(self):
        # ShapesSpec represents `8*s1` directly and asserts the relation.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "shapes.json").write_text(json.dumps({
                "symbols": {"s0": {"hint": 4, "range": [2, None]},
                            "s1": {"hint": 8, "range": [2, None]}},
                "guards": [],
                "points": [
                    {"shape_hash": "aaaa1111", "captured_dynamic": True,
                     "bindings": {"s0": 4, "s1": 8},
                     "models": {"m": {"occurrences": 1}},
                     "inputs": [[["s0*s1", 3], "f32"]]},
                ]}))
            repro = d / "repro.py"
            repro.write_text("# stub")
            results = _run(str(repro), ["--freeze", "s0=8"])
            (_key, row), = results.items()
            self.assertEqual(row["frozen_symbols"], {"s0": 8})


class TestExactBindings(unittest.TestCase):
    """Explicit bindings are EXACT family shapes — no silent
    inheritance from an arbitrary saved point."""

    def test_incomplete_run_at_errors_naming_missing(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as cm:
                _run(_two_symbol_family(Path(td)), ["--run-at", "s0=16"])
            self.assertIn("missing ['s1']", str(cm.exception))

    def test_incomplete_static_run_at_errors(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as cm:
                _run(_two_symbol_family(Path(td)),
                     ["--static", "--run-at", "s0=16"])
            self.assertIn("missing ['s1']", str(cm.exception))

    def test_incomplete_compile_at_errors(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as cm:
                _run(_two_symbol_family(Path(td)),
                     ["--compile-at", "s0=16",
                      "--run-at", "s0=16,s1=32"])
            self.assertIn("missing ['s1']", str(cm.exception))

    def test_freeze_completes_a_partial_run_at(self):
        # --freeze s0=8 --run-at s1=512 is complete and valid; the row and
        # compile history carry the COMPLETE effective bindings.
        with tempfile.TemporaryDirectory() as td:
            results = _run(_two_symbol_family(Path(td)),
                           ["--freeze", "s0=8", "--run-at", "s1=512"])
            (_key, row), = results.items()
            self.assertEqual(row["binding"], {"s0": 8, "s1": 512})
            for wb in row["compile_bindings"]:
                self.assertEqual(sorted(wb), ["s0", "s1"])
                self.assertEqual(wb["s0"], 8)

    def test_complete_bindings_carry_full_json_values(self):
        with tempfile.TemporaryDirectory() as td:
            results = _run(_two_symbol_family(Path(td)),
                           ["--compile-at", "s0=8,s1=16",
                            "--run-at", "s0=16,s1=32"])
            (_key, row), = results.items()
            self.assertEqual(row["binding"], {"s0": 16, "s1": 32})
            self.assertEqual(row["compile_bindings"],
                             [{"s0": 8, "s1": 16}])
            self.assertEqual(row["compile_bindings_source"], "explicit")


class TestDuplicateAssignments(unittest.TestCase):
    """Duplicated symbols are typos, never last-value-wins."""

    def test_duplicate_in_run_at_errors(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as cm:
                _run(_one_symbol_family(Path(td)),
                     ["--run-at", "s0=8,s0=9"])
            self.assertIn("assigns 's0' twice", str(cm.exception))

    def test_duplicate_in_compile_at_errors(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as cm:
                _run(_one_symbol_family(Path(td)),
                     ["--compile-at", "s0=8,s0=9", "--run-at", "s0=16"])
            self.assertIn("assigns 's0' twice", str(cm.exception))

    def test_conflicting_freeze_errors_identical_ok(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as cm:
                _run(_two_symbol_family(Path(td)),
                     ["--freeze", "s0=4", "--freeze", "s0=8"])
            self.assertIn("conflicting values", str(cm.exception))
        with tempfile.TemporaryDirectory() as td:
            results = _run(_two_symbol_family(Path(td)),
                           ["--freeze", "s0=8", "--freeze", "s0=8"])
            (_key, row), = results.items()
            self.assertEqual(row["frozen_symbols"], {"s0": 8})


if __name__ == "__main__":
    unittest.main()
