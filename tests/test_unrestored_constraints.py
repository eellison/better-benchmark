"""Tests for the dynamic bench's constraint-fidelity accounting.

PR80 re-review P1: `--dynamic` restores recorded symbol ranges via
mark_dynamic(min=, max=) where dynamo can express them (fully-concrete
bounds on bare tensor dims); EVERYTHING else it cannot restore — informative
half-open ranges, ranges on composite-dim-only or symint-value symbols, and
residual guards — must be disclosed, machine-readably, on every result row
(structured `unrestored_constraints` entries + `constraints_fully_restored`
stamped true/false on EVERY dynamic row, plus `dynamic_mode` and a
`::dynamic/<mode>` row key), not just guards. A guard-free family with an
unrestorable range previously produced an unqualified number.

The bench-level tests run `_run_bound_benchmark` end to end on CPU with
compilation/timing mocked (count_kernels, timed_min_us, torch.compile), so
they pin the WIRING — the caveat actually reaching the rows — not just the
helper.

Usage:
    python tests/test_unrestored_constraints.py
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
from repro_harness import (  # noqa: E402
    benchmark_repro,
    _dynamic_dim_bounds_for_repro,
    _unrestored_constraints_for_repro,
)


class _Scale(torch.nn.Module):
    def forward(self, x):
        return x * 2


def _mk(shape_config=None):
    return [torch.zeros(4, 3)]


def _write_family(d: Path, srange, guards=(), inputs=None) -> str:
    (d / "shapes.json").write_text(json.dumps({
        "symbols": {"s0": {"hint": 4, "range": list(srange)}},
        "guards": list(guards),
        "points": [{
            "shape_hash": "aaaa1111",
            "captured_dynamic": True,
            "bindings": {"s0": 4},
            "models": {"probe/infer/m": {"occurrences": 1}},
            "inputs": inputs if inputs is not None else [[["s0", 3], "f32"]],
        }],
    }))
    repro = d / "repro.py"
    repro.write_text("# stub")
    return str(repro)


class TestUnrestoredConstraintsHelper(unittest.TestCase):
    """Unit coverage over the re-review's four probe rows + guards/trivial."""

    def test_every_fidelity_loss_is_reported(self):
        shapes = {
            "symbols": {
                "s0": {"hint": 8, "range": [4, 64]},    # bare dim: RESTORED
                "s1": {"hint": 8, "range": [4, None]},  # bare dim, half-open
                "s2": {"hint": 8, "range": [4, 64]},    # composite-dim only
                "s3": {"hint": 8, "range": [4, 64]},    # symint VALUE only
                "s4": {"hint": 8, "range": [2, None]},  # trivial (== plain mark)
            },
            "guards": ["Eq(Mod(s0, 4), 0)"],
            "points": [{
                "shape_hash": "x", "captured_dynamic": True,
                "bindings": {"s0": 8, "s1": 8, "s2": 8, "s3": 8, "s4": 8},
                "models": {"m": {"occurrences": 1}},
                "inputs": [
                    [["s0", "s1", "s4"], "f32"],
                    [["2*s2"], "f32"],
                    ["I", 8, "s3"],
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "shapes.json").write_text(json.dumps(shapes))
            (d / "repro.py").write_text("# stub")
            f = str(d / "repro.py")
            bounds = _dynamic_dim_bounds_for_repro(f)
            self.assertEqual(bounds, {(0, 0): (4, 64)})
            out = _unrestored_constraints_for_repro(f, bounds)
            # Structured entries (alignment §3b): typed objects, not
            # presentation strings — JSON is the machine interface.
            self.assertEqual(out, [
                {"kind": "range", "symbol": "s1", "lower": 4, "upper": None},
                {"kind": "range", "symbol": "s2", "lower": 4, "upper": 64},
                {"kind": "range", "symbol": "s3", "lower": 4, "upper": 64},
                {"kind": "guard", "expr": "Eq(Mod(s0, 4), 0)"},
            ])

    def test_trivial_exemption_is_placement_aware(self):
        """Third-review P1: suppressing a [2, None] range is valid ONLY when
        the symbol itself occurs as a bare marked tensor dim — that is where
        a plain mark's implicit floor-2/no-ceiling range lands. A composite-
        only or symint-value symbol has no dim carrying its range (a marked
        '2*s0' dim gets a fresh [2, inf) symbol), and a 0/1-capable floor is
        WIDER than the mark's 0/1-specialized floor — all must report."""
        cases = [
            # (inputs, srange, expect_reported)
            ([[["s0", 3], "f32"]], [2, None], False),  # bare: suppressed
            ([[["2*s0"], "f32"]], [2, None], True),    # composite-only
            ([["I", 4, "s0"]], [2, None], True),       # symint-value only
            ([[["s0", 3], "f32"]], [0, None], True),   # zero-capable floor
            ([[["s0", 3], "f32"]], [1, None], True),   # one-capable floor
        ]
        for inputs, srange, expect in cases:
            with tempfile.TemporaryDirectory() as td:
                f = _write_family(Path(td), srange=srange, inputs=inputs)
                bounds = _dynamic_dim_bounds_for_repro(f)
                self.assertEqual(bounds, {}, (inputs, srange))
                out = _unrestored_constraints_for_repro(f, bounds)
                if expect:
                    self.assertEqual(
                        out, [{"kind": "range", "symbol": "s0",
                               "lower": srange[0], "upper": srange[1]}],
                        (inputs, srange))
                else:
                    self.assertEqual(out, [], (inputs, srange))

    def test_fully_restored_guardless_family_has_no_caveat(self):
        with tempfile.TemporaryDirectory() as td:
            f = _write_family(Path(td), srange=[4, 64])
            bounds = _dynamic_dim_bounds_for_repro(f)
            self.assertEqual(bounds, {(0, 0): (4, 64)})
            self.assertEqual(_unrestored_constraints_for_repro(f, bounds), [])

    def test_static_repro_has_no_caveat(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "shapes.json").write_text(json.dumps({"points": [{
                "shape_hash": "aa", "inputs": [[[8, 4], "f32"]],
                "models": {"m": {"occurrences": 1}}}]}))
            (d / "repro.py").write_text("# stub")
            f = str(d / "repro.py")
            self.assertEqual(_unrestored_constraints_for_repro(f, {}), [])


class TestRowStamping(unittest.TestCase):
    """Bench-level: the caveat must reach the result rows (mocked compile)."""

    def _run(self, repro_file):
        fake_compile = lambda m, **kw: m  # noqa: E731 - eager stand-in
        with mock.patch.object(rh, "count_kernels",
                               return_value=(1, ["k"])), \
                mock.patch.object(rh, "timed_min_us", return_value=1.0), \
                mock.patch.object(torch, "compile", fake_compile):
            return benchmark_repro(
                repro_file, _Scale, _mk,
                args=["--dynamic", "--bind", "s0=4", "--no-gpu-lock"])

    def test_half_open_range_stamps_rows(self):
        with tempfile.TemporaryDirectory() as td:
            repro_file = _write_family(Path(td), srange=[4, None])
            results = self._run(repro_file)
            self.assertTrue(results)
            # Row keys carry the artifact mechanism so a diagnostic
            # compile_fx --update-perf run can never overwrite these rows.
            for key in results:
                self.assertTrue(key.endswith("::dynamic/mark_dynamic"), key)
            for row in results.values():
                self.assertEqual(row["dynamic_mode"], "mark_dynamic")
                self.assertIs(row["constraints_fully_restored"], False)
                self.assertEqual(row["unrestored_constraints"], [
                    {"kind": "range", "symbol": "s0",
                     "lower": 4, "upper": None}])

    def test_restored_range_stamps_fully_restored_true(self):
        # constraints_fully_restored is ALWAYS stamped on a dynamic row —
        # absence would be ambiguous with "not assessed" (alignment §3b).
        with tempfile.TemporaryDirectory() as td:
            repro_file = _write_family(Path(td), srange=[4, 64])
            results = self._run(repro_file)
            self.assertTrue(results)
            for row in results.values():
                self.assertEqual(row["dynamic_mode"], "mark_dynamic")
                self.assertIs(row["constraints_fully_restored"], True)
                self.assertNotIn("unrestored_constraints", row)

    def test_compile_fx_rows_key_apart_and_restore_nothing(self):
        """compile_fx threads NO bounds into its make_fx artifact, so its
        accounting runs with an EMPTY restored map: a fully-concrete range
        that mark_dynamic WOULD restore is unrestored here. And its row key
        carries `dynamic/compile_fx`, so a diagnostic --update-perf run can
        never overwrite the mark_dynamic row for the same binding
        (alignment §3b)."""
        with tempfile.TemporaryDirectory() as td:
            repro_file = _write_family(Path(td), srange=[4, 64])
            fake_artifact = lambda *args: args  # noqa: E731
            with mock.patch.object(rh, "_build_compile_fx_dynamic_artifact",
                                   return_value=fake_artifact), \
                    mock.patch.object(rh, "timed_min_us", return_value=1.0):
                results = benchmark_repro(
                    repro_file, _Scale, _mk,
                    args=["--dynamic", "--dynamic-mode", "compile_fx",
                          "--bind", "s0=4", "--no-gpu-lock"])
            self.assertTrue(results)
            for key, row in results.items():
                self.assertTrue(key.endswith("::dynamic/compile_fx"), key)
                self.assertEqual(row["dynamic_mode"], "compile_fx")
                self.assertIs(row["constraints_fully_restored"], False)
                self.assertEqual(row["unrestored_constraints"], [
                    {"kind": "range", "symbol": "s0",
                     "lower": 4, "upper": 64}])


if __name__ == "__main__":
    unittest.main()
