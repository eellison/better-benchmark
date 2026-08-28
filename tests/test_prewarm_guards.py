"""CPU tests for --prewarm flag validation in repro_harness.

--prewarm only means something for the faithful dynamic bench path
(--dynamic --dynamic-mode mark_dynamic with recorded symbolic dims): it
steers the warm sequence of the ONE dynamic artifact timed across --bind
points. Every other mode either compiles per-binding (static) or builds a
single artifact with no dynamo warm loop (compile_fx), so a --prewarm there
would be silently discarded — making "--prewarm A" and "--prewarm B" report
identical numbers and inviting a false hint/order conclusion.

These guards fail loudly BEFORE any GPU/config work, so they are CPU-only:
  - legacy path (no --bind/--dynamic): parser.error -> SystemExit(2)
  - static mode (--bind, no --dynamic): ValueError
  - compile_fx dynamic mode: ValueError (before config resolution)
Plus: _parse_bind_args attributes a --prewarm typo to --prewarm, not --bind.

Usage:
    python tests/test_prewarm_guards.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402


from repro_harness import benchmark_repro, _parse_bind_args  # noqa: E402


class _Dummy(torch.nn.Module):
    def forward(self, x):  # noqa: D401 - trivial fixture
        return x


def _mk(shape_config=None):
    return [torch.zeros(2)]


class TestParseBindArgsAttribution(unittest.TestCase):
    def test_prewarm_typo_names_prewarm_not_bind(self):
        with self.assertRaises(ValueError) as cm:
            _parse_bind_args(["s0=bad"], flag_name="--prewarm")
        msg = str(cm.exception)
        self.assertIn("--prewarm", msg)
        self.assertNotIn("--bind", msg)

    def test_missing_equals_names_the_flag(self):
        with self.assertRaises(ValueError) as cm:
            _parse_bind_args(["nope"], flag_name="--prewarm")
        self.assertIn("--prewarm", str(cm.exception))

    def test_default_flag_name_is_bind(self):
        with self.assertRaises(ValueError) as cm:
            _parse_bind_args(["nope"])
        self.assertIn("--bind", str(cm.exception))

    def test_empty_binding_names_the_flag(self):
        with self.assertRaises(ValueError) as cm:
            _parse_bind_args([","], flag_name="--prewarm")
        self.assertIn("--prewarm", str(cm.exception))


class TestPrewarmGuards(unittest.TestCase):
    def test_prewarm_without_dynamic_or_bind_errors(self):
        err = io.StringIO()
        with self.assertRaises(SystemExit) as cm, \
                contextlib.redirect_stderr(err):
            benchmark_repro("x.py", _Dummy, _mk, args=["--prewarm", "s0=8"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--prewarm only applies to the dynamic bench path",
                      err.getvalue())

    def test_prewarm_in_static_mode_errors(self):
        # §7: a repro with NO symbols table runs static even with --bind, so
        # a warm-order flag is meaningless — rejected at the parser (the
        # in-function static-mode ValueError remains as defense-in-depth for
        # programmatic callers).
        err = io.StringIO()
        with self.assertRaises(SystemExit) as cm, \
                contextlib.redirect_stderr(err):
            benchmark_repro(
                "x.py", _Dummy, _mk,
                args=["--bind", "s0=8", "--prewarm", "s0=8", "--no-gpu-lock"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("only applies to the dynamic bench path",
                      err.getvalue())

    def test_prewarm_in_compile_fx_mode_errors_before_config(self):
        # Fires before _resolve_bound_configs, so no shapes.json is needed on
        # disk for the guard to trip — a flag mismatch shouldn't require a
        # valid fixture to surface.
        with self.assertRaises(ValueError) as cm:
            benchmark_repro(
                "x.py", _Dummy, _mk,
                args=["--dynamic", "--dynamic-mode", "compile_fx",
                      "--bind", "s0=8", "--prewarm", "s0=8", "--no-gpu-lock"])
        self.assertIn("not supported with --dynamic-mode compile_fx",
                      str(cm.exception))


if __name__ == "__main__":
    unittest.main()
