"""Focused tests for repro format versioning and upgrade decisions.

Usage:
    python scripts/test_repro_versioning.py
"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import test_repro_format
import upgrade_repros
from repro_harness import (
    CURRENT_REPRO_VERSION,
    UNVERSIONED_REPRO_VERSION,
    parse_repro_version,
    read_repro_version,
)


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


class ReproVersioningTests(unittest.TestCase):
    def test_parse_current_and_unversioned(self):
        self.assertEqual(
            parse_repro_version(f"_repro_version = {CURRENT_REPRO_VERSION}\n"),
            CURRENT_REPRO_VERSION,
        )
        self.assertEqual(
            parse_repro_version(
                f"_repro_version = {CURRENT_REPRO_VERSION}  # current\n"
            ),
            CURRENT_REPRO_VERSION,
        )
        self.assertEqual(
            parse_repro_version("_shapes_config = \"(T([1], f32),)\"\n"),
            UNVERSIONED_REPRO_VERSION,
        )

    def test_invalid_marker_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "top-level integer"):
            parse_repro_version('_repro_version = "2"\n')
        with self.assertRaisesRegex(ValueError, "exactly once"):
            parse_repro_version("_repro_version = 1\n_repro_version = 2\n")

    def test_format_check_accepts_current_or_unversioned_only(self):
        self.assertIsNone(
            test_repro_format.check_repro_version_marker(
                f"_repro_version = {CURRENT_REPRO_VERSION}\n"
            )
        )
        self.assertIsNone(
            test_repro_format.check_repro_version_marker("_shapes_config = 'x'\n")
        )
        self.assertIn(
            "unexpected _repro_version",
            test_repro_format.check_repro_version_marker("_repro_version = 1\n"),
        )

    def test_format_check_requires_real_shapes_config_assignment(self):
        self.assertEqual(
            test_repro_format.check_has_shapes_config(
                "# _shapes_config = 'x'\n"
                "class Repro:\n"
                "    def forward(self, x):\n"
                "        return x\n"
            ),
            "missing _shapes_config",
        )
        self.assertIsNone(
            test_repro_format.check_has_shapes_config("_shapes_config = 'x'\n")
        )

    def test_forward_arg_count_uses_repro_class(self):
        content = (
            "def forward(self, x, y, z):\n"
            "    return x\n\n"
            "class Repro:\n"
            "    def forward(self, x):\n"
            "        return x\n"
        )
        self.assertIsNone(test_repro_format.check_input_count(
            "_shapes_config = \"(T([1], f32),)\"\n" + content
        ))

    def test_capture_hook_template_emits_current_marker(self):
        source = (ROOT / "capture_hook.py").read_text()
        # The marker is DERIVED from CURRENT_REPRO_VERSION (interpolated into
        # the generated-repro f-string template), not a hardcoded literal --
        # so a future format bump that touches only the constant can never
        # leave this writer stamping a stale version (the exact bug the v3
        # migration hit). Pin the derivation, and confirm the retired inline
        # _shapes_config assignment is gone.
        self.assertIn("_repro_version = {CURRENT_REPRO_VERSION}\n", source)
        self.assertNotIn("\n_shapes_config =", source)

    def test_capture_hook_derived_marker_resolves_to_current(self):
        # Prove the derived template actually resolves to the current version:
        # the module-level constant capture_hook interpolates must equal the
        # harness constant the rest of the toolchain reads.
        import capture_hook

        self.assertEqual(capture_hook.CURRENT_REPRO_VERSION, CURRENT_REPRO_VERSION)

    def test_dry_run_marks_unversioned_repro_outdated(self):
        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp) / "legacy_repro"
            _write(repro_dir / "repro.py", "_shapes_config = 'x'\n")

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                changed = upgrade_repros.upgrade_one(repro_dir, dry_run=True)

            self.assertTrue(changed)
            self.assertIn(
                f"WOULD UPGRADE legacy_repro (v0 -> v{CURRENT_REPRO_VERSION})",
                out.getvalue(),
            )

    def test_current_repro_is_not_reupgraded(self):
        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp) / "current_repro"
            _write(
                repro_dir / "repro.py",
                f"_repro_version = {CURRENT_REPRO_VERSION}\n_shapes_config = 'x'\n",
            )

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                changed = upgrade_repros.upgrade_one(repro_dir, dry_run=True)

            self.assertFalse(changed)
            self.assertEqual(out.getvalue(), "")

    def test_copy_validated_repro_requires_target_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = _write(
                tmp_path / "canonical" / "repro.py",
                f"_repro_version = {CURRENT_REPRO_VERSION}\nold = True\n",
            )
            current_capture = _write(
                tmp_path / "capture_current.py",
                f"_repro_version = {CURRENT_REPRO_VERSION}\nnew = True\n",
            )
            legacy_capture = _write(tmp_path / "capture_legacy.py", "new = True\n")

            upgrade_repros.copy_validated_repro(
                current_capture,
                target,
                CURRENT_REPRO_VERSION,
            )
            self.assertEqual(read_repro_version(target), CURRENT_REPRO_VERSION)
            self.assertIn("new = True", target.read_text())

            with self.assertRaisesRegex(ValueError, "expected"):
                upgrade_repros.copy_validated_repro(
                    legacy_capture,
                    target,
                    CURRENT_REPRO_VERSION,
                )
            self.assertIn("new = True", target.read_text())

    def test_upgrade_refuses_empty_or_multi_region_capture_index(self):
        self.assertEqual(
            upgrade_repros.select_single_capture_file([{"file": "/tmp/repro.py"}]),
            "/tmp/repro.py",
        )
        with self.assertRaisesRegex(ValueError, "got 0"):
            upgrade_repros.select_single_capture_file([])
        with self.assertRaisesRegex(ValueError, "got 2"):
            upgrade_repros.select_single_capture_file([
                {"file": "/tmp/a.py"},
                {"file": "/tmp/b.py"},
            ])
        with self.assertRaisesRegex(ValueError, "missing file"):
            upgrade_repros.select_single_capture_file([{}])

    def test_unsupported_target_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "target version"):
            upgrade_repros.validate_target_version(CURRENT_REPRO_VERSION + 1)


if __name__ == "__main__":
    unittest.main()
