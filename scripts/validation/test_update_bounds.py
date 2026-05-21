#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

import update_bounds


def _write_fake_repro(root: Path) -> Path:
    repro_dir = root / "fake_index_repro"
    repro_dir.mkdir()
    (repro_dir / "repro.py").write_text(
        """
import torch


_shapes_config = "(T([4], i64, gen=Index(10)), T([3], f32))"


class Repro(torch.nn.Module):
    def forward(self, idx, table):
        return table[idx]


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    from repro_harness import make_inputs_from_config
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()
""".lstrip()
    )
    (repro_dir / "shapes.txt").write_text(
        "bad: (T([4], i64, gen=Index(10)), T([3], f32))\n"
        "good: (T([4], i64, gen=Index(3)), T([3], f32))\n"
    )
    return repro_dir / "repro.py"


def _write_non_index_failure_repro(root: Path) -> Path:
    repro_dir = root / "fake_non_index_failure_repro"
    repro_dir.mkdir()
    (repro_dir / "repro.py").write_text(
        """
import torch


_shapes_config = "(T([4], i64, gen=Index(10)), T([3], f32))"


class Repro(torch.nn.Module):
    def forward(self, idx, table):
        raise RuntimeError("eager failed before indexing")


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    from repro_harness import make_inputs_from_config
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()
""".lstrip()
    )
    return repro_dir / "repro.py"


def test_detects_exact_bad_shape_and_bound_repair():
    with tempfile.TemporaryDirectory() as tmp:
        repro = _write_fake_repro(Path(tmp))

        summary = update_bounds.validate_repro(
            repro,
            all_shapes=True,
            shape_label="bad",
            device="cpu",
            isolate=False,
        )

        assert summary["points"] == 1
        result = summary["results"][0]
        assert result["shape"] == "bad"
        assert result["source"] == "shapes.txt"
        assert result["original_ok"] is False
        assert result["failure_class"] == "index_bounds"
        assert result["needs_bound_repair"] is True
        assert result["repaired_ok"] is True
        assert result["repairs"] == [
            {
                "input_index": 0,
                "old_high": 10,
                "new_high": 3,
                "low": 0,
                "shape": [4],
                "dtype": "torch.int64",
            }
        ]


def test_write_repairs_bad_shape_and_revalidates():
    with tempfile.TemporaryDirectory() as tmp:
        repro = _write_fake_repro(Path(tmp))

        summary = update_bounds.validate_repro(
            repro,
            all_shapes=True,
            shape_label="bad",
            device="cpu",
            isolate=False,
            write=True,
        )
        assert summary["updated"] == 1

        shapes = (repro.parent / "shapes.txt").read_text().splitlines()
        assert shapes[0] == "bad: (T([4], i64, gen=Index(3)), T([3], f32))"
        assert shapes[1] == "good: (T([4], i64, gen=Index(3)), T([3], f32))"

        recheck = update_bounds.validate_repro(
            repro,
            all_shapes=True,
            shape_label="bad",
            device="cpu",
            isolate=False,
        )
        result = recheck["results"][0]
        assert result["original_ok"] is True
        assert result["repairs"] == []


def test_write_repairs_default_shapes_config_only():
    with tempfile.TemporaryDirectory() as tmp:
        repro = _write_fake_repro(Path(tmp))
        original_shapes = (repro.parent / "shapes.txt").read_text()

        summary = update_bounds.validate_repro(
            repro,
            device="cpu",
            isolate=False,
            write=True,
        )
        assert summary["updated"] == 1

        result = summary["results"][0]
        assert result["shape"] == "default"
        assert result["source"] == "default"
        assert result["failure_class"] == "index_bounds"
        assert result["updated_expr"] == "(T([4], i64, gen=Index(3)), T([3], f32))"

        assert '_shapes_config = "(T([4], i64, gen=Index(3)), T([3], f32))"' in (
            repro.read_text()
        )
        assert (repro.parent / "shapes.txt").read_text() == original_shapes

        recheck = update_bounds.validate_repro(
            repro,
            device="cpu",
            isolate=False,
        )
        result = recheck["results"][0]
        assert result["original_ok"] is True
        assert result["repairs"] == []


def test_all_shapes_includes_default_and_shapes_txt_points():
    with tempfile.TemporaryDirectory() as tmp:
        repro = _write_fake_repro(Path(tmp))

        points = update_bounds.discover_shape_points(repro, all_shapes=True)

        assert [(point.label, point.source) for point in points] == [
            ("default", "default"),
            ("bad", "shapes.txt"),
            ("good", "shapes.txt"),
        ]


def test_non_index_eager_failure_is_classified_but_not_rewritten():
    with tempfile.TemporaryDirectory() as tmp:
        repro = _write_non_index_failure_repro(Path(tmp))
        original_repro = repro.read_text()

        summary = update_bounds.validate_repro(
            repro,
            device="cpu",
            isolate=False,
            write=True,
        )

        assert summary["updated"] == 0
        result = summary["results"][0]
        assert result["shape"] == "default"
        assert result["source"] == "default"
        assert result["original_ok"] is False
        assert result["failure_class"] == "eager_failure"
        assert result["index_inputs"] == [
            {
                "input_index": 0,
                "high": 10,
                "low": 0,
                "shape": [4],
                "dtype": "torch.int64",
            }
        ]
        assert result["needs_bound_repair"] is False
        assert result["repairs"] == []
        assert result["updated_expr"] is None
        assert repro.read_text() == original_repro


def test_shape_label_not_found_raises_value_error():
    with tempfile.TemporaryDirectory() as tmp:
        repro = _write_fake_repro(Path(tmp))

        try:
            update_bounds.validate_repro(
                repro,
                shape_label="missing",
                device="cpu",
                isolate=False,
            )
        except ValueError as exc:
            assert "shape label 'missing' not found" in str(exc)
        else:
            raise AssertionError("missing shape label should raise ValueError")


def test_no_shape_points_raises_value_error():
    with tempfile.TemporaryDirectory() as tmp:
        repro_dir = Path(tmp) / "empty_shape_repro"
        repro_dir.mkdir()
        repro = repro_dir / "repro.py"
        repro.write_text(
            """
import torch


class Repro(torch.nn.Module):
    def forward(self, x):
        return x
""".lstrip()
        )

        try:
            update_bounds.validate_repro(repro, device="cpu", isolate=False)
        except ValueError as exc:
            assert "no shape configs found" in str(exc)
        else:
            raise AssertionError("missing shape configs should raise ValueError")


if __name__ == "__main__":
    test_detects_exact_bad_shape_and_bound_repair()
    test_write_repairs_bad_shape_and_revalidates()
    test_write_repairs_default_shapes_config_only()
    test_all_shapes_includes_default_and_shapes_txt_points()
    test_non_index_eager_failure_is_classified_but_not_rewritten()
    test_shape_label_not_found_raises_value_error()
    test_no_shape_points_raises_value_error()
    print("All update_bounds tests passed.")
