"""Tests for shapes.json writer and loader.

Covers:
  1. Writer idempotence (re-merge no-op)
  2. Multi-model accumulation on one point
  3. Loader preference (json over txt)
  4. Loader fallback (txt-only repro still works)
  5. Parse round-trip (json point signature -> make_inputs_from_config -> tensors
     with correct shapes/dtypes/strides on meta device)

Usage:
    python scripts/test_shapes_json.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from merge_captures import (
    _extract_shapes_config,
    _write_shapes_json,
    merge_one_capture,
)
from repro_harness import (
    _eval_signature,
    _parse_shapes_json,
    load_shape_configs,
    make_inputs_from_config,
)


class TestShapesJsonWriter(unittest.TestCase):
    """Test _write_shapes_json and its integration in merge_one_capture."""

    def test_writer_creates_correct_schema(self):
        """shapes.json has the settled schema with points array."""
        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp) / "repro"
            repro_dir.mkdir()
            _write_shapes_json(
                repro_dir,
                shape_hash="75902420",
                signature="(T([32, 320, 7, 7], bf16), T([32, 1280], bf16))",
                model_key="timm/infer/mobilenetv2_100",
            )
            data = json.loads((repro_dir / "shapes.json").read_text())
            self.assertIn("points", data)
            self.assertEqual(len(data["points"]), 1)
            point = data["points"][0]
            self.assertEqual(point["shape_hash"], "75902420")
            self.assertEqual(point["signature"], "(T([32, 320, 7, 7], bf16), T([32, 1280], bf16))")
            # NOTE: the per-point "source" field was retired from shapes.json
            # in the v3 migration (76c4fb8c8) and is read by no consumer; the
            # writer no longer emits it, so the test no longer asserts it.
            self.assertIn("timm/infer/mobilenetv2_100", point["models"])
            self.assertIsNone(point["models"]["timm/infer/mobilenetv2_100"]["occurrences"])

    def test_writer_idempotent_same_model_same_hash(self):
        """Re-merging the same (model, shape_hash) does not duplicate."""
        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp) / "repro"
            repro_dir.mkdir()
            for _ in range(3):
                _write_shapes_json(
                    repro_dir,
                    shape_hash="abcd1234",
                    signature="(T([64, 128], f32),)",
                    model_key="hf/train/bert",
                )
            data = json.loads((repro_dir / "shapes.json").read_text())
            self.assertEqual(len(data["points"]), 1)
            self.assertEqual(len(data["points"][0]["models"]), 1)

    def test_writer_multi_model_accumulation(self):
        """New model on an existing point adds to models dict."""
        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp) / "repro"
            repro_dir.mkdir()
            sig = "(T([32, 320, 7, 7], bf16),)"
            _write_shapes_json(repro_dir, "aabb1122", sig, "timm/infer/mobilenetv2_100")
            _write_shapes_json(repro_dir, "aabb1122", sig, "timm/infer/resnet50")
            _write_shapes_json(repro_dir, "aabb1122", sig, "timm/train/efficientnet_b0")

            data = json.loads((repro_dir / "shapes.json").read_text())
            self.assertEqual(len(data["points"]), 1)
            models = data["points"][0]["models"]
            self.assertEqual(len(models), 3)
            self.assertIn("timm/infer/mobilenetv2_100", models)
            self.assertIn("timm/infer/resnet50", models)
            self.assertIn("timm/train/efficientnet_b0", models)

    def test_writer_multiple_points(self):
        """Different shape_hashes create separate points."""
        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp) / "repro"
            repro_dir.mkdir()
            _write_shapes_json(repro_dir, "aaaa1111", "(T([32, 64], bf16),)", "timm/infer/a")
            _write_shapes_json(repro_dir, "bbbb2222", "(T([64, 128], bf16),)", "timm/infer/b")

            data = json.loads((repro_dir / "shapes.json").read_text())
            self.assertEqual(len(data["points"]), 2)
            hashes = {p["shape_hash"] for p in data["points"]}
            self.assertEqual(hashes, {"aaaa1111", "bbbb2222"})


class TestShapesJsonLoader(unittest.TestCase):
    """Test load_shape_configs with shapes.json."""

    def test_loader_prefers_json_over_txt(self):
        """When both shapes.json and shapes.txt exist, json wins."""
        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp)
            (repro_dir / "repro.py").write_text("# dummy")
            (repro_dir / "shapes.json").write_text(json.dumps({
                "points": [{
                    "shape_hash": "json1234",
                    "signature": "(T([1, 2, 3], bf16),)",
                    "models": {"timm/infer/json_model": {"occurrences": None}},
                    "source": "captured",
                }]
            }))
            (repro_dir / "shapes.txt").write_text(
                "txt_model_deadbeef: (T([99, 99], f32),)\n"
            )
            configs = load_shape_configs(str(repro_dir / "repro.py"))
            labels = list(configs.keys())
            self.assertIn("json_model_json1234", labels)
            self.assertNotIn("txt_model_deadbeef", labels)

    def test_loader_fallback_to_txt(self):
        """When only shapes.txt exists, it is loaded correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp)
            (repro_dir / "repro.py").write_text("# dummy")
            (repro_dir / "shapes.txt").write_text(
                "model_a_12345678: (T([32, 64], bf16), T([32, 128], f32))\n"
                "model_b_87654321: (T([16, 32], f16),)\n"
            )
            configs = load_shape_configs(str(repro_dir / "repro.py"))
            self.assertEqual(len(configs), 2)
            self.assertIn("model_a_12345678", configs)
            self.assertIn("model_b_87654321", configs)

    def test_loader_existing_corpus_repro(self):
        """A real existing repro with shapes.txt still loads correctly."""
        real_repro = ROOT / "repros" / "canonical" / "sum_77824d392401" / "repro.py"
        if not real_repro.exists():
            self.skipTest("Real repro not available")
        configs = load_shape_configs(str(real_repro))
        self.assertGreater(len(configs), 0)
        for label, cfg in configs.items():
            self.assertIn("inputs", cfg)
            self.assertGreater(len(cfg["inputs"]), 0)

    def test_loader_empty_json(self):
        """shapes.json with no points returns empty dict."""
        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp)
            (repro_dir / "repro.py").write_text("# dummy")
            (repro_dir / "shapes.json").write_text(json.dumps({"points": []}))
            configs = load_shape_configs(str(repro_dir / "repro.py"))
            self.assertEqual(configs, {})

    def test_loader_legacy_json_format(self):
        """Legacy shapes.json with 'configs' key still works."""
        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp)
            (repro_dir / "repro.py").write_text("# dummy")
            (repro_dir / "shapes.json").write_text(json.dumps({
                "configs": {
                    "legacy_label": {
                        "inputs": [{"kind": "tensor", "shape": [4, 4],
                                    "dtype": "torch.float32", "stride": None,
                                    "device": "cuda"}]
                    }
                }
            }))
            configs = load_shape_configs(str(repro_dir / "repro.py"))
            self.assertIn("legacy_label", configs)


class TestParseRoundTrip(unittest.TestCase):
    """Test that json point signature -> make_inputs_from_config -> correct tensors."""

    def _roundtrip(self, signature: str, expected_shapes: list, expected_dtypes: list):
        """Parse signature, create inputs on meta device, verify shapes/dtypes."""
        specs = _eval_signature(signature)
        self.assertGreater(len(specs), 0)
        config = {"inputs": specs}
        # Use meta device to avoid needing CUDA
        for spec in config["inputs"]:
            if spec.get("kind") != "shape":
                spec["device"] = "meta"
        inputs = make_inputs_from_config(config)
        tensor_inputs = [i for i in inputs if hasattr(i, "shape")]
        self.assertEqual(len(tensor_inputs), len(expected_shapes))
        for tensor, exp_shape, exp_dtype in zip(tensor_inputs, expected_shapes, expected_dtypes):
            self.assertEqual(list(tensor.shape), exp_shape)
            self.assertEqual(tensor.dtype, exp_dtype)

    def test_bf16_tensors(self):
        import torch
        self._roundtrip(
            "(T([32, 320, 7, 7], bf16), T([32, 1280], bf16))",
            [[32, 320, 7, 7], [32, 1280]],
            [torch.bfloat16, torch.bfloat16],
        )

    def test_mixed_dtypes(self):
        import torch
        self._roundtrip(
            "(T([8192], i64), T([8192, 262144], bf16), T([8192, 1], f32))",
            [[8192], [8192, 262144], [8192, 1]],
            [torch.int64, torch.bfloat16, torch.float32],
        )

    def test_with_stride(self):
        import torch
        specs = _eval_signature("(T([4, 8], f32, stride=(1, 4)),)")
        config = {"inputs": specs}
        for spec in config["inputs"]:
            spec["device"] = "meta"
        inputs = make_inputs_from_config(config)
        self.assertEqual(list(inputs[0].shape), [4, 8])
        self.assertEqual(list(inputs[0].stride()), [1, 4])

    def test_with_shape_params(self):
        """S() entries produce plain lists, not tensors."""
        specs = _eval_signature("(T([32, 64], bf16), S([32, 64]))")
        config = {"inputs": specs}
        for spec in config["inputs"]:
            if spec.get("kind") != "shape":
                spec["device"] = "meta"
        inputs = make_inputs_from_config(config)
        self.assertEqual(len(inputs), 2)
        import torch
        self.assertIsInstance(inputs[0], torch.Tensor)
        self.assertIsInstance(inputs[1], list)
        self.assertEqual(inputs[1], [32, 64])

    def test_scalar_tensor(self):
        import torch
        self._roundtrip(
            "(T([], bf16),)",
            [[]],
            [torch.bfloat16],
        )

    def test_full_complex_signature(self):
        """Test a real-world complex signature from the corpus."""
        import torch
        sig = "(T([], bf16), T([8192], i64), T([8192, 262144], bf16), T([8192, 1], f32), T([8192, 1], f32), S([8192]), S([1, 262144]), S([8192, 262144]))"
        specs = _eval_signature(sig)
        config = {"inputs": specs}
        for spec in config["inputs"]:
            if spec.get("kind") != "shape":
                spec["device"] = "meta"
        inputs = make_inputs_from_config(config)
        # 5 tensors + 3 shape params = 8 inputs
        self.assertEqual(len(inputs), 8)
        tensors = [i for i in inputs if hasattr(i, "shape")]
        self.assertEqual(len(tensors), 5)
        self.assertEqual(list(tensors[0].shape), [])
        self.assertEqual(tensors[0].dtype, torch.bfloat16)
        self.assertEqual(list(tensors[1].shape), [8192])
        self.assertEqual(tensors[1].dtype, torch.int64)


class TestExtractShapesConfig(unittest.TestCase):
    """Test _extract_shapes_config regex extraction."""

    def test_extracts_from_v2_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "repro.py"
            src.write_text('''
_repro_version = 2
_shapes_config = "(T([32, 320, 7, 7], bf16), T([32, 1280], bf16))"

class Repro:
    def forward(self, *args):
        pass

def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)
''')
            result = _extract_shapes_config(src)
            self.assertEqual(result, "(T([32, 320, 7, 7], bf16), T([32, 1280], bf16))")

    def test_returns_none_for_v1_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "repro.py"
            src.write_text('''
def make_inputs():
    return [torch.randn(32, 64, device="cuda")]
''')
            result = _extract_shapes_config(src)
            self.assertIsNone(result)

    def test_returns_none_for_missing_file(self):
        result = _extract_shapes_config(Path("/nonexistent/path/repro.py"))
        self.assertIsNone(result)

    def test_extracts_from_real_v2_repro(self):
        """Test extraction from an actual v2 repro in the corpus."""
        real = ROOT / "repros" / "canonical" / "sum_77824d392401" / "repro.py"
        if not real.exists():
            self.skipTest("Real repro not available")
        result = _extract_shapes_config(real)
        self.assertIsNotNone(result)
        self.assertIn("T(", result)


class TestMergeIntegration(unittest.TestCase):
    """Integration tests: merge_one_capture writes shapes.json correctly."""

    def _make_v2_capture(self, root: Path, *, pattern_hash: str,
                         shape_hash: str, signature: str) -> Path:
        """Create a fake v2 capture with proper source file."""
        capture_dir = root / "capture"
        capture_dir.mkdir(parents=True, exist_ok=True)
        src_file = capture_dir / "src_repro.py"
        src_file.write_text(f'''
_repro_version = 2
_shapes_config = "{signature}"

class Repro:
    def forward(self, *args):
        pass

def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)
''')
        (capture_dir / "index.json").write_text(json.dumps([{
            "pattern_hash": pattern_hash,
            "shape_hash": shape_hash,
            "kind": "pointwise",
            "reduction_types": [],
            "n_ops": 2,
            "origin_ops": ["aten.mul.Tensor"],
            "file": str(src_file),
        }]) + "\n")
        return capture_dir

    def test_merge_writes_shapes_json_not_txt(self):
        """merge_one_capture writes shapes.json, not shapes.txt."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            capture = self._make_v2_capture(
                root,
                pattern_hash="deadbeef12345678",
                shape_hash="abcdef0123456789",
                signature="(T([32, 64], bf16),)",
            )
            merge_one_capture(capture, output, "mobilenetv2_100", suite="timm", mode="infer")
            repro_dir = output / "canonical" / "pointwise_deadbeef12345678"
            self.assertTrue((repro_dir / "shapes.json").exists())
            self.assertFalse((repro_dir / "shapes.txt").exists())

    def test_merge_idempotent(self):
        """Re-merging same capture does not duplicate points."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            capture = self._make_v2_capture(
                root,
                pattern_hash="aaa111bbb222",
                shape_hash="cccc3333dddd4444",
                signature="(T([16, 32], f32),)",
            )
            merge_one_capture(capture, output, "resnet50", suite="timm", mode="infer")
            merge_one_capture(capture, output, "resnet50", suite="timm", mode="infer")
            repro_dir = output / "canonical" / "pointwise_aaa111bbb222"
            data = json.loads((repro_dir / "shapes.json").read_text())
            self.assertEqual(len(data["points"]), 1)
            self.assertEqual(len(data["points"][0]["models"]), 1)

    def test_merge_multi_model(self):
        """Different models on same pattern accumulate in shapes.json."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            capture = self._make_v2_capture(
                root,
                pattern_hash="fff000eee111",
                shape_hash="1234567890abcdef",
                signature="(T([8, 16], bf16),)",
            )
            merge_one_capture(capture, output, "mobilenetv2_100", suite="timm", mode="infer")
            merge_one_capture(capture, output, "resnet50", suite="timm", mode="infer")
            merge_one_capture(capture, output, "efficientnet_b0", suite="timm", mode="train")

            repro_dir = output / "canonical" / "pointwise_fff000eee111"
            data = json.loads((repro_dir / "shapes.json").read_text())
            self.assertEqual(len(data["points"]), 1)
            models = data["points"][0]["models"]
            self.assertEqual(len(models), 3)
            self.assertIn("timm/infer/mobilenetv2_100", models)
            self.assertIn("timm/infer/resnet50", models)
            self.assertIn("timm/train/efficientnet_b0", models)


if __name__ == "__main__":
    unittest.main()
