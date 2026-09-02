from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import repro_harness  # noqa: E402
import oracle_harness  # noqa: E402
from bench_parallel import (  # noqa: E402
    _aggregate_oracle_timings,
    _compile_policy_allows_overlay,
    _merge_into_baseline_locked,
    _persistent_worker_script,
    _tagged_perf_context_allows_overlay,
    _worker_script,
)
from bench_report import compute_deltas, load_results  # noqa: E402
_COMPILE_POLICY = {
    "inductor": {"emulate_precision_casts": True},
}


class _PlainRepro:
    pass


class ReproPolicyTests(unittest.TestCase):
    def test_compile_repro_applies_policy_without_leaking(self):
        calls = []

        def fake_compile(repro, **kwargs):
            calls.append((repro, kwargs))
            return repro

        with mock.patch.object(
            repro_harness.torch,
            "compile",
            side_effect=fake_compile,
        ):
            plain = _PlainRepro()
            self.assertIs(
                repro_harness.compile_repro(
                    plain,
                    compile_policy=_COMPILE_POLICY,
                ),
                plain,
            )
            self.assertIs(repro_harness.compile_repro(plain), plain)

        self.assertEqual(
            calls[0][1]["options"],
            {"emulate_precision_casts": True},
        )
        self.assertNotIn("options", calls[1][1])

    def test_compile_repro_merges_compatible_explicit_options(self):
        captured = {}

        def fake_compile(repro, **kwargs):
            captured.update(kwargs)
            return repro

        with mock.patch.object(
            repro_harness.torch,
            "compile",
            side_effect=fake_compile,
        ):
            repro_harness.compile_repro(
                _PlainRepro(),
                compile_policy=_COMPILE_POLICY,
                options={
                    "emulate_precision_casts": True,
                    "coordinate_descent_tuning": True,
                },
            )
        self.assertEqual(
            captured["options"],
            {
                "emulate_precision_casts": True,
                "coordinate_descent_tuning": True,
            },
        )

    def test_compile_repro_normalizes_compatible_option_alias(self):
        captured = {}

        def fake_compile(repro, **kwargs):
            captured.update(kwargs)
            return repro

        with mock.patch.object(
            repro_harness.torch,
            "compile",
            side_effect=fake_compile,
        ):
            repro_harness.compile_repro(
                _PlainRepro(),
                compile_policy=_COMPILE_POLICY,
                options={
                    "emulate-precision-casts": True,
                    "coordinate-descent-tuning": True,
                },
            )
        self.assertEqual(
            captured["options"],
            {
                "emulate_precision_casts": True,
                "coordinate_descent_tuning": True,
            },
        )

    def test_compile_environment_restores_triton_libdevice_state(self):
        try:
            from triton import knobs
        except ImportError:
            self.skipTest("Triton is unavailable")

        original_knob = knobs.nvidia.libdevice_path
        try:
            knobs.nvidia.libdevice_path = "before-libdevice"
            with mock.patch.dict(
                os.environ,
                {"TRITON_LIBDEVICE_PATH": "before-libdevice"},
            ):
                with repro_harness.preserve_compile_environment():
                    os.environ["TRITON_LIBDEVICE_PATH"] = "leaked-libdevice"
                    knobs.nvidia.libdevice_path = "leaked-libdevice"
                self.assertEqual(
                    os.environ["TRITON_LIBDEVICE_PATH"],
                    "before-libdevice",
                )
                self.assertEqual(
                    knobs.nvidia.libdevice_path,
                    "before-libdevice",
                )
        finally:
            knobs.nvidia.libdevice_path = original_knob

    def test_compile_repro_rejects_policy_conflicts(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            repro_harness.compile_repro(
                _PlainRepro(),
                compile_policy=_COMPILE_POLICY,
                options={"emulate_precision_casts": False},
            )

    def test_compile_repro_rejects_hyphenated_policy_conflict(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            repro_harness.compile_repro(
                _PlainRepro(),
                compile_policy=_COMPILE_POLICY,
                options={"emulate-precision-casts": False},
            )

    def test_compile_repro_rejects_non_inductor_backend(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            repro_harness.compile_repro(
                _PlainRepro(),
                compile_policy=_COMPILE_POLICY,
                backend="eager",
            )

    def test_compile_repro_rejects_non_mapping_options(self):
        with self.assertRaisesRegex(TypeError, "mapping"):
            repro_harness.compile_repro(
                _PlainRepro(),
                options="emulate_precision_casts=True",
            )

    def test_invalid_compile_policy_is_rejected(self):
        cases = [
            (
                {"unknown": {}},
                "unsupported compile policy keys",
            ),
            (
                {"inductor": {"unknown": True}},
                "unsupported per-point",
            ),
            (
                {"inductor": {"emulate_precision_casts": 1}},
                "must be a bool",
            ),
        ]
        for policy, error in cases:
            with self.subTest(policy=policy):
                with self.assertRaisesRegex((TypeError, ValueError), error):
                    repro_harness.normalize_compile_policy(policy)

    def test_compile_policy_accepts_read_only_mappings(self):
        policy = MappingProxyType({
            "inductor": MappingProxyType({
                "emulate_precision_casts": True,
            }),
        })
        self.assertEqual(
            repro_harness.normalize_compile_policy(policy),
            _COMPILE_POLICY,
        )

    def test_invalid_point_policy_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp)
            (repro_dir / "shapes.json").write_text(json.dumps({
                "points": [{
                    "shape_hash": "deadbeef",
                    "signature": "T(1, f32)",
                    "compile_policy": {"unsupported": True},
                }],
            }))
            with self.assertRaisesRegex(ValueError, "unsupported"):
                repro_harness.load_shape_configs(str(repro_dir / "repro.py"))

    def test_audited_input_points_declare_compile_policy(self):
        expected = {
            "sum_sum_34b54dfb6c54": {"77702286"},
            "sum_sum_643db2887a01": {"18f48531"},
            "sum_sum_sum_c2ae956520f9": {"cf12eab2"},
            "sum_sum_sum_d0cb44a92f6d": {"09b68f1c"},
            "var_mean_mean_5f6d60b04d02": {
                "79146166",
                "8881253b",
                "c99f0cec",
                "77734290",
                "f7eda15e",
            },
        }
        for repro, expected_hashes in expected.items():
            with self.subTest(repro=repro):
                repro_dir = ROOT / "repros" / "canonical" / repro
                data = json.loads((repro_dir / "shapes.json").read_text())
                actual = {
                    point["shape_hash"]
                    for point in data["points"]
                    if point.get("compile_policy") == _COMPILE_POLICY
                }
                self.assertEqual(actual, expected_hashes)
                self.assertNotIn(
                    "compile_policy",
                    (repro_dir / "repro.py").read_text(),
                )

                configs = repro_harness.load_shape_configs(
                    str(repro_dir / "repro.py")
                )
                configured = {
                    label.rsplit("_", 1)[-1]
                    for label, config in configs.items()
                    if repro_harness.compile_policy_from_config(config)
                    == _COMPILE_POLICY
                }
                self.assertEqual(configured, expected_hashes)

    def test_recapture_merge_preserves_point_compile_policy(self):
        from merge_captures import _write_shapes_json

        with tempfile.TemporaryDirectory() as tmp:
            repro_dir = Path(tmp)
            shapes_path = repro_dir / "shapes.json"
            shapes_path.write_text(json.dumps({
                "points": [{
                    "shape_hash": "deadbeef",
                    "models": {
                        "suite/train/model": {"occurrences": 1},
                    },
                    "inputs": [],
                    "compile_policy": _COMPILE_POLICY,
                }],
            }))
            _write_shapes_json(
                repro_dir,
                "deadbeef",
                "()",
                "suite/train/model",
                occurrences=2,
                inputs=[],
            )
            point = json.loads(shapes_path.read_text())["points"][0]
            self.assertEqual(point["compile_policy"], _COMPILE_POLICY)

    def test_generated_workers_use_compile_repro(self):
        args = {
            "root": str(ROOT),
            "all_shapes": False,
            "no_cd": True,
            "n_warmup": 1,
            "n_rep": 1,
            "strict_gpu_lock": False,
            "workload_kind": "repro",
        }
        persistent = _persistent_worker_script("0", args)
        ast.parse(persistent)
        self.assertIn(
            "compile_policy_from_config",
            persistent,
        )
        self.assertIn("@preserve_compile_environment()", persistent)
        self.assertEqual(
            persistent.count("compiled = compile_repro("),
            1,
        )
        self.assertEqual(
            persistent.count("compiled_cd = compile_repro("),
            1,
        )
        self.assertIn(
            'options={"coordinate_descent_tuning": True}',
            persistent,
        )
        self.assertNotIn(
            "inductor_config.coordinate_descent_tuning =",
            persistent,
        )
        self.assertEqual(
            persistent.count('"compile_policy": compile_policy'),
            1,
        )

        repro_path = (
            ROOT
            / "repros"
            / "canonical"
            / "var_mean_mean_5f6d60b04d02"
            / "repro.py"
        )
        one_shot = _worker_script(str(repro_path), "0", args)
        ast.parse(one_shot)
        self.assertEqual(
            one_shot.count("compiled = compile_repro("),
            1,
        )
        self.assertEqual(
            one_shot.count("compiled_cd = compile_repro("),
            1,
        )
        self.assertIn(
            'options={"coordinate_descent_tuning": True}',
            one_shot,
        )
        self.assertNotIn(
            "inductor_config.coordinate_descent_tuning =",
            one_shot,
        )
        self.assertEqual(
            one_shot.count('"compile_policy": compile_policy'),
            1,
        )

    def test_local_comparison_rejects_compile_policy_mismatch(self):
        base = {
            "example": {
                "compiled_us": 10.0,
            },
        }
        head = {
            "example": {
                "compile_policy": {
                    "inductor": {
                        "emulate_precision_casts": True,
                    },
                },
                "compiled_us": 9.0,
            },
        }
        with self.assertRaisesRegex(ValueError, "refusing to compare"):
            compute_deltas(base, head)

    def test_local_comparison_treats_missing_and_empty_policy_as_default(self):
        base = {
            "example": {
                "compiled_us": 10.0,
            },
        }
        head = {
            "example": {
                "compile_policy": {},
                "compiled_us": 9.0,
            },
        }
        deltas = compute_deltas(base, head)
        self.assertEqual(len(deltas), 1)
        self.assertAlmostEqual(deltas[0].delta_pct, -0.1)

    def test_merge_rejects_mixed_semantic_benchmark_config(self):
        baseline = {
            "_metadata": {
                "benchmark_config": {
                    "combo_kernels": False,
                },
            },
            "repros/canonical/base/repro.py": {
                "default": {
                    "compiled_us": 10.0,
                },
            },
        }
        new_results = {
            "repros/canonical/head/repro.py": {
                "default": {
                    "compiled_us": 9.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError, "different benchmark"):
                _merge_into_baseline_locked(
                    baseline_path,
                    new_results,
                    {},
                    workload_kind="repro",
                    config_metadata={
                        "combo_kernels": True,
                    },
                )

    def test_merge_rejects_cd_and_no_cd_results(self):
        baseline = {
            "_metadata": {
                "benchmark_config": {
                    "coordinate_descent": True,
                },
            },
            "repros/canonical/base/repro.py": {
                "default": {
                    "compiled_us": 10.0,
                    "coord_descent_us": 9.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError, "different benchmark"):
                _merge_into_baseline_locked(
                    baseline_path,
                    {
                        "repros/canonical/head/repro.py": {
                            "default": {
                                "compiled_us": 8.0,
                                "coord_descent_us": None,
                            },
                        },
                    },
                    {},
                    workload_kind="repro",
                    config_metadata={
                        "coordinate_descent": False,
                    },
                )

    def test_failure_only_merge_does_not_relabel_existing_results(self):
        baseline = {
            "_metadata": {
                "benchmark_config": {
                    "combo_kernels": False,
                },
            },
            "repros/canonical/base/repro.py": {
                "default": {
                    "compiled_us": 10.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline))
            _merge_into_baseline_locked(
                baseline_path,
                {},
                {
                    "repros/canonical/failed/repro.py": {
                        "status": "failed",
                        "error": "compile failed",
                    },
                },
                workload_kind="repro",
                config_metadata={
                    "combo_kernels": True,
                },
            )
            merged = json.loads(baseline_path.read_text())

        self.assertEqual(
            merged["_metadata"]["benchmark_config"],
            {"combo_kernels": False},
        )

    def test_merge_preserves_other_points_when_compile_policy_changes(self):
        repro_path = "repros/canonical/example/repro.py"
        baseline = {
            "_metadata": {
                "benchmark_config": {
                    "coordinate_descent": True,
                },
            },
            repro_path: {
                "old_shape": {
                    "compile_policy": {},
                    "compiled_us": 10.0,
                },
                "other_old_shape": {
                    "compile_policy": {},
                    "compiled_us": 11.0,
                },
            },
        }
        policy = {
            "inductor": {
                "emulate_precision_casts": True,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline))
            _merge_into_baseline_locked(
                baseline_path,
                {
                    repro_path: {
                        "old_shape": {
                            "compile_policy": policy,
                            "compiled_us": 9.0,
                        },
                    },
                },
                {},
                workload_kind="repro",
                partial_repros={repro_path},
                config_metadata={
                    "coordinate_descent": True,
                },
            )
            merged = json.loads(baseline_path.read_text())

        self.assertEqual(
            set(merged[repro_path]),
            {"old_shape", "other_old_shape"},
        )
        self.assertEqual(
            merged[repro_path]["old_shape"]["compile_policy"],
            policy,
        )

    def test_perf_overlay_preserves_other_point_policies(self):
        policy = {
            "inductor": {
                "emulate_precision_casts": True,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            repro_path = Path(tmp) / "repro.py"
            perf_path = Path(tmp) / "perf.json"
            perf_path.write_text(
                json.dumps(
                    {
                        "H100": {
                            "old_shape": {
                                "compile_policy": {},
                                "compiled_us": 10.0,
                            },
                            "other_old_shape": {
                                "compile_policy": {},
                                "compiled_us": 11.0,
                            },
                        },
                    },
                ),
            )
            repro_harness._save_perf(
                str(repro_path),
                "H100",
                "old_shape",
                {
                    "compile_policy": policy,
                    "compiled_us": 9.0,
                },
            )
            perf = json.loads(perf_path.read_text())

        self.assertEqual(
            set(perf["H100"]),
            {"old_shape", "other_old_shape"},
        )
        self.assertEqual(
            perf["H100"]["old_shape"]["compile_policy"],
            policy,
        )

    def test_tagged_perf_overlay_allows_point_local_policy_change(self):
        prior = {
            "old_shape": {
                "compile_policy": {},
                "compiled_us": 10.0,
            },
            "other_old_shape": {
                "compile_policy": {},
                "compiled_us": 11.0,
            },
        }
        incoming = {
            "old_shape": {
                "compile_policy": {
                    "inductor": {
                        "emulate_precision_casts": True,
                    },
                },
                "compiled_us": 9.0,
            },
        }
        self.assertTrue(
            _compile_policy_allows_overlay(prior, incoming),
        )

    def test_tagged_perf_overlay_rejects_config_change(self):
        prior = {
            "old_shape": {
                "compile_policy": {},
                "benchmark_config": {},
                "compiled_us": 10.0,
            },
            "other_old_shape": {
                "compile_policy": {},
                "benchmark_config": {},
                "compiled_us": 11.0,
            },
        }
        incoming = {
            "old_shape": {
                "compile_policy": {},
                "compiled_us": 9.0,
            },
        }
        self.assertFalse(
            _tagged_perf_context_allows_overlay(
                prior,
                incoming,
                {"coordinate_descent": False},
            ),
        )
        self.assertTrue(
            _tagged_perf_context_allows_overlay(
                prior,
                incoming,
                {"coordinate_descent": True},
            ),
        )

    def test_oracle_aggregation_preserves_point_compile_policy(self):
        policy = {
            "inductor": {
                "emulate_precision_casts": True,
            },
        }
        aggregated = _aggregate_oracle_timings(
            {
                "repros/canonical/example": {
                    "model_deadbeef": {
                        "compile_policy": policy,
                        "oracle_us": 5.0,
                        "compile_us": 10.0,
                        "ratio": 2.0,
                        "status": "GOOD",
                    },
                },
            },
        )

        entry = aggregated["example"]
        self.assertEqual(entry["compile_policy"], policy)
        self.assertEqual(
            entry["points"]["model_deadbeef"]["compile_policy"],
            policy,
        )
        self.assertEqual(
            entry["points_by_shape"]["deadbeef"]["compile_policy"],
            policy,
        )

    def test_oracle_dispatch_failure_preserves_compile_policy(self):
        with mock.patch.object(
            oracle_harness,
            "resolve_oracle",
            side_effect=oracle_harness.OracleDispatchError("no match"),
        ):
            with mock.patch("builtins.print"):
                result = oracle_harness.bench_oracle(
                    lambda _inputs: None,
                    _PlainRepro(),
                    [],
                    "example",
                    compile_policy=_COMPILE_POLICY,
                )

        self.assertEqual(result["status"], "NO_ORACLE_FOR_SHAPE")
        self.assertEqual(
            result["compile_policy"],
            _COMPILE_POLICY,
        )


if __name__ == "__main__":
    unittest.main()
