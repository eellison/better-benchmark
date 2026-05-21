"""Focused tests for scripts/report_repro_corpus.py.

Usage:
    python scripts/test_report_repro_corpus.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report_repro_corpus


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _write_repro(
    repo_root: Path,
    name: str,
    pattern_hash: str,
    source: str,
) -> Path:
    repro_dir = repo_root / "repros" / "canonical" / name
    _write(repro_dir / "repro.py", source)
    _write(repro_dir / "meta.json", f'{{"pattern_hash": "{pattern_hash}"}}\n')
    return repro_dir


class ReportReproCorpusTests(unittest.TestCase):
    def test_fake_corpus_counts_versions_manifests_and_full_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write(repo / "repro_harness.py", "CURRENT_REPRO_VERSION = 2\n")
            _write_repro(
                repo,
                "pointwise_aaaaaaaaaaaa",
                "aaaaaaaaaaaa",
                '_repro_version = 2\n_shapes_config = "x"\n',
            )
            _write_repro(
                repo,
                "sum_bbbbbbbbbbbb",
                "bbbbbbbbbbbb",
                '_shapes_config = "x"\n',
            )
            _write_repro(
                repo,
                "var_mean_cccccccccccc",
                "cccccccccccc",
                "_repro_version = 2\n",
            )
            _write_repro(
                repo,
                "pointwise_eeeeeeeeeeee",
                "eeeeeeeeeeee",
                '_repro_version = 1\n_shapes_config = "x"\n',
            )

            _write(
                repo / "repros" / "models" / "hf" / "infer" / "ModelA" / "manifest.json",
                '{"patterns": ["aaaaaaaaaaaa", "bbbbbbbbbbbb", "dddddddddddd"]}\n',
            )
            _write(
                repo / "repros" / "models" / "hf" / "infer" / "ModelA" / "full_graph_000.py",
                "# recaptured graph\n",
            )
            _write(
                repo / "repros" / "models" / "hf" / "train" / "ModelB" / "manifest.json",
                '{"patterns": ["bbbbbbbbbbbb"]}\n',
            )
            _write(
                repo / "repros" / "models" / "timm" / "infer" / "timm_fake_infer.json",
                '{"patterns": ["cccccccccccc"]}\n',
            )

            report = report_repro_corpus.build_report(
                repo_root=repo,
                current_version=2,
                max_examples=10,
            )

            canonical = report["canonical"]
            self.assertEqual(canonical["total_repros"], 4)
            self.assertEqual(canonical["current_version"]["count"], 2)
            self.assertEqual(canonical["unversioned"]["count"], 1)
            self.assertEqual(canonical["old_versioned"]["count"], 1)
            self.assertEqual(canonical["missing_shapes_config"]["count"], 1)

            manifests = report["model_manifest_references"]
            self.assertEqual(manifests["manifest_files"], 3)
            self.assertEqual(manifests["directory_manifest_files"], 2)
            self.assertEqual(manifests["flat_manifest_files"], 1)
            self.assertEqual(manifests["total_pattern_references"], 5)
            self.assertEqual(manifests["unique_pattern_references"], 4)
            self.assertEqual(
                manifests["references_missing_canonical"]["examples"],
                ["dddddddddddd"],
            )

            full_graph = report["full_graph_availability"]
            self.assertEqual(full_graph["model_dirs"], 2)
            self.assertEqual(full_graph["with_full_graph"], 1)
            self.assertEqual(full_graph["without_full_graph"], 1)
            rows = {
                (row["suite"], row["mode"]): row
                for row in full_graph["by_suite_mode"]
            }
            self.assertEqual(rows[("hf", "infer")]["with_full_graph"], 1)
            self.assertEqual(rows[("hf", "train")]["without_full_graph"], 1)

            followup = report["followup"]
            self.assertEqual(
                followup["old_or_incomplete_canonical_repros"]["count"],
                3,
            )
            self.assertEqual(
                followup["referenced_old_or_incomplete_canonical_repros"]["count"],
                2,
            )
            self.assertEqual(
                followup["unreferenced_old_or_incomplete_canonical_repros"]["count"],
                1,
            )

    def test_render_report_includes_safe_read_only_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _write(repo / "repro_harness.py", "CURRENT_REPRO_VERSION = 2\n")
            _write_repro(
                repo,
                "pointwise_aaaaaaaaaaaa",
                "aaaaaaaaaaaa",
                '_repro_version = 2\n_shapes_config = "x"\n',
            )
            _write(
                repo / "repros" / "models" / "genai" / "Kernel" / "manifest.json",
                '{"patterns": ["aaaaaaaaaaaa"]}\n',
            )

            report = report_repro_corpus.build_report(repo_root=repo, max_examples=3)
            rendered = report_repro_corpus.render_report(report)

            self.assertIn("current-version repros (v2): 1", rendered)
            self.assertIn("Full graph availability", rendered)
            self.assertIn("genai/-", rendered)
            self.assertIn("Safe deletion/recapture sequence", rendered)
            self.assertIn("No deletion or recapture is performed", rendered)

    def test_shapes_config_detection_ignores_comments(self):
        self.assertFalse(
            report_repro_corpus.has_top_level_assignment(
                "# _shapes_config = 'x'\nclass Repro: pass\n",
                "_shapes_config",
            )
        )
        self.assertTrue(
            report_repro_corpus.has_top_level_assignment(
                "_shapes_config = 'x'\n",
                "_shapes_config",
            )
        )


if __name__ == "__main__":
    unittest.main()
