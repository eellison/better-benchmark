from pathlib import Path

from scripts.validate_corpus_invariants import ROOT, _display_path


def test_display_path_supports_external_corpus(tmp_path: Path):
    external = tmp_path / "repros" / "models" / "all" / "infer" / "model"

    assert _display_path(ROOT / "repros" / "canonical") == "repros/canonical"
    assert _display_path(external) == str(external)
