"""corpus_relative_path: one identity for a workload across checkouts."""
from scripts.benchmark_provenance import corpus_relative_path


def test_in_repo_and_external_corpora_share_identity():
    assert corpus_relative_path(
        "/home/a/better-benchmark/repros/canonical/sum_abc/repro.py"
    ) == "canonical/sum_abc/repro.py"
    assert corpus_relative_path(
        "/scratch/bundle/canonical/sum_abc/repro.py"
    ) == "canonical/sum_abc/repro.py"
    assert corpus_relative_path(
        "repros/models/hf/infer/Bert/full_graph_000.py"
    ) == corpus_relative_path("/mnt/corpus/models/hf/infer/Bert/full_graph_000.py")


def test_checkout_under_a_models_directory_anchors_at_repros():
    assert corpus_relative_path(
        "/mnt/models/better-benchmark/repros/models/hf/infer/Bert/full_graph_000.py"
    ) == "models/hf/infer/Bert/full_graph_000.py"
    assert corpus_relative_path(
        "/mnt/models/better-benchmark/repros/canonical/sum_abc/repro.py"
    ) == "canonical/sum_abc/repro.py"


def test_paths_outside_a_corpus_have_no_identity():
    assert corpus_relative_path("/tmp/scratch/repro.py") is None
