import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.build_model_accounting import validate_occurrence_manifest
from scripts.generate_occurrence_sidecars import (
    _cache_key,
    _metadata_validation_error,
    build_sidecar,
    discover_models,
    generate,
    load_cache,
    sidecar_filename,
)


def _model(root: Path, suite: str, mode: str, model: str) -> Path:
    directory = root / "models" / suite / mode / model
    directory.mkdir(parents=True)
    (directory / "full_graph_000.py").write_text("# graph\n")
    return directory


def test_discovery_preserves_nested_models_and_genai_mode(tmp_path):
    _model(tmp_path, "hf", "infer", "org/model")
    _model(tmp_path, "genai", "static", "SoftmaxForward")

    specs = discover_models(tmp_path)

    assert [spec["identity"] for spec in specs] == [
        "genai/static/SoftmaxForward",
        "hf/infer/org/model",
    ]


def test_sidecar_filenames_are_stable_and_collision_resistant():
    left = sidecar_filename("hf/infer/org/model")
    right = sidecar_filename("hf/infer/org__model")

    assert left == sidecar_filename("hf/infer/org/model")
    assert left != right
    assert left.endswith(".json")


def test_cache_rejects_different_environment(tmp_path):
    path = tmp_path / "cache.json"
    cache = load_cache(path, {"device_kind": "B200", "torch": "a"})
    path.write_text(json.dumps(cache))

    with pytest.raises(ValueError, match="different environment"):
        load_cache(path, {"device_kind": "B200", "torch": "b"})


def test_failed_roundtrip_metadata_is_preserved_as_trace_error(tmp_path):
    graph = tmp_path / "models" / "torchbench" / "infer" / "model" / "full_graph_000.py"
    graph.parent.mkdir(parents=True)
    graph.write_text("# graph\n")
    graph.with_suffix(".meta.json").write_text(
        json.dumps({"roundtrip": "failed: stride mismatch"})
    )

    assert _metadata_validation_error(graph, tmp_path) == {
        "graph": "models/torchbench/infer/model/full_graph_000.py",
        "error": "failed: stride mismatch",
    }


def test_build_sidecar_preserves_counts_and_unpriced_extern():
    identity = "hf/infer/model"
    spec = {"suite": "hf", "mode": "infer", "model": "model"}
    signature = "((T([4],torch.float32,[1])),{})"
    key = _cache_key("aten.mm.default", signature)
    inventory = {
        "graphs": [Path("full_graph_000.py")],
        "fusible": Counter({("pattern", "shape"): 3}),
        "extern": Counter({("aten.mm.default", signature): 2}),
        "trace_errors": [],
        "source_digest": "source",
    }
    cache = {
        "entries": {
            key: {
                "target": "aten.mm.default",
                "signature": signature,
                "us": None,
                "error": "failed",
            }
        }
    }

    sidecar = build_sidecar(spec, inventory, cache)

    assert sidecar["fusible"] == {"pattern": {"shape": 3}}
    assert sidecar["extern"] == [
        {
            "target": "aten.mm.default",
            "signature": signature,
            "signature_hash": key[:12],
            "count": 2,
            "baseline_us": None,
            "benchmark_error": "failed",
        }
    ]
    assert identity == f"{sidecar['suite']}/{sidecar['mode']}/{sidecar['model']}"


def test_generate_writes_complete_manifest_and_resumes(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "repros"
    directory = _model(corpus, "hf", "infer", "model")
    spec = {
        "identity": "hf/infer/model",
        "suite": "hf",
        "mode": "infer",
        "model": "model",
        "directory": directory,
    }
    inventory = {
        "graphs": [directory / "full_graph_000.py"],
        "fusible": Counter({("pattern", "shape"): 1}),
        "extern": Counter(),
        "handles": {},
        "trace_errors": [],
        "source_digest": "source",
    }
    monkeypatch.setattr(
        "scripts.generate_occurrence_sidecars.inventory_model",
        lambda *_args: inventory,
    )
    monkeypatch.setattr(
        "scripts.generate_occurrence_sidecars.price_externs",
        lambda *_args, **_kwargs: None,
    )
    output = tmp_path / "occurrences"
    cache = tmp_path / "extern-cache.json"
    fingerprint = {"device_kind": "B200"}

    first = generate(
        [spec],
        corpus_root=corpus,
        output_dir=output,
        cache_path=cache,
        fingerprint=fingerprint,
        resume=False,
        retry_failures=False,
    )
    second = generate(
        [spec],
        corpus_root=corpus,
        output_dir=output,
        cache_path=cache,
        fingerprint=fingerprint,
        resume=True,
        retry_failures=False,
    )

    assert first["status"] == "complete"
    assert second["status"] == "complete"
    manifest = json.loads((output / "_metadata.json").read_text())
    assert manifest["status"] == "complete"
    assert set(manifest["sidecar_digests"]) == {"hf/infer/model"}
    sidecar_path = output / manifest["expected_sidecars"]["hf/infer/model"]
    assert json.loads(sidecar_path.read_text())["source_digest"] == "source"
    validated, manifest_digest = validate_occurrence_manifest(output)
    assert validated["hardware_key"] == "B200"
    assert len(manifest_digest) == 64

    sidecar_path.write_text("{}")
    with pytest.raises(ValueError, match="does not match manifest digest"):
        validate_occurrence_manifest(output)


def test_generate_requires_resume_for_existing_sidecars(tmp_path):
    output = tmp_path / "occurrences"
    output.mkdir()
    (output / "old.json").write_text("{}")

    with pytest.raises(ValueError, match="pass --resume"):
        generate(
            [],
            corpus_root=tmp_path,
            output_dir=output,
            cache_path=tmp_path / "cache.json",
            fingerprint={"device_kind": "B200"},
            resume=False,
            retry_failures=False,
        )
