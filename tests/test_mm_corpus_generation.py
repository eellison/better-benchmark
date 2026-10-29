from __future__ import annotations

import json

from scripts import extract_matmul_corpus


def test_public_generator_exports_core_only_and_replaces_stale_patterns(
    tmp_path, monkeypatch
):
    """Exercise actual capture, filtering, dedup and atomic publication."""
    source = tmp_path / "repros/models/test/infer/demo/full_graph_000.py"
    source.parent.mkdir(parents=True)
    source.write_text("""import torch
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[4, 8]", w: "f32[8, 8]", v: "f32[8, 8]"):
        mm = torch.ops.aten.mm.default(x, w)
        relu = torch.ops.aten.relu.default(mm)
        padded = torch.ops.aten.constant_pad_nd.default(v, [0, 2])
        mm_1 = torch.ops.aten.mm.default(x, padded)
        cropped = torch.ops.aten.slice.Tensor(mm_1, 1, 0, -2)
        return relu, cropped
""")
    output = tmp_path / "mm_repro"
    stale = output / "canonical/stale"
    stale.mkdir(parents=True)
    (stale / "repro.py").write_text("stale = True\n")
    (output / "report.json").write_text(json.dumps({"unique_repros": 1}))
    monkeypatch.setattr(extract_matmul_corpus, "ROOT", tmp_path)
    result = extract_matmul_corpus.generate_corpus(output, [source], workers=1)
    assert result["unique_repros"] == 1 and result["shape_points"] == 1
    assert not stale.exists()
    repros = list((output / "canonical").glob("*/repro.py"))
    assert len(repros) == 1 and "relu" in repros[0].read_text()
    assert all("control" not in p.name for p in tmp_path.iterdir())
    assert not list(tmp_path.glob(".mm_repro_build_*"))
    rows = json.loads((output / "occurrences.json").read_text())
    assert sum(len(r["regions"]) for r in rows) == 1
    assert rows[0]["source"] == "repros/models/test/infer/demo/full_graph_000.py"
