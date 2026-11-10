from __future__ import annotations

import json
import runpy

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from roundtrip_validation import fake_tensor_ctx
from scripts import extract_matmul_corpus


def test_public_generator_replaces_stale_patterns_and_exports_replayable_core(tmp_path, monkeypatch):
    source = tmp_path / "repros/models/test/infer/demo/full_graph_000.py"
    source.parent.mkdir(parents=True)
    source.write_text("""import torch
class GraphModule(torch.nn.Module):
    def forward(self, x, w, v):
        mm = torch.ops.aten.mm.default(x, w)
        relu = torch.ops.aten.relu.default(mm)
        padded = torch.ops.aten.constant_pad_nd.default(v, [0, 2])
        mm_1 = torch.ops.aten.mm.default(x, padded)
        cropped = torch.ops.aten.slice.Tensor(mm_1, 1, 0, -2)
        return relu, cropped
""")
    specs = [{"kind": "tensor", "name": name, "shape": shape, "dtype": "float32", "device": "cpu"}
             for name, shape in [("x", [4, 8]), ("w", [8, 8]), ("v", [8, 8])]]
    source.with_suffix(".meta.json").write_text(json.dumps({"inputs": specs, "tensor_attrs": {}}))
    output = tmp_path / "mm_repro"
    stale = output / "canonical/stale"
    stale.mkdir(parents=True)
    (stale / "repro.py").write_text("stale = True\n")
    (output / "report.json").write_text(json.dumps({"unique_repros": 1}))
    monkeypatch.setattr(extract_matmul_corpus, "ROOT", tmp_path)
    result = extract_matmul_corpus.generate_corpus(output, [source], workers=1)
    # relu(mm) is an epilogue; constant_pad_nd has no Tag.pointwise but no
    # reduction/scan either, so padded feeding mm_1 is a prologue.
    assert result["unique_repros"] == 2 and result["shape_points"] == 2
    assert not stale.exists()
    repros = {p.parent.parent.name: p for p in (output / "canonical").glob("*/*/repro.py")}
    assert sorted(repros) == ["epilogue", "prologue"]
    epilogue = repros["epilogue"]
    namespace = runpy.run_path(str(epilogue))
    with fake_tensor_ctx():
        inputs = namespace["make_inputs"]()
        assert all(t.device.type == "cpu" for t in inputs)
        gm = make_fx(namespace["Repro"](), tracing_mode="fake")(*inputs)
        assert any(n.target is torch.ops.aten.relu.default for n in gm.graph.nodes)
    points = json.loads(epilogue.with_name("shapes.json").read_text())["points"]
    assert points[0]["models"] == {"test/infer/demo": {"occurrences": 1}}
    assert result["region_occurrences"] == 2
    assert not list(tmp_path.glob(".mm_repro_build_*"))
    assert {p.name for p in output.iterdir()} == {"canonical", "report.json", "README.md"}
    calls = []
    import mm_replay
    monkeypatch.setattr(mm_replay, "benchmark_repro_safe", lambda *args: calls.append(args))
    runpy.run_path(str(epilogue), run_name="__main__")
    assert len(calls) == 1 and calls[0][0] == str(epilogue)
    assert isinstance(calls[0][1], type)
