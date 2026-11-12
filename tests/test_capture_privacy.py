"""install_capture_hook(strip_source_locations=True) writes graphs and repros
without the model's source file paths, source lines and user annotations."""

import glob
import json
import os
import sys
from pathlib import Path

import torch
import torch._dynamo
import torch.fx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture_hook import _print_readable, install_capture_hook, uninstall_capture_hook


def test_print_readable_can_strip_source_locations():
    gm = torch.fx.symbolic_trace(lambda x: x + 1)
    node = next(node for node in gm.graph.nodes if node.op == "call_function")
    trace = '  File "/src/secret_model.py", line 12, in forward\n    y = x + 1\n'
    node.meta["stack_trace"] = trace
    node.meta["custom"] = {"owner": "secret_annotation"}

    unstripped = _print_readable(gm, False)
    assert "secret_model.py" in unstripped
    stripped = _print_readable(gm, True)
    assert "secret_model.py" not in stripped
    assert "secret_annotation" not in stripped
    assert node.meta["stack_trace"] == trace
    assert node.meta["custom"] == {"owner": "secret_annotation"}


def _model(x, w):
    y = (x @ w).relu()
    return y.sum(-1) + y.amax(-1)


def _capture(tmp_path, name, strip):
    torch._dynamo.reset()
    output_dir, graph_dir = tmp_path / name, tmp_path / f"{name}_graphs"
    install_capture_hook(str(output_dir), label=name, validate=False,
                         graph_dir=str(graph_dir), strip_source_locations=strip)
    try:
        torch.compile(_model, backend="inductor")(torch.randn(8, 16), torch.randn(16, 32))
    finally:
        uninstall_capture_hook()
    index = json.loads((output_dir / "index.json").read_text())
    repros = [Path(p).read_text() for p in sorted(glob.glob(str(output_dir / "*.py")))]
    graphs = [Path(p).read_text() for p in sorted(glob.glob(str(graph_dir / "*.py")))]
    metas = [json.loads(Path(p).read_text())
             for p in sorted(glob.glob(str(graph_dir / "*.meta.json")))]
    return index, repros, graphs, metas


def test_capture_strips_source_locations_from_graphs_and_repros(tmp_path):
    this_file = os.path.basename(__file__)

    index, repros, graphs, metas = _capture(tmp_path, "plain", strip=False)
    assert repros and graphs
    assert index["source_locations_stripped"] is False
    assert all(meta["source_locations_stripped"] is False for meta in metas)
    assert any(this_file in text for text in repros + graphs)

    s_index, s_repros, s_graphs, s_metas = _capture(tmp_path, "stripped", strip=True)
    assert s_index["source_locations_stripped"] is True
    assert s_metas and all(meta["source_locations_stripped"] is True for meta in s_metas)
    for text in s_repros + s_graphs:
        assert this_file not in text
        assert "# File:" not in text

    # Stripping changes only comments: the same regions are captured.
    assert len(s_repros) == len(repros) and len(s_graphs) == len(graphs)
    assert ([entry["pattern_hash"] for entry in s_index["captured"]]
            == [entry["pattern_hash"] for entry in index["captured"]])
