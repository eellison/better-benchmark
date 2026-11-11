"""A captured graph that launches user-defined Triton kernels must never be
replayed through another process's kernel_side_table.

The saved graph refers to each kernel by its index into the capturing
process's table. A process that has compiled some other model with user
kernels holds different kernels at those indices; replayed there, the graph
runs them and returns a different result without an error.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="user-defined Triton kernels need CUDA"
)

_CAPTURE = textwrap.dedent('''
    import sys
    sys.path[:0] = [{root!r}, {scripts!r}]
    import torch
    import triton
    import triton.language as tl
    from capture_hook import install_capture_hook, uninstall_capture_hook

    @triton.jit
    def bbtest_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

    def model(x):
        y = x * 2.0
        out = torch.empty_like(y)
        bbtest_add_kernel[(triton.cdiv(y.numel(), 256),)](y, y, out, y.numel(), BLOCK=256)
        return out.relu() + 1.0

    x = torch.randn(4096, device="cuda")
    install_capture_hook({capture!r}, label="user_triton", graph_dir={graphs!r})
    torch.compile(model)(x)
    uninstall_capture_hook()
    torch.save({{"x": x.cpu(), "out": model(x).cpu()}}, {reference!r})
''')

_REPLAY_AFTER_OTHER_MODEL = textwrap.dedent('''
    import json, sys
    sys.path[:0] = [{root!r}, {scripts!r}]
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def bbtest_other_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) * 100.0, mask=m)

    def other_model(a):
        b = torch.empty_like(a)
        bbtest_other_kernel[(triton.cdiv(a.numel(), 256),)](a, a, b, a.numel(), BLOCK=256)
        return b

    torch.compile(other_model)(torch.randn(4096, device="cuda"))

    from bench_parallel import _preflight_full_graphs
    from full_graph_harness import UserTritonKernelReplayError, load_full_graph

    graph = {graph!r}
    queueable, failures = _preflight_full_graphs([graph])
    try:
        load_full_graph(graph, default_device="cuda")
        refused = False
    except UserTritonKernelReplayError:
        refused = True

    # What the guards prevent: bypassing them, this process runs its own
    # kernel at the captured index and silently returns a different result.
    from full_graph_harness import instantiate_full_graph, load_full_graph_definition

    reference = torch.load({reference!r})
    instance = instantiate_full_graph(load_full_graph_definition(graph))
    with torch.no_grad():
        (bypassed,) = instance(reference["x"].cuda())
    print(json.dumps({{
        "queued": [str(p) for p in queueable],
        "category": failures.get(str(graph), {{}}).get("category"),
        "masked": failures.get(str(graph), {{}}).get("masked"),
        "refused": refused,
        "bypass_matches_capture": bool(torch.equal(bypassed.cpu(), reference["out"])),
    }}))
''')

# A fresh process (empty kernel_side_table) can still repartition the saved
# graph, so masking keeps every partition of a user-Triton graph.
_REPARTITION_FRESH = textwrap.dedent('''
    import json, sys
    sys.path[:0] = [{root!r}, {scripts!r}]
    from pathlib import Path
    from capture_hook import _CaptureState
    from repartition_from_graphs import load_graph_module

    gm = load_graph_module(Path({graph!r}))
    state = _CaptureState({out!r}, label="repartition", graph_dir={out_graphs!r})
    state.process_graph(gm)
    print(json.dumps({{
        "regions": sorted(sorted(c["origin_ops"]) for c in state.captured),
        "dropped": state.dropped,
    }}))
''')


def _run(source: str, tmp_path: Path) -> str:
    script = tmp_path / "script.py"
    script.write_text(source)
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout


def test_user_triton_graph_is_masked_and_never_replayed(tmp_path):
    paths = {
        "root": str(ROOT),
        "scripts": str(ROOT / "scripts"),
        "capture": str(tmp_path / "capture"),
        "graphs": str(tmp_path / "graphs"),
        "reference": str(tmp_path / "reference.pt"),
    }
    _run(_CAPTURE.format(**paths), tmp_path)
    graph = tmp_path / "graphs" / "full_graph_000.py"
    sidecar = json.loads(graph.with_suffix(".meta.json").read_text())
    assert sidecar["replay_dependencies"]["user_triton_kernels"] == [
        {"kernel_idx": 0, "kernel": "bbtest_add_kernel", "calls": 1}
    ]
    mask = {"reasons": ["user_triton"], "scope": "graph"}
    assert sidecar["masked"] == mask
    assert sidecar["node_accounting"]["ops"]["masked"] == {
        "torch.ops.higher_order.triton_kernel_wrapper_functional": {
            "count": 1,
            "reasons": ["user_triton"],
        }
    }
    index = json.loads((tmp_path / "capture" / "index.json").read_text())
    regions = sorted(sorted(entry["origin_ops"]) for entry in index["captured"])
    assert regions == [["aten.add.Tensor", "aten.relu.default"], ["aten.mul.Tensor"]]

    out = _run(_REPLAY_AFTER_OTHER_MODEL.format(graph=str(graph), **paths), tmp_path)
    verdict = json.loads(out.strip().splitlines()[-1])

    assert verdict == {
        "queued": [],
        "category": "masked",
        "masked": mask,
        "refused": True,
        "bypass_matches_capture": False,
    }

    out = _run(
        _REPARTITION_FRESH.format(
            graph=str(graph),
            out=str(tmp_path / "repartition"),
            out_graphs=str(tmp_path / "repartition_graphs"),
            **paths,
        ),
        tmp_path,
    )
    repartitioned = json.loads(out.strip().splitlines()[-1])
    assert repartitioned == {"regions": regions, "dropped": []}
    resaved = tmp_path / "repartition_graphs" / "full_graph_000.meta.json"
    assert json.loads(resaved.read_text())["masked"] == mask
