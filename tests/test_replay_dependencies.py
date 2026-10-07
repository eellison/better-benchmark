import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import torch
import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch.fx.experimental.proxy_tensor import make_fx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import replay_dependencies as rd
from bench_parallel import _preflight_full_graphs
from full_graph_harness import load_full_graph_sidecar, write_full_graph_metadata
from replay_dependencies import check_replay_dependencies, op_available, replay_dependencies


@torch.library.custom_op("bbtest::scale", mutates_args=())
def _scale(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@torch.library.custom_op("bbtest::bump", mutates_args=("x",))
def _bump(x: torch.Tensor) -> None:
    x.add_(1)


_LIB = torch.library.Library("bbtest_lib", "DEF")
_LIB.define("double(Tensor x) -> Tensor")
_LIB.impl("double", lambda x: x * 2, "CompositeExplicitAutograd")
_LIB.define("double.Tensor(Tensor x, Tensor y) -> Tensor")
_LIB.impl("double.Tensor", lambda x, y: x * 2 + y, "CompositeExplicitAutograd")


def _ops(deps):
    return {entry["op"]: entry["provider"] for entry in deps.get("custom_ops", [])}


def test_records_custom_op_called_directly_with_python_provider():
    gm = make_fx(lambda x: torch.ops.bbtest.scale(x).sin())(torch.randn(4))

    deps = replay_dependencies(gm)

    assert deps["schema_version"] == 1
    assert _ops(deps) == {"bbtest::scale.default": {"python_module": __name__}}
    assert "distributed" not in deps


def test_records_custom_op_passed_to_higher_order_op():
    graph = fx.Graph()
    x = graph.placeholder("x")
    out = graph.call_function(
        auto_functionalized, (torch.ops.bbtest.bump.default,), {"x": x}
    )
    graph.output(out)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    assert set(_ops(replay_dependencies(gm))) == {"bbtest::bump.default"}


def test_records_library_registered_op_by_registering_module():
    gm = make_fx(lambda x: torch.ops.bbtest_lib.double(x))(torch.randn(4))

    assert _ops(replay_dependencies(gm)) == {
        "bbtest_lib::double.default": {"python_module": __name__}
    }


def test_stock_graph_records_nothing_and_is_ready():
    gm = make_fx(lambda x: (x.relu() + 1).sum())(torch.randn(4))

    deps = replay_dependencies(gm)

    assert deps == {"schema_version": 1}
    assert check_replay_dependencies(deps) == {"status": "ready", "blocked": []}


def test_library_names_are_basenames_only(monkeypatch):
    monkeypatch.setattr(rd, "_python_provider", lambda name: None)
    monkeypatch.setattr(rd, "_registered_from_python", lambda name: False)
    monkeypatch.setattr(
        torch.ops, "loaded_libraries", {os.path.join("some", "dir", "libbbtest_ops.so")}
    )
    gm = make_fx(lambda x: torch.ops.bbtest.scale(x))(torch.randn(4))

    provider = _ops(replay_dependencies(gm))["bbtest::scale.default"]

    assert provider == {"python_module": None, "libraries": ["libbbtest_ops.so"]}


def test_missing_op_is_blocked_with_provider_hint():
    deps = {
        "schema_version": 1,
        "custom_ops": [{
            "op": "bbtest_missing::nope.default",
            "provider": {"python_module": "bbtest_missing_pkg.ops"},
        }],
    }

    result = check_replay_dependencies(deps, import_providers=True)

    assert result["status"] == "blocked"
    [item] = result["blocked"]
    assert item["category"] == "missing_custom_op"
    assert item["item"] == "bbtest_missing::nope.default"
    assert "import bbtest_missing_pkg.ops" in item["hint"]


def test_registered_op_is_ready():
    deps = {
        "schema_version": 1,
        "custom_ops": [{"op": "bbtest::scale.default", "provider": {"python_module": None}}],
    }

    assert check_replay_dependencies(deps)["status"] == "ready"
    assert op_available("bbtest::scale")
    assert not op_available("bbtest::scale.no_such_overload")


def test_import_providers_registers_missing_op(tmp_path, monkeypatch):
    (tmp_path / "bbtest_late_provider.py").write_text(textwrap.dedent("""
        import torch

        @torch.library.custom_op("bbtest_late::op", mutates_args=())
        def op(x: torch.Tensor) -> torch.Tensor:
            return x.clone()
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    deps = {
        "schema_version": 1,
        "custom_ops": [{
            "op": "bbtest_late::op.default",
            "provider": {"python_module": "bbtest_late_provider"},
        }],
    }

    assert check_replay_dependencies(deps)["status"] == "blocked"
    assert check_replay_dependencies(deps, import_providers=True)["status"] == "ready"


def test_capture_without_block_is_unknown():
    assert check_replay_dependencies(None)["status"] == "unknown"
    assert check_replay_dependencies({"schema_version": 2, "inputs": []})["status"] == "unknown"


def test_collectives_record_group_size_and_block_without_process_group():
    graph = fx.Graph()
    x = graph.placeholder("x")
    gathered = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default, (x, 4, "0")
    )
    waited = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (gathered,))
    graph.output(waited)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    deps = replay_dependencies(gm)

    assert deps["distributed"] == {
        "collectives": ["_c10d_functional::all_gather_into_tensor.default"],
        "group_size": 4,
        "groups": [{"name": "0", "size": 4}],
    }
    assert "custom_ops" not in deps
    result = check_replay_dependencies(deps)
    assert result["status"] == "blocked"
    assert result["blocked"][0]["category"] == "distributed"
    assert "process group of size 4" in result["blocked"][0]["hint"]


def test_torch_namespaces_and_null_block_are_not_custom():
    graph = fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_function(torch.ops.c10d.allreduce_.default, ([x],)))

    deps = replay_dependencies(fx.GraphModule(torch.nn.Module(), graph))

    assert "custom_ops" not in deps
    assert deps["distributed"]["collectives"] == ["c10d::allreduce_.default"]
    assert check_replay_dependencies({"replay_dependencies": None})["status"] == "unknown"


def test_named_overload_is_recorded_once_and_found():
    gm = make_fx(lambda x: torch.ops.bbtest_lib.double.Tensor(x, x))(torch.randn(4))

    deps = replay_dependencies(gm)

    assert list(_ops(deps)) == ["bbtest_lib::double.Tensor"]
    assert check_replay_dependencies(deps)["status"] == "ready"


def test_collective_on_a_group_this_process_lacks_is_blocked(tmp_path):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        subgroup = dist.new_group([0])
        name = subgroup.group_name
        deps = {
            "schema_version": 1,
            "distributed": {
                "collectives": ["_c10d_functional::all_reduce.default"],
                "group_size": 1,
                "groups": [{"name": name, "size": 1}],
            },
        }
        assert check_replay_dependencies(deps)["status"] == "ready"
        dist.destroy_process_group(subgroup)
        result = check_replay_dependencies(deps)
        assert result["status"] == "blocked"
        assert name in result["blocked"][0]["reason"]
    finally:
        dist.destroy_process_group()


def test_sidecar_round_trip_records_dependencies():
    gm = make_fx(lambda x: torch.ops.bbtest.scale(x))(torch.randn(4))
    with tempfile.TemporaryDirectory() as tmp:
        graph_path = Path(tmp) / "full_graph_000.py"
        graph_path.write_text("")
        write_full_graph_metadata(
            graph_path, gm, extra={"replay_dependencies": replay_dependencies(gm)}
        )
        sidecar = load_full_graph_sidecar(graph_path)

    assert set(_ops(sidecar["replay_dependencies"])) == {"bbtest::scale.default"}
    assert check_replay_dependencies(sidecar)["status"] == "ready"


_GRAPH_SOURCE = '''
import torch

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[4]cpu"):
        return (torch.ops.{op}(x),)
'''


def _write_graph(tmp: Path, op: str, provider: dict) -> Path:
    graph = tmp / "full_graph_000.py"
    graph.write_text(_GRAPH_SOURCE.format(op=op))
    graph.with_suffix(".meta.json").write_text(json.dumps({
        "schema_version": 1,
        "replay_dependencies": {
            "schema_version": 1,
            "custom_ops": [{"op": f"{op.replace('.', '::', 1)}.default", "provider": provider}],
        },
    }))
    return graph


def test_preflight_skips_graph_with_missing_op():
    with tempfile.TemporaryDirectory() as tmp:
        graph = _write_graph(
            Path(tmp), "bbtest_absent.op", {"python_module": "bbtest_absent_pkg"}
        )

        queueable, failures = _preflight_full_graphs([graph])

    assert queueable == []
    assert failures[str(graph)]["status"] == "skipped"
    assert failures[str(graph)]["category"] == "missing_custom_op"
    assert "import bbtest_absent_pkg" in failures[str(graph)]["hint"]


def test_preflight_with_worker_init_queues_graph_with_missing_op(tmp_path):
    graph = _write_graph(tmp_path, "bbtest_winit.op", {"python_module": None})

    queueable, failures = _preflight_full_graphs(
        [graph], worker_init=["bbtest_worker_init_ops:register"]
    )

    assert failures == {}
    assert queueable == [graph]
    assert not op_available("bbtest_winit::op.default")
