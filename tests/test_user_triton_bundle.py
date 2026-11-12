"""A full graph whose sidecar carries a user-Triton kernel bundle replays its
kernels through the resolver registered in the replaying process, and never
through the kernel_side_table: not at the captured index, and not after
torch._dynamo.reset() has emptied the table."""
import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from user_triton import ResolvedBundle, rewrite_hop_calls  # noqa: E402

_GRAPH = textwrap.dedent('''
    class Repro(torch.nn.Module):
        def forward(self, x):
            out = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 3, constant_args_idx = 5, grid = [(16, 1, 1), (16, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'x_ptr': x}, tensors_to_clone = ['x_ptr'])
            return (out['x_ptr'],)
''')


def _graph_class(source: str = _GRAPH) -> ast.ClassDef:
    return ast.parse(source).body[0]


def test_rewrite_needs_every_kernel_and_constant_args():
    for bundle in (ResolvedBundle({}, {5: {}}), ResolvedBundle({3: object()}, {})):
        cls = _graph_class()
        assert not rewrite_hop_calls(cls, bundle)
        assert "triton_kernel_wrapper_functional" in ast.unparse(cls)

    cls = _graph_class()
    assert rewrite_hop_calls(cls, ResolvedBundle({3: object()}, {5: {}}))
    text = ast.unparse(cls)
    assert "triton_kernel_wrapper" not in text
    assert "launch_user_triton(kernel_idx=3, constant_args_idx=5, grid=[(16, 1, 1)]" in text


class _TwoConfigs:
    configs = (object(), object())


def test_rewrite_keeps_per_config_grids_only_when_they_match_the_configs():
    per_config = _GRAPH.replace("[(16, 1, 1), (16, 1, 1)]", "[(16, 1, 1), (8, 1, 1)]")
    cls = _graph_class(per_config)
    assert not rewrite_hop_calls(cls, ResolvedBundle({3: object()}, {5: {}}))
    cls = _graph_class(per_config)
    assert rewrite_hop_calls(cls, ResolvedBundle({3: _TwoConfigs()}, {5: {}}))
    assert "grid=[(16, 1, 1), (8, 1, 1)]" in ast.unparse(cls)
    mutation = _GRAPH.replace("triton_kernel_wrapper_functional", "triton_kernel_wrapper_mutation")
    assert not rewrite_hop_calls(_graph_class(mutation), ResolvedBundle({3: object()}, {5: {}}))


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="user-defined Triton kernels need CUDA"
)
@pytest.mark.parametrize("chosen", [0, 1])
def test_launcher_uses_the_grid_of_the_launched_config(chosen):
    import triton
    import triton.language as tl
    from user_triton import make_launcher

    @triton.autotune(
        configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})],
        key=[],
        prune_configs_by={"early_config_prune": lambda configs, nargs, **kw: [configs[chosen]]},
    )
    @triton.jit
    def count_kernel(x_ptr, count_ptr, BLOCK: tl.constexpr):
        tl.atomic_add(count_ptr, 1)

    n = 4096
    grid = [(n // 64, 1, 1), (n // 128, 1, 1)]
    launch = make_launcher(ResolvedBundle({0: count_kernel}, {0: {}}))

    def run(x):
        count = torch.zeros(1, dtype=torch.int32, device=x.device)
        out = launch(kernel_idx=0, constant_args_idx=0, grid=grid, tma_descriptor_metadata={},
                     kwargs={"x_ptr": x, "count_ptr": count}, tensors_to_clone=["count_ptr"])
        return out["count_ptr"]

    x = torch.randn(n, device="cuda")
    assert run(x).item() == grid[chosen][0]
    torch._dynamo.reset()
    assert torch.compile(run, fullgraph=True)(x).item() == grid[chosen][0]


def test_rewrite_refuses_calls_the_launcher_cannot_replay():
    bundle = ResolvedBundle({3: object()}, {5: {}})
    host_tma = _GRAPH.replace("tma_descriptor_metadata = {}", "tma_descriptor_metadata = {'x_ptr': ('stable', ([16], [4]))}")
    tensor_bases = _GRAPH.replace("tensors_to_clone = ['x_ptr']", "tensors_to_clone = ['x_ptr'], tensor_bases = {'x_ptr': x}")
    spoofed = _GRAPH.replace("return (out['x_ptr'],)", "launch_user_triton(kernel_idx = 3)\n        return (out['x_ptr'],)")
    for source in (host_tma, tensor_bases, spoofed):
        cls = _graph_class(source)
        assert not rewrite_hop_calls(cls, bundle)
        assert "triton_kernel_wrapper_functional" in ast.unparse(cls)


def test_resolved_kernels_are_cached_per_graph_never_across_graphs(monkeypatch):
    import types

    from user_triton import RESOLVER_ENV, resolve_bundle

    # A resolved autotuner caches its tuned config by its key, which can miss
    # shapes; graphs with identical bundle entries must not share one.
    resolver = types.ModuleType("bbtest_resolver")
    resolver.resolve = lambda entry: object()
    monkeypatch.setitem(sys.modules, "bbtest_resolver", resolver)
    monkeypatch.setenv(RESOLVER_ENV, "bbtest_resolver:resolve")
    sidecar = {
        "user_triton_bundle": {
            "schema_version": 1,
            "kernels": {"3": {"name": "k", "kid": "x", "configs": []}},
            "constant_args": {"5": {}},
        }
    }

    def kernel(scope):
        return resolve_bundle(sidecar, scope=scope).kernels[3]

    assert kernel("/g/recompile_0.py") is kernel("/g/recompile_0.py")
    assert kernel("/g/recompile_0.py") is not kernel("/g/recompile_1.py")
    assert kernel(None) is not kernel(None)


_UPCAST_FAILS = [False]
_EAGER_RUNS = [0]


def launch_user_triton(x, unstable=False):
    # Stands in for the launcher a rewritten graph calls; the gate keys on the
    # name. Rounds to bf16 after every step, like a kernel that stores bf16.
    if x.dtype == torch.float64 or (x.dtype == torch.float32 and _UPCAST_FAILS[0]):
        raise RuntimeError(f"this kernel does not run in {x.dtype}")
    _EAGER_RUNS[0] += 1
    for _ in range(20):
        x = x * 1.01
    return x + (_EAGER_RUNS[0] if unstable else 0)


class _UserKernelGraph(torch.nn.Module):
    def forward(self, x):
        return (launch_user_triton(x),)


class _UnstableUserKernelGraph(torch.nn.Module):
    def forward(self, x):
        return (launch_user_triton(x, unstable=True),)


def _gate(graph, compiled):
    from full_graph_harness import run_numerics_gate

    def make_inputs():
        return [torch.linspace(1, 8, 64).to(torch.bfloat16)]

    return run_numerics_gate(graph, compiled, make_inputs, label="c", variant="default", seeds=1)


def _exact(x):
    # One rounding: what an fp32-accumulating compiled kernel returns.
    return ((x.float() * 1.01**20).to(torch.bfloat16),)


@pytest.fixture
def upcast_fails():
    _UPCAST_FAILS[0] = True
    yield
    _UPCAST_FAILS[0] = False


def test_user_kernel_graph_is_judged_against_the_float32_reference():
    from full_graph_harness import FullGraphNumericsError

    # Eager's 20 bf16 roundings miss compiled's single rounding by more than
    # bf16 tolerance; the float32 reference (kernel run in float32) is right.
    summary, eager = _gate(_UserKernelGraph(), _exact)
    assert summary["status"] == "pass"
    assert summary["reference"] == "fp32"
    assert summary["label"] == "pass"
    assert eager is not None
    with pytest.raises(FullGraphNumericsError) as info:
        _gate(_UserKernelGraph(), lambda x: (_exact(x)[0] + 0.5,))
    assert info.value.numerics["status"] == "fail"
    assert info.value.numerics["reference"] == "fp32"


class _StatsGraph(torch.nn.Module):
    def forward(self, x):
        # A float32 output computed from bf16 data, like var_mean statistics.
        return (launch_user_triton(x).float().sum(0, keepdim=True),)


def test_float32_output_falls_back_to_the_float32_reference_without_float64():
    # The kernel has no float64 run, so the float32 output is judged against
    # the float32 run, where the bf16 roundings are gone.
    def compiled(x):
        return (_exact(x)[0].float().sum(0, keepdim=True),)

    summary, _ = _gate(_StatsGraph(), compiled)
    assert summary["status"] == "pass"
    assert summary["reference"] == "fp32"


def test_user_kernel_without_float32_run_is_judged_against_eager(upcast_fails):
    from full_graph_harness import FullGraphNumericsError

    summary, _ = _gate(_UserKernelGraph(), lambda x: (launch_user_triton(x),))
    assert summary["status"] == "pass"
    with pytest.raises(FullGraphNumericsError) as info:
        _gate(_UserKernelGraph(), _exact)
    assert info.value.numerics["status"] == "fail"
    assert info.value.numerics["reference"] == "eager"
    # No upcast run and eager does not reproduce itself: no usable reference.
    summary, _ = _gate(_UnstableUserKernelGraph(), _exact)
    assert summary["status"] == "unanchored_user_kernel"
    assert summary["label"] is None


_CAPTURE = textwrap.dedent('''
    import json, sys
    sys.path[:0] = [{root!r}, {scripts!r}]
    import torch
    import triton
    from capture_hook import install_capture_hook, uninstall_capture_hook
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
    from bbtest_kernels import bbtest_add_kernel

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
    print(json.dumps(kernel_side_table.constant_args))
''')

_KERNELS = textwrap.dedent('''
    import triton
    import triton.language as tl

    @triton.jit
    def bbtest_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

    @triton.jit
    def bbtest_other_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) * 100.0, mask=m)

    def resolve(entry):
        from user_triton import UserTritonResolutionError

        if entry["name"] != "bbtest_add_kernel":
            raise UserTritonResolutionError("unknown kernel")
        return bbtest_add_kernel
''')

_REPLAY = textwrap.dedent('''
    import json, os, sys
    sys.path[:0] = [{root!r}, {scripts!r}]
    os.environ["BETTER_BENCHMARK_USER_TRITON_RESOLVER"] = "bbtest_kernels:resolve"
    import torch
    import triton
    from bbtest_kernels import bbtest_other_kernel
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    def poison():
        # Other kernels at the captured indices, as in a worker that already
        # compiled other graphs.
        for _ in range(8):
            kernel_side_table.add_kernel(bbtest_other_kernel)
            kernel_side_table.add_constant_args({{}})

    poison()
    from bench_parallel import _preflight_full_graphs
    from full_graph_harness import load_full_graph

    graph = {graph!r}
    queueable, failures = _preflight_full_graphs([graph])
    print("preflight masked:", json.dumps([f.get("masked") for f in failures.values()]), flush=True)
    instance, inputs, _ = load_full_graph(graph, default_device="cuda")
    reference = torch.load({reference!r})
    x = reference["x"].cuda()
    with torch.no_grad():
        (eager,) = instance(x)
    torch._dynamo.reset()
    poison()
    with torch.no_grad():
        (compiled,) = torch.compile(instance, fullgraph=True)(x)
    print(json.dumps({{
        "queued": [str(p) for p in queueable],
        "eager_matches_capture": bool(torch.equal(eager.cpu(), reference["out"])),
        "compiled_matches_capture": bool(torch.allclose(compiled.cpu(), reference["out"])),
    }}))
''')


def _run(source: str, tmp_path: Path) -> str:
    script = tmp_path / "script.py"
    script.write_text(source)
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=600, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="user-defined Triton kernels need CUDA"
)
def test_bundled_graph_replays_resolved_kernel_after_reset(tmp_path):
    (tmp_path / "bbtest_kernels.py").write_text(_KERNELS)
    paths = {
        "root": str(ROOT),
        "scripts": str(ROOT / "scripts"),
        "capture": str(tmp_path / "capture"),
        "graphs": str(tmp_path / "graphs"),
        "reference": str(tmp_path / "reference.pt"),
    }
    constant_args = json.loads(_run(_CAPTURE.format(**paths), tmp_path).strip().splitlines()[-1])
    graph = tmp_path / "graphs" / "full_graph_000.py"
    meta = graph.with_suffix(".meta.json")
    sidecar = json.loads(meta.read_text())
    # What ingest's resolver-side code writes once every kernel resolves.
    sidecar["user_triton_bundle"] = {
        "schema_version": 1,
        "kernels": {"0": {"name": "bbtest_add_kernel"}},
        "constant_args": {"0": constant_args["0"]},
    }
    del sidecar["masked"]
    del sidecar["replay_dependencies"]["user_triton_kernels"]
    meta.write_text(json.dumps(sidecar))

    out = _run(_REPLAY.format(graph=str(graph), **paths), tmp_path)
    assert json.loads(out.strip().splitlines()[-1]) == {
        "queued": [str(graph)],
        "eager_matches_capture": True,
        "compiled_matches_capture": True,
    }

    # Without a resolvable bundle entry the graph stays refused.
    sidecar["user_triton_bundle"]["kernels"]["0"]["name"] = "bbtest_other_kernel"
    meta.write_text(json.dumps(sidecar))
    script = _REPLAY.format(graph=str(graph), **paths)
    (tmp_path / "script.py").write_text(script)
    result = subprocess.run(
        [sys.executable, str(tmp_path / "script.py")], capture_output=True, text=True,
        timeout=600, cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "UserTritonKernelReplayError" in result.stderr
    assert "kernel bundle did not resolve here" in result.stderr
    assert (
        'preflight masked: [{"reasons": ["user_triton_unresolved"], "scope": "graph"}]'
        in result.stdout
    )
