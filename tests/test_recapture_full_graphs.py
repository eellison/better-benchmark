"""Focused tests for recapturing from saved full_graph artifacts.

Usage:
    python scripts/test_recapture_full_graphs.py
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import recapture_full_graphs as recapture
from full_graph_harness import load_graph_class_from_source
from ingest_tlparse import (
    _evaluate_int_expression,
    _parse_input_shapes_from_reader,
    load_graph_module,
)


FAKE_FULL_GRAPH = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]"):
        return (torch.ops.aten.relu.default(x),)
"""


FULL_GRAPH_WITH_TENSOR_CONSTANT = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]"):
        _tensor_constant0: "f32[]" = self._tensor_constant0
        lift_fresh_copy: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(x, lift_fresh_copy);  x = lift_fresh_copy = None
        return (add,)
"""


FULL_GRAPH_WITH_PRIMS_FMA = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]", y: "f32[2, 2]", z: "f32[2, 2]"):
        fma: "f32[2, 2]" = torch.ops.prims.fma.default(x, y, z);  x = y = z = None
        return (fma,)
"""


FULL_GRAPH_WITH_STRIDED_INPUT = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 3][1, 2]cpu"):
        add: "f32[2, 3][1, 2]cpu" = torch.ops.aten.add.Tensor(x, 1.0);  x = None
        return (add,)
"""


FULL_GRAPH_WITH_NO_INPUTS = """
class GraphModule(torch.nn.Module):
    def forward(self):
        iota: "i64[4][1]cpu" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        add: "i64[4][1]cpu" = torch.ops.aten.add.Tensor(iota, 1);  iota = None
        return (add,)
"""


FULL_GRAPH_WITH_SYM_INPUT = """
class GraphModule(torch.nn.Module):
    def forward(self, s0: "Sym(s0)", x: "f32[2, s0]cpu"):
        view: "f32[2, s0]cpu" = torch.ops.aten.reshape.default(x, [2, s0]);  x = s0 = None
        return (view,)
"""


def _write(path: Path, source: str = FAKE_FULL_GRAPH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


class RecaptureFullGraphsTests(unittest.TestCase):
    def test_int_true_div_requires_an_integral_result(self):
        self.assertEqual(_evaluate_int_expression("IntTrueDiv(8, 2)", {}), 4)
        with self.assertRaisesRegex(ValueError, "not integral"):
            _evaluate_int_expression("IntTrueDiv(7, 2)", {})

    def test_reader_const_preserves_literal_value(self):
        specs = _parse_input_shapes_from_reader(
            """
def load_args(reader):
    reader.const(("mode", 3, True))  # config
"""
        )

        self.assertEqual(specs[0]["value"], ("mode", 3, True))

    def test_infer_target_preserves_suite_mode_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(
                root / "hf" / "train" / "BertForMaskedLM" / "full_graph_000.py"
            )

            target = recapture.infer_target(graph, models_root=root)

            self.assertEqual(target.suite, "hf")
            self.assertEqual(target.mode, "train")
            self.assertEqual(target.model, "BertForMaskedLM")
            self.assertEqual(target.graph_name, "full_graph_000")

    def test_infer_target_preserves_modeless_suite_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(root / "vllm" / "facebook_opt-125m" / "full_graph_001.py")

            target = recapture.infer_target(graph, models_root=root)

            self.assertEqual(target.suite, "vllm")
            self.assertIsNone(target.mode)
            self.assertEqual(target.model, "facebook_opt-125m")

    def test_find_full_graphs_accepts_files_and_directories_with_dedupe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            model_dir = root / "torchbench" / "infer" / "alexnet"
            graph0 = _write(model_dir / "full_graph_000.py")
            graph1 = _write(model_dir / "full_graph_001.py")
            _write(model_dir / "region_000_graph.py")

            found = recapture.find_full_graphs([graph0, model_dir], models_root=root)

            self.assertEqual(found, [graph0, graph1])

    def test_dry_run_does_not_load_or_process_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(root / "timm" / "infer" / "resnet18" / "full_graph_000.py")
            targets = recapture.discover_targets([graph], models_root=root)

            def fail_load(_path):
                raise AssertionError("dry run should not load")

            def fail_process(_gm, _target, _canonical_root):
                raise AssertionError("dry run should not process")

            results = recapture.recapture_targets(
                targets,
                Path(tmp) / "out",
                dry_run=True,
                load_fn=fail_load,
                process_fn=fail_process,
            )

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].ok)
            self.assertEqual(results[0].regions, 0)

    def test_recapture_uses_loader_processor_and_path_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(
                root / "torchbench" / "train" / "densenet121" / "full_graph_000.py"
            )
            targets = recapture.discover_targets([graph], models_root=root)
            calls = []

            fake_gm = object()

            def load_fn(path):
                calls.append(("load", path))
                return fake_gm

            def process_fn(gm, target, canonical_root):
                calls.append(
                    (
                        "process",
                        gm,
                        target.suite,
                        target.mode,
                        target.model,
                        canonical_root,
                    )
                )
                return 7

            canonical_root = Path(tmp) / "canonical_out"
            results = recapture.recapture_targets(
                targets,
                canonical_root,
                load_fn=load_fn,
                process_fn=process_fn,
            )

            self.assertEqual(results[0].regions, 7)
            self.assertEqual(
                calls,
                [
                    ("load", graph),
                    (
                        "process",
                        fake_gm,
                        "torchbench",
                        "train",
                        "densenet121",
                        canonical_root,
                    ),
                ],
            )

    def test_recapture_validates_by_default_and_can_disable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(root / "vllm" / "facebook_opt-125m" / "full_graph_000.py")
            targets = recapture.discover_targets([graph], models_root=root)
            calls = []
            original = recapture.process_graph_for_target

            def load_fn(_path):
                return object()

            def fake_process_graph_for_target(_gm, _target, _canonical_root, *, validate):
                calls.append(validate)
                return 1

            recapture.process_graph_for_target = fake_process_graph_for_target
            try:
                recapture.recapture_targets(
                    targets,
                    Path(tmp) / "out",
                    load_fn=load_fn,
                )
                recapture.recapture_targets(
                    targets,
                    Path(tmp) / "out",
                    validate=False,
                    load_fn=load_fn,
                )
            finally:
                recapture.process_graph_for_target = original

            self.assertEqual(calls, [True, False])

    def test_recapture_fail_fast_returns_only_attempted_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph0 = _write(root / "genai" / "KernelA" / "full_graph_000.py")
            graph1 = _write(root / "genai" / "KernelB" / "full_graph_000.py")
            targets = recapture.discover_targets([graph0, graph1], models_root=root)

            def load_fn(_path):
                raise RuntimeError("load failed")

            results = recapture.recapture_targets(
                targets,
                Path(tmp) / "out",
                fail_fast=True,
                load_fn=load_fn,
            )

            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].ok)
            self.assertIn("load failed", results[0].error)

    def test_isolated_recapture_reports_graph_load_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repros" / "models"
            graph = _write(root / "genai" / "Broken" / "full_graph_000.py", "x = 1\n")
            target = recapture.infer_target(graph, models_root=root)

            result = recapture.recapture_target_isolated(
                target,
                Path(tmp) / "out",
                validate=False,
            )

            self.assertFalse(result.ok)
            self.assertIn("graph load returned None", result.error)

    def test_loader_recreates_saved_tensor_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_TENSOR_CONSTANT,
            )

            gm = load_graph_module(graph)

            self.assertIsNotNone(gm)
            self.assertTrue(hasattr(gm, "graph"))

    def test_loader_keeps_float8_attribute_dtypes(self):
        source = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]"):
        _frozen_param0: "f8e4m3fn[2, 2][2, 1]cpu" = self._frozen_param0
        to: "f32[2, 2]" = torch.ops.prims.convert_element_type.default(_frozen_param0, torch.float32)
        return (torch.ops.aten.add.Tensor(x, to),)
"""
        with tempfile.TemporaryDirectory() as tmp:
            gm = load_graph_module(_write(Path(tmp) / "full_graph_000.py", source), tracing_mode="fake")
            bogus = load_graph_module(
                _write(Path(tmp) / "full_graph_001.py", source.replace("f8e4m3fn[", "zz9[")),
                tracing_mode="fake",
            )

        self.assertIsNotNone(gm)
        float8 = [n for n in gm.graph.nodes if n.op == "get_attr"]
        self.assertEqual([torch.float8_e4m3fn], [getattr(gm, n.target).dtype for n in float8])
        self.assertIsNone(bogus)

    def test_loader_traces_constant_buffers_registered_in_init(self):
        source = """
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_frozen_param0", tensor([2, 0, 1], dtype=torch.int64, device=device(type="cpu")))

    def forward(self, x: "f32[4, 3]"):
        _frozen_param0: "i64[3][1]cpu" = self._frozen_param0
        return (torch.ops.aten.index.Tensor(x, [None, _frozen_param0]),)
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "full_graph_000.py", source)
            graphs = [load_graph_module(path, tracing_mode=mode) for mode in ("fake", "symbolic")]

        for gm in graphs:
            self.assertIsNotNone(gm)
            constants = [getattr(gm, n.target) for n in gm.graph.nodes if n.op == "get_attr"]
            self.assertEqual([[2, 0, 1]], [c.tolist() for c in constants])

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_loader_loads_cuda_constant_buffers_before_cuda_init(self):
        # A fresh process: the graph's __init__ is the first CUDA use.
        source = """
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_frozen_param0", tensor([2, 0, 1], dtype=torch.int64, device=device(type="cuda", index=0)))

    def forward(self, x: "f32[4, 3]"):
        _frozen_param0: "i64[3][1]cuda:0" = self._frozen_param0
        return (torch.ops.aten.index.Tensor(x, [None, _frozen_param0]),)
"""
        script = (
            "import sys; from pathlib import Path\n"
            f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'scripts')!r}]\n"
            "import torch; from ingest_tlparse import load_graph_module\n"
            "assert not torch.cuda.is_initialized()\n"
            "gm = load_graph_module(Path(sys.argv[1]), tracing_mode='symbolic')\n"
            "print([getattr(gm, n.target).tolist() for n in gm.graph.nodes if n.op == 'get_attr'])\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "full_graph_000.py", source)
            result = subprocess.run(
                [sys.executable, "-c", script, str(path)],
                capture_output=True,
                text=True,
                timeout=600,
            )

        self.assertEqual(0, result.returncode, result.stderr[-2000:])
        self.assertEqual("[[2, 0, 1]]", result.stdout.strip().splitlines()[-1])

    def test_loader_leaves_cuda_uninitialized_for_forward_only_devices(self):
        # forward() names CUDA but __init__ does not: loading must not touch
        # CUDA, so a CPU-only host or a parent that later forks still works.
        source = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[4, 3]"):
        full: "f32[3][1]cuda:0" = torch.ops.aten.full.default([3], 1.0, dtype=torch.float32, layout=torch.strided, device=device(type="cuda", index=0), pin_memory=False)
        return (torch.ops.aten.add.Tensor(x, 1),)
"""
        script = (
            "import os, sys; from pathlib import Path\n"
            "if sys.argv[2:]: os.environ['CUDA_VISIBLE_DEVICES'] = ''\n"
            f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'scripts')!r}]\n"
            "import torch; from ingest_tlparse import load_graph_module\n"
            "assert load_graph_module(Path(sys.argv[1]), tracing_mode='fake') is not None\n"
            "print(torch.cuda.is_initialized())\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "full_graph_000.py", source)
            # With and without CUDA devices visible.
            for hide_cuda in ([], ["hide_cuda"]):
                result = subprocess.run(
                    [sys.executable, "-c", script, str(path), *hide_cuda],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                self.assertEqual(0, result.returncode, result.stderr[-2000:])
                self.assertEqual("False", result.stdout.strip().splitlines()[-1])

    def test_loader_registers_inductor_prims_for_fma(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_PRIMS_FMA,
            )

            gm = load_graph_module(graph)

            self.assertIsNotNone(gm)
            self.assertTrue(hasattr(gm, "graph"))

    def test_loader_recreates_printed_input_strides(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_STRIDED_INPUT,
            )

            gm = load_graph_module(graph)

            self.assertIsNotNone(gm)
            placeholder = next(n for n in gm.graph.nodes if n.op == "placeholder")
            self.assertEqual(tuple(placeholder.meta["val"].stride()), (1, 2))

    def test_loader_traces_zero_input_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_NO_INPUTS,
            )

            gm = load_graph_module(graph)

            self.assertIsNotNone(gm)
            self.assertFalse(any(n.op == "placeholder" for n in gm.graph.nodes))

    def test_loader_synthesizes_symint_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_SYM_INPUT,
            )

            gm = load_graph_module(graph)

            self.assertIsNotNone(gm)
            self.assertTrue(hasattr(gm, "graph"))

    def test_graph_loader_ignores_module_level_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "executed"
            source = f'''open({str(marker)!r}, "w").write("bad")
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]cpu"):
        return (torch.ops.aten.relu.default(x),)
'''

            graph_cls = load_graph_class_from_source(source, "test_graph.py")

            self.assertFalse(marker.exists())
            self.assertEqual(graph_cls.__name__, "GraphModule")

    def test_graph_loader_rejects_unsafe_method_calls(self):
        source = '''
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]cpu"):
        open("executed", "w").write("bad")
        return (x,)
'''

        with self.assertRaisesRegex(ValueError, "unsafe|unsupported call"):
            load_graph_class_from_source(source, "test_graph.py")

    def test_graph_loader_rejects_mutation_and_unknown_operator_namespaces(self):
        sources = (
            '''
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]cpu"):
        self.value = x
        return (x,)
''',
            '''
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]cpu"):
        return (torch.ops.unapproved.execute.default(x),)
''',
            '''
class GraphModule(
    torch.nn.Module,
    torch.serialization.os.system("touch should-not-exist") or object,
):
    def forward(self, x: "f32[2, 2]cpu"):
        return (x,)
''',
            '''
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]cpu"):
        leaked = torch.serialization.os.environ["BB_REVIEW_SECRET"]
        return (torch.ops.aten.scalar_tensor.default(leaked == "secret"),)
''',
            '''
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]cpu"):
        alias = torch
        leaked = alias.serialization.os.environ["BB_REVIEW_SECRET"]
        return (torch.ops.aten.scalar_tensor.default(leaked == "secret"),)
''',
        )

        for source in sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    ValueError,
                    "mutate|unsupported call|unsupported torch|one Module base",
                ):
                    load_graph_class_from_source(source, "test_graph.py")

    def test_graph_loader_allows_dunder_operator_overloads(self):
        source = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "i64[4]cpu"):
        return (torch.ops.aten.__rshift__.Scalar(x, 1),)
"""
        graph_cls = load_graph_class_from_source(source, "test_graph.py")

        (out,) = graph_cls()(torch.tensor([8, 4, 2, 1]))
        self.assertEqual([4, 2, 1, 0], out.tolist())

    def test_graph_loader_rejects_dunder_escapes(self):
        escapes = (
            "torch.ops.aten.__class__",
            "torch.ops.aten.__dict__",
            "torch.ops.aten.__getattribute__('add')",
            "torch.ops.aten.__rshift__.__globals__",
            "torch.ops.aten.__rshift__.Scalar.__class__",
            "torch.ops.unapproved.__rshift__.Scalar(x, 1)",
            "x.__class__.__subclasses__()",
            "torch.__builtins__",
            "self.__init__.__globals__",
        )
        for escape in escapes:
            source = f"""
class GraphModule(torch.nn.Module):
    def forward(self, x: "i64[4]cpu"):
        y = {escape}
        return (x,)
"""
            with self.subTest(escape=escape):
                with self.assertRaisesRegex(
                    ValueError, "unsafe|unsupported call|unsupported torch"
                ):
                    load_graph_class_from_source(source, "test_graph.py")

        shadowed = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "i64[4]cpu"):
        torch = x
        return (torch.ops.aten.__rshift__.Scalar(x, 1),)
"""
        with self.assertRaisesRegex(ValueError, "unsafe"):
            load_graph_class_from_source(shadowed, "test_graph.py")

    def test_loader_traces_random_constant_buffers_registered_in_init(self):
        # fx_graph_runnable's form for buffers too large to print.
        source = """
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_tensor_constant0', torch.randint(1, size=[3], dtype=torch.int64))
        self.register_buffer('_tensor_constant1', torch.randn([3], dtype=torch.float32))

    def forward(self, x: "f32[4, 3]"):
        _tensor_constant0: "i64[3][1]cpu" = self._tensor_constant0
        _tensor_constant1: "f32[3][1]cpu" = self._tensor_constant1
        index = torch.ops.aten.index.Tensor(x, [None, _tensor_constant0])
        return (index, torch.ops.aten.mul.Tensor(_tensor_constant1, 2))
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "full_graph_000.py", source)
            graphs = [load_graph_module(path, tracing_mode=mode) for mode in ("fake", "symbolic")]

        for gm in graphs:
            self.assertIsNotNone(gm)
            constants = [getattr(gm, n.target) for n in gm.graph.nodes if n.op == "get_attr"]
            self.assertEqual([torch.int64, torch.float32], [c.dtype for c in constants])
            self.assertEqual([0, 0, 0], constants[0].tolist())

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_loader_loads_random_cuda_constant_buffers_before_cuda_init(self):
        source = """
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_tensor_constant0', torch.randint(1, size=[3], dtype=torch.int64).to("cuda:0"))

    def forward(self, x: "f32[4, 3]"):
        _tensor_constant0: "i64[3][1]cuda:0" = self._tensor_constant0
        return (torch.ops.aten.index.Tensor(x, [None, _tensor_constant0]),)
"""
        script = (
            "import sys; from pathlib import Path\n"
            f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'scripts')!r}]\n"
            "import torch; from ingest_tlparse import load_graph_module\n"
            "assert not torch.cuda.is_initialized()\n"
            "gm = load_graph_module(Path(sys.argv[1]), tracing_mode='symbolic')\n"
            "print([str(getattr(gm, n.target).device) for n in gm.graph.nodes if n.op == 'get_attr'])\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "full_graph_000.py", source)
            result = subprocess.run(
                [sys.executable, "-c", script, str(path)],
                capture_output=True,
                text=True,
                timeout=600,
            )

        self.assertEqual(0, result.returncode, result.stderr[-2000:])
        self.assertEqual("['cuda:0']", result.stdout.strip().splitlines()[-1])

    def test_graph_loader_limits_tensor_factories_to_buffer_constants(self):
        sources = (
            # Outside register_buffer.
            """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[3]cpu"):
        return (torch.randn([3]),)
""",
            # Outside __init__.
            """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[3]cpu"):
        self.register_buffer('b', torch.randn([3]))
        return (x,)
""",
            # Not one of the factories fx_graph_runnable writes.
            """
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.load('weights.pt'))
""",
            # A method other than .to on the factory result.
            """
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.randn([3]).numpy())
""",
            # .to with a non-constant argument.
            """
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.randn([3]).to(torch.serialization.os.getcwd()))
""",
            # Keywords fx_graph_runnable does not write.
            """
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.randn([3], device='cuda'))
""",
            """
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.randn(*[3]))
""",
            # A factory argument that escapes.
            """
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.randn([3], dtype=torch.serialization.os.system('true')))
""",
        )
        for source in sources:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    ValueError, "unsupported call|unsupported torch"
                ):
                    load_graph_class_from_source(source, "test_graph.py")

        for device, error in (("nowhere", "invalid device"), ("meta", "unsupported device")):
            source = f"""
class GraphModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.randn([3]).to("{device}"))
"""
            with self.subTest(device=device):
                with self.assertRaisesRegex(ValueError, error):
                    load_graph_class_from_source(source, "test_graph.py")

    def test_loader_resolves_and_binds_symint_inputs_from_expressions(self):
        # primals_1 has no written value but its symbol is bound at module
        # level; primals_2 is an expression of two symbols.
        source = """
s0 = 6
s1 = 2


class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, primals_1, primals_2, primals_3):
        full = torch.ops.aten.full.default([primals_2], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        cat = torch.ops.aten.cat.default([primals_3, full]);  primals_3 = full = None
        mul = primals_1 * 2;  primals_1 = None
        return (cat, mul)


def load_args(reader):
    reader.symint(None, expr='s0')  # primals_1
    reader.symint(8, expr='s0 + s1')  # primals_2
    buf0 = reader.storage(None, 4*s0, device=device(type='cpu'), dtype_hint=torch.float32)
    reader.tensor(buf0, (s0,), is_leaf=True)  # primals_3
load_args._version = 0
mod = Repro()
"""
        specs = _parse_input_shapes_from_reader(source)
        self.assertEqual([6, 8], [spec["value"] for spec in specs[:2]])

        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "fx_graph_runnable.py", source)
            graphs = [load_graph_module(path, tracing_mode=mode) for mode in ("real", "symbolic")]

        real, symbolic = graphs
        self.assertIsNotNone(real)
        self.assertIsNotNone(symbolic)
        self.assertTrue(symbolic.meta["better_benchmark_captured_dynamic"])
        placeholders = [n.meta["val"] for n in symbolic.graph.nodes if n.op == "placeholder"]
        self.assertTrue(all(isinstance(value, torch.SymInt) for value in placeholders[:2]))

    def test_loader_rejects_symint_expression_inputs_that_disagree_with_shapes(self):
        source = """
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(2*s0)", primals_2: "f32[2, s0]"):
        view = torch.ops.aten.view.default(primals_2, [primals_1]);  primals_1 = primals_2 = None
        return (view,)
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "full_graph_000.py", source)
            self.assertIsNone(load_graph_module(path, tracing_mode="symbolic"))

    def test_ingest_writes_graphs_with_constant_symint_inputs(self):
        # A SymInt specialized to a constant (expr='4') traces as a plain int,
        # so its input spec must be a scalar like the captured placeholder.
        source = """
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, primals_1, primals_2):
        view = torch.ops.aten.view.default(primals_2, [primals_1, 2]);  primals_1 = primals_2 = None
        relu = torch.ops.aten.relu.default(view);  view = None
        return (relu,)


def load_args(reader):
    reader.symint(4, expr='4')  # primals_1
    buf0 = reader.storage(None, 32, device=device(type='cpu'), dtype_hint=torch.float32)
    reader.tensor(buf0, (8,), is_leaf=True)  # primals_2
load_args._version = 0
mod = Repro()
"""
        specs = _parse_input_shapes_from_reader(source)
        self.assertEqual(("scalar", 4), (specs[0]["kind"], specs[0]["value"]))

        from capture_hook import _CaptureState

        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "fx_graph_runnable.py", source)
            graph_module = load_graph_module(path, tracing_mode="symbolic")
            state = _CaptureState(str(Path(tmp) / "capture"), label="g", graph_dir=str(Path(tmp) / "g"))
            state.process_graph(graph_module)
            state.finalize()
            self.assertTrue((Path(tmp) / "g" / "full_graph_000.py").exists())

    def test_loader_still_rejects_unresolvable_symint_inputs(self):
        source = """
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, primals_1):
        mul = primals_1 * 2;  primals_1 = None
        return (mul,)


def load_args(reader):
    reader.symint(None, expr='s7')  # primals_1
load_args._version = 0
mod = Repro()
"""
        specs = _parse_input_shapes_from_reader(source)
        self.assertIsNone(specs[0]["value"])

    def test_loader_copies_only_views_that_eager_cannot_perform(self):
        # A padded matmul's output sliced back to its width has row stride 8,
        # so merging its rows cannot be a view; splitting its last dim can.
        source = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[6, 8][8, 1]cpu"):
        slice_1: "f32[6, 4][8, 1]cpu" = torch.ops.aten.slice.Tensor(x, 1, 0, -4);  x = None
        view: "f32[2, 12][12, 1]cpu" = torch.ops.aten.view.default(slice_1, [-1, 12])
        view_1: "f32[6, 2, 2][8, 2, 1]cpu" = torch.ops.aten.view.default(slice_1, [6, 2, 2]);  slice_1 = None
        return (view, view_1)
"""
        x = torch.arange(48, dtype=torch.float32).reshape(6, 8)
        expected = (x[:, :4].reshape(2, 12), x[:, :4].view(6, 2, 2))
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "full_graph_000.py", source)
            graphs = {
                mode: load_graph_module(path, tracing_mode=mode)
                for mode in ("real", "fake", "symbolic")
            }

        for mode, gm in graphs.items():
            with self.subTest(mode=mode):
                self.assertIsNotNone(gm)
                targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
                self.assertEqual(1, targets.count(torch.ops.aten.view.default))
                self.assertIn(torch.ops.aten.clone.default, targets)
        out = graphs["real"](x)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(out, expected)))
        # The view eager can perform still aliases the input.
        self.assertEqual(x.data_ptr(), out[1].data_ptr())

    def test_loader_rejects_copied_views_whose_aliasing_is_observed(self):
        prefix = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[6, 8][8, 1]cpu"):
        slice_1: "f32[6, 4][8, 1]cpu" = torch.ops.aten.slice.Tensor(x, 1, 0, -4)
        view: "f32[24][1]cpu" = torch.ops.aten.view.default(slice_1, [-1]);  slice_1 = None
"""
        bodies = {
            "in-place op": """
        add_: "f32[24][1]cpu" = torch.ops.aten.add_.Tensor(view, 1);  view = None
        return (x,)
""",
            "as_strided": """
        as_strided: "f32[3, 4][8, 1]cpu" = torch.ops.aten.as_strided.default(view, [3, 4], [8, 1], 0);  view = None
        return (as_strided,)
""",
            "as_strided on a view": """
        view_1: "f32[2, 12][12, 1]cpu" = torch.ops.aten.view.default(view, [2, 12]);  view = None
        as_strided: "f32[3, 4][8, 1]cpu" = torch.ops.aten.as_strided.default(view_1, [3, 4], [8, 1], 0);  view_1 = None
        return (as_strided,)
""",
        }
        for case, body in bodies.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                path = _write(Path(tmp) / "full_graph_000.py", prefix + body)
                self.assertIsNone(load_graph_module(path, tracing_mode="fake"))

        # as_strided on a fresh result is unaffected by the copy.
        source = prefix + """
        mul: "f32[24][1]cpu" = torch.ops.aten.mul.Tensor(view, 2);  view = None
        as_strided: "f32[3, 4][8, 1]cpu" = torch.ops.aten.as_strided.default(mul, [3, 4], [8, 1], 0);  mul = None
        return (as_strided,)
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "full_graph_000.py", source)
            self.assertIsNotNone(load_graph_module(path, tracing_mode="fake"))


if __name__ == "__main__":
    unittest.main()
