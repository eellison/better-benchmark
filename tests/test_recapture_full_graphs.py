"""Focused tests for recapturing from saved full_graph artifacts.

Usage:
    python scripts/test_recapture_full_graphs.py
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import recapture_full_graphs as recapture
from ingest_tlparse import load_graph_module
from saved_graph_replay import _build_symbolic_inputs, _sidecar_symbol_hints


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


FULL_GRAPH_SYM_POINTWISE = """
class GraphModule(torch.nn.Module):
    def forward(self, s0: "Sym(s0)", x: "f32[2, s0]cpu"):
        view: "f32[2, s0]cpu" = torch.ops.aten.reshape.default(x, [2, s0]);  x = None
        add: "f32[2, s0]cpu" = torch.ops.aten.add.Tensor(view, 1.0);  view = None
        mul: "f32[2, s0]cpu" = torch.ops.aten.mul.Tensor(add, 2);  add = s0 = None
        return (mul,)
"""


FULL_GRAPH_WITH_FLOORDIV_DIM = """
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, s0//2]cpu"):
        add: "f32[2, s0//2]cpu" = torch.ops.aten.add.Tensor(x, 1.0);  x = None
        return (add,)
"""


def _write(path: Path, source: str = FAKE_FULL_GRAPH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


class RecaptureFullGraphsTests(unittest.TestCase):
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

    def test_loader_preserves_dynamism_symint_shares_tensor_dim(self):
        """Idempotence regression: a saved graph with a symint input shared
        with a tensor dynamic dim (s0 in both `Sym(s0)` and `f32[2, s0]`) MUST
        re-trace SYMBOLICALLY with the symint and the tensor dim as the SAME
        symbol. Real-mode tracing baked them to hints, so a dynamic saved graph
        recaptured as a STATIC region (different pattern hash, frozen dims,
        lifted _shape_param const list). This asserts the dim stays symbolic
        and the two placeholders share one symbol."""
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_SYM_INPUT,
            )
            gm = load_graph_module(graph)
            self.assertIsNotNone(gm)

            phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
            symint_ph = next(n for n in phs
                             if isinstance(n.meta.get("val"), torch.SymInt))
            tensor_ph = next(n for n in phs
                             if torch.is_tensor(n.meta.get("val")))
            tdim = tensor_ph.meta["val"].shape[1]
            # the tensor's 2nd dim must still be symbolic (NOT baked to a hint)
            self.assertIsInstance(tdim, torch.SymInt)
            # and it must be the SAME symbol as the symint input
            self.assertEqual(str(symint_ph.meta["val"].node.expr),
                             str(tdim.node.expr))

    def test_loader_preserves_floordiv_dims(self):
        """PR80 review finding 4 (end to end): a saved annotation whose dim
        uses the closed grammar beyond +/-/* ('f32[2, s0//2]') must re-trace
        SYMBOLICALLY — pre-fix it silently fell back to a static real-mode
        trace (dim baked to the concrete hint, dynamism lost)."""
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_FLOORDIV_DIM,
            )
            gm = load_graph_module(graph)
            self.assertIsNotNone(gm)
            ph = next(n for n in gm.graph.nodes if n.op == "placeholder")
            self.assertIsInstance(ph.meta["val"].shape[1], torch.SymInt)

    def test_loader_refuses_static_recapture_of_dynamic_graph(self):
        """PR80 review finding 4 (refusal half): a graph with symbolic inputs
        whose rebuild genuinely fails (a dim folding to a non-integer at its
        hint) must FAIL ingestion (None) — never silently produce a static
        GraphModule for a dynamic family (f(f(x)) != f(x), corrupted point
        identity)."""
        with tempfile.TemporaryDirectory() as tmp:
            graph = _write(
                Path(tmp) / "full_graph_000.py",
                FULL_GRAPH_WITH_FLOORDIV_DIM.replace("s0//2", "s0*1.5"),
            )
            self.assertIsNone(load_graph_module(graph))

    def test_recapture_is_fixed_point_for_dynamic_graph(self):
        """recapture(recapture(x)) == recapture(x): a dynamic saved graph,
        recaptured twice, yields byte-identical canonical content. Guards the
        whole idempotence chain (load -> partition -> merge) for symbolic
        shapes, not just that the loader doesn't crash."""
        import json
        canon_a = self._recapture_once(FULL_GRAPH_WITH_SYM_INPUT, "a")
        canon_b = self._recapture_once(FULL_GRAPH_WITH_SYM_INPUT, "b")
        self.assertEqual(sorted(canon_a), sorted(canon_b),
                         "pattern-hash region set not stable across recapture")
        for region in canon_a:
            for fname in ("repro.py", "shapes.json"):
                fa = canon_a[region] / fname
                fb = canon_b[region] / fname
                if fa.exists() and fb.exists():
                    self.assertEqual(fa.read_text(), fb.read_text(),
                                     f"{region}/{fname} not byte-stable")

    def test_recapture_composition_is_fixed_point(self):
        """f(f(x)) == f(x) THROUGH the pipeline (alignment §6b): the sibling
        fixed-point test runs the FIRST generation twice, which proves
        determinism, not composition. Here the LOADED symbolic gm is re-traced
        a SECOND time (generation 2 = f(f(x))) and BOTH generations are
        processed through partition/merge into fresh canonical roots — same
        region-dir set, byte-identical repro.py / shapes.json, and the
        symint/tensor-dim symbol sharing survives generation 2."""
        import torch
        from torch.fx.experimental.proxy_tensor import make_fx

        tmp = Path(tempfile.mkdtemp(prefix="fixedpoint_compose_"))
        models_root = tmp / "repros" / "models"
        gpath = models_root / "torchbench" / "infer" / "m" / "full_graph_000.py"
        # A fixture that actually CAPTURES a region: the bare-reshape source
        # the sibling test uses partitions to zero regions, which made an
        # empty-vs-empty comparison pass vacuously.
        _write(gpath, FULL_GRAPH_SYM_POINTWISE)
        target = recapture.infer_target(gpath, models_root=models_root)

        gm1 = load_graph_module(gpath)
        self.assertIsNotNone(gm1)

        # Generation 2: symbolic re-trace OF THE LOADED gm, driving its own
        # placeholder values back through it under their fake mode.
        vals = [n.meta["val"] for n in gm1.graph.nodes
                if n.op == "placeholder"]
        fake_mode = torch._guards.detect_fake_mode(vals)
        self.assertIsNotNone(fake_mode)
        with fake_mode:
            gm2 = make_fx(gm1, tracing_mode="symbolic")(*vals)

        # The composed generation still shares ONE symbol between the symint
        # input and the tensor dynamic dim — dynamism survived f(f(x)).
        phs2 = [n for n in gm2.graph.nodes if n.op == "placeholder"]
        symint_ph = next(n for n in phs2
                         if isinstance(n.meta.get("val"), torch.SymInt))
        tensor_ph = next(n for n in phs2
                         if torch.is_tensor(n.meta.get("val")))
        tdim = tensor_ph.meta["val"].shape[1]
        self.assertIsInstance(tdim, torch.SymInt)
        self.assertEqual(str(symint_ph.meta["val"].node.expr),
                         str(tdim.node.expr))

        def _process(gm, tag):
            root = tmp / f"out_{tag}"
            recapture.process_graph_for_target(gm, target, root,
                                               validate=True)
            canon = root / "canonical"
            return ({d.name: d for d in canon.iterdir()}
                    if canon.exists() else {})

        canon_1 = _process(gm1, "gen1")
        canon_2 = _process(gm2, "gen2")
        self.assertTrue(canon_1)
        self.assertEqual(sorted(canon_1), sorted(canon_2),
                         "region set not stable under composition")
        for region in canon_1:
            for fname in ("repro.py", "shapes.json"):
                fa = canon_1[region] / fname
                fb = canon_2[region] / fname
                self.assertEqual(fa.exists(), fb.exists(),
                                 f"{region}/{fname} presence differs")
                if fa.exists():
                    self.assertEqual(
                        fa.read_text(), fb.read_text(),
                        f"{region}/{fname} not byte-stable under composition")

    def _recapture_once(self, source: str, tag: str) -> dict:
        """Recapture one saved-graph source into a fresh root; return
        {region_dir_name: canonical_dir_path}. Helper for the fixed-point test."""
        tmp = Path(tempfile.mkdtemp(prefix=f"fixedpoint_{tag}_"))
        models_root = tmp / "repros" / "models"
        gpath = models_root / "torchbench" / "infer" / "m" / "full_graph_000.py"
        gpath.parent.mkdir(parents=True, exist_ok=True)
        _write(gpath, source)
        canonical_root = tmp / "out"
        target = recapture.infer_target(gpath, models_root=models_root)
        gm = load_graph_module(gpath)
        self.assertIsNotNone(gm)
        recapture.process_graph_for_target(gm, target, canonical_root, validate=True)
        canon = canonical_root / "canonical"
        return {d.name: d for d in canon.iterdir()} if canon.exists() else {}


class TestBuildSymbolicInputs(unittest.TestCase):
    """Unit tests for _build_symbolic_inputs symbol resolution (CPU, no CUDA).

    Regression coverage for the symint/stride/hint resolution bugs where a
    dynamic quantity silently collapsed to a constant (losing dynamism):
      B3  constant symint (Sym(256)) and composite symint (Sym(s0*s1))
      B4  a symbol appearing ONLY in a stride expr
      B5  a symbol whose hint is 0 or 1 (ShapeEnv 0/1-specialization)
    """

    def _build(self, specs, hints=None):
        import torch
        return _build_symbolic_inputs(specs, torch.device("cpu"), hints or {})

    def test_constant_symint_keeps_value_not_default(self):
        import torch
        specs = [
            {"kind": "tensor", "name": "x", "shape": ["s0"],
             "dtype": "float32", "stride": ["1"], "device": "cpu"},
            {"kind": "symint", "name": "n", "value": 256},
        ]
        _mode, inputs = self._build(specs, {"s0": 8})
        self.assertEqual(inputs[1], 256)  # not collapsed to the default hint

    def test_composite_symint_stays_symbolic(self):
        import torch
        specs = [
            {"kind": "tensor", "name": "x", "shape": ["s0", "s1"],
             "dtype": "float32", "stride": ["s1", "1"], "device": "cpu"},
            {"kind": "symint", "name": "n", "expr": "s0*s1"},
        ]
        _mode, inputs = self._build(specs, {"s0": 8, "s1": 4})
        self.assertIsInstance(inputs[1], torch.SymInt)

    def test_stride_only_symbol_is_seeded(self):
        import torch
        # s2 appears ONLY in the stride, never in a shape dim or the sidecar.
        specs = [
            {"kind": "tensor", "name": "x", "shape": ["s0", "4"],
             "dtype": "float32", "stride": ["s2", "s0"], "device": "cpu"},
        ]
        res = self._build(specs, {})
        self.assertIsNotNone(res, "stride-only symbol dropped -> rebuild None")
        _mode, inputs = res
        self.assertIsInstance(inputs[0].stride()[0], torch.SymInt)

    def test_hint_one_does_not_specialize(self):
        import torch
        specs = [
            {"kind": "tensor", "name": "x", "shape": ["s0"],
             "dtype": "float32", "stride": ["1"], "device": "cpu"},
        ]
        _mode, inputs = self._build(specs, {"s0": 1})  # falsy hint
        self.assertIsInstance(inputs[0].shape[0], torch.SymInt)

    def test_composite_grammar_dims_stay_symbolic(self):
        """PR80 review finding 4: valid saved dims using the closed torch
        grammar beyond +/-/* (floordiv, CeilToInt, PythonMod) must rebuild
        SYMBOLICALLY over the SAME base symbol — the old evaluator's charset
        gate returned None for them and the whole graph silently recaptured
        static."""
        import torch
        for dim in ("s0//2", "CeilToInt(s0/3)", "PythonMod(s0, 4)"):
            specs = [{"kind": "tensor", "name": "x", "shape": ["s0", dim],
                      "dtype": "float32", "stride": None, "device": "cpu"}]
            res = self._build(specs, {"s0": 8})
            self.assertIsNotNone(res, dim)
            _mode, inputs = res
            d0, d1 = inputs[0].shape[0], inputs[0].shape[1]
            self.assertIsInstance(d1, torch.SymInt, dim)
            self.assertIn(str(d0.node.expr), str(d1.node.expr), dim)

    def test_non_integral_dim_refuses(self):
        """A dim expr that folds to a NON-integer at the hint ('s0*1.5' at
        s0=8 -> 12.0) cannot be traced at a concrete size — the rebuild must
        return None (caller fails ingestion), never guess a coercion."""
        specs = [{"kind": "tensor", "name": "x", "shape": ["s0*1.5"],
                  "dtype": "float32", "stride": None, "device": "cpu"}]
        self.assertIsNone(self._build(specs, {"s0": 8}))


class TestSidecarSymbolHints(unittest.TestCase):
    """B7: sidecar hint coercion. A native per-symbol hint is the join key
    back to live occurrence counts, so a hint stored in a slightly-off type
    must not be silently dropped (orphaning the accounting join) nor let a
    bool through as a 0/1-specializing size."""

    def _hints(self, symbols):
        with tempfile.TemporaryDirectory() as tmp:
            g = Path(tmp) / "full_graph_000.py"
            g.write_text("# stub")
            (g.with_name("full_graph_000.meta.json")).write_text(
                __import__("json").dumps({"symbols": symbols}))
            return _sidecar_symbol_hints(g)

    def test_int_hint_is_kept(self):
        self.assertEqual(self._hints({"s0": {"hint": 16}}), {"s0": 16})

    def test_bool_hint_is_rejected(self):
        # isinstance(True, int) is True; a bool must not seed a size-1 hint.
        self.assertEqual(self._hints({"s0": {"hint": True}}), {})

    def test_integral_float_is_coerced(self):
        r = self._hints({"s0": {"hint": 4.0}})
        self.assertEqual(r, {"s0": 4})
        self.assertIsInstance(r["s0"], int)

    def test_non_integral_and_garbage_are_dropped(self):
        self.assertEqual(
            self._hints({"a": {"hint": 3.5}, "b": {"hint": "8"},
                         "c": {"hint": None}, "d": "notadict"}),
            {})


if __name__ == "__main__":
    unittest.main()
