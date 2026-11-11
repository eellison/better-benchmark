"""Native ShapesSpec replay tests.

Capture records ranges, guards, and symbolic input relationships. These tests
prove that replay installs the same contract in Dynamo's ShapeEnv, including
runtime guard enforcement, raw-SymInt correspondence, composite dimensions,
and symbolic tensor metadata.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from dynamic_shape_replay import (  # noqa: E402
    _backed_args,
    _backed_replay_plan_from_contract,
    _backed_replay_plan_for_repro,
    _backed_repro,
    _eval_shape_env_expr,
    _dynamic_replay_args,
    _dynamic_replay_config,
    _prepare_dynamic_replay_from_contract,
    _shape_env_repro,
    _shape_env_spec_from_contract,
    _shape_env_spec_for_repro,
)


class TestShapeEnvPlayback(unittest.TestCase):
    def _write_guarded(self, d: Path) -> str:
        (d / "shapes.json").write_text(json.dumps({
            "symbols": {"s0": {"hint": 8, "range": [2, None]}},
            "guards": ["Eq(Mod(s0, 4), 0)"],
            "points": [{
                "shape_hash": "guarded",
                "captured_dynamic": True,
                "bindings": {"s0": 8},
                "models": {"m": {"occurrences": 1}},
                "inputs": [[["s0", 3], "f32"], ["I", 8, "s0"]],
            }],
        }))
        repro = d / "repro.py"
        repro.write_text("# stub")
        return str(repro)

    def test_guard_and_tensor_symint_correspondence_run_in_shape_env(self):
        class UsesInt(torch.nn.Module):
            def forward(self, x, n):
                return x + n

        with tempfile.TemporaryDirectory() as td:
            repro = self._write_guarded(Path(td))
            (spec, live_names, input_count,
             metadata_checks, frozen) = _shape_env_spec_for_repro(repro)
            self.assertEqual(live_names, ("s0",))
            self.assertEqual(input_count, 2)
            # Core's config/metrics serializer must see data, not live SymInts.
            json.dumps(spec.to_jsonable())

            graphs = []

            def backend(gm, _example_inputs):
                graphs.append(gm)
                return gm.forward

            torch._dynamo.reset()
            compiled = torch.compile(
                _shape_env_repro(
                    UsesInt(), input_count, spec, live_names,
                    metadata_checks, frozen),
                backend=backend,
            )
            compiled(torch.zeros(8, 3), 8, 8)
            compiled(torch.zeros(12, 3), 12, 12)
            self.assertEqual(len(graphs), 1)
            self.assertTrue(any(
                node.op == "call_function"
                and "_assert_scalar" in str(node.target)
                for node in graphs[0].graph.nodes
            ))

            # Bypass the harness's external binding validation: these failures
            # prove the compiled artifact itself enforces both invariants.
            with self.assertRaises((AssertionError, RuntimeError)):
                compiled(torch.zeros(8, 3), 7, 8)
            with self.assertRaises((AssertionError, RuntimeError)):
                compiled(torch.zeros(6, 3), 6, 6)
            torch._dynamo.reset()

    def test_backed_replay_derives_symint_and_keeps_guard_out_of_graph(self):
        class UsesInt(torch.nn.Module):
            def forward(self, x, n):
                return x + n

        with tempfile.TemporaryDirectory() as td:
            repro = self._write_guarded(Path(td))
            plan = _backed_replay_plan_for_repro(repro)
            self.assertIsNotNone(plan)
            self.assertEqual(plan["kept"], (0,))

            graphs = []

            def backend(gm, _example_inputs):
                graphs.append(gm)
                return gm.forward

            compiled = torch.compile(
                _backed_repro(UsesInt(), plan), backend=backend)

            def invoke(size, raw_int):
                inputs = [torch.zeros(size, 3), raw_int]
                return compiled(*_backed_args(inputs, plan))

            invoke(8, 8)
            invoke(12, 12)
            self.assertEqual(len(graphs), 1)
            self.assertFalse(any(
                node.op == "call_function"
                and "_assert_scalar" in str(node.target)
                for node in graphs[0].graph.nodes
            ))
            # The lifted raw int is not a compiled argument: the callable
            # re-derives it from x.size(0), preserving the model relationship.
            self.assertTrue(torch.equal(
                invoke(8, 999), torch.full((8, 3), 8.0)))
            with self.assertRaises(RuntimeError):
                invoke(6, 6)  # violates Eq(Mod(s0, 4), 0)
            torch._dynamo.reset()

    def test_backed_replay_marks_repeated_and_affine_tensor_occurrences(self):
        """Every occurrence must stay dynamic, not only the root source.

        An unmarked second ``s0`` specializes to the first value and its
        relation check then forces the explicitly dynamic source to the same
        constant. An affine ``2*s0`` occurrence has the same failure mode.
        """
        class UsesAll(torch.nn.Module):
            def forward(self, x, repeated, affine):
                return x.sum() + repeated.sum() + affine.sum()

        contract = {
            "symbols": {"s0": {"hint": 4, "range": [2, None]}},
            "guards": [],
            "points": [{
                "captured_dynamic": True,
                "bindings": {"s0": 4},
                "inputs": [
                    [["s0", 3], "f32"],
                    [["s0", 3], "f32"],
                    [["2*s0", 3], "f32"],
                ],
            }],
        }
        plan = _backed_replay_plan_from_contract(contract)
        self.assertIsNotNone(plan)
        self.assertEqual(
            tuple((pos, dim) for pos, dim, _bounds in plan["markings"]),
            ((0, 0), (1, 0), (2, 0)),
        )

        graphs = []

        def backend(gm, _example_inputs):
            graphs.append(gm)
            return gm.forward

        torch._dynamo.reset()
        compiled = torch.compile(
            _backed_repro(UsesAll(), plan), backend=backend)
        for size in (4, 6, 4):
            inputs = [
                torch.zeros(size, 3),
                torch.ones(size, 3),
                torch.ones(2 * size, 3),
            ]
            compiled(*_backed_args(inputs, plan))
        self.assertEqual(len(graphs), 1)
        torch._dynamo.reset()

    def test_unbacked_metadata_prevents_backed_replay(self):
        """A tensor occurrence is a possible source, not symbol provenance."""
        with tempfile.TemporaryDirectory() as td:
            repro = self._write_guarded(Path(td))
            shapes_path = Path(repro).parent / "shapes.json"
            data = json.loads(shapes_path.read_text())
            data["symbols"]["s0"]["observed_value"] = (
                data["symbols"]["s0"].pop("hint"))
            data["symbols"]["s0"]["unbacked"] = True
            shapes_path.write_text(json.dumps(data))
            self.assertIsNone(_backed_replay_plan_for_repro(repro))

    def test_unbacked_optimization_hint_is_not_a_point(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "repro.py").write_text("# stub")
            (d / "shapes.json").write_text(json.dumps({
                "symbols": {
                    "s0": {
                        "range": [0, 64],
                        "unbacked": True,
                        "optimization_hint": 16,
                    },
                },
                "guards": [],
                "points": [{
                    "shape_hash": "template",
                    "captured_dynamic": True,
                    "bindings": {},
                    "requires_binding": ["s0"],
                    "models": {"m": {"occurrences": 1}},
                    "inputs": [[["s0", 3], "f32"], ["I", None, "s0"]],
                }],
            }))
            spec, names, *_ = _shape_env_spec_for_repro(
                str(d / "repro.py"))
            self.assertEqual(names, ("s0",))
            serialized = spec.to_jsonable()
            self.assertIn(
                '"optimization_hint": 16',
                json.dumps(serialized, sort_keys=True),
            )

    def test_symfloat_typed_roundtrip_is_idempotent_and_reuses_one_graph(self):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from full_graph_harness import (
            _harvest_shape_env,
            _scalar_spec_from_value,
        )
        from input_codec import compact_from_spec, spec_from_compact

        shape_env = ShapeEnv()
        symfloat = shape_env.create_unbacked_symfloat()
        shape_env.real_tensor_prop_unbacked_vals[symfloat.node.expr] = 1.5
        symbol_block = _harvest_shape_env(shape_env)
        self.assertEqual(
            symbol_block["symbols"]["zuf0"]["kind"], "symfloat")

        spec = _scalar_spec_from_value("scale", symfloat)
        self.assertEqual(spec["value"], 1.5)
        entry1 = compact_from_spec(spec)
        entry2 = compact_from_spec(spec_from_compact(entry1))
        entry3 = compact_from_spec(spec_from_compact(entry2))
        self.assertEqual(entry1, entry2)
        self.assertEqual(entry2, entry3)
        json.dumps(entry3)  # persisted representation is plain typed data

        class Scale(torch.nn.Module):
            def forward(self, x, scale):
                return x * scale

        contract = {
            "symbols": symbol_block["symbols"],
            "guards": [],
            "points": [{
                "captured_dynamic": True,
                "bindings": {"zuf0": 1.5},
                "inputs": [[[3], "f32"], entry3],
            }],
        }
        prepared = _prepare_dynamic_replay_from_contract(
            Scale(), contract)
        self.assertEqual(prepared["kind"], "symfloat")
        graphs = []

        def backend(gm, _example_inputs):
            graphs.append(gm)
            return gm.forward

        with _dynamic_replay_config(prepared):
            compiled = torch.compile(
                prepared["module"], backend=backend, fullgraph=True)
            for value in (1.5, 2.25, 1.5):
                args = _dynamic_replay_args(
                    [torch.ones(3), value],
                    {"zuf0": value},
                    prepared,
                )
                torch.testing.assert_close(
                    compiled(*args),
                    torch.full((3,), value),
                )
        self.assertEqual(len(graphs), 1)
        torch._dynamo.reset()

    def test_capture_repro_emits_typed_symfloat_entry(self):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from capture_hook import _CaptureState, canonicalize_subgraph
        from full_graph_harness import _harvest_shape_env

        shape_env = ShapeEnv()
        symfloat = shape_env.create_unbacked_symfloat()
        shape_env.real_tensor_prop_unbacked_vals[
            symfloat.node.expr] = 1.5
        block = _harvest_shape_env(shape_env)
        dtype = block["symbols"]["zuf0"]["dtype"]

        graph = torch.fx.Graph()
        scale = graph.placeholder("scale")
        scale.meta["val"] = symfloat
        graph.output(scale)
        gm = torch.fx.GraphModule({}, graph)
        info = {
            "scale": {
                "shape": [],
                "stride": [],
                "dtype": "symfloat",
                "device": "cpu",
                "hint": 1.5,
                "expr": "zuf0",
                "float_dtype": dtype,
            },
        }

        canonical, canonical_info, _ = canonicalize_subgraph(gm, info)
        self.assertIs(canonical, gm)
        self.assertEqual(canonical_info, info)
        with tempfile.TemporaryDirectory() as td:
            state = _CaptureState(td, validate=False)
            _, signature, compact, _ = state._generate_repro_file(
                gm,
                info,
                {"pattern_hash": "pattern", "shape_hash": "shape"},
                "repro.py",
                shape_env_block=block,
            )
        self.assertEqual(compact, [["F", 1.5, "zuf0", dtype]])
        self.assertEqual(signature, "(Sf(1.5))")

    def test_live_partition_capture_keeps_observed_symfloat(self):
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from capture_hook import extract_partition_subgraph

        shape_env = ShapeEnv()
        symfloat = shape_env.create_unbacked_symfloat()
        shape_env.real_tensor_prop_unbacked_vals[
            symfloat.node.expr] = 1.75

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.ones(3)
        scale = graph.placeholder("scale")
        scale.meta["val"] = symfloat
        mul = graph.call_function(
            torch.ops.aten.mul.Scalar, (x, scale))
        mul.meta["val"] = torch.ones(3)
        graph.output(mul)
        gm = torch.fx.GraphModule({}, graph)

        _, info, _, block = extract_partition_subgraph([mul], gm)
        self.assertEqual(info["scale"]["hint"], 1.75)
        self.assertEqual(info["scale"]["expr"], "zuf0")
        self.assertEqual(
            block["symbols"]["zuf0"]["observed_value"], 1.75)

    def test_symfloat_merge_canonicalization_is_a_fixed_point(self):
        from merge_captures import (
            _canonicalize_symbols,
            _family_constraints_component,
            _hintfree_inputs_signature,
        )

        symbols = {
            "zuf9": {
                "kind": "symfloat",
                "dtype": "float32",
                "range": [None, None],
                "unbacked": True,
                "observed_value": 1.25,
            },
        }
        inputs = [[
            "F", 1.25, "2.0*zuf9", "float32",
        ]]
        first = _canonicalize_symbols(
            symbols, inputs, ["zuf9 > 0.0"], {"zuf9": 1.25})
        second = _canonicalize_symbols(*first[:3], bindings=first[3])
        self.assertEqual(first, second)
        self.assertEqual(first[1][0][2], "2.0*s0")
        self.assertEqual(first[0]["s0"]["kind"], "symfloat")

        int_family = {
            "s0": {"range": [None, None], "unbacked": True},
        }
        self.assertNotEqual(
            _family_constraints_component(first[0], first[2]),
            _family_constraints_component(int_family, first[2]),
        )
        self.assertEqual(
            _hintfree_inputs_signature(
                [["F", 1.25, "s0", "float32"]]),
            _hintfree_inputs_signature(
                [["F", 2.5, "s0", "float32"]]),
        )

        from dynamic_shape_replay import (
            _distinct_dynamic_bindings_for_contract,
        )
        explicit_only = {
            "zuf0": {
                "kind": "symfloat",
                "dtype": "float32",
                "range": [0.0, 10.0],
                "unbacked": True,
            },
        }
        warm = _distinct_dynamic_bindings_for_contract(
            explicit_only, [], [{"zuf0": 1.25}], n=2)
        self.assertGreaterEqual(len(warm), 1)
        self.assertEqual(warm[0]["zuf0"], 1.25)

    def test_symfloat_capture_merge_load_is_a_fixed_point(self):
        from input_codec import spec_from_compact
        from merge_captures import _write_shapes_json
        from repro_harness import (
            load_shape_configs,
            make_inputs_from_config,
        )

        def write_point(root, shape_hash, observed):
            _write_shapes_json(
                root,
                shape_hash,
                "",
                "suite/infer/model",
                occurrences=1,
                inputs=[["F", observed, "zuf9", "float32"]],
                symbols={
                    "zuf9": {
                        "kind": "symfloat",
                        "dtype": "float32",
                        "range": [0.0, 4.0],
                        "unbacked": True,
                        "observed_value": observed,
                    },
                },
                guards=["zuf9 >= 0.0"],
            )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "repro.py").write_text("# stub\n")
            for shape_hash, observed in (
                ("point15", 1.5),
                ("point25", 2.5),
            ):
                write_point(root, shape_hash, observed)
            first = (root / "shapes.json").read_text()

            for shape_hash, observed in (
                ("point15", 1.5),
                ("point25", 2.5),
            ):
                write_point(root, shape_hash, observed)
            second = (root / "shapes.json").read_text()
            self.assertEqual(first, second)

            payload = json.loads(second)
            self.assertEqual(set(payload["symbols"]), {"s0"})
            self.assertEqual(
                [point["bindings"]["s0"] for point in payload["points"]],
                [1.5, 2.5],
            )
            self.assertEqual(
                [point["inputs"][0][2] for point in payload["points"]],
                ["s0", "s0"],
            )

            configs = load_shape_configs(root / "repro.py")
            values = sorted(
                make_inputs_from_config(config)[0]
                for config in configs.values()
            )
            self.assertEqual(values, [1.5, 2.5])

        malformed = (
            ["F", 1.0, "s0"],
            ["F", True, "s0", "float32"],
            ["F", 1.0, "", "float32"],
            ["F", 1.0, "s0", "float16"],
            ["flt", None, "float32"],
        )
        for entry in malformed:
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError):
                    spec_from_compact(entry)
                if entry[0] == "F":
                    from input_codec import evaluate_symbolic_entry
                    with self.assertRaises(ValueError):
                        evaluate_symbolic_entry(
                            entry, {"s0": 1.0},
                            {"s0": {"kind": "symfloat"}},
                        )

    def test_backed_symint_and_independent_symfloats_reuse_one_graph(self):
        class Mixed(torch.nn.Module):
            def forward(self, x, scale, bias):
                return x * scale + bias

        contract = {
            "symbols": {
                "s0": {"hint": 4, "range": [2, None]},
                "zuf0": {
                    "kind": "symfloat",
                    "dtype": "float32",
                    "range": [None, None],
                    "unbacked": True,
                    "observed_value": 1.5,
                },
                "zuf1": {
                    "kind": "symfloat",
                    "dtype": "float32",
                    "range": [None, None],
                    "unbacked": True,
                    "observed_value": 1.5,
                },
            },
            "guards": ["zuf0 > 0.0", "zuf1 < s0"],
            "points": [{
                "captured_dynamic": True,
                "bindings": {"s0": 4, "zuf0": 1.5, "zuf1": 1.5},
                "inputs": [
                    [["s0", 2], "f32"],
                    ["F", 1.5, "zuf0", "float32"],
                    ["F", 1.5, "zuf1", "float32"],
                ],
            }],
        }
        prepared = _prepare_dynamic_replay_from_contract(Mixed(), contract)
        self.assertEqual(prepared["kind"], "backed")
        self.assertEqual(
            prepared["float"]["float_names"], ("zuf0", "zuf1"))
        graphs = []

        def backend(gm, _example_inputs):
            graphs.append(gm)
            return gm.forward

        old_capture = torch._dynamo.config.capture_scalar_outputs
        old_fp64 = torch._inductor.config._use_fp64_for_unbacked_floats
        with _dynamic_replay_config(prepared):
            self.assertTrue(torch._dynamo.config.capture_scalar_outputs)
            self.assertFalse(
                torch._inductor.config._use_fp64_for_unbacked_floats)
            compiled = torch.compile(
                prepared["module"], backend=backend, fullgraph=True)
            for size, scale, bias in (
                (4, 1.5, 1.5),
                (6, 2.25, 0.5),
                (4, 1.5, 1.5),
            ):
                x = torch.ones(size, 2)
                binding = {"s0": size, "zuf0": scale, "zuf1": bias}
                args = _dynamic_replay_args(
                    [x, scale, bias], binding, prepared)
                torch.testing.assert_close(
                    compiled(*args),
                    torch.full_like(x, scale + bias),
                )
            invalid = _dynamic_replay_args(
                [torch.ones(4, 2), -1.0, 1.5],
                {"s0": 4, "zuf0": -1.0, "zuf1": 1.5},
                prepared,
            )
            with self.assertRaises((AssertionError, RuntimeError)):
                compiled(*invalid)
        self.assertEqual(
            torch._dynamo.config.capture_scalar_outputs, old_capture)
        self.assertEqual(
            torch._inductor.config._use_fp64_for_unbacked_floats, old_fp64)
        self.assertEqual(len(graphs), 1)
        torch._dynamo.reset()

    def test_unbacked_symint_and_symfloat_mixed_guard_reuses_one_graph(self):
        class Mixed(torch.nn.Module):
            def forward(self, x, scale):
                return x * scale

        contract = {
            "symbols": {
                "u0": {
                    "range": [2, None],
                    "unbacked": True,
                    "observed_value": 4,
                },
                "zuf0": {
                    "kind": "symfloat",
                    "dtype": "float32",
                    "range": [0.0, 10.0],
                    "unbacked": True,
                    "observed_value": 1.5,
                },
            },
            "guards": ["zuf0 < u0"],
            "points": [{
                "captured_dynamic": True,
                "bindings": {"u0": 4, "zuf0": 1.5},
                "inputs": [
                    [["u0", 2], "f32"],
                    ["F", 1.5, "zuf0", "float32"],
                ],
            }],
        }
        prepared = _prepare_dynamic_replay_from_contract(Mixed(), contract)
        self.assertEqual(prepared["kind"], "shape_env")
        graphs = []

        def backend(gm, _example_inputs):
            graphs.append(gm)
            return gm.forward

        with _dynamic_replay_config(prepared):
            compiled = torch.compile(
                prepared["module"], backend=backend, fullgraph=True)
            for size, scale in ((4, 1.5), (7, 2.5), (4, 1.5)):
                x = torch.ones(size, 2)
                binding = {"u0": size, "zuf0": scale}
                args = _dynamic_replay_args(
                    [x, scale], binding, prepared)
                torch.testing.assert_close(
                    compiled(*args), torch.full_like(x, scale))
        self.assertEqual(len(graphs), 1)
        torch._dynamo.reset()

    def test_symfloat_fp64_derived_expression_and_range_guard(self):
        class Scale(torch.nn.Module):
            def forward(self, x, scale):
                return x * scale

        contract = {
            "symbols": {
                "zuf0": {
                    "kind": "symfloat",
                    "dtype": "float64",
                    "range": [0.0, 3.0],
                    "unbacked": True,
                    "observed_value": 1.25,
                },
            },
            "guards": ["Ne(zuf0, 2.0)"],
            "points": [{
                "captured_dynamic": True,
                "inputs": [
                    [[2], "f64"],
                    ["F", 3.0, "2.0*zuf0 + 0.5", "float64"],
                ],
            }],
        }
        prepared = _prepare_dynamic_replay_from_contract(
            Scale(), contract)
        graphs = []

        def backend(gm, _example_inputs):
            graphs.append(gm)
            return gm.forward

        old_fp64 = torch._inductor.config._use_fp64_for_unbacked_floats
        with _dynamic_replay_config(prepared):
            self.assertTrue(
                torch._inductor.config._use_fp64_for_unbacked_floats)
            compiled = torch.compile(
                prepared["module"], backend=backend, fullgraph=True)
            for root in (1.25, 1.75, 1.25):
                scale = 2.0 * root + 0.5
                args = _dynamic_replay_args(
                    [torch.ones(2, dtype=torch.float64), scale],
                    {"zuf0": root},
                    prepared,
                )
                self.assertEqual(args[-1].dtype, torch.float64)
                torch.testing.assert_close(
                    compiled(*args),
                    torch.full((2,), scale, dtype=torch.float64),
                )
            invalid = _dynamic_replay_args(
                [torch.ones(2, dtype=torch.float64), 8.5],
                {"zuf0": 4.0},
                prepared,
            )
            with self.assertRaises((AssertionError, RuntimeError)):
                compiled(*invalid)
        self.assertEqual(
            torch._inductor.config._use_fp64_for_unbacked_floats,
            old_fp64,
        )
        self.assertEqual(len(graphs), 1)
        torch._dynamo.reset()

    def test_symfloat_reciprocal_expression_reuses_one_graph(self):
        class Scale(torch.nn.Module):
            def forward(self, x, scale):
                return x * scale

        contract = {
            "symbols": {
                "zuf0": {
                    "kind": "symfloat",
                    "dtype": "float32",
                    "range": [0.5, 3.0],
                    "unbacked": True,
                    "observed_value": 1.0,
                },
            },
            "guards": ["Ne(zuf0, 0.0)"],
            "points": [{
                "captured_dynamic": True,
                "inputs": [
                    [[3], "f32"],
                    ["F", 1.0, "1.0/zuf0", "float32"],
                ],
            }],
        }
        prepared = _prepare_dynamic_replay_from_contract(
            Scale(), contract)
        graphs = []

        def backend(gm, _example_inputs):
            graphs.append(gm)
            return gm.forward

        with _dynamic_replay_config(prepared):
            compiled = torch.compile(
                prepared["module"], backend=backend, fullgraph=True)
            for root in (1.0, 2.0, 1.0):
                args = _dynamic_replay_args(
                    [torch.ones(3), 1.0 / root],
                    {"zuf0": root},
                    prepared,
                )
                torch.testing.assert_close(
                    compiled(*args),
                    torch.full((3,), 1.0 / root),
                )
        self.assertEqual(len(graphs), 1)
        torch._dynamo.reset()

    def test_symfloat_guard_only_range_and_malformed_dtype(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x

        contract = {
            "symbols": {
                "zuf0": {
                    "kind": "symfloat",
                    "dtype": "float32",
                    "range": [0.0, 2.0],
                    "unbacked": True,
                    "observed_value": 1.0,
                },
            },
            "guards": [],
            "points": [{
                "captured_dynamic": True,
                "inputs": [[[2], "f32"]],
            }],
        }
        prepared = _prepare_dynamic_replay_from_contract(
            Identity(), contract)
        self.assertEqual(prepared["kind"], "symfloat")
        with _dynamic_replay_config(prepared):
            compiled = torch.compile(prepared["module"], fullgraph=True)
            valid = _dynamic_replay_args(
                [torch.ones(2)], {"zuf0": 1.0}, prepared)
            torch.testing.assert_close(compiled(*valid), torch.ones(2))
            invalid = _dynamic_replay_args(
                [torch.ones(2)], {"zuf0": 3.0}, prepared)
            with self.assertRaises((AssertionError, RuntimeError)):
                compiled(*invalid)
        torch._dynamo.reset()

        malformed = json.loads(json.dumps(contract))
        malformed["symbols"]["zuf0"]["dtype"] = "float16"
        with self.assertRaisesRegex(ValueError, "unsupported replay dtypes"):
            _prepare_dynamic_replay_from_contract(Identity(), malformed)
        with self.assertRaisesRegex(ValueError, "is above"):
            _prepare_dynamic_replay_from_contract(
                Identity(), contract, frozen={"zuf0": 3.0})
        with self.assertRaisesRegex(ValueError, "not numeric"):
            _dynamic_replay_args(
                [torch.ones(2)], {"zuf0": True}, prepared)

    def test_saved_full_graph_symfloat_roundtrip_and_execution(self):
        from full_graph_harness import (
            load_full_graph_sidecar,
            prepare_full_graph_execution,
        )
        from input_codec import compact_from_spec

        with tempfile.TemporaryDirectory() as td:
            graph_path = Path(td) / "full_graph_000.py"
            graph_path.write_text("""
import torch
class Repro(torch.nn.Module):
    def forward(self, x: "f32[3]cpu", scale: "Sym(zuf0)"):
        return torch.ops.aten.mul.Scalar(x, scale)
""")
            raw_inputs = [
                [[3], "f32", {
                    "n": "x",
                    "dev": "cpu",
                    "x": {"storage_nbytes": 12},
                }],
                ["F", 1.5, "zuf0", "float32"],
            ]
            graph_path.with_suffix(".meta.json").write_text(json.dumps({
                "schema_version": 2,
                "inputs": raw_inputs,
                "outputs": [[[3], "f32", {"dev": "cpu"}]],
                "tensor_attrs": {},
                "symbols": {
                    "zuf0": {
                        "kind": "symfloat",
                        "dtype": "float32",
                        "range": [None, None],
                        "unbacked": True,
                        "observed_value": 1.5,
                    },
                },
                "guards": [],
                "captured_dynamic": True,
            }))

            loaded = load_full_graph_sidecar(graph_path)
            reencoded = [
                compact_from_spec(spec, include_name=True)
                for spec in loaded["inputs"]
            ]
            loaded_again = [
                compact_from_spec(spec, include_name=True)
                for spec in load_full_graph_sidecar(graph_path)["inputs"]
            ]
            self.assertEqual(reencoded, loaded_again)

            execution = prepare_full_graph_execution(
                graph_path, default_device="cpu")
            graphs = []

            def backend(gm, _example_inputs):
                graphs.append(gm)
                return gm.forward

            compiled = execution.compile(
                backend=backend, fullgraph=True)
            for value in (1.5, 2.5, 1.5):
                args = execution.args({"zuf0": value})
                torch.testing.assert_close(
                    compiled(*args),
                    args[0] * value,
                )
            self.assertEqual(len(graphs), 1)
            torch._dynamo.reset()

    def test_saved_full_graph_fp64_with_frozen_and_live_symfloats(self):
        from full_graph_harness import prepare_full_graph_execution

        with tempfile.TemporaryDirectory() as td:
            graph_path = Path(td) / "full_graph_001.py"
            graph_path.write_text("""
import torch
class Repro(torch.nn.Module):
    def forward(
        self,
        x: "f64[2]cpu",
        scale: "Sym(zuf0)",
        bias: "Sym(zuf1)",
    ):
        mul = torch.ops.aten.mul.Scalar(x, scale)
        return torch.ops.aten.add.Scalar(mul, bias)
""")
            graph_path.with_suffix(".meta.json").write_text(json.dumps({
                "schema_version": 2,
                "inputs": [
                    [[2], "f64", {
                        "n": "x",
                        "dev": "cpu",
                        "x": {"storage_nbytes": 16},
                    }],
                    ["F", 1.25, "zuf0", "float64"],
                    ["F", 0.5, "zuf1", "float64"],
                ],
                "outputs": [[[2], "f64", {"dev": "cpu"}]],
                "tensor_attrs": {},
                "symbols": {
                    "zuf0": {
                        "kind": "symfloat",
                        "dtype": "float64",
                        "range": [0.0, 2.0],
                        "unbacked": True,
                        "observed_value": 1.25,
                    },
                    "zuf1": {
                        "kind": "symfloat",
                        "dtype": "float64",
                        "range": [0.0, 1.0],
                        "unbacked": True,
                        "observed_value": 0.5,
                    },
                },
                "guards": ["zuf0 + zuf1 < 3.0"],
                "captured_dynamic": True,
            }))

            execution = prepare_full_graph_execution(
                graph_path,
                default_device="cpu",
                frozen={"zuf1": 0.5},
            )
            graphs = []
            trace_configs = []

            def backend(gm, _example_inputs):
                graphs.append(gm)
                trace_configs.append((
                    torch._dynamo.config.capture_scalar_outputs,
                    torch._inductor.config
                    ._use_fp64_for_unbacked_floats,
                ))
                return gm.forward

            old_capture = torch._dynamo.config.capture_scalar_outputs
            old_fp64 = (
                torch._inductor.config._use_fp64_for_unbacked_floats)
            compiled = execution.compile(
                backend=backend, fullgraph=True)
            for root in (1.25, 1.75, 1.25):
                args = execution.args({"zuf0": root})
                self.assertEqual(args[-1].dtype, torch.float64)
                torch.testing.assert_close(
                    compiled(*args),
                    args[0] * root + 0.5,
                )
            self.assertEqual(len(graphs), 1)
            self.assertEqual(trace_configs, [(True, True)])
            self.assertEqual(
                torch._dynamo.config.capture_scalar_outputs,
                old_capture,
            )
            self.assertEqual(
                torch._inductor.config._use_fp64_for_unbacked_floats,
                old_fp64,
            )
            with self.assertRaisesRegex(ValueError, "above range max"):
                execution.args({"zuf0": 3.0})
            with self.assertRaisesRegex(ValueError, "contradicts frozen"):
                execution.args({"zuf0": 1.0, "zuf1": 0.25})
            torch._dynamo.reset()

    def test_shape_env_expression_parser_reuses_safe_boundary(self):
        from torch.fx.experimental.dynamic_spec import IntVar

        s0 = IntVar("s0", min=0)
        with self.assertRaises(ValueError):
            _eval_shape_env_expr("__import__(1)", {"s0": s0})

    def test_symbolic_stride_and_offset_are_runtime_invariants(self):
        class Scale(torch.nn.Module):
            def forward(self, x):
                return x * 2

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "repro.py").write_text("# stub")
            (d / "shapes.json").write_text(json.dumps({
                "symbols": {
                    "s0": {"hint": 4, "range": [0, None]},
                    "s1": {"hint": 4, "range": [2, None]},
                },
                "guards": [],
                "points": [{
                    "shape_hash": "metadata",
                    "captured_dynamic": True,
                    "bindings": {"s0": 4, "s1": 4},
                    "models": {"m": {"occurrences": 1}},
                    "inputs": [[
                        ["s1", 4], "f32",
                        {"st": ["2*s0", 1], "off": "s0"},
                    ]],
                }],
            }))
            (spec, names, input_count,
             checks, frozen) = _shape_env_spec_for_repro(str(d / "repro.py"))
            self.assertEqual(names, ("s0", "s1"))
            self.assertEqual([c[1] for c in checks],
                             ["stride", "storage_offset"])

            graphs = []

            def backend(gm, _example_inputs):
                graphs.append(gm)
                return gm.forward

            compiled = torch.compile(
                _shape_env_repro(
                    Scale(), input_count, spec, names, checks, frozen),
                backend=backend,
            )
            storage = torch.empty(4096)

            def invoke(s0, s1, stride=None, offset=None):
                stride = 2 * s0 if stride is None else stride
                offset = s0 if offset is None else offset
                tensor = storage.as_strided((s1, 4), (stride, 1), offset)
                return compiled(tensor, s0, s1)

            invoke(4, 4)
            invoke(6, 8)
            self.assertEqual(len(graphs), 1)
            with self.assertRaises((AssertionError, RuntimeError)):
                invoke(6, 8, stride=13)
            with self.assertRaises((AssertionError, RuntimeError)):
                invoke(6, 8, offset=7)
            torch._dynamo.reset()

    def test_stride_only_symbol_on_static_shape_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "repro.py").write_text("# stub")
            (d / "shapes.json").write_text(json.dumps({
                "symbols": {"s0": {"hint": 8, "range": [2, None]}},
                "guards": [],
                "points": [{
                    "shape_hash": "stride-only",
                    "captured_dynamic": True,
                    "bindings": {"s0": 8},
                    "models": {"m": {"occurrences": 1}},
                    "inputs": [[[4, 4], "f32", {"st": ["s0", 1]}]],
                }],
            }))
            with self.assertRaisesRegex(
                    ValueError, "cannot make stride/offset metadata dynamic"):
                _shape_env_spec_for_repro(str(d / "repro.py"))

    def test_partial_freeze_binds_composite_only_live_root(self):
        class Sum(torch.nn.Module):
            def forward(self, x):
                return x.sum()

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "repro.py").write_text("# stub")
            (d / "shapes.json").write_text(json.dumps({
                "symbols": {
                    "s0": {"hint": 4, "range": [2, None]},
                    "s1": {"hint": 8, "range": [2, None]},
                },
                "guards": [],
                "points": [{
                    "shape_hash": "composite",
                    "captured_dynamic": True,
                    "bindings": {"s0": 4, "s1": 8},
                    "models": {"m": {"occurrences": 1}},
                    "inputs": [[["s0*s1", 3], "f32"]],
                }],
            }))
            (spec, names, input_count,
             checks, frozen) = _shape_env_spec_for_repro(
                 str(d / "repro.py"), frozen={"s0": 4})
            self.assertIsNone(_backed_replay_plan_for_repro(
                str(d / "repro.py"), frozen={"s0": 4}))
            self.assertEqual(names, ("s1",))
            graphs = []

            def backend(gm, _example_inputs):
                graphs.append(gm)
                return gm.forward

            compiled = torch.compile(
                _shape_env_repro(
                    Sum(), input_count, spec, names, checks, frozen),
                backend=backend,
            )
            compiled(torch.empty(32, 3), 8)
            compiled(torch.empty(36, 3), 9)
            self.assertEqual(len(graphs), 1)
            with self.assertRaises(AssertionError):
                compiled(torch.empty(34, 3), 9)
            torch._dynamo.reset()

    def test_multiple_private_roots_are_read_in_decorated_forward_frame(self):
        """The first private scalar root must not specialize in a child frame."""
        class Sum(torch.nn.Module):
            def forward(self, x):
                return x.sum()

        contract = {
            "symbols": {
                "s0": {
                    "hint": 4,
                    "range": [2, None],
                    "unbacked": True,
                },
                "s1": {
                    "hint": 5,
                    "range": [2, None],
                    "unbacked": True,
                },
            },
            "guards": [],
            "points": [{
                "captured_dynamic": True,
                "bindings": {"s0": 4, "s1": 5},
                "inputs": [[["s0*s1", 3], "f32"]],
            }],
        }
        spec, names, count, checks, frozen = (
            _shape_env_spec_from_contract(contract)
        )
        graphs = []

        def backend(gm, _example_inputs):
            graphs.append(gm)
            return gm.forward

        torch._dynamo.reset()
        compiled = torch.compile(
            _shape_env_repro(Sum(), count, spec, names, checks, frozen),
            backend=backend,
        )
        for s0, s1 in ((4, 5), (6, 7), (4, 5)):
            compiled(torch.empty(s0 * s1, 3), s0, s1)
        self.assertEqual(len(graphs), 1)
        torch._dynamo.reset()


if __name__ == "__main__":
    unittest.main()
