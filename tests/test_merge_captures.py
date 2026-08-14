"""Focused tests for merge_captures.py.

Usage:
    python scripts/test_merge_captures.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from merge_captures import merge_one_capture, temporary_capture_for_merge


def _write_capture(
    root: Path,
    name: str,
    *,
    pattern_hash: str,
    reduction_types: list[str],
) -> Path:
    capture_dir = root / name
    capture_dir.mkdir(parents=True, exist_ok=True)
    (capture_dir / "index.json").write_text(
        json.dumps(
            [
                {
                    "pattern_hash": pattern_hash,
                    "shape_hash": name,
                    "kind": "reduction",
                    "reduction_types": reduction_types,
                    "n_ops": 1,
                    "origin_ops": ["aten.sum.default"],
                    "file": str(root / f"missing_{name}.py"),
                }
            ]
        )
        + "\n"
    )
    return capture_dir


def _write_dynamic_capture(
    root: Path,
    name: str,
    *,
    pattern_hash: str,
    shape_hash: str,
    split_factor: int,
    hint: int,
    guards: list[str],
    inputs: list | None = None,
    srange: list | None = None,
) -> Path:
    """A synthetic DYNAMIC capture: one region whose emitted graph bakes
    `split_factor` into a reshape (the constant pattern_hash cannot see) and
    carries symbols/guards. Mirrors the opacus GroupNorm over-merge shape:
    same ops+wiring for every split_factor, different emitted graph. Like the
    real emitter, every assignment is annotated with the HINT-CONCRETIZED
    shape — the family identity must ignore those (they differ between two
    bindings of one family)."""
    capture_dir = root / name
    capture_dir.mkdir(parents=True, exist_ok=True)
    src = capture_dir / f"{name}.py"
    ann = f"f32[64, 32, {split_factor}, {hint * hint}]"
    src.write_text(
        "_repro_version = 2\n"
        "import torch\n"
        "\n"
        "\n"
        "class Repro(torch.nn.Module):\n"
        f"    def forward(self, arg0_1: \"f32[64, 64, {hint}, {hint}]\"):\n"
        f"        view: \"{ann}\" = torch.ops.aten.reshape.default("
        f"arg0_1, [64, 32, {split_factor}, -1])\n"
        f"        mul: \"{ann}\" = torch.ops.aten.mul.Tensor(view, 2)\n"
        "        return (mul,)\n"
    )
    entry = {
        "pattern_hash": pattern_hash,
        "shape_hash": shape_hash,
        "kind": "pointwise",
        "reduction_types": [],
        "n_ops": 2,
        "origin_ops": ["aten.reshape.default", "aten.mul.Tensor"],
        "file": str(src),
        "signature": "(T([64, 64, s0, s0], f32),)",
        "inputs": inputs if inputs is not None else [[[64, 64, "s0", "s0"], "f32"]],
        "symbols": {"s0": {"hint": hint, "range": srange or [2, None]}},
        "guards": guards,
        "captured_dynamic": True,
        "occurrences": 1,
    }
    (capture_dir / "index.json").write_text(
        json.dumps({"captured": [entry], "dropped": []}) + "\n"
    )
    return capture_dir


class MergeCapturesTests(unittest.TestCase):
    def test_same_pattern_hash_reuses_existing_canonical_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            pattern_hash = "abcdef123456"
            first = _write_capture(
                root,
                "first",
                pattern_hash=pattern_hash,
                reduction_types=["amax", "sum"],
            )
            second = _write_capture(
                root,
                "second",
                pattern_hash=pattern_hash,
                reduction_types=["sum", "amax"],
            )

            merge_one_capture(first, output, "ModelA", suite="hf", mode="train")
            merge_one_capture(second, output, "ModelB", suite="hf", mode="train")

            kept = output / "canonical" / f"amax_sum_{pattern_hash}"
            duplicate = output / "canonical" / f"sum_amax_{pattern_hash}"
            self.assertTrue(kept.exists())
            self.assertFalse(duplicate.exists())

            meta = json.loads((kept / "meta.json").read_text())
            self.assertEqual(meta["pattern_hash"], pattern_hash)
            # Model keys are fully qualified as suite/mode/model (matches the
            # shapes.json "models" keying); the bare-name form is retired.
            self.assertEqual(meta["models"], ["hf/train/ModelA", "hf/train/ModelB"])

    def test_temporary_capture_for_merge_removes_raw_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            pattern_hash = "deadbeef1234"

            with temporary_capture_for_merge(
                output,
                "ModelA",
                suite="hf",
                mode="infer",
                prefix="test_capture_",
            ) as capture:
                raw_dir = capture.capture_dir
                _write_capture(
                    raw_dir.parent,
                    raw_dir.name,
                    pattern_hash=pattern_hash,
                    reduction_types=["sum"],
                )

                self.assertEqual(capture.merge(), 1)
                self.assertTrue(raw_dir.exists())

            repro_dir = output / "canonical" / f"sum_{pattern_hash}"
            manifest = output / "models" / "hf" / "infer" / "ModelA" / "manifest.json"

            self.assertFalse(raw_dir.exists())
            self.assertTrue(repro_dir.exists())
            self.assertTrue(manifest.exists())
            self.assertFalse((output / "captures").exists())

    def test_temporary_capture_for_merge_requires_explicit_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"

            with temporary_capture_for_merge(
                output,
                "ModelA",
                suite="hf",
                mode="infer",
                prefix="test_capture_",
            ) as capture:
                raw_dir = capture.capture_dir
                _write_capture(
                    raw_dir.parent,
                    raw_dir.name,
                    pattern_hash="cafebabe1234",
                    reduction_types=["sum"],
                )

            self.assertFalse(raw_dir.exists())
            self.assertFalse((output / "canonical").exists())

    def test_graph_only_capture_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            capture_dir = root / "empty_capture"
            capture_dir.mkdir()
            model_dir = output / "models" / "hf" / "infer" / "ModelA"
            model_dir.mkdir(parents=True)
            (model_dir / "full_graph_000.py").write_text("# graph\n")
            (model_dir / "full_graph_000.meta.json").write_text("{}\n")

            merged = merge_one_capture(capture_dir, output, "ModelA", suite="hf", mode="infer")

            manifest = json.loads((model_dir / "manifest.json").read_text())
            self.assertEqual(merged, 0)
            self.assertEqual(manifest["patterns"], [])
            self.assertEqual(manifest["graphs"], ["full_graph_000.py"])
            self.assertEqual(
                manifest["graph_metadata"],
                {"full_graph_000.py": "full_graph_000.meta.json"},
            )

    def test_dynamic_divergent_graphs_split_dirs(self):
        """Two dynamic captures with the SAME pattern_hash but DIFFERENT
        emitted graphs (a baked reshape split factor) must land in separate
        canonical dirs — each with its own repro.py and its own guards. The
        over-merge regression: one dir froze point-0's repro.py onto siblings
        and pooled their contradictory guards."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            pattern_hash = "feedface0001"
            cap_a = _write_dynamic_capture(
                root, "dyn_a", pattern_hash=pattern_hash,
                shape_hash="aaaa1111", split_factor=2, hint=16,
                guards=["Eq(Mod(s0, 2), 0)"],
            )
            cap_b = _write_dynamic_capture(
                root, "dyn_b", pattern_hash=pattern_hash,
                shape_hash="bbbb2222", split_factor=4, hint=16,
                guards=["Eq(Mod(s0, 4), 0)"],
            )

            merge_one_capture(cap_a, output, "ModelA", suite="hf", mode="train")
            merge_one_capture(cap_b, output, "ModelB", suite="hf", mode="train")

            primary = output / "canonical" / f"pointwise_{pattern_hash}"
            split = output / "canonical" / f"pointwise_{pattern_hash}__2"
            self.assertTrue(primary.exists())
            self.assertTrue(split.exists())

            # Each family keeps ITS OWN emitted graph.
            self.assertIn("[64, 32, 2, -1]", (primary / "repro.py").read_text())
            self.assertIn("[64, 32, 4, -1]", (split / "repro.py").read_text())

            # Guards are NOT pooled across families.
            shapes_a = json.loads((primary / "shapes.json").read_text())
            shapes_b = json.loads((split / "shapes.json").read_text())
            self.assertEqual(shapes_a["guards"], ["Eq(Mod(s0, 2), 0)"])
            self.assertEqual(shapes_b["guards"], ["Eq(Mod(s0, 4), 0)"])
            self.assertEqual(len(shapes_a["points"]), 1)
            self.assertEqual(len(shapes_b["points"]), 1)

            # Both dirs carry the shared pattern_hash in meta.json.
            for d in (primary, split):
                meta = json.loads((d / "meta.json").read_text())
                self.assertEqual(meta["pattern_hash"], pattern_hash)

            # Idempotence: re-merging capture A joins its existing family —
            # no third dir, no new point.
            merge_one_capture(cap_a, output, "ModelA", suite="hf", mode="train")
            self.assertFalse(
                (output / "canonical" / f"pointwise_{pattern_hash}__3").exists()
            )
            shapes_a2 = json.loads((primary / "shapes.json").read_text())
            self.assertEqual(len(shapes_a2["points"]), 1)
            self.assertEqual(shapes_a2["guards"], ["Eq(Mod(s0, 2), 0)"])

    def test_dynamic_same_graph_different_binding_merges_as_points(self):
        """Two dynamic captures of the SAME family at different bindings
        (hints) must still merge into ONE dir as two points — the identity
        hash covers the graph body, not the per-point hint."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            pattern_hash = "feedface0002"
            cap_a = _write_dynamic_capture(
                root, "dyn_p16", pattern_hash=pattern_hash,
                shape_hash="cccc3333", split_factor=2, hint=16,
                guards=["Eq(Mod(s0, 2), 0)"],
            )
            cap_b = _write_dynamic_capture(
                root, "dyn_p32", pattern_hash=pattern_hash,
                shape_hash="dddd4444", split_factor=2, hint=32,
                guards=["Eq(Mod(s0, 2), 0)"],
            )

            merge_one_capture(cap_a, output, "ModelA", suite="hf", mode="train")
            merge_one_capture(cap_b, output, "ModelB", suite="hf", mode="train")

            primary = output / "canonical" / f"pointwise_{pattern_hash}"
            self.assertTrue(primary.exists())
            self.assertFalse(
                (output / "canonical" / f"pointwise_{pattern_hash}__2").exists()
            )

            shapes = json.loads((primary / "shapes.json").read_text())
            self.assertEqual(len(shapes["points"]), 2)
            bindings = sorted(p["bindings"]["s0"] for p in shapes["points"])
            self.assertEqual(bindings, [16, 32])
            self.assertEqual(shapes["guards"], ["Eq(Mod(s0, 2), 0)"])

    def test_dynamic_same_body_different_symbolization_splits(self):
        """Two dynamic captures whose forward BODIES are identical but whose
        symbolic INPUT structure differs ([64,64,s0,s0] vs [64,s0,s1,s2])
        must split — a pointwise body references no shapes, so the input
        signature is the only thing telling the families apart. This is the
        opacus e129 collision: two different families whose hint-concretized
        shapes coincide."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            pattern_hash = "feedface0004"
            cap_a = _write_dynamic_capture(
                root, "dyn_chan_static", pattern_hash=pattern_hash,
                shape_hash="eeee5555", split_factor=2, hint=8,
                guards=["Eq(Mod(s0, 2), 0)"],
            )
            cap_b = _write_dynamic_capture(
                root, "dyn_chan_dynamic", pattern_hash=pattern_hash,
                shape_hash="ffff6666", split_factor=2, hint=8,
                guards=["Eq(Mod(s0, 2), 0)"],
                inputs=[[[64, "s0", "s1", "s2"], "f32"]],
            )

            merge_one_capture(cap_a, output, "ModelA", suite="hf", mode="train")
            merge_one_capture(cap_b, output, "ModelB", suite="hf", mode="train")

            primary = output / "canonical" / f"pointwise_{pattern_hash}"
            split = output / "canonical" / f"pointwise_{pattern_hash}__2"
            self.assertTrue(primary.exists())
            self.assertTrue(split.exists())
            shapes_a = json.loads((primary / "shapes.json").read_text())
            shapes_b = json.loads((split / "shapes.json").read_text())
            self.assertEqual(len(shapes_a["points"]), 1)
            self.assertEqual(len(shapes_b["points"]), 1)
            self.assertEqual(shapes_a["points"][0]["shape_hash"], "eeee5555")
            self.assertEqual(shapes_b["points"][0]["shape_hash"], "ffff6666")

    def test_dynamic_same_body_different_guards_splits(self):
        """PR80 review finding 1: an identical body + input symbolization but
        a DIFFERENT guard set came from a different compiled artifact — it
        gets its own family dir. Pre-fix the second capture JOINED the dir
        and the collision namespacer split the 'family' into disjoint
        per-point symbols (s0 vs s0__hash), breaking family-wide binding
        semantics (--bind s0=N must retarget EVERY point of a family) and
        pooling separate constraint domains under one repro/oracle."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            ph = "feedface0005"
            cap_a = _write_dynamic_capture(
                root, "dyn_guarded", pattern_hash=ph,
                shape_hash="aaaa0001", split_factor=2, hint=8,
                guards=["Eq(Mod(s0, 8), 0)"])
            cap_b = _write_dynamic_capture(
                root, "dyn_unguarded", pattern_hash=ph,
                shape_hash="bbbb0002", split_factor=2, hint=8,
                guards=[])

            merge_one_capture(cap_a, output, "ModelA", suite="hf", mode="train")
            merge_one_capture(cap_b, output, "ModelB", suite="hf", mode="train")

            primary = output / "canonical" / f"pointwise_{ph}"
            split = output / "canonical" / f"pointwise_{ph}__2"
            self.assertTrue(primary.exists())
            self.assertTrue(split.exists())
            shapes_a = json.loads((primary / "shapes.json").read_text())
            shapes_b = json.loads((split / "shapes.json").read_text())
            # ONE family per dir: canonical symbols only, never namespaced.
            self.assertEqual(set(shapes_a["symbols"]), {"s0"})
            self.assertEqual(set(shapes_b["symbols"]), {"s0"})
            self.assertEqual(shapes_a["guards"], ["Eq(Mod(s0, 8), 0)"])
            self.assertEqual(shapes_b.get("guards", []), [])
            for shapes in (shapes_a, shapes_b):
                for p in shapes["points"]:
                    self.assertEqual(set(p["bindings"]), {"s0"})

    def test_dynamic_same_body_different_ranges_splits(self):
        """PR80 review finding 1 (range half): the same body + symbolization
        with a DIFFERENT symbol range ([2, None] vs [4, 64]) is a different
        constraint domain -> its own family dir, not a namespaced join."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            ph = "feedface0006"
            cap_a = _write_dynamic_capture(
                root, "dyn_open", pattern_hash=ph,
                shape_hash="aaaa0003", split_factor=2, hint=8,
                guards=[], srange=[2, None])
            cap_b = _write_dynamic_capture(
                root, "dyn_pinned", pattern_hash=ph,
                shape_hash="bbbb0004", split_factor=2, hint=8,
                guards=[], srange=[4, 64])

            merge_one_capture(cap_a, output, "ModelA", suite="hf", mode="train")
            merge_one_capture(cap_b, output, "ModelB", suite="hf", mode="train")

            primary = output / "canonical" / f"pointwise_{ph}"
            split = output / "canonical" / f"pointwise_{ph}__2"
            self.assertTrue(primary.exists())
            self.assertTrue(split.exists())
            shapes_a = json.loads((primary / "shapes.json").read_text())
            shapes_b = json.loads((split / "shapes.json").read_text())
            self.assertEqual(shapes_a["symbols"]["s0"]["range"], [2, None])
            self.assertEqual(shapes_b["symbols"]["s0"]["range"], [4, 64])
            self.assertNotIn("s0__bbbb", shapes_a["symbols"])

    def test_duplicate_guard_recording_does_not_split_family(self):
        """PR80 re-review P2: ShapeEnv can record the same guard once per
        evaluation, and _write_shapes_json dedups on persist — multiplicity
        is not constraint semantics. A duplicate-guard capture must JOIN the
        single-guard capture's family (identity hashes the guard SET), not
        mint __2 and defeat recapture coalescing."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            ph = "feedface0008"
            cap_a = _write_dynamic_capture(
                root, "dyn_dupguard", pattern_hash=ph,
                shape_hash="aaaa0007", split_factor=2, hint=8,
                guards=["Eq(Mod(s0, 2), 0)", "Eq(Mod(s0, 2), 0)"])
            cap_b = _write_dynamic_capture(
                root, "dyn_oneguard", pattern_hash=ph,
                shape_hash="bbbb0008", split_factor=2, hint=16,
                guards=["Eq(Mod(s0, 2), 0)"])

            merge_one_capture(cap_a, output, "ModelA", suite="hf", mode="train")
            merge_one_capture(cap_b, output, "ModelB", suite="hf", mode="train")

            primary = output / "canonical" / f"pointwise_{ph}"
            self.assertTrue(primary.exists())
            self.assertFalse(
                (output / "canonical" / f"pointwise_{ph}__2").exists())
            shapes = json.loads((primary / "shapes.json").read_text())
            self.assertEqual(len(shapes["points"]), 2)
            self.assertEqual(shapes["guards"], ["Eq(Mod(s0, 2), 0)"])
            self.assertEqual(set(shapes["symbols"]), {"s0"})

    def test_family_identity_persisted_and_preferred(self):
        """The identity a dynamic dir was routed by is PERSISTED in
        shapes.json and preferred over recomputation — reconstructing it
        later from a shared (possibly evolved) symbol table is fragile.
        Re-merge is a fixed point: same dir, same recorded identity."""
        from merge_captures import _dir_family_identity

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            ph = "feedface0007"
            cap_a = _write_dynamic_capture(
                root, "dyn_persist", pattern_hash=ph,
                shape_hash="aaaa0005", split_factor=2, hint=8,
                guards=["Eq(Mod(s0, 2), 0)"])

            merge_one_capture(cap_a, output, "ModelA", suite="hf", mode="train")
            primary = output / "canonical" / f"pointwise_{ph}"
            shapes = json.loads((primary / "shapes.json").read_text())
            ident = shapes.get("family_identity")
            self.assertTrue(ident)
            self.assertEqual(_dir_family_identity(primary), ident)

            # Persisted value wins over recomputation.
            tampered = dict(shapes)
            tampered["family_identity"] = "feedfeedfeed"
            (primary / "shapes.json").write_text(json.dumps(tampered))
            self.assertEqual(_dir_family_identity(primary), "feedfeedfeed")
            (primary / "shapes.json").write_text(json.dumps(shapes))

            # Idempotent re-merge: joins the same dir, identity unchanged.
            merge_one_capture(cap_a, output, "ModelA", suite="hf", mode="train")
            self.assertFalse(
                (output / "canonical" / f"pointwise_{ph}__2").exists())
            again = json.loads((primary / "shapes.json").read_text())
            self.assertEqual(again["family_identity"], ident)
            self.assertEqual(len(again["points"]), 1)

    def test_canonicalize_symbols_covers_trace_internal_names(self):
        """Symbols in the table that no input expr or guard references
        (trace-internal) must canonicalize too — they used to keep their raw
        dynamo names, leaking s31/s79 into bindings. Referenced symbols get
        s0.. by slot appearance; leftovers follow, name-sorted; and the
        result is a fixed point (idempotent)."""
        from merge_captures import _canonicalize_symbols

        symbols = {
            "s82": {"hint": 4, "range": [2, None]},   # referenced (dim)
            "s99": {"hint": 7, "range": [2, None]},   # trace-internal
            "s41": {"hint": 3, "range": [2, None]},   # trace-internal
        }
        inputs = [[[64, "s82"], "f32", {"st": ["s82", 1]}]]
        guards = []

        c_symbols, c_inputs, c_guards, _ = _canonicalize_symbols(
            symbols, inputs, guards)
        self.assertEqual(set(c_symbols), {"s0", "s1", "s2"})
        self.assertEqual(c_symbols["s0"]["hint"], 4)      # s82 -> s0 (slot)
        self.assertEqual(c_symbols["s1"]["hint"], 3)      # s41 -> s1 (name-sorted)
        self.assertEqual(c_symbols["s2"]["hint"], 7)      # s99 -> s2
        self.assertEqual(c_inputs, [[[64, "s0"], "f32", {"st": ["s0", 1]}]])

        again = _canonicalize_symbols(c_symbols, c_inputs, c_guards)
        self.assertEqual(again[0], c_symbols)
        self.assertEqual(again[1], c_inputs)

    def test_collision_namespacing_is_idempotent(self):
        """Re-merging a point whose symbol was namespaced (collision with an
        existing s0 of a DIFFERENT range) must REUSE that namespaced name, not
        bump past it (s0__hash -> s0__hash_1 -> ...) every merge. Otherwise the
        symbol set drifts, the point is re-added as a duplicate, and shapes.json
        never reaches a fixed point (D4/D3)."""
        from merge_captures import _write_shapes_json

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_shapes_json(
                d, "aaaa1111", "(T([s0], f32),)", "hf/train/m",
                occurrences=1, inputs=[[["s0"], "f32"]],
                symbols={"s0": {"hint": 8, "range": [2, None]}}, guards=[])

            def merge_b():
                _write_shapes_json(
                    d, "bbbb2222", "(T([s0], f32),)", "hf/train/m",
                    occurrences=1, inputs=[[["s0"], "f32"]],
                    symbols={"s0": {"hint": 4, "range": [4, None]}}, guards=[])

            merge_b()
            first = json.loads((d / "shapes.json").read_text())
            merge_b()
            merge_b()
            again = json.loads((d / "shapes.json").read_text())

            self.assertEqual(first, again, "shapes.json not a fixed point")
            self.assertIn("s0__bbbb", again["symbols"])
            self.assertNotIn("s0__bbbb_1", again["symbols"])  # no proliferation
            self.assertEqual(len(again["points"]), 2)  # no duplicate point

    def test_same_range_different_guards_is_a_collision(self):
        """The collision predicate compares range AND guard participation, not
        range alone. Two same-range symbols with DIFFERENT guard sets must
        namespace (the constrained point's guard must not be pooled onto the
        unconstrained one, which would invalidate its bindings) (D2)."""
        from merge_captures import _write_shapes_json

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_shapes_json(
                d, "aaaa1111", "(T([s0], f32),)", "hf/train/m",
                occurrences=1, inputs=[[["s0"], "f32"]],
                symbols={"s0": {"hint": 8, "range": [2, None]}},
                guards=["Eq(Mod(s0, 8), 0)"])
            _write_shapes_json(
                d, "bbbb2222", "(T([s0], f32),)", "hf/train/m",
                occurrences=1, inputs=[[["s0"], "f32"]],
                symbols={"s0": {"hint": 8, "range": [2, None]}},
                guards=[])  # same range, NO guard

            shapes = json.loads((d / "shapes.json").read_text())
            self.assertIn("s0__bbbb", shapes["symbols"])
            b_pt = next(p for p in shapes["points"]
                        if p["shape_hash"] == "bbbb2222")
            # B's unconstrained dim references its own namespaced symbol, so the
            # Mod-8 guard stays bound to A's s0 only.
            self.assertEqual(b_pt["inputs"], [[["s0__bbbb"], "f32"]])

    def test_empty_symbols_dynamic_entry_does_not_proliferate_dirs(self):
        """A captured_dynamic entry with an EMPTY symbols table is effectively
        static (no dynamic family). It must group like a static capture — one
        dir, stable across re-merges — not mint a fresh __N dir each time
        because its entry identity never matched the dir side (D5)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            ph = "feedface5550"
            cap = root / "dyn_empty"
            cap.mkdir(parents=True)
            src = cap / "dyn_empty.py"
            src.write_text(
                "_repro_version = 2\nimport torch\n\n\n"
                "class Repro(torch.nn.Module):\n"
                '    def forward(self, arg0_1: "f32[64, 64]"):\n'
                "        mul: \"f32[64, 64]\" = "
                "torch.ops.aten.mul.Tensor(arg0_1, 2)\n"
                "        return (mul,)\n")
            entry = {
                "pattern_hash": ph, "shape_hash": "abcd0001",
                "kind": "pointwise", "reduction_types": [], "n_ops": 1,
                "origin_ops": ["aten.mul.Tensor"], "file": str(src),
                "signature": "(T([64, 64], f32),)",
                "inputs": [[[64, 64], "f32"]],
                "symbols": {}, "guards": [], "captured_dynamic": True,
                "occurrences": 1,
            }
            (cap / "index.json").write_text(
                json.dumps({"captured": [entry], "dropped": []}) + "\n")

            for _ in range(3):
                merge_one_capture(cap, output, "ModelA",
                                  suite="hf", mode="train")

            dirs = sorted(p.name for p in (output / "canonical").iterdir())
            self.assertEqual(dirs, [f"pointwise_{ph}"])

    def test_static_capture_grouping_unchanged_by_dynamic_split(self):
        """Static captures (no symbols/guards) stay on the pattern-hash-only
        path: same pattern_hash -> same dir, even when their (missing-file)
        sources cannot be identity-hashed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            pattern_hash = "feedface0003"
            first = _write_capture(
                root, "static_one", pattern_hash=pattern_hash,
                reduction_types=["sum"],
            )
            second = _write_capture(
                root, "static_two", pattern_hash=pattern_hash,
                reduction_types=["sum"],
            )

            merge_one_capture(first, output, "ModelA", suite="hf", mode="train")
            merge_one_capture(second, output, "ModelB", suite="hf", mode="train")

            kept = output / "canonical" / f"sum_{pattern_hash}"
            self.assertTrue(kept.exists())
            self.assertEqual(
                len(list((output / "canonical").iterdir())), 1
            )
            meta = json.loads((kept / "meta.json").read_text())
            self.assertEqual(meta["models"], ["hf/train/ModelA", "hf/train/ModelB"])

    def test_explicit_suite_preserves_prefixed_model_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "repros"
            capture_dir = root / "empty_capture"
            capture_dir.mkdir()
            model_dir = output / "models" / "hf" / "train" / "hf_Foo_train"
            model_dir.mkdir(parents=True)
            (model_dir / "full_graph_000.py").write_text("# graph\n")

            merge_one_capture(
                capture_dir,
                output,
                "hf_Foo_train",
                suite="hf",
                mode="train",
            )

            self.assertTrue((model_dir / "manifest.json").exists())
            self.assertFalse(
                (output / "models" / "hf" / "train" / "Foo" / "manifest.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
