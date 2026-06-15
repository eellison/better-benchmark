"""
Merge captured repros into the canonical set.

Takes a capture directory (produced by capture_hook.py or extract_reductions.py)
and upserts regions into the canonical repro set.

Usage:
    python merge_captures.py /tmp/captures/my_model --canonical-dir repros/ --model-name my_model --suite hf --mode train

    # Merge multiple capture directories
    python merge_captures.py /tmp/captures/model_a /tmp/captures/model_b --canonical-dir repros/
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import re
import tempfile
from pathlib import Path


from typing import Iterator

from canonicalize_repros import (
    extract_docstring,
    extract_imports,
    extract_make_inputs_body,
    extract_repro_class,
    generate_canonical_repro,
    parse_make_inputs,
    _spec_to_T,
)


# Regex to extract _shapes_config from v2 repro source files.
# Matches: _shapes_config = "(...)"  (single or double quotes, possibly multi-line)
_SHAPES_CONFIG_RE = re.compile(
    r'^_shapes_config\s*=\s*["\'](.+?)["\']\s*$',
    re.MULTILINE,
)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via temp + rename so a SIGKILL mid-write (run_recapture kills
    workers at timeout+60) can never leave a truncated file in the SHARED
    canonical tree — a poisoned meta.json/shapes.json fails every later
    model that touches the same pattern dir."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _extract_shapes_config(src_path: Path) -> str | None:
    """Extract the _shapes_config string from a v2 repro source file.

    This is the 0d fix: instead of trying to execute parse_shapes_config
    (which requires repro_harness importable), we extract the literal string
    directly from the source via regex.
    """
    if not src_path.exists():
        return None
    text = src_path.read_text()
    m = _SHAPES_CONFIG_RE.search(text)
    if m:
        return m.group(1)
    return None


def _compute_shape_hash(signature: str) -> str:
    """Compute the 8-hex-char shape hash from the signature string."""
    return hashlib.sha256(signature.encode()).hexdigest()[:8]


def _write_shapes_json(
    repro_dir: Path,
    shape_hash: str,
    signature: str,
    model_key: str,
    occurrences: int | None = None,
    inputs: list | None = None,
    alias_group_nbytes: list | None = None,
    symbols: dict | None = None,
    guards: list | None = None,
) -> None:
    """Write or update shapes.json for a canonical repro directory.

    Idempotent: re-merging the same (model_key, shape_hash) updates that
    model's entry in place. A new model on an existing point adds a key
    under "models". `occurrences` is the EXACT pre-dedup count of this
    (pattern, shape) point in the model's graphs (counted by the capture
    hook) — the accounting joins on it without needing a GPU retrace.

    `inputs` is the compact structured encoding (input_codec) — the DATA
    consumers parse. `signature` is its human-readable T()/S() rendering,
    kept for documentation and the repro.py default; never text-parsed
    when `inputs` is present.

    Schema (static/degenerate case — omits symbols/guards/bindings):
    {
      "points": [
        {"shape_hash": "<8hex>",
         "inputs": [[[128,512,7,7], "bf16", {"st": [...]}], ["S", [128]]],
         "signature": "<rendered T()/S() doc string>",
         "models": {"<suite>/<mode>/<model>": {"occurrences": 7}},
         "source": "captured"}
      ]
    }

    Dynamic case (a region captured from a dynamic compilation): `symbols`
    and `guards` are GRAPH-LEVEL (top-level, shared across points — that is
    how _parse_shapes_json reads them), and each dynamic point carries its
    hint `bindings` + `captured_dynamic`. `inputs` then hold expr strings in
    symbolic slots (['I', hint, expr], '64*s0*s53' strides, ['S',[...,expr]]).
    {
      "symbols": {"s0": {"hint": 16, "range": [2, null]}, ...},
      "guards":  ["Eq(s0*s53*4096, 4096*s0*s53)", ...],
      "points": [
        {"shape_hash": "<8hex>", "inputs": [...], "captured_dynamic": true,
         "bindings": {"s0": 16, "s53": 16},
         "models": {...}}
      ]
    }
    """
    shapes_path = repro_dir / "shapes.json"

    if shapes_path.exists():
        data = json.loads(shapes_path.read_text())
    else:
        data = {"points": []}

    # Dynamic capture: symbols/guards are GRAPH-LEVEL (shared across points).
    # Merge them at the top of shapes.json — the consumer reads them there.
    # Per-symbol entries are keyed by name, so a re-merge that sees the same
    # symbols is idempotent; a genuinely new symbol is added.
    if symbols:
        data.setdefault("symbols", {}).update(symbols)
    if guards:
        existing_guards = data.setdefault("guards", [])
        for g in guards:
            if g not in existing_guards:
                existing_guards.append(g)
    # The hint binding for this point (one value per symbol) — what every
    # expr evaluates under to reproduce the captured snapshot shape.
    bindings = ({name: s["hint"] for name, s in symbols.items()}
                if symbols else None)

    # Find existing point by shape_hash
    existing_point = None
    for point in data["points"]:
        if point.get("shape_hash") == shape_hash:
            existing_point = point
            break

    if existing_point is not None:
        # Point exists — add/update this model's entry (a recapture with a
        # real count REPLACES a stale null from an older merge).
        models = existing_point.setdefault("models", {})
        if model_key not in models or occurrences is not None:
            models[model_key] = {"occurrences": occurrences}
        if inputs is not None and "inputs" not in existing_point:
            existing_point["inputs"] = inputs
        elif inputs is not None and existing_point.get("inputs") != inputs:
            # Richer recapture (alias tags) REPLACES a pre-alias inputs
            # list — never silently discard fidelity (review bug #2).
            def _has_alias(entries):
                return any(isinstance(e, list) and len(e) > 2
                           and isinstance(e[2], dict) and "alias" in e[2]
                           for e in entries)
            if _has_alias(inputs) and not _has_alias(
                    existing_point.get("inputs") or []):
                existing_point["inputs"] = inputs
        if alias_group_nbytes and not existing_point.get("alias_group_nbytes"):
            existing_point["alias_group_nbytes"] = alias_group_nbytes
        if bindings is not None:
            existing_point["bindings"] = bindings
            existing_point["captured_dynamic"] = True
    else:
        # New point. "inputs" is THE data (compact codec); render the
        # human-readable string on demand via input_codec.render_signature
        # — it is not stored (settled: no duplicative signature field).
        # "signature" is written ONLY as a legacy fallback when the entry
        # carries no structured inputs (pre-codec capture dirs).
        new_point = {
            "shape_hash": shape_hash,
            "models": {model_key: {"occurrences": occurrences}},
        }
        if inputs is not None:
            new_point["inputs"] = inputs
        else:
            new_point["signature"] = signature
        if alias_group_nbytes:
            # True allocation size (bytes) per alias group, captured from
            # the live storage — consumers allocate group buffers directly,
            # never re-derive size by scanning member offsets/spans.
            new_point["alias_group_nbytes"] = alias_group_nbytes
        if bindings is not None:
            new_point["bindings"] = bindings
            new_point["captured_dynamic"] = True
        data["points"].append(new_point)

    import copy as _copy
    from full_graph_harness import _OneLine, dumps_with_onelines

    marked = _copy.deepcopy(data)
    for point in marked.get("points", []):
        if isinstance(point.get("inputs"), list):
            point["inputs"] = [_OneLine(e) for e in point["inputs"]]
    _atomic_write_text(shapes_path, dumps_with_onelines(marked) + "\n")


@dataclass
class CaptureMerge:
    """Temporary raw-capture state for one canonical merge."""

    capture_dir: Path
    canonical_dir: Path
    model_name: str
    suite: str | None = None
    mode: str | None = None
    merged: int = 0

    def merge(self) -> int:
        """Merge the temporary capture into the canonical repro root."""
        self.merged = merge_one_capture(
            self.capture_dir,
            self.canonical_dir,
            self.model_name,
            suite=self.suite,
            mode=self.mode,
        )
        return self.merged


@contextmanager
def temporary_capture_for_merge(
    canonical_dir: Path,
    model_name: str,
    *,
    suite: str | None = None,
    mode: str | None = None,
    prefix: str = "repro_capture_",
) -> Iterator[CaptureMerge]:
    """Create temporary raw-capture state for a canonical merge.

    Pass ``capture.capture_dir`` to ``install_capture_hook`` or ``_CaptureState``.
    After a successful capture, call ``capture.merge()`` while still inside the
    context. Raw capture files are temporary state and are removed automatically
    when the context exits.
    """
    canonical_dir = Path(canonical_dir)
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        capture = CaptureMerge(
            capture_dir=Path(tmp),
            canonical_dir=canonical_dir,
            model_name=model_name,
            suite=suite,
            mode=mode,
        )
        yield capture


def _format_compact_config(label: str, input_specs: list[dict]) -> str:
    """Format input specs as compact one-liner: label: (T([...], f32), S([...]), ...)"""
    parts = []
    for spec in input_specs:
        if spec.get("kind") == "shape":
            parts.append(f"S({spec['dims']})")
        else:
            parts.append(_spec_to_T(spec))
    return f"{label}: ({', '.join(parts)})"


def _model_output_dir(
    canonical_dir: Path,
    model_name: str,
    suite: str = "other",
    mode: str | None = None,
) -> Path:
    models_dir = canonical_dir / "models"
    if mode:
        return models_dir / suite / mode / model_name
    return models_dir / suite / model_name


def _resolve_model_identity(
    model_name: str,
    suite: str | None,
    mode: str | None,
) -> tuple[str, str | None, str]:
    if suite is None:
        suite, inferred_mode, clean_name = _infer_suite_mode(model_name)
        if mode is None:
            mode = inferred_mode
    else:
        clean_name = model_name
    return suite, mode, clean_name


def _write_model_json(canonical_dir: Path, model_name: str, patterns: list[str],
                      suite: str = "other", mode: str | None = None):
    """Write per-model manifest.json into the model directory.

    Writes to models/<suite>/<mode>/<model_name>/manifest.json (or
    models/<suite>/<model_name>/manifest.json if mode is None).
    """
    out_dir = _model_output_dir(canonical_dir, model_name, suite, mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = out_dir / "manifest.json"

    # Merge with existing if present
    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text())
        patterns = sorted(set(existing.get("patterns", []) + patterns))
        graphs = existing.get("graphs", [])
    else:
        patterns = sorted(set(patterns))
        graphs = []

    # Discover full_graph files in the model dir
    graph_files = sorted(f.name for f in out_dir.glob("full_graph_*.py"))
    if graph_files:
        graphs = graph_files
    graph_metadata = {
        graph: f"{Path(graph).stem}.meta.json"
        for graph in graphs
        if (out_dir / f"{Path(graph).stem}.meta.json").exists()
    }

    manifest_data = {
        "schema_version": 1,
        "source": suite,
        "model": model_name,
        "patterns": patterns,
        "graphs": graphs,
    }
    if graph_metadata:
        manifest_data["graph_metadata"] = graph_metadata
    if mode:
        manifest_data["mode"] = mode

    _atomic_write_text(manifest_file, json.dumps(manifest_data, indent=2) + "\n")
    return out_dir


def _find_existing_pattern_dir(canonical_path: Path, pattern_hash: str) -> Path | None:
    """Return an existing canonical dir for pattern_hash, if one exists."""
    for meta_path in sorted(canonical_path.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("pattern_hash") == pattern_hash:
            return meta_path.parent
    return None


def _infer_suite_mode(model_name: str) -> tuple[str, str | None, str]:
    """Infer suite, mode, and clean name from a model label."""
    name = model_name

    if name.startswith("dynamo_"):
        name = name.replace("dynamo_", "")
        if "_inference" in name:
            return "hf", "infer", name.replace("_inference", "")
        return "hf", "train", name

    if name.startswith("hf_"):
        name = name.replace("hf_", "")
        if "_train" in name:
            return "hf", "train", name.replace("_train", "")
        return "hf", "infer", name

    if name.startswith("timm_"):
        name = name.replace("timm_", "")
        if "_training" in name:
            return "timm", "train", name.replace("_training", "")
        if "_inference" in name:
            return "timm", "infer", name.replace("_inference", "")
        return "timm", "infer", name

    if name.startswith("vllm_"):
        name = name.replace("vllm_", "")
        if "_inference" in name:
            name = name.replace("_inference", "")
        return "vllm", None, name

    if name.startswith("genai_") or name.startswith("tritonbench_"):
        return "genai", None, name

    if name.startswith("tlparse_"):
        return "tlparse", None, name.replace("tlparse_", "")

    # torchvision or other
    if "_training" in name:
        return "torchvision", "train", name.replace("_training", "")
    if "_inference" in name:
        return "torchvision", "infer", name.replace("_inference", "")

    return "other", None, name


def merge_one_capture(capture_dir: Path, canonical_dir: Path, model_name: str,
                      suite: str | None = None, mode: str | None = None):
    """Merge a single capture directory into the canonical set."""
    capture_dir = Path(capture_dir)
    canonical_dir = Path(canonical_dir)
    suite, mode, clean_name = _resolve_model_identity(model_name, suite, mode)

    index_path = capture_dir / "index.json"
    if not index_path.exists():
        model_dir = _model_output_dir(canonical_dir, clean_name, suite, mode)
        if any(model_dir.glob("full_graph_*.py")):
            model_dir = _write_model_json(canonical_dir, clean_name, [], suite, mode)
            print(f"  Model dir: {model_dir}")
        else:
            print(f"  No index.json in {capture_dir}, skipping")
        return 0

    with open(index_path) as f:
        index = json.load(f)
    # index.json v2 is {"captured": [...], "dropped": [...]}; v1 was a bare
    # list. Drops FAIL the merge: every drop ever observed was a pipeline
    # bug, and a partial model must never enter the canonical corpus.
    if isinstance(index, dict):
        entries = index.get("captured", [])
        dropped = index.get("dropped", [])
        if dropped:
            raise RuntimeError(
                f"refusing to merge {capture_dir}: {len(dropped)} dropped "
                f"region(s) in index.json — fix the capture bug and re-run. "
                f"First: {dropped[0].get('reason', '?')[:200]}"
            )
    else:
        entries = index

    canonical_path = canonical_dir / "canonical"
    canonical_path.mkdir(parents=True, exist_ok=True)

    merged_patterns = []
    merged = 0

    for entry in entries:
        pattern_hash = entry.get("pattern_hash")
        if not pattern_hash:
            continue

        shape_hash = entry.get("shape_hash", "unknown")
        kind = entry.get("kind", "region")
        reduction_types = entry.get("reduction_types", [])
        kind_label = "_".join(reduction_types[:3]) if reduction_types else kind
        dir_name = f"{kind_label}_{pattern_hash}"
        repro_dir = _find_existing_pattern_dir(canonical_path, pattern_hash)
        if repro_dir is None:
            repro_dir = canonical_path / dir_name
        repro_dir.mkdir(parents=True, exist_ok=True)

        # Update shapes.json. The signature travels as DATA in the index
        # entry (capture_hook stamps it at generation time) — never re-derive
        # it by regexing generated source (lossy text-parsing, banned).
        src_file = Path(entry["file"])
        signature = entry.get("signature")
        if signature is None:
            # Legacy capture dirs (pre-signature index entries) only.
            signature = _extract_shapes_config(src_file)
        if signature is None:
            # Fallback for v1 repros: build signature from input_specs
            input_specs = parse_make_inputs(src_file) if src_file.exists() else []
            if input_specs:
                parts = []
                for spec in input_specs:
                    if spec.get("kind") == "shape":
                        parts.append(f"S({spec['dims']})")
                    else:
                        parts.append(_spec_to_T(spec))
                signature = f"({', '.join(parts)})"

        if signature:
            # Build model key: suite/mode/model_name
            model_key = f"{suite}/{mode}/{clean_name}" if mode else f"{suite}/{clean_name}"
            point_hash = shape_hash[:8] if len(shape_hash) >= 8 else shape_hash
            _write_shapes_json(repro_dir, point_hash, signature, model_key,
                               occurrences=entry.get("occurrences"),
                               inputs=entry.get("inputs"),
                               alias_group_nbytes=entry.get("alias_group_nbytes"),
                               symbols=entry.get("symbols"),
                               guards=entry.get("guards"))

        # Update meta.json. Models recorded by QUALIFIED key
        # (suite/mode/name) — the same key shapes.json uses; bare names
        # collide across suites/modes (resnet18 exists in timm AND
        # torchbench).
        qual_key = f"{suite}/{mode}/{clean_name}" if mode else f"{suite}/{clean_name}"
        meta_path = repro_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if qual_key not in meta.get("models", []):
                meta["models"].append(qual_key)
                meta["models"].sort()
                meta["n_models"] = len(meta["models"])
                _atomic_write_text(meta_path, json.dumps(meta, indent=2))
        else:
            meta = {
                "pattern_hash": pattern_hash,
                "kind": kind,
                "reduction_types": reduction_types,
                "n_ops": entry.get("n_ops"),
                "origin_ops": entry.get("origin_ops", []),
                "n_models": 1,
                "models": [qual_key],
            }
            _atomic_write_text(meta_path, json.dumps(meta, indent=2))

        # Write canonical repro.py if it doesn't exist or is older-format
        # than the capture's. Version compare, NOT equality-with-v2 — the
        # old check treated v3 files as "stale v1" and shoved them through
        # the legacy text-extraction rebuild, corrupting them.
        _ver_re = re.compile(r"^_repro_version\s*=\s*(\d+)", re.MULTILINE)

        def _version_of(text: str) -> int:
            m = _ver_re.search(text)
            return int(m.group(1)) if m else 1

        repro_py = repro_dir / "repro.py"
        src_file = Path(entry["file"])
        src_text = src_file.read_text() if src_file.exists() else None
        src_ver = _version_of(src_text) if src_text else 1
        needs_write = not repro_py.exists()
        if not needs_write:
            needs_write = _version_of(repro_py.read_text()) < src_ver
        if needs_write:
            if src_text is not None:
                try:
                    # Capture-produced files (v2+) are AUTHORITATIVE: copy
                    # verbatim. The extraction rebuild below is for true
                    # v1 legacy captures only.
                    if src_ver >= 2:
                        _atomic_write_text(repro_py, src_text)
                    else:
                        repro_class = extract_repro_class(src_file)
                        docstring = extract_docstring(src_file)
                        imports = extract_imports(src_file)
                        fallback = extract_make_inputs_body(src_file)
                        # Compute shapes_config from input specs
                        input_specs = parse_make_inputs(src_file)
                        shapes_config = None
                        if input_specs:
                            parts = []
                            for spec in input_specs:
                                if spec.get("kind") == "shape":
                                    parts.append(f"S({spec['dims']})")
                                else:
                                    parts.append(_spec_to_T(spec))
                            shapes_config = f"({', '.join(parts)})"
                        if repro_class:
                            code = generate_canonical_repro(
                                repro_class, docstring, imports, fallback,
                                shapes_config=shapes_config,
                            )
                            _atomic_write_text(repro_py, code)
                except Exception as e:
                    print(f"  Warning: could not generate canonical repro for {dir_name}: {e}")

        merged_patterns.append(pattern_hash)
        merged += 1

    # Write per-model manifest
    model_dir = _model_output_dir(canonical_dir, clean_name, suite, mode)
    if merged_patterns or any(model_dir.glob("full_graph_*.py")):
        model_dir = _write_model_json(canonical_dir, clean_name, merged_patterns, suite, mode)
        print(f"  Model dir: {model_dir}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge capture directories into canonical repro set")
    parser.add_argument("capture_dirs", nargs="+", type=Path,
                        help="One or more capture directories (containing index.json)")
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros"),
                        help="Path to canonical repro set root")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name for manifest. If not set, inferred from directory name.")
    parser.add_argument("--suite", type=str, default=None,
                        choices=["hf", "timm", "vllm", "torchvision", "genai", "tritonbench", "torchbench", "other"],
                        help="Suite to categorize this model under.")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["train", "infer"],
                        help="Mode (training or inference). If not set, inferred from model name.")
    args = parser.parse_args()

    total = 0
    for capture_dir in args.capture_dirs:
        if not capture_dir.is_dir():
            print(f"Skipping {capture_dir} (not a directory)")
            continue

        model_name = args.model_name or capture_dir.name
        print(f"Merging {capture_dir} as '{model_name}'...")
        n = merge_one_capture(capture_dir, args.canonical_dir, model_name,
                              suite=args.suite, mode=args.mode)
        total += n
        print(f"  Merged {n} regions")

    print(f"\nTotal: {total} regions merged into {args.canonical_dir}/canonical/")


if __name__ == "__main__":
    main()
