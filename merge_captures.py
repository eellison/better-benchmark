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
import ast
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


def _symbol_alloc_key(name: str):
    """Sort key that orders symbol names by ShapeEnv ALLOCATION order, not
    lexicographically. Dynamo allocates symbols off a monotonic counter, so the
    trailing integer reflects allocation (and thus role) order; a plain string
    sort puts 's10' before 's9' ('1' < '9'), which flips the canonical
    assignment across two dynamo namings of the SAME family. Key = (non-digit
    prefix, trailing int, name) so 's9' < 's10' and 'u0' groups apart from
    's0'; a name with no trailing digits sorts by name alone."""
    i = len(name)
    while i > 0 and name[i - 1].isdigit():
        i -= 1
    prefix = name[:i]
    num = int(name[i:]) if i < len(name) else -1
    return (prefix, num, name)


def _symbols_in_order(text) -> list:
    """Free symbol NAMES in an expr string, in a DETERMINISTIC order. sympy
    gives free_symbols as an unordered SET, so we order by ALLOCATION order
    (_symbol_alloc_key) — stable across dynamo namings and (unlike a str.find on
    the rendering) with no substring hazard (find('s1') would match inside
    's15'). NOT sympy's sort_key: that is lexicographic on the name, so within a
    compound slot where two symbols FIRST co-appear (a flattened 's_b*s_s' dim)
    it orders 's10' before 's9' and a re-run whose names straddle a digit
    boundary gets a different canonical assignment -> a different shape_hash for
    the same family. A bare int or non-symbolic slot yields []. For the common
    case (a symbol first appears as a BARE shape dim) the slot has one symbol
    and order is trivial."""
    if not isinstance(text, str):
        return []
    from input_codec import _sympify_expr
    expr = _sympify_expr(text)
    return [s.name for s in sorted(expr.free_symbols,
                                   key=lambda s: _symbol_alloc_key(s.name))]


def _canonical_symbol_rename(inputs, guards, extra_names=None) -> dict:
    """Build {original_name -> 's0','s1',...} by FIRST APPEARANCE across the
    inputs (then guards), left-to-right — the same discipline as ordering
    outputs by definition order. Makes the saved symbolic structure stable
    regardless of the names dynamo's ShapeEnv happened to allocate (s17/s15
    one run, s7/s92 another). Walks the compact input entries in forward
    order, each slot left-to-right: tensor shape dims, then strides, then
    ['I',hint,expr] symint exprs and ['S',[dims]] shape-param dims; finally
    guards. `extra_names` (the symbols-table keys) canonicalize LAST, name-
    sorted: trace-internal symbols referenced by no input expr or guard used
    to keep their raw names (s31/s79 leaking into bindings). Returns {} if
    there is nothing to rename."""
    order: list[str] = []
    seen: set[str] = set()

    def note(text):
        for name in _symbols_in_order(text):
            if name not in seen:
                seen.add(name)
                order.append(name)

    for e in (inputs or []):
        if not isinstance(e, list) or not e:
            continue
        if isinstance(e[0], list):                 # [shape, dtype, opts?]
            for d in e[0]:
                note(d)
            if len(e) > 2 and isinstance(e[2], dict):
                for s in e[2].get("st", []):
                    note(s)
                note(e[2].get("off"))
        elif e[0] == "I":                          # ['I', hint, expr]
            if len(e) > 2:
                note(e[2])
        elif e[0] == "S":                          # ['S', [dims...]]
            for d in e[1]:
                note(d)
    for g in (guards or []):
        note(g)
    for nm in sorted(extra_names or []):
        if nm not in seen:
            seen.add(nm)
            order.append(nm)

    # Already canonical (s0,s1,... in order) -> empty rename (idempotent).
    target = [f"s{i}" for i in range(len(order))]
    if order == target:
        return {}
    return {old: new for old, new in zip(order, target)}


def _canonicalize_symbols(symbols, inputs, guards, bindings=None):
    """Canonicalize a dynamic point's symbol NAMES to s0,s1,... by first
    appearance (see _canonical_symbol_rename), rewriting the coupled set
    (symbols table, inputs exprs, guards, bindings) consistently via the
    sympy-based renamers (substring-safe, simultaneous). Returns the rewritten
    (symbols, inputs, guards, bindings). A no-op when already canonical."""
    rename = _canonical_symbol_rename(inputs, guards,
                                      extra_names=(symbols or {}).keys())
    if not rename:
        return symbols, inputs, guards, bindings
    symbols = {rename.get(n, n): d for n, d in (symbols or {}).items()}
    inputs = _rename_symbols_in_inputs(inputs, rename)
    guards = [_rename_symbols_in_expr(g, rename) for g in (guards or [])] or guards
    if bindings is not None:
        bindings = {rename.get(n, n): v for n, v in bindings.items()}
    return symbols, inputs, guards, bindings


def _rename_symbols_in_expr(text, rename: dict) -> str:
    """Rename symbols in a sympy-printable expr string via sympy substitution
    (NOT string replace — 's0' must not corrupt 's0_other' or '64*s0'). A bare
    int passes through. Used to namespace a colliding capture's guards."""
    if not isinstance(text, str) or not rename:
        return text
    from input_codec import _sympify_expr
    import sympy
    expr = _sympify_expr(text)
    # Match free symbols BY NAME — _sympify_expr stamps assumptions
    # (integer/nonnegative) onto them, so a plain Symbol(old) won't compare
    # equal in a subs dict. Map the actual free symbols whose name is renamed.
    # xreplace, not subs: subs(dict) applies rules SEQUENTIALLY, so a swap
    # rename ({s1->s0, s0->s1}, possible when raw names are already sN in
    # permuted positions) would chain-corrupt; xreplace is exact-node and
    # simultaneous.
    sub = {s: sympy.Symbol(rename[s.name]) for s in expr.free_symbols
           if s.name in rename}
    return str(expr.xreplace(sub)) if sub else text


def _rename_symbols_in_inputs(inputs, rename: dict):
    """Rename symbols inside compact input entries (shape/stride expr strings,
    ['S',[...]] shape-param dims, ['I',hint,expr] symint exprs). Ints/static
    entries untouched. sympy-based per-slot so no substring corruption."""
    if not inputs or not rename:
        return inputs

    def _slot(d):
        return _rename_symbols_in_expr(d, rename) if isinstance(d, str) else d

    out = []
    for e in inputs:
        if not isinstance(e, list) or not e:
            out.append(e)
            continue
        if e[0] == "I":                      # ['I', hint, expr]
            out.append([e[0], e[1],
                        _slot(e[2]) if len(e) > 2 else e[2]] if len(e) > 2
                       else e)
        elif e[0] == "S":                    # ['S', [dims...]]
            out.append([e[0], [_slot(d) for d in e[1]]])
        elif isinstance(e[0], list):         # [shape, dtype, opts?]
            shape = [_slot(d) for d in e[0]]
            ne = [shape, e[1]]
            if len(e) > 2 and isinstance(e[2], dict):
                opts = dict(e[2])
                if "st" in opts:
                    opts["st"] = [_slot(s) for s in opts["st"]]
                if "off" in opts:
                    # A symbolic storage offset ('off') is part of the coupled
                    # set — _canonical_symbol_rename reads it for ordering, so
                    # it MUST be rewritten too, else it retains a raw dynamo
                    # name (unbound-symbol error at eval) or, worse, a name that
                    # now denotes a DIFFERENT canonical symbol (silent
                    # wrong-offset tensor).
                    opts["off"] = _slot(opts["off"])
                ne.append(opts)
            out.append(ne)
        else:
            out.append(e)
    return out


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
    family_identity: str | None = None,
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

    # Find existing point by shape_hash
    existing_point = None
    for point in data["points"]:
        if point.get("shape_hash") == shape_hash:
            existing_point = point
            break

    # CANONICALIZE symbol names to s0,s1,... by first appearance BEFORE anything
    # else touches them. dynamo's ShapeEnv allocates names off a global counter
    # (s17/s15 one run, s7/s92 another) so the SAME family recaptured twice
    # otherwise lands different symbol names -> non-idempotent shapes.json (the
    # recapture symbol-drift residual). Canonicalizing here makes the saved
    # symbolic structure name-stable, so collision detection + the binding
    # below compare canonical-to-canonical. Idempotent: already-canonical input
    # renames to itself (no-op).
    if symbols:
        symbols, inputs, guards, _ = _canonicalize_symbols(
            symbols, inputs, guards)

    # A dynamic point's symbols/guards/bindings/inputs are a COUPLED set: the
    # inputs' expr strings reference exactly these symbol names. Dynamo
    # REALLOCATES symbol names per trace context, so a second capture of the
    # same shape_hash can reuse `s0` for a DIFFERENT dim with a different
    # range/guard. A naive name-keyed union would clobber the existing s0's
    # range (rejecting/widening the first point) and cross-contaminate guards.
    # So: if any incoming symbol name collides with an existing top-level
    # symbol whose DEFINITION DIFFERS, namespace this capture's colliding
    # symbols (rename in symbols+bindings+guards+inputs) to fresh names before
    # merge. Same-definition reuse is genuine -> no rename (idempotent).
    if symbols:
        existing_syms_table = data.get("symbols") or {}
        existing_guards_list = data.get("guards") or []
        incoming_guards_list = guards or []

        def _guard_participation(nm, glist):
            # The set of guard exprs mentioning symbol `nm`. Compared name-to-
            # name (both sides use the same canonical name), so string equality
            # is meaningful. Two same-range symbols that participate in
            # DIFFERENT guards are structurally different — a real clash the
            # shared table can't hold, else the pooled `guards` union below
            # would apply one point's constraint to an unrelated point.
            return frozenset(g for g in glist if nm in _symbols_in_order(g))

        rename = {}
        for name, defn in symbols.items():
            other = existing_syms_table.get(name)
            # Collision = same canonical name but a STRUCTURALLY different
            # symbol. Compare RANGE *and* guard participation, NOT the hint:
            # post-canonicalization every point of the same family gets
            # s0,s1,... so two points differing only in BINDING (hint 8 vs 4)
            # are the SAME symbol at different points — they MUST share the name
            # (the whole point of canonicalization), with the per-point
            # `bindings` carrying the hint. A different range OR a different
            # guard set is a genuine clash the shared symbols table can't hold.
            if other is not None and (
                    _symbol_family_definition(other)
                    != _symbol_family_definition(defn)
                    or _guard_participation(name, existing_guards_list)
                    != _guard_participation(name, incoming_guards_list)):
                # Collision with a different definition -> fresh name. REUSE
                # this point's own prior namespacing instead of bumping past it
                # every re-merge: a candidate already in the table with the SAME
                # range is our previous rename (range is name-independent, so
                # guard strings — which carry the renamed name — don't factor
                # into the reuse test). Only a DIFFERENT-range occupant forces a
                # bump. This keeps namespacing idempotent (no sN__hash_i creep).
                cand = f"{name}__{shape_hash[:4]}"
                i = 1
                while ((cand in existing_syms_table
                        and _symbol_family_definition(
                            existing_syms_table[cand])
                        != _symbol_family_definition(defn))
                       or cand in symbols):
                    cand = f"{name}__{shape_hash[:4]}_{i}"
                    i += 1
                rename[name] = cand
        if rename:
            symbols = {rename.get(n, n): d for n, d in symbols.items()}
            guards = [_rename_symbols_in_expr(g, rename) for g in (guards or [])]
            inputs = _rename_symbols_in_inputs(inputs, rename)

    # The hint binding for this point — computed from the (possibly renamed)
    # symbols, so a namespaced symbol binds its NEW name.
    from input_codec import _symbol_point_value
    bindings = ({
        name: value
        for name, definition in symbols.items()
        if isinstance(
            (value := _symbol_point_value(definition)), int)
        and not isinstance(value, bool)
    } if symbols else None)
    required_bindings = (
        sorted(set(symbols) - set(bindings or {})) if symbols else [])

    # A second capture with a DIFFERENT OBSERVED symbol set is a distinct
    # template/point under the same shape hash. Find its prior entry on
    # re-merge (idempotence) rather than always comparing with the first hash
    # match and appending duplicates.
    existing_syms = set((existing_point or {}).get("bindings") or {})
    if symbols and existing_point is not None \
            and existing_syms != set(bindings or {}):
        existing_point = next(
            (
                point for point in data["points"]
                if point.get("shape_hash") == shape_hash
                and set(point.get("bindings") or {}) == set(bindings or {})
            ),
            None,
        )

    # Graph-level symbols/guards (shared across points). After collision
    # renaming above, names are now genuinely distinct OR identically-defined,
    # so update() is a safe union; guards de-duped. Each point's bindings
    # select its own symbols.
    if symbols:
        symbol_table = data.setdefault("symbols", {})
        for name, definition in symbols.items():
            if name not in symbol_table:
                symbol_table[name] = definition
                continue
            prior_hint = symbol_table[name].get("hint")
            prior_observed = symbol_table[name].get("observed_value")
            symbol_table[name].update(definition)
            if prior_hint is not None:
                symbol_table[name]["hint"] = prior_hint
            if prior_observed is not None:
                symbol_table[name]["observed_value"] = prior_observed
        if guards:
            existing_guards = data.setdefault("guards", [])
            for g in guards:
                if g not in existing_guards:
                    existing_guards.append(g)

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
        if bindings is not None and not existing_point.get("bindings"):
            # First dynamic write establishes the symbol<->input binding;
            # never overwrite it from a later (consistent) capture.
            existing_point["bindings"] = bindings
            existing_point["captured_dynamic"] = True
        if required_bindings:
            existing_point["requires_binding"] = required_bindings
        else:
            existing_point.pop("requires_binding", None)
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
        if symbols:
            new_point["bindings"] = bindings
            new_point["captured_dynamic"] = True
            if required_bindings:
                new_point["requires_binding"] = required_bindings
        data["points"].append(new_point)

    # Persist the family identity the dir was routed by, so later merges match
    # against the RECORDED identity instead of re-deriving it from a shared
    # (possibly namespaced) symbol table. Joins re-write the same value (they
    # only reach here by matching it); an identity-uncomputable entry passes
    # None and never clobbers a recorded one. INTERNAL routing cache, not a
    # public identifier (alignment §3c): consumers must not depend on the
    # digest algorithm, its width, or the identity components — changing the
    # algorithm requires versioning/migrating persisted values, never
    # comparing new digests against old ones.
    if family_identity:
        data["family_identity"] = family_identity

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


class _DropShapeAnnotations(ast.NodeTransformer):
    """Rewrite AnnAssign -> Assign (drop bare annotations) so the dump of a
    forward body is hint-blind. Emitted repro bodies annotate every assignment
    with the HINT-CONCRETIZED shape ('f32[64, 128, 4, 4]'), which differs
    between two bindings of the same family and can coincide between two
    different families — the opposite of what a family identity needs."""

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None  # pure annotation: no computation, no structure
        return ast.copy_location(
            ast.Assign(targets=[node.target], value=node.value,
                       type_comment=None),
            node)


def _forward_graph_dump(source_text: str) -> str | None:
    """Hint-blind structural dump of a repro's forward graph — ops, wiring,
    AND the baked constants (reshape targets, group split factors) that
    pattern_hash deliberately omits.

    pattern_hash is 'ops + wiring, IGNORING shapes', so two genuinely
    different symbolic graphs collide under it — e.g. a GroupNorm reshape to
    [64,32,2,mul] (channels=64) vs [64,32,4,mul] (channels=128). The dump
    keeps those call-arg constants but strips everything hint-valued: the
    forward's argument annotations AND the per-statement shape annotations
    (both render the hint binding, which must not split a family). Returns
    None when no forward is found OR the source does not parse — a truncated/
    corrupt capture file (a SIGKILL'd worker can leave one) must fall back to
    legacy grouping, not abort the whole model's merge mid-loop the way an
    escaping SyntaxError would (the legacy source-extraction path is likewise
    guarded)."""
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, ValueError):
        return None
    forward = next((n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "forward"),
                   None)
    if forward is None:
        return None
    body = [_DropShapeAnnotations().visit(stmt) for stmt in forward.body]
    body = [stmt for stmt in body if stmt is not None]
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]  # drop a leading docstring (documentation, not structure)
    return "".join(ast.dump(stmt) for stmt in body)


def _hintfree_inputs_signature(inputs) -> str:
    """Canonical-symbol input structure with per-point hint values removed.

    This is the second identity component: a pointwise region's forward BODY
    references no shapes at all (only annotations do), so [64,128,s0,s1] and
    [64,s0,s1,s2] emit byte-identical bodies. Their symbolic input structure
    is what tells the families apart. ['I', hint, expr] symint entries drop
    the hint slot (a binding value); tensor dims/strides are already exprs or
    family-invariant ints."""
    entries = []
    for e in (inputs or []):
        if isinstance(e, list) and e and e[0] == "I":
            entries.append(["I", e[2] if len(e) > 2 else None])
        else:
            entries.append(e)
    return json.dumps(entries, sort_keys=True)


def _symbol_family_definition(definition: dict | None) -> dict:
    """Hint-blind symbol metadata that changes the compiled family."""
    definition = definition or {}
    return {
        "range": definition.get("range"),
        "unbacked": definition.get("unbacked") is True,
        "optimization_hint": definition.get("optimization_hint"),
    }


def _family_constraints_component(symbols, guards) -> str:
    """Canonical, hint-free rendering of a family's CONSTRAINT surface:
    per-symbol ranges + the guard set. This is the third identity component
    (PR80 review finding 1): two captures with an identical body and input
    symbolization but different constraints (a divisibility guard, a pinned
    range) came from DIFFERENT compiled artifacts. Merging them pooled
    contradictory constraints in one dir and drove the symbol-collision
    namespacer to split the "family" into disjoint per-point symbols
    (s0 vs s0__hash), which breaks family-wide binding semantics — --bind
    s0=N must retarget EVERY point of a family. Hints are excluded (binding
    values, not structure); guards hash as a sorted SET — ShapeEnv records a
    guard once per evaluation, so a re-traced capture can carry the same
    guard twice, and _write_shapes_json already dedups on persist:
    multiplicity is not constraint semantics and must not split a family
    (PR80 re-review P2)."""
    definitions = {
        name: _symbol_family_definition(defn)
        for name, defn in (symbols or {}).items()
    }
    return (json.dumps(definitions, sort_keys=True)
            + "||" + json.dumps(sorted(set(guards or []))))


def _entry_family_identity(entry: dict, src_file: Path) -> str | None:
    """Family identity of an incoming dynamic capture: hint-blind forward
    body + canonical hint-free symbolic input signature + canonical
    constraint surface (ranges/guards). None when the source is missing/has
    no forward (caller falls back to legacy grouping)."""
    if not src_file.exists():
        return None
    dump = _forward_graph_dump(src_file.read_text())
    if dump is None:
        return None
    symbols = entry.get("symbols")
    if not symbols:
        # No symbolic dims => no dynamic FAMILY to split by. Return None so the
        # caller groups it like a static capture (legacy first-dir path). A
        # non-None identity here would never match the dir side — which reports
        # None for a symbol-less point (its write path leaves captured_dynamic
        # unset) — so every merge would mint a fresh __N dir (D5).
        return None
    inputs = entry.get("inputs")
    guards = entry.get("guards")
    symbols, inputs, guards, _ = _canonicalize_symbols(symbols, inputs, guards)
    sig = _hintfree_inputs_signature(inputs)
    constraints = _family_constraints_component(symbols, guards)
    return hashlib.md5(
        f"{dump}||{sig}||{constraints}".encode()).hexdigest()[:12]


def _dir_family_identity(repro_dir: Path) -> str | None:
    """Family identity of an existing canonical dir. Prefers the identity
    PERSISTED at merge time (shapes.json "family_identity") — recomputing it
    from a merged dir is fragile once the symbol table is shared (a pre-fix
    dir may hold namespaced symbols from a mistaken join). Falls back to
    recomputation from repro.py + the first dynamic point's (already
    canonical) inputs + the top-level constraint tables, for dirs written
    before persistence existed. None for static dirs (no dynamic point) — a
    dynamic capture never joins one, so an oracle-bearing static dir keeps
    its name and dynamic families split off."""
    repro_py = repro_dir / "repro.py"
    shapes_path = repro_dir / "shapes.json"
    if not repro_py.exists() or not shapes_path.exists():
        return None
    try:
        shapes = json.loads(shapes_path.read_text())
    except Exception:
        return None
    persisted = shapes.get("family_identity")
    if persisted:
        return persisted
    dump = _forward_graph_dump(repro_py.read_text())
    if dump is None:
        return None
    for point in shapes.get("points", []):
        if point.get("captured_dynamic"):
            sig = _hintfree_inputs_signature(point.get("inputs"))
            constraints = _family_constraints_component(
                shapes.get("symbols"), shapes.get("guards"))
            return hashlib.md5(
                f"{dump}||{sig}||{constraints}".encode()).hexdigest()[:12]
    return None


def _pattern_dirs(canonical_path: Path, pattern_hash: str) -> list[Path]:
    """Every canonical dir recorded under this pattern_hash (sorted, stable)."""
    dirs = []
    for meta_path in sorted(canonical_path.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("pattern_hash") == pattern_hash:
            dirs.append(meta_path.parent)
    return dirs


def _resolve_dynamic_family_dir(canonical_path: Path, pattern_hash: str,
                                dir_name: str,
                                entry_identity: str | None,
                                entry_is_symbolic: bool = False) -> Path:
    """Pick the canonical dir for a DYNAMIC capture, splitting by graph identity.

    A dynamic capture may only JOIN a pattern dir with the IDENTICAL family
    identity (same symbolic graph — its points then differ only by binding).
    A capture with a different identity (a baked constant or a different
    input symbolization that pattern_hash can't see) gets its OWN dir:
    `dir_name` if free, else `dir_name__2`, `__3`, ... The first family of a
    pattern keeps the plain name, so an oracle-bearing dir stays put.

    entry_identity=None (no computable identity): a NON-symbolic capture is
    effectively static and takes the legacy first-dir path. But a genuinely
    SYMBOLIC capture whose identity couldn't be computed (e.g. its source
    failed to parse) must NOT be poured into a STATIC oracle dir — that freezes
    a static repro.py over dynamic points and pools guards. Route it instead to
    the first EXISTING dynamic dir (one with a computable identity), else its
    own dir; this stays idempotent because the dir it writes gains a computable
    identity and is re-selected on the next merge (D6).

    CONCURRENCY: the __N suffix is allocated by reading the dirs on disk, so it
    is only race-free under the tool's single-process invocation — merge_one_
    capture materializes each resolved dir (mkdir) before the next entry
    resolves (main()'s loop is sequential). Two merge_captures processes writing
    the SAME --canonical-dir at once could both pick the same __N and clobber;
    callers running merges in parallel must shard by canonical root or serialize
    them (there is no cross-process lock here by design)."""
    existing = _pattern_dirs(canonical_path, pattern_hash)

    def _own_dir() -> Path:
        candidate = canonical_path / dir_name
        n = 2
        while candidate in existing or candidate.exists():
            candidate = canonical_path / f"{dir_name}__{n}"
            n += 1
        return candidate

    if entry_identity is None:
        if not entry_is_symbolic:
            return existing[0] if existing else canonical_path / dir_name
        # Symbolic but identity-uncomputable: prefer any existing DYNAMIC dir
        # (non-None identity) over a static one; else our own dir.
        for d in existing:
            if _dir_family_identity(d) is not None:
                return d
        return _own_dir()
    for d in existing:
        if _dir_family_identity(d) == entry_identity:
            return d  # same family -> join as a point
    return _own_dir()


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
        # Dynamic captures need graph-identity-aware grouping: pattern_hash is
        # shape-blind ("ops+wiring, IGNORING shapes"), so two symbolic graphs
        # that differ only in a baked constant (a GroupNorm reshape split
        # factor, a channel literal) collide under it. Merging them into one
        # dir freezes point-0's repro.py onto its siblings and pools their
        # contradictory guards. Route dynamic entries through identity-split;
        # leave the static corpus on the unchanged pattern-hash path.
        is_dynamic = bool(entry.get("captured_dynamic")
                          or entry.get("symbols") or entry.get("guards"))
        entry_identity = None
        if is_dynamic:
            entry_identity = _entry_family_identity(entry, Path(entry["file"]))
            repro_dir = _resolve_dynamic_family_dir(
                canonical_path, pattern_hash, dir_name, entry_identity,
                entry_is_symbolic=bool(entry.get("symbols")))
        else:
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
                               guards=entry.get("guards"),
                               family_identity=entry_identity)

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
