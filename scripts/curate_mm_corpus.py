"""Curate captured MM regions into the four-category mm_repro corpus.

Normal-partition prologues/epilogues retain their whole views/casts and have no
window-size filter. Their families are the normal BB structural pattern hashes.
For horizontal/horizontal_combine, the existing metadata-proven graph
normalization, size limits and compute-DAG families remain unchanged.

Scalar-sensitive graph matching keeps executable variants distinct. One member
with the most source occurrences represents each family; only that member's
compatible shape points are exported. Different literals are never replayed
using the representative's code. The output is canonical/, report.json and
README.md, with no benchmark/study or diagnostic artifact trees.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch
from mm_corpus_dedup import (
    graph_digest,
    match_graphs,
    normalize_graph,
    semantic_graph,
    tensor_metadata,
    trace_repro,
)
from mm_corpus_io import (
    admits,
    filter_patterns,
    graph_metrics,
    load_records,
    write_json,
)
from torch.utils._pytree import tree_leaves

from capture_hook import canonical_node_order, lift_shape_params, pattern_hash_for_subgraph
from input_codec import compact_from_spec, spec_from_compact
from matmul_partitioner import (
    CATEGORIES,
    MatmulLimits,
    classify_pattern_graph,
    structure_key,
)
from mm_input_constraints import annotate_external_sqrt_inputs
from repro_harness import load_shape_configs, make_inputs_from_config
from roundtrip_validation import fake_tensor_ctx


def outputs(gm):
    output = next(n for n in gm.graph.nodes if n.op == "output")
    return [n for n in tree_leaves(output.args[0]) if isinstance(n, torch.fx.Node)]


def input_entries(gm, infos, nbytes, order=None):
    phs = order or [n for n in gm.graph.nodes if n.op == "placeholder"]
    groups, entries = {}, []
    for node in phs:
        if node.name not in infos:
            entries.append(["S", list(node.meta["val"])])
            continue
        info = copy.deepcopy(infos[node.name])
        info = {k: v for k, v in info.items() if not k.startswith("_")}
        gid = info.get("alias_group")
        if gid is not None:
            groups.setdefault(gid, len(groups))
            info["alias_group"] = groups[gid]
        entries.append(compact_from_spec(info))
    sizes = [nbytes[old] for old, _ in sorted(groups.items(), key=lambda kv: kv[1])]
    return entries, sizes


def canonicalize_seed_slots(gm):
    """Read every inductor_lookup_seed from slot 0.

    Which slot of the seed buffer a dropout reads does not change its kernel,
    but the slot number is a literal that would split identical patterns.
    """
    changed = False
    for node in gm.graph.nodes:
        if node.target is torch.ops.prims.inductor_lookup_seed.default and node.args[1] != 0:
            node.args = (node.args[0], 0)
            changed = True
    if changed:
        gm.recompile()


def shape_identity(inputs, nbytes, compile_policy=None):
    signature = {"inputs": inputs, "alias_group_nbytes": nbytes}
    if compile_policy is not None:
        signature["compile_policy"] = compile_policy
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:8], payload


def metadata_contract(gm):
    nodes = outputs(gm)
    values = [n.meta["val"] for n in nodes]
    return {
        "outputs": [tensor_metadata(v) for v in values],
        "alias_pairs": [
            [i, j]
            for i in range(len(values))
            for j in range(i + 1, len(values))
            if torch._C._is_alias_of(values[i], values[j])
        ],
    }


def validate_candidate(group, candidate, mapping, config):
    reference = group["gm"]
    phs = [n for n in reference.graph.nodes if n.op == "placeholder"]
    candidate_phs = [mapping[n] for n in phs]
    entries, sizes = input_entries(
        candidate["gm"],
        candidate["infos"],
        config.get("alias_group_nbytes", []),
        candidate_phs,
    )
    replay = {"inputs": [spec_from_compact(e) for e in entries]}
    if sizes:
        replay["alias_group_nbytes"] = sizes
    vals = reference(*make_inputs_from_config(replay))
    result = list(tree_leaves(vals))
    source_nodes = [mapping[n] for n in outputs(reference)]
    expected = [n.meta["val"] for n in source_nodes]
    if group.get("normal") and [str(v.device) for v in result] != [str(v.device) for v in expected]:
        return None
    if [tensor_metadata(v) for v in result] != [tensor_metadata(v) for v in expected]:
        return None
    aliases = lambda values: [
        [i, j]
        for i in range(len(values))
        for j in range(i + 1, len(values))
        if torch._C._is_alias_of(values[i], values[j])
    ]
    if aliases(result) != aliases(expected):
        return None
    return entries, sizes


def write_repro(gm, path, pattern):
    code = torch.fx.GraphModule(
        gm, copy.deepcopy(gm.graph), class_name="Repro"
    ).print_readable(print_output=False)
    header = f'''"""Curated MM repro. Pattern hash: {pattern}."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

'''
    tail = """
def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
"""
    path.write_text(header + code + tail)


def family_representatives(groups):
    """Group repros by (category, structure_key); pick one representative.

    structure_key includes the input/output dtypes. The representative is the
    member with the most source occurrences; only it is exported.
    """
    occurrences = lambda g: sum(
        m["occurrences"] for p in g["points"].values() for m in p["models"].values()
    )
    families = defaultdict(list)
    for group in groups:
        families[(group["pattern_category"], group["family"])].append(group)
    return {
        (g["pattern_category"], g["hash"])
        for members in families.values()
        for g in [min(members, key=lambda g: (-occurrences(g), g["hash"]))]
    }


def export_core(output, groups, rows, admitted_points, source_report):
    output.mkdir(parents=True)
    members = groups
    representatives = family_representatives(groups)
    groups = [g for g in groups if (g["pattern_category"], g["hash"]) in representatives]
    for group in groups:
        prefix = "_".join(
            sorted(
                {
                    str(n.target).split(".")[1]
                    for n in group["gm"].graph.nodes
                    if str(n.target).startswith("aten.")
                    and str(n.target).split(".")[1]
                    in {
                        "mm",
                        "bmm",
                        "addmm",
                        "baddbmm",
                        "mv",
                        "addmv",
                        "matmul",
                        "_scaled_mm",
                        "_int_mm",
                        "_grouped_mm",
                    }
                }
            )
        )
        directory = output / "canonical" / group["pattern_category"] / (prefix + "_" + group["hash"])
        directory.mkdir(parents=True)
        write_repro(group["gm"], directory / "repro.py", group["hash"])
        points = list(group["points"].values())
        write_json(directory / "shapes.json", {"points": points})
        meta = {
            "pattern_hash": group["hash"],
            "pattern_hash_mode": "scalar_literals_v1",
            "shape_hash_mode": "ordered_input_specs_v1",
            "kind": "matmul",
            "shape_points": len(points),
        }
        meta["pattern_category"] = group["pattern_category"]
        meta["pattern_family"] = group["family"]
        write_json(directory / "meta.json", meta)
    report = {
        "policy": source_report["policy"],
        "policy_id": source_report["policy_id"],
        "curation": {
            "input_source": "repros/models/**/full_graph_*.py",
            "version": "normal_absorption_and_metadata_windows_v2",
            "dependency_id": hashlib.sha256(
                b"".join(
                    (ROOT / name).read_bytes()
                    for name in (
                        "scripts/curate_mm_corpus.py",
                        "scripts/mm_corpus_dedup.py",
                        "scripts/mm_corpus_io.py",
                        "capture_hook.py",
                        "mm_input_constraints.py",
                        "input_codec.py",
                        "repro_harness.py",
                    )
                )
            ).hexdigest()[:16],
        },
        "unique_repros": len(groups),
        "shape_points": sum(len(g["points"]) for g in groups),
        "family_members": len(members),
        "deduplicated_repros": len(members) - len(groups),
        "region_occurrences": sum(
            (region["pattern_hash"], region["shape_hash"]) in admitted_points
            for row in rows for region in row["regions"]
        ),
        "sources": len(rows),
        "original_mm": sum(r["original_mm"] for r in rows),
    }
    report["pattern_categories"] = {
        category: {
            "repros": sum(g["pattern_category"] == category for g in groups),
            "shape_points": sum(
                len(g["points"]) for g in groups if g["pattern_category"] == category
            ),
            "family_members": sum(g["pattern_category"] == category for g in members),
        }
        for category in CATEGORIES
    }
    report["families"] = len(groups)
    write_json(output / "report.json", report)
    per_category = (
        "\n".join(
            f"- `{c}`: {v['repros']} repros (one per family; {v['family_members']} before dedup) / {v['shape_points']} shape points"
            for c, v in report["pattern_categories"].items()
        )
        + "\n\n"
    )
    (output / "README.md").write_text(
        f"# MM pattern corpus\n\n{report['unique_repros']} repros / {report['shape_points']} shape points / {report['region_occurrences']} source occurrences.\n\n{per_category}Regenerate with `python scripts/extract_matmul_corpus.py`. Prologue/epilogue use whole normal BB partitions with producer (epilogue) absorption first and one MM per augmented partition. Epilogues have no reduction-count limit. Horizontal windows retain their existing policy. Each family exports one scalar-preserving representative and only its compatible input points. There is no corpus-count cap.\n\n[Report](report.json)\n"
    )
    return report


def curate_corpus(source: Path, output: Path):
    """Build a fresh core-only corpus from the source capture staging tree."""
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("Use a fresh curation output directory")
    # Final size limits of an exported repro (the search allowed more).
    limits = MatmulLimits()
    records, rows = load_records(source)
    kept = filter_patterns(records, limits)
    raw_report = json.loads((source / "report.json").read_text())
    groups, buckets, admitted_points = [], defaultdict(list), set()
    torch.set_num_threads(1)
    processed = 0
    for record in kept:
        path = record["directory"] / "repro.py"
        points = json.loads(path.with_name("shapes.json").read_text())["points"]
        configs = list(load_shape_configs(str(path)).values())
        with fake_tensor_ctx():
            for point, config in zip(points, configs):
                gm, infos = trace_repro(path, config)
                original_contract = metadata_contract(gm)
                extracted = point["partition_metrics"]["category"]
                normal = extracted in {"prologue", "epilogue"}
                if normal:
                    # Keep whole normal partitions, including their views/casts.
                    # Restore shape parameters bound by trace_repro, without the
                    # old single-MM window normalization or size/quality filters.
                    gm, _shape_params = lift_shape_params(gm)
                    gm.graph.set_codegen(torch.fx.graph.CodeGen())
                    gm.recompile()
                else:
                    gm, infos, _shape_params = normalize_graph(gm, infos)
                annotate_external_sqrt_inputs(gm, infos)
                if not normal:
                    canonicalize_seed_slots(gm)
                candidate = {"gm": gm, "infos": infos}
                contract = metadata_contract(gm)
                if contract != original_contract:
                    raise ValueError(
                        (record["id"], "normalization changed output metadata/aliases")
                    )
                metrics = graph_metrics(gm)
                category = classify_pattern_graph(gm, expected=extracted if normal else None)
                if category is None or category != extracted:
                    raise ValueError(
                        (record["id"], "pattern category changed", extracted, category)
                    )
                if not normal and not admits(metrics, limits):
                    raise ValueError(
                        (record["id"], "normalized region violates size/topology limits")
                    )
                family = point["partition_metrics"]["normal_pattern_hash"] if normal else structure_key(gm)
                # Find an equivalent group whose code replays this point
                # exactly: for horizontal windows, same WL digest and an
                # isomorphism of the DAGs; for normal partitions, the same
                # pattern hash, whose canonical node order maps the graphs
                # onto each other whatever their FX node or output order.
                graph = semantic_graph(gm)
                digest = graph_digest(graph)
                pattern = pattern_hash_for_subgraph(gm, scalar_sensitive=True)
                bucket = (category, family, pattern) if normal else digest
                group = None
                mapping = None
                validation = None
                for existing in buckets[bucket]:
                    mapping = (
                        dict(zip(
                            canonical_node_order(existing["gm"], scalar_sensitive=True),
                            canonical_node_order(gm, scalar_sensitive=True),
                        ))
                        if normal
                        else match_graphs(existing["graph"], graph)
                    )
                    if mapping is None:
                        continue
                    validation = validate_candidate(
                        existing, candidate, mapping, config
                    )
                    if validation is not None:
                        group = existing
                        break
                if group is None:
                    if any(g["hash"] == pattern and g["pattern_category"] == category for g in groups):
                        raise ValueError(("unmatched strict hash collision", category, pattern))
                    group = {
                        "hash": pattern,
                        "gm": gm,
                        "graph": graph,
                        "points": {},
                        "pattern_category": category,
                        "family": family,
                        "normal": normal,
                    }
                    groups.append(group)
                    buckets[bucket].append(group)
                    mapping = {node: node for node in (gm.graph.nodes if normal else graph)}
                    validation = validate_candidate(group, candidate, mapping, config)
                    if validation is None:
                        raise ValueError(
                            "new representative fails own metadata contract"
                        )
                entries, sizes = validation
                shape, payload = shape_identity(
                    entries, sizes, config.get("compile_policy")
                )
                if (
                    shape in group["points"]
                    and group["points"][shape]["_signature"] != payload
                ):
                    raise ValueError(("shape hash collision", shape))
                target = group["points"].setdefault(
                    shape,
                    {
                        "shape_hash": shape,
                        "inputs": entries,
                        "models": {},
                        "partition_metrics": metrics,
                        "_signature": payload,
                    },
                )
                if sizes:
                    target["alias_group_nbytes"] = sizes
                if config.get("compile_policy"):
                    target["compile_policy"] = config["compile_policy"]
                for model, counts in point["models"].items():
                    item = target["models"].setdefault(model, {"occurrences": 0})
                    item["occurrences"] += counts.get("occurrences") or 0
                admitted_points.add((record["id"], point["shape_hash"]))
                processed += 1
        if processed % 50 < len(points):
            print(
                f"Normalized {processed} raw shape points; {len(groups)} groups",
                flush=True,
            )
    for group in groups:
        for point in group["points"].values():
            point.pop("_signature")
    report = export_core(output, groups, rows, admitted_points, raw_report)
    report["curation"]["raw_size_admitted_repros"] = len(kept)
    report["curation"]["raw_shape_points"] = processed
    write_json(output / "report.json", report)
    return report
