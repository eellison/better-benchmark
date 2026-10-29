"""Normalize and export only meaningful core MM fusion regions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch
from mm_corpus_dedup import (
    graph_digest,
    has_rng,
    match_graphs,
    normalize_graph,
    semantic_graph,
    tensor_metadata,
    trace_repro,
)
from mm_corpus_features import analyze_graph
from mm_corpus_io import (
    admits,
    filter_patterns,
    load_records,
    write_json,
)
from mm_corpus_quality import classify_graph_quality
from torch.utils._pytree import tree_leaves

from capture_hook import pattern_hash_for_subgraph
from input_codec import compact_from_spec, spec_from_compact
from matmul_partitioner import MatmulLimits
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
    candidate_graph = candidate["graph"]
    reference_edges = sorted(
        (
            (data["output_index"], node)
            for node, data in group["graph"].nodes(data=True)
            if "output_index" in data
        ),
        key=lambda item: item[0],
    )
    permutation = (
        [
            candidate_graph.nodes[mapping[node]]["output_index"]
            for _, node in reference_edges
        ]
        if reference_edges
        else [0]
    )
    return entries, sizes, candidate_phs, permutation


def write_repro(gm, path, pattern):
    code = gm.print_readable(print_output=False)
    code = re.sub(
        r"^class \S+\(torch\.nn\.Module\):",
        "class Repro(torch.nn.Module):",
        code,
        count=1,
        flags=re.MULTILINE,
    )
    header = f'''"""Curated MM repro. Pattern hash: {pattern}."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

'''
    tail = """
def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
"""
    path.write_text(header + code + tail)


def export_core(output, groups, rows, mappings, source_report, limits):
    output.mkdir(parents=True)
    selected = groups
    hashes = {g["hash"] for g in selected}
    inventory, model_patterns = [], defaultdict(set)
    for group in selected:
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
                    }
                }
            )
        )
        directory = output / "canonical" / (prefix + "_" + group["hash"])
        directory.mkdir(parents=True)
        write_repro(group["gm"], directory / "repro.py", group["hash"])
        points = list(group["points"].values())
        write_json(directory / "shapes.json", {"points": points})
        meta = {
            "pattern_hash": group["hash"],
            "pattern_hash_mode": "scalar_literals_v1",
            "shape_hash_mode": "ordered_input_specs_v1",
            "kind": "matmul",
            "curation": "metadata_normalization_v1",
            "category": "core",
            "source_patterns": sorted(group["raw_hashes"]),
            "legacy_pattern_hashes": sorted(group["legacy_hashes"]),
            "normalizations": dict(group["changes"]),
            "origin_ops": sorted(
                str(n.target)
                for n in group["gm"].graph.nodes
                if n.op == "call_function"
            ),
            "shape_points": len(points),
            "coverage": sorted(group["coverage"]),
        }
        write_json(directory / "meta.json", meta)
        inventory.append(
            {
                "pattern_hash": group["hash"],
                "repro": str((directory / "repro.py").relative_to(output)),
                "shape_points": len(points),
                "traits": sorted(group["coverage"]),
            }
        )
        for point in points:
            for model in point["models"]:
                model_patterns[model].add(group["hash"])
    filtered_rows = copy.deepcopy(rows)
    for row in filtered_rows:
        kept = []
        for region in row["regions"]:
            entry = mappings.get((region["pattern_hash"], region["shape_hash"]))
            if entry is None or entry["pattern_hash"] not in hashes:
                continue
            kept.append(
                {
                    **region,
                    "raw_pattern_hash": region["pattern_hash"],
                    "raw_shape_hash": region["shape_hash"],
                    **entry,
                }
            )
        row["regions"] = kept
        row["source_partition_selected_mm"] = row.pop("selected_mm")
        row["exported_regions"] = len(kept)
        if "diagnostics" in row:
            row["source_partition_diagnostics"] = row.pop("diagnostics")
    write_json(output / "occurrences.json", filtered_rows)
    for model, patterns in model_patterns.items():
        write_json(
            output / "models" / model / "manifest.json",
            {"patterns": sorted(patterns), "category": "core"},
        )
    write_json(output / "inventory.json", inventory)
    coverage = {
        key: sum(key in g["coverage"] for g in selected)
        for key in {k for g in groups for k in g["coverage"]}
    }
    report = {
        "policy": source_report["policy"],
        "policy_id": source_report["policy_id"],
        "curation": {
            "input_source": "repros/models/**/full_graph_*.py",
            "size_limits": asdict(limits),
            "category": "core",
            "version": "metadata_normalization_v1",
            "dependency_id": hashlib.sha256(
                b"".join(
                    (ROOT / name).read_bytes()
                    for name in (
                        "scripts/curate_mm_corpus.py",
                        "scripts/mm_corpus_dedup.py",
                        "scripts/mm_corpus_quality.py",
                        "capture_hook.py",
                        "mm_input_constraints.py",
                        "input_codec.py",
                        "repro_harness.py",
                    )
                )
            ).hexdigest()[:16],
            "corpus_count_limit": None,
        },
        "unique_repros": len(selected),
        "shape_points": sum(len(g["points"]) for g in selected),
        "region_occurrences": sum(len(r["regions"]) for r in filtered_rows),
        "sources": len(rows),
        "original_mm": sum(r["original_mm"] for r in rows),
        "coverage_repros": dict(sorted(coverage.items())),
        "failures": [],
    }
    write_json(output / "report.json", report)
    write_json(
        output / "curation_mapping.json",
        [
            {"raw_pattern_hash": p, "raw_shape_hash": s, **m}
            for (p, s), m in mappings.items()
            if m["pattern_hash"] in hashes
        ],
    )
    (output / "README.md").write_text(
        f"# Core MM fusion corpus\n\n{report['unique_repros']} repros / {report['shape_points']} shape points / {report['region_occurrences']} source occurrences.\n\nRegenerate with `python scripts/extract_matmul_corpus.py`. Scalar values, input domains, strides, offsets and output aliases are preserved. Alignment/copy-only and infinity-mask regions are excluded by the partitioner. There is no corpus-count cap.\n\n[Report](report.json) · [Source input/output mappings](curation_mapping.json) · [Source occurrences](occurrences.json)\n"
    )
    return report


def curate_corpus(source: Path, output: Path):
    """Build a fresh core-only corpus from the source capture staging tree."""
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("Use a fresh curation output directory")
    limits = MatmulLimits(
        max_mm=3, max_compute=32, max_nodes=42, max_inputs=8, max_outputs=2
    )
    cache = Path("/tmp") / (
        "bb_curate_features_"
        + hashlib.sha256(str(source).encode()).hexdigest()[:16]
        + ".json"
    )
    records, rows = load_records(source, cache)
    kept = filter_patterns(records, limits)
    raw_report = json.loads((source / "report.json").read_text())
    groups, buckets, mappings = [], defaultdict(list), {}
    torch.set_num_threads(1)
    processed = 0
    for record in kept:
        path = record["directory"] / "repro.py"
        raw_meta = json.loads(path.with_name("meta.json").read_text())
        points = json.loads(path.with_name("shapes.json").read_text())["points"]
        configs = list(load_shape_configs(str(path)).values())
        with fake_tensor_ctx():
            for point, config in zip(points, configs):
                gm, infos, positions = trace_repro(path, config)
                original_contract = metadata_contract(gm)
                gm, infos, _shape_params, transforms, changes = normalize_graph(
                    gm, infos
                )
                annotate_external_sqrt_inputs(gm, infos)
                candidate = {"gm": gm, "infos": infos}
                contract = metadata_contract(gm)
                if contract != original_contract:
                    raise ValueError(
                        (record["id"], "normalization changed output metadata/aliases")
                    )
                quality = classify_graph_quality(gm)
                if quality["category"] != "core":
                    processed += 1
                    continue
                analysis = analyze_graph(gm)
                if not analysis["horizontal_shapes_match"] or not admits(
                    analysis["metrics"], limits
                ):
                    raise ValueError(
                        (
                            record["id"],
                            "normalized region violates size/topology limits",
                        )
                    )
                graph = semantic_graph(gm)
                candidate["graph"] = graph
                digest = graph_digest(graph)
                group = None
                mapping = None
                validation = None
                for existing in buckets[digest]:
                    if has_rng(gm) and record["id"] not in existing["raw_hashes"]:
                        continue
                    mapping = match_graphs(existing["graph"], graph)
                    if mapping is None:
                        continue
                    validation = validate_candidate(
                        existing, candidate, mapping, config
                    )
                    if validation is not None:
                        group = existing
                        break
                if group is None:
                    pattern = pattern_hash_for_subgraph(gm, scalar_sensitive=True)
                    if any(g["hash"] == pattern for g in groups):
                        raise ValueError(("unmatched strict hash collision", pattern))
                    group = {
                        "hash": pattern,
                        "gm": gm,
                        "graph": graph,
                        "points": {},
                        "raw_hashes": set(),
                        "legacy_hashes": set(),
                        "changes": Counter(),
                        "coverage": set(),
                    }
                    groups.append(group)
                    buckets[digest].append(group)
                    mapping = {node: node for node in graph}
                    validation = validate_candidate(group, candidate, mapping, config)
                    if validation is None:
                        raise ValueError(
                            "new representative fails own metadata contract"
                        )
                entries, sizes, mapped_phs, permutation = validation
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
                        "partition_metrics": analysis["metrics"],
                        "coverage": analysis["coverage"],
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
                group["raw_hashes"].add(record["id"])
                group["legacy_hashes"].add(
                    raw_meta.get("legacy_pattern_hash", record["id"])
                )
                group["changes"].update(changes)
                group["coverage"].update(analysis["coverage"])
                mappings[(record["id"], point["shape_hash"])] = {
                    "pattern_hash": group["hash"],
                    "shape_hash": shape,
                    "input_mapping": [
                        {
                            "raw_input_index": positions.get(n.name),
                            "shape_value": list(n.meta["val"])
                            if n.name not in infos
                            else None,
                            "views": transforms.get(n.name, []),
                        }
                        for n in mapped_phs
                    ],
                    "output_permutation": permutation,
                    "normalizations": changes,
                }
                processed += 1
        if processed % 50 < len(points):
            print(
                f"Normalized {processed} raw shape points; {len(groups)} groups",
                flush=True,
            )
    for group in groups:
        for point in group["points"].values():
            point.pop("_signature")
    report = export_core(output, groups, rows, mappings, raw_report, limits)
    report["curation"]["raw_size_admitted_repros"] = len(kept)
    report["curation"]["raw_shape_points"] = processed
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "mm_repro")
    args = parser.parse_args()
    report = curate_corpus(args.source, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
