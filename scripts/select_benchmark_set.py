"""
Select a representative benchmark set from all canonical repros + shapes.

Strategy:
- Pointwise: group by (stride_pattern, size_quantile), pick 1 per group
- Reductions: group by (reduction_dim_quantile, grid_size_quantile, stride_pattern),
  pick 1 per group — more aggressive sampling since heuristic decisions depend on reduction dim

Uses quantile-based bucketing (no hardcoded thresholds) so the selection
adapts to whatever distribution of shapes exists.

Output: a frozen JSON manifest of (repro_dir, shape_config_key) pairs.
"""
import json
import math
from collections import defaultdict
from pathlib import Path


def _quantile_buckets(values: list[int | float], n_buckets: int) -> list[float]:
    """Compute bucket boundaries from data distribution."""
    if not values:
        return []
    s = sorted(values)
    boundaries = []
    for i in range(1, n_buckets):
        idx = int(len(s) * i / n_buckets)
        boundaries.append(s[idx])
    return boundaries


def _assign_bucket(value: int | float, boundaries: list[float]) -> int:
    """Assign a value to a bucket given boundaries."""
    for i, b in enumerate(boundaries):
        if value <= b:
            return i
    return len(boundaries)


def _stride_pattern(shape: list, stride: list | None) -> str:
    """Classify stride pattern."""
    if not stride:
        return "contiguous"
    if len(shape) == 4 and len(stride) == 4 and stride[1] == 1:
        return "channels_last"
    if 0 in stride:
        return "broadcast"
    return "strided"


def select_benchmark_set(canonical_dir: Path, n_size_buckets: int = 4,
                         n_rdim_buckets: int = 6,
                         max_shapes_pointwise: int = 3,
                         max_shapes_reduction: int = 8) -> list[dict]:
    """Select representative (repro, shape) pairs."""

    # First pass: collect all shape characteristics for quantile computation
    all_sizes = []
    all_rdims = []
    all_grid_sizes = []

    repro_data = []

    for d in sorted(canonical_dir.iterdir()):
        if not (d / 'repro.py').exists():
            continue
        meta_path = d / 'meta.json'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        kind = meta.get('kind', 'pointwise')
        is_reduction = kind == 'reduction'

        configs = {}
        sj = d / 'shapes.json'
        if sj.exists():
            data = json.loads(sj.read_text())
            configs = data.get('configs', {})

        config_info = []
        for config_key, config in configs.items():
            inputs = config.get('inputs', [])
            total_bytes = 0
            main_shape = []
            main_stride = None

            for inp in inputs:
                if inp.get('kind') == 'shape':
                    continue
                shape = inp.get('shape', [])
                dtype_str = inp.get('dtype', 'torch.float32')
                elem_size = 4 if 'float32' in dtype_str or 'int32' in dtype_str else 2 if 'float16' in dtype_str or 'bfloat16' in dtype_str else 8
                total_bytes += math.prod(shape) * elem_size if shape else 0
                if not main_shape and shape:
                    main_shape = shape
                    main_stride = inp.get('stride')

            rdim = main_shape[-1] if main_shape else 0
            grid_size = math.prod(main_shape[:-1]) if len(main_shape) > 1 else 1

            all_sizes.append(total_bytes)
            if is_reduction:
                all_rdims.append(rdim)
                all_grid_sizes.append(grid_size)

            config_info.append({
                'key': config_key,
                'total_bytes': total_bytes,
                'rdim': rdim,
                'grid_size': grid_size,
                'stride_pat': _stride_pattern(main_shape, main_stride),
            })

        repro_data.append({
            'dir': d,
            'name': d.name,
            'kind': kind,
            'is_reduction': is_reduction,
            'configs': config_info,
        })

    # Compute quantile boundaries from actual data
    size_boundaries = _quantile_buckets(all_sizes, n_size_buckets)
    rdim_boundaries = _quantile_buckets(all_rdims, n_rdim_buckets)
    grid_boundaries = _quantile_buckets(all_grid_sizes, n_size_buckets)

    # Second pass: group and select
    selected = []

    for repro in repro_data:
        if not repro['configs']:
            selected.append({"repro": repro['name'], "shape": None, "kind": repro['kind']})
            continue

        if len(repro['configs']) == 1:
            selected.append({"repro": repro['name'], "shape": repro['configs'][0]['key'], "kind": repro['kind']})
            continue

        if repro['is_reduction']:
            # Reductions: include ALL shapes (heuristic decisions are shape-sensitive)
            for cfg in repro['configs']:
                selected.append({"repro": repro['name'], "shape": cfg['key'], "kind": repro['kind']})
        else:
            # Pointwise: group by (size_quantile, stride_pattern), pick median per group
            groups = defaultdict(list)
            for cfg in repro['configs']:
                size_bkt = _assign_bucket(cfg['total_bytes'], size_boundaries)
                stride_pat = cfg['stride_pat']
                group_key = (size_bkt, stride_pat)
                groups[group_key].append(cfg)

            picks = []
            for group_key in sorted(groups.keys()):
                members = sorted(groups[group_key], key=lambda x: x['total_bytes'])
                median_idx = len(members) // 2
                picks.append(members[median_idx]['key'])

            for config_key in picks[:max_shapes_pointwise]:
                selected.append({"repro": repro['name'], "shape": config_key, "kind": repro['kind']})

    return selected


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Select benchmark set")
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros/canonical"))
    parser.add_argument("--max-pointwise", type=int, default=3,
                        help="Max shapes per pointwise repro")
    parser.add_argument("--max-reduction", type=int, default=8,
                        help="Max shapes per reduction repro")
    parser.add_argument("--size-buckets", type=int, default=4,
                        help="Number of quantile buckets for total data size")
    parser.add_argument("--rdim-buckets", type=int, default=6,
                        help="Number of quantile buckets for reduction dimension")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/v1.json"),
                        help="Output benchmark set manifest")
    args = parser.parse_args()

    selected = select_benchmark_set(
        args.canonical_dir,
        n_size_buckets=args.size_buckets,
        n_rdim_buckets=args.rdim_buckets,
        max_shapes_pointwise=args.max_pointwise,
        max_shapes_reduction=args.max_reduction,
    )

    n_repros = len(set(s['repro'] for s in selected))
    n_reduction = sum(1 for s in selected if s['kind'] == 'reduction')
    n_pointwise = sum(1 for s in selected if s['kind'] != 'reduction')

    print(f"Selected {len(selected)} benchmark points from {n_repros} repros")
    print(f"  Reductions: {n_reduction}")
    print(f"  Pointwise: {n_pointwise}")
    print(f"  Estimated time (2 GPUs): {len(selected) * 0.5 / 60 / 2:.0f} min")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "version": "v1",
        "description": f"Auto-selected: {len(selected)} points from {n_repros} repros",
        "selection_params": {
            "max_pointwise": args.max_pointwise,
            "max_reduction": args.max_reduction,
            "size_buckets": args.size_buckets,
            "rdim_buckets": args.rdim_buckets,
        },
        "benchmarks": selected,
    }, indent=2) + "\n")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
