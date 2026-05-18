"""
Select a representative benchmark set from all canonical repros + shapes.

Strategy:
- Pointwise: group by (stride_pattern, size_bucket), pick 1 per group
- Reductions: group by (reduction_dim_size, non_reduction_size_bucket, stride_pattern),
  pick 1 per group — more aggressive sampling since heuristic decisions depend on reduction dim

Output: a frozen JSON manifest of (repro_dir, shape_config_key) pairs.
"""
import json
import math
from collections import defaultdict
from pathlib import Path


def _size_bucket(total_bytes: int) -> str:
    """Coarse size bucket."""
    if total_bytes < 1_000_000:
        return "tiny"     # <1MB (launch-bound)
    elif total_bytes < 10_000_000:
        return "small"    # 1-10MB
    elif total_bytes < 100_000_000:
        return "medium"   # 10-100MB
    elif total_bytes < 1_000_000_000:
        return "large"    # 100MB-1GB
    else:
        return "huge"     # >1GB


def _reduction_dim_bucket(rdim: int) -> str:
    """Bucket the reduction dimension size."""
    if rdim <= 64:
        return "r_tiny"
    elif rdim <= 256:
        return "r_small"
    elif rdim <= 1024:
        return "r_medium"
    elif rdim <= 4096:
        return "r_large"
    else:
        return "r_huge"


def _stride_pattern(shape: list, stride: list | None) -> str:
    """Classify stride pattern."""
    if not stride:
        return "contiguous"
    # Check channels-last (stride[1] == 1 for 4D NHWC)
    if len(shape) == 4 and len(stride) == 4 and stride[1] == 1:
        return "channels_last"
    # Check if any stride is 0 (broadcast)
    if 0 in stride:
        return "broadcast"
    return "strided"


def _infer_reduction_dim(repro_path: Path) -> int | None:
    """Infer the reduction dimension size from the repro's forward method."""
    content = repro_path.read_text()

    # Look for var_mean/mean/sum with dim argument
    import re
    # Pattern: var_mean.correction(tensor, [2], ...) or mean.dim(tensor, [-1], ...)
    for m in re.finditer(r'(?:var_mean\.correction|mean\.dim|sum\.dim_IntList)\([^,]+,\s*\[([^\]]+)\]', content):
        dims_str = m.group(1)
        # Get the tensor shape from annotation
        # Find the first tensor arg annotation before this op
        break

    # Simpler: look at the forward signature for the main input shape
    fwd_match = re.search(r'def forward\(self,\s*\w+:\s*"[^"]*\[([^\]]+)\]"', content)
    if fwd_match:
        shape_str = fwd_match.group(1)
        try:
            dims = [int(x.strip()) for x in shape_str.split(',') if x.strip().isdigit()]
            if dims:
                # For reductions, last dim is usually the reduction dim
                return dims[-1]
        except:
            pass
    return None


def select_benchmark_set(canonical_dir: Path, max_shapes_pointwise: int = 3,
                         max_shapes_reduction: int = 5) -> list[dict]:
    """Select representative (repro, shape) pairs."""

    selected = []

    for d in sorted(canonical_dir.iterdir()):
        if not (d / 'repro.py').exists():
            continue

        meta_path = d / 'meta.json'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())

        kind = meta.get('kind', 'pointwise')
        is_reduction = kind == 'reduction'

        # Get all shape configs
        configs = {}
        sj = d / 'shapes.json'
        if sj.exists():
            data = json.loads(sj.read_text())
            configs = data.get('configs', {})

        if not configs:
            # Just default shape
            selected.append({"repro": d.name, "shape": None, "kind": kind})
            continue

        if len(configs) == 1:
            key = list(configs.keys())[0]
            selected.append({"repro": d.name, "shape": key, "kind": kind})
            continue

        # Multiple shapes — need to select representatives
        # Group by characteristics
        groups = defaultdict(list)

        for config_key, config in configs.items():
            inputs = config.get('inputs', [])

            # Compute total bytes
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

            size_bkt = _size_bucket(total_bytes)
            stride_pat = _stride_pattern(main_shape, main_stride)

            if is_reduction:
                # Group by reduction dim size + non-reduction size + strides
                rdim = main_shape[-1] if main_shape else 0
                non_rdim = math.prod(main_shape[:-1]) if len(main_shape) > 1 else 1
                rdim_bkt = _reduction_dim_bucket(rdim)
                non_rdim_bkt = _size_bucket(non_rdim * 4)  # rough bytes for grid size
                group_key = (rdim_bkt, non_rdim_bkt, stride_pat)
            else:
                # Pointwise: group by size + strides only
                group_key = (size_bkt, stride_pat)

            groups[group_key].append((config_key, total_bytes))

        # Pick one representative per group
        max_picks = max_shapes_reduction if is_reduction else max_shapes_pointwise
        picks = []
        for group_key, members in sorted(groups.items()):
            # Pick the median-sized member
            members.sort(key=lambda x: x[1])
            median_idx = len(members) // 2
            picks.append(members[median_idx][0])

        # Cap at max
        for config_key in picks[:max_picks]:
            selected.append({"repro": d.name, "shape": config_key, "kind": kind})

    return selected


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Select benchmark set")
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros/canonical"))
    parser.add_argument("--max-pointwise", type=int, default=3,
                        help="Max shapes per pointwise repro")
    parser.add_argument("--max-reduction", type=int, default=5,
                        help="Max shapes per reduction repro")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/v1.json"),
                        help="Output benchmark set manifest")
    args = parser.parse_args()

    selected = select_benchmark_set(args.canonical_dir, args.max_pointwise, args.max_reduction)

    # Stats
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
        "description": f"Auto-selected benchmark set: {len(selected)} points from {n_repros} repros",
        "selection_params": {
            "max_pointwise": args.max_pointwise,
            "max_reduction": args.max_reduction,
        },
        "benchmarks": selected,
    }, indent=2) + "\n")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
