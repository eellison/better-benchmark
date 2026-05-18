"""
Fix all shapes.json files to include shape param (S()) entries.
V3: Enhanced with better split inference using all config tensors and forward signature.
"""
import json
import re
from pathlib import Path
from math import prod
from copy import deepcopy

BASE = Path('/tmp/scratch_space/better_benchmark/repros/canonical')


def parse_default_make_inputs(content):
    """Parse _default_make_inputs() to extract ordered list of items."""
    match = re.search(r'def _default_make_inputs\(\):\s*\n\s*return \[(.+?)\n\s*\]', content, re.DOTALL)
    if not match:
        return None
    
    body = match.group(1)
    items = []
    tensor_idx = 0
    shape_param_idx = 0
    item_pos = 0

    for line in body.split('\n'):
        line = line.strip().rstrip(',')
        if not line or line.startswith('#'):
            continue

        tensor_match = re.match(r'^torch\.(randn|randint|zeros|ones|empty|rand)', line)

        if tensor_match:
            if 'randint' in line:
                shape_m = re.search(r'randint\([^,]+,\s*[^,]+,\s*\[([^\]]*)\]', line)
            else:
                shape_m = re.search(r'(?:randn|zeros|ones|empty|rand)\(\[([^\]]*)\]', line)

            if shape_m:
                dims_str = shape_m.group(1)
                shape = [int(x.strip()) for x in dims_str.split(',') if x.strip()] if dims_str.strip() else []
                items.append({"type": "tensor", "shape": shape, "tensor_idx": tensor_idx, "item_pos": item_pos})
                tensor_idx += 1
                item_pos += 1
        elif re.match(r'^\[', line):
            dims_match = re.match(r'^\[([^\]]*)\]', line)
            if dims_match and dims_match.group(1).strip():
                try:
                    dims = [int(x.strip()) for x in dims_match.group(1).split(',')]
                    items.append({"type": "shape_param", "value": dims, "sp_idx": shape_param_idx, "item_pos": item_pos})
                    shape_param_idx += 1
                    item_pos += 1
                except ValueError:
                    pass

    return items


def align_config_to_default(default_tensors, config_tensors):
    """Align config tensors to default tensors, handling missing scalars/seeds."""
    n_def = len(default_tensors)
    n_cfg = len(config_tensors)
    
    if n_def == n_cfg:
        return {i: i for i in range(n_def)}
    
    if n_cfg > n_def:
        return {i: i for i in range(n_def)}
    
    # Config has fewer tensors - greedy match
    mapping = {}
    cfg_idx = 0
    
    for def_idx in range(n_def):
        if cfg_idx >= n_cfg:
            mapping[def_idx] = None
            continue
        
        def_shape = default_tensors[def_idx]["shape"]
        cfg_shape = config_tensors[cfg_idx]["shape"]
        
        if def_shape == cfg_shape:
            mapping[def_idx] = cfg_idx
            cfg_idx += 1
        elif len(def_shape) == len(cfg_shape):
            # Same ndim - likely corresponds
            mapping[def_idx] = cfg_idx
            cfg_idx += 1
        elif len(def_shape) <= 1 or (def_shape and prod(def_shape) <= 1):
            # Scalar or tiny - likely omitted
            mapping[def_idx] = None
        elif len(def_shape) == 1 and def_shape[0] <= 256:
            # Small 1D tensor (like seed tensor) - might be omitted
            # Check if cfg tensor matches a later default better
            found_later = False
            for future_def in range(def_idx + 1, min(def_idx + 3, n_def)):
                if len(default_tensors[future_def]["shape"]) == len(cfg_shape):
                    mapping[def_idx] = None
                    found_later = True
                    break
            if not found_later:
                mapping[def_idx] = cfg_idx
                cfg_idx += 1
        else:
            # Check if skipping this default gives better alignment
            found_later = False
            for future_def in range(def_idx + 1, min(def_idx + 3, n_def)):
                if (len(default_tensors[future_def]["shape"]) == len(cfg_shape) or
                    default_tensors[future_def]["shape"] == cfg_shape):
                    mapping[def_idx] = None
                    found_later = True
                    break
            if not found_later:
                mapping[def_idx] = cfg_idx
                cfg_idx += 1
    
    # Handle remaining
    for def_idx in range(len(mapping), n_def):
        if cfg_idx < n_cfg:
            mapping[def_idx] = cfg_idx
            cfg_idx += 1
        else:
            mapping[def_idx] = None
    
    return mapping


def compute_shape_params(default_items, config_tensors):
    """Compute shape param values for a config given its tensors."""
    default_tensors = [item for item in default_items if item["type"] == "tensor"]
    shape_params = [item for item in default_items if item["type"] == "shape_param"]
    
    if not shape_params:
        return []
    
    # Align config tensors to default tensors
    mapping = align_config_to_default(default_tensors, config_tensors)
    
    # Check if ALL mapped config tensor shapes match default tensor shapes
    all_same = True
    for def_idx, cfg_idx in mapping.items():
        if cfg_idx is not None:
            if default_tensors[def_idx]["shape"] != config_tensors[cfg_idx]["shape"]:
                all_same = False
                break
    
    if all_same:
        return [sp["value"] for sp in shape_params]
    
    # Collect all config tensor shapes for heuristic splitting
    all_cfg_shapes = [ct["shape"] for ct in config_tensors]
    
    # Derive each shape param
    results = []
    for sp in shape_params:
        sp_val = sp["value"]
        sp_item_pos = sp.get("item_pos", 999)
        new_val = derive_shape_param(sp_val, default_tensors, config_tensors, mapping, all_cfg_shapes, sp_item_pos)
        if new_val is None:
            return None
        results.append(new_val)

    return results


def derive_shape_param(sp_val, default_tensors, config_tensors, mapping, all_cfg_shapes, sp_item_pos=999):
    """Derive a single shape param value for a config."""
    has_neg = -1 in sp_val

    # Strategy 1: Direct shape match with a default tensor
    # Only consider tensors that appear BEFORE this shape param in the items list
    # (shape params reshape preceding tensors, not following ones)
    for def_idx, def_t in enumerate(default_tensors):
        if def_t["shape"] == sp_val and def_t.get("item_pos", 0) < sp_item_pos:
            cfg_idx = mapping.get(def_idx)
            if cfg_idx is not None:
                return list(config_tensors[cfg_idx]["shape"])
            else:
                return list(sp_val)
    
    # Strategy 2: Product match
    found_product_match = False
    if not has_neg:
        sp_prod = prod(sp_val) if sp_val else 0

        for def_idx, def_t in enumerate(default_tensors):
            def_prod = prod(def_t["shape"]) if def_t["shape"] else 0
            if def_prod == sp_prod and def_prod > 0:
                found_product_match = True
                cfg_idx = mapping.get(def_idx)
                if cfg_idx is None:
                    continue

                cfg_shape = config_tensors[cfg_idx]["shape"]
                cfg_prod = prod(cfg_shape) if cfg_shape else 0

                if cfg_prod == sp_prod:
                    if cfg_shape == def_t["shape"]:
                        return list(sp_val)
                    result = derive_reshape(def_t["shape"], sp_val, cfg_shape, all_cfg_shapes,
                                           default_tensors, config_tensors, mapping)
                    if result is not None:
                        return result
                    continue

                result = derive_reshape(def_t["shape"], sp_val, cfg_shape, all_cfg_shapes,
                                       default_tensors, config_tensors, mapping)
                if result is not None:
                    return result
                # Config tensor has different product and we couldn't derive -> fail
                return None
    else:
        # Has -1
        known_dims = [x for x in sp_val if x != -1]
        known_prod = prod(known_dims) if known_dims else 0

        if known_prod == 0:
            return list(sp_val)

        # Try each candidate tensor (multiple may match by divisibility)
        for def_idx, def_t in enumerate(default_tensors):
            def_prod = prod(def_t["shape"]) if def_t["shape"] else 0
            if def_prod > 0 and def_prod % known_prod == 0:
                found_product_match = True
                cfg_idx = mapping.get(def_idx)
                if cfg_idx is None:
                    continue

                cfg_shape = config_tensors[cfg_idx]["shape"]
                cfg_prod = prod(cfg_shape) if cfg_shape else 0

                if cfg_prod == def_prod and cfg_shape == def_t["shape"]:
                    return list(sp_val)

                result = derive_reshape_with_neg(def_t["shape"], sp_val, cfg_shape, all_cfg_shapes,
                                               default_tensors, config_tensors, mapping)
                if result is not None:
                    return result
                # Try next candidate

    # If we found a product-matching tensor but couldn't derive, it means
    # the shape param depends on tensor sizes and can't be kept as default
    if found_product_match:
        return None

    # No product match found with any default tensor.
    # This shape param is likely:
    # (a) A static expand/view shape that doesn't depend on input tensor sizes, OR
    # (b) A cross-dimension regrouping that we can't derive
    #
    # Heuristic: if sp_val's product matches a CONFIG tensor's product, it probably
    # needs to be derived (case b) -> return None.
    # Otherwise, it's likely static (case a) -> use default.
    sp_prod_check = prod(x for x in sp_val if x != -1) if sp_val else 0
    if sp_prod_check > 0:
        for ct in config_tensors:
            ct_prod = prod(ct["shape"]) if ct.get("shape") else 0
            if ct_prod == sp_prod_check and ct["shape"] != sp_val:
                # Matches a config tensor's product but different shape -> needs derivation
                return None
            # Also check if neg-resolved product matches
            if has_neg and ct_prod > 0 and ct_prod % sp_prod_check == 0:
                return None

    return list(sp_val)


def derive_reshape(def_tensor_shape, sp_val, cfg_tensor_shape, all_cfg_shapes,
                   default_tensors, config_tensors, mapping):
    """Derive new shape param when reshaping a tensor with different shape."""
    if def_tensor_shape == cfg_tensor_shape:
        return list(sp_val)
    
    # Try unflatten mapping
    unflatten_map = find_unflatten_mapping(def_tensor_shape, sp_val)
    if unflatten_map is not None:
        result = apply_unflatten(unflatten_map, cfg_tensor_shape, sp_val, all_cfg_shapes,
                                default_tensors, config_tensors, mapping)
        if result is not None:
            return result
    
    # Try flatten mapping (sp_val has fewer dims)
    flatten_map = find_unflatten_mapping(sp_val, def_tensor_shape)
    if flatten_map is not None:
        if len(cfg_tensor_shape) == len(def_tensor_shape):
            # Apply flatten directly
            result = []
            for start, end in flatten_map:
                group_product = prod(cfg_tensor_shape[start:end])
                result.append(group_product)
            return result
    
    # Same ndim case: only apply if it's a simple permutation/identity of dims
    # (NOT a cross-dim regrouping like [batch, seq, heads*hd] -> [batch*heads, seq, hd])
    if len(def_tensor_shape) == len(sp_val) == len(cfg_tensor_shape):
        # Only handle the trivial case where unflatten_map would have been 1:1
        # i.e., each dim of tensor maps to exactly one dim of sp_val
        # This means def_tensor_shape == sp_val (handled above) OR
        # they differ only by a dimension that also changed in the config
        # Skip this for safety - it's handled by the product match above
        pass
    
    # Try: find config tensor with same ndim as sp_val and same product as cfg_tensor
    cfg_prod = prod(cfg_tensor_shape) if cfg_tensor_shape else 0
    for ct_shape in all_cfg_shapes:
        if len(ct_shape) == len(sp_val) and prod(ct_shape) == cfg_prod:
            return list(ct_shape)
    
    return None


def derive_reshape_with_neg(def_tensor_shape, sp_val, cfg_tensor_shape, all_cfg_shapes,
                           default_tensors, config_tensors, mapping):
    """Like derive_reshape but shape param has -1."""
    if def_tensor_shape == cfg_tensor_shape:
        return list(sp_val)
    
    # Resolve -1
    known_prod = prod(x for x in sp_val if x != -1)
    def_prod = prod(def_tensor_shape)
    if known_prod == 0:
        return list(sp_val)
    neg_val = def_prod // known_prod
    resolved = [neg_val if x == -1 else x for x in sp_val]
    
    # Derive using resolved
    new_resolved = derive_reshape(def_tensor_shape, resolved, cfg_tensor_shape, all_cfg_shapes,
                                  default_tensors, config_tensors, mapping)
    if new_resolved is None:
        return None
    
    # Put -1 back
    result = list(new_resolved)
    for i, x in enumerate(sp_val):
        if x == -1 and i < len(result):
            result[i] = -1
    return result


def find_unflatten_mapping(flat_shape, unflat_shape):
    """Find how unflat_shape dims group into flat_shape dims."""
    if not flat_shape or not unflat_shape:
        return None
    
    mapping = []
    ui = 0
    
    for fd in flat_shape:
        p = 1
        start = ui
        while ui < len(unflat_shape):
            p *= unflat_shape[ui]
            ui += 1
            if p == fd:
                break
            if p > fd:
                return None
        if p != fd:
            return None
        mapping.append((start, ui))
    
    if ui != len(unflat_shape):
        return None
    
    return mapping


def apply_unflatten(unflatten_map, cfg_flat_shape, sp_val, all_cfg_shapes,
                    default_tensors, config_tensors, mapping):
    """Apply unflatten mapping to config tensor to get new shape param."""
    if len(cfg_flat_shape) != len(unflatten_map):
        return None
    
    result = []
    
    for fi, (start, end) in enumerate(unflatten_map):
        n_dims = end - start
        cfg_dim = cfg_flat_shape[fi]
        
        if n_dims == 1:
            result.append(cfg_dim)
        else:
            default_group = sp_val[start:end]
            split = find_dim_split(cfg_dim, default_group, all_cfg_shapes,
                                  default_tensors, config_tensors, mapping)
            if split is None:
                return None
            result.extend(split)
    
    return result


def find_dim_split(cfg_dim, default_group, all_cfg_shapes,
                   default_tensors, config_tensors, mapping):
    """Figure out how to split cfg_dim into parts following the pattern of default_group."""
    n = len(default_group)
    def_prod = prod(default_group)
    
    if cfg_dim == def_prod:
        # Same value - use same split
        return list(default_group)
    
    if n == 2:
        d0, d1 = default_group

        # Key insight: d0 is typically "batch" (first dim), d1 is typically "seq" or "heads".
        # We use multiple heuristics with careful ordering.

        # Strategy A: Check if d0 (batch) appears as first dim of a 3D+ config tensor.
        # If so, batch stays constant and d1 scales.
        batch_candidates = set()
        for ct_shape in all_cfg_shapes:
            if len(ct_shape) >= 3:
                batch_candidates.add(ct_shape[0])

        if d0 in batch_candidates and cfg_dim % d0 == 0 and cfg_dim != d0:
            return [d0, cfg_dim // d0]

        # Strategy B: If d1 is small (looks like num_heads, <= 32) and we can find the
        # config's num_heads from a 2D tensor (position_bias pattern [positions, heads])
        if d1 <= 32:
            for ct_shape in all_cfg_shapes:
                if len(ct_shape) == 2 and ct_shape[1] != d1 and ct_shape[1] <= 32:
                    candidate_heads = ct_shape[1]
                    if candidate_heads > 1 and cfg_dim % candidate_heads == 0 and candidate_heads != cfg_dim:
                        return [cfg_dim // candidate_heads, candidate_heads]

        # Strategy C: d1 preserved across configs (appears in some config tensor shape)
        d1_in_config = any(d1 in ct_shape for ct_shape in all_cfg_shapes)
        if d1_in_config and cfg_dim % d1 == 0 and cfg_dim != d1:
            return [cfg_dim // d1, d1]

        # Strategy D: Keep d0 (batch typically stays constant across configs)
        if d0 > 0 and d0 < cfg_dim and cfg_dim % d0 == 0:
            return [d0, cfg_dim // d0]

        # Strategy E: Use a 3D+ tensor's first dim as batch (only if d0 doesn't divide)
        for batch in sorted(batch_candidates):
            if batch > 0 and batch < cfg_dim and cfg_dim % batch == 0:
                return [batch, cfg_dim // batch]

        # Strategy F: Keep d1
        if d1 > 1 and d1 < cfg_dim and cfg_dim % d1 == 0:
            return [cfg_dim // d1, d1]

        return None
    
    elif n == 3:
        d0, d1, d2 = default_group
        
        # Common patterns:
        # [batch, heads, seq] or [batch, height, width]
        
        # Strategy: keep last dim (often head_dim or width), derive rest
        if cfg_dim % d2 == 0:
            remaining = cfg_dim // d2
            if remaining % d1 == 0:
                return [remaining // d1, d1, d2]
            if remaining % d0 == 0:
                return [d0, remaining // d0, d2]
            # Look for batch in config tensors
            for ct_shape in all_cfg_shapes:
                if len(ct_shape) >= 3:
                    batch = ct_shape[0]
                    if batch > 0 and remaining % batch == 0:
                        return [batch, remaining // batch, d2]
        
        # Try keeping d0 and d2
        if cfg_dim % (d0 * d2) == 0:
            return [d0, cfg_dim // (d0 * d2), d2]
        
        # Try keeping d0 and d1
        if cfg_dim % (d0 * d1) == 0:
            return [d0, d1, cfg_dim // (d0 * d1)]
        
        return None
    
    elif n == 4:
        d0, d1, d2, d3 = default_group
        if cfg_dim % (d1 * d2 * d3) == 0:
            return [cfg_dim // (d1 * d2 * d3), d1, d2, d3]
        if cfg_dim % (d0 * d2 * d3) == 0:
            return [d0, cfg_dim // (d0 * d2 * d3), d2, d3]
        if cfg_dim % (d0 * d1 * d3) == 0:
            return [d0, d1, cfg_dim // (d0 * d1 * d3), d3]
        return None
    
    return None


def process_repro(repro_dir):
    """Process a single repro directory."""
    repro_path = repro_dir / 'repro.py'
    shapes_path = repro_dir / 'shapes.json'
    
    if not repro_path.exists() or not shapes_path.exists():
        return 0, 0, 0
    
    content = repro_path.read_text()
    if '_shape_param' not in content:
        return 0, 0, 0
    
    shapes_data = json.loads(shapes_path.read_text())
    
    items = parse_default_make_inputs(content)
    if items is None:
        return 0, 0, 0
    
    shape_params = [item for item in items if item["type"] == "shape_param"]
    if not shape_params:
        return 0, 0, 0
    
    configs_updated = 0
    configs_skipped = 0
    modified = False
    
    for config_name, config in shapes_data.get("configs", {}).items():
        if any(inp.get("kind") == "shape" for inp in config.get("inputs", [])):
            continue
        
        config_tensors = config["inputs"]
        
        new_shape_params = compute_shape_params(items, config_tensors)
        
        if new_shape_params is None or len(new_shape_params) != len(shape_params):
            configs_skipped += 1
            continue
        
        # Validate: check that products are consistent
        valid = True
        for sp_val in new_shape_params:
            if any(d == 0 for d in sp_val if d != -1):
                valid = False
                break
        if not valid:
            configs_skipped += 1
            continue
        
        # Build new inputs list
        default_tensors = [item for item in items if item["type"] == "tensor"]
        tensor_mapping = align_config_to_default(default_tensors, config_tensors)
        
        new_inputs = []
        sp_idx = 0
        
        for item in items:
            if item["type"] == "tensor":
                def_ti = item["tensor_idx"]
                cfg_ti = tensor_mapping.get(def_ti)
                if cfg_ti is not None:
                    new_inputs.append(config_tensors[cfg_ti])
            elif item["type"] == "shape_param":
                new_inputs.append({"kind": "shape", "dims": new_shape_params[sp_idx]})
                sp_idx += 1
        
        expected_tensors = sum(1 for di, ci in tensor_mapping.items() if ci is not None)
        expected_total = expected_tensors + len(new_shape_params)
        
        if len(new_inputs) == expected_total and sp_idx == len(new_shape_params):
            config["inputs"] = new_inputs
            configs_updated += 1
            modified = True
        else:
            configs_skipped += 1
    
    if modified:
        shapes_path.write_text(json.dumps(shapes_data, indent=2) + "\n")
        return 1, configs_updated, configs_skipped
    
    return 0, configs_updated, configs_skipped


def main():
    total_fixed = 0
    total_configs_updated = 0
    total_configs_skipped = 0
    
    for d in sorted(BASE.iterdir()):
        if not d.is_dir():
            continue
        fixed, updated, skipped = process_repro(d)
        total_fixed += fixed
        total_configs_updated += updated
        total_configs_skipped += skipped
    
    print(f"Files fixed: {total_fixed}")
    print(f"Configs updated: {total_configs_updated}")
    print(f"Configs skipped: {total_configs_skipped}")


if __name__ == "__main__":
    main()
