"""
Retroactive bounds inference for full_graph .meta.json files.

Parses full_graph_*.py source files to infer valid index bounds for integer
tensor inputs that currently have `constraint_source: "annotation_default"`.

Strategy:
  1. Parse forward() signature to extract all parameter names + shapes + dtypes
  2. Parse assignment statements to build a dataflow graph (var -> op + args)
  3. Follow integer inputs through passthrough ops (reshape, view, expand, etc.)
  4. When a consumer is found (embedding, gather, scatter, index_select, etc.),
     infer the bound from the source/target tensor's shape annotation
  5. Update the .meta.json with inferred bounds

This avoids needing to trace the graph as an FX module (which fails for saved
full_graph files that reference self.config, etc.).
"""

import json
import re
import sys
from pathlib import Path


# --- Source parsing utilities ---

def parse_forward_params(source: str) -> list[dict]:
    """Extract parameter info from the forward() signature.

    Returns list of dicts with keys: name, dtype_short, shape, is_int
    """
    # Find the forward def line (may span multiple lines but usually one very long line)
    # Pattern: def forward(self, name: "dtype[shape]", ...)
    m = re.search(r'def forward\(self,\s*(.*?)\)\s*:', source, re.DOTALL)
    if not m:
        return []

    sig = m.group(1)
    params = []
    # Match each parameter: name: "dtype[dim1, dim2, ...]"
    for pm in re.finditer(r'(\w+):\s*"(\w+)\[([^\]]*)\]"', sig):
        name = pm.group(1)
        dtype_short = pm.group(2)
        dims_str = pm.group(3).strip()
        shape = [int(d.strip()) for d in dims_str.split(',') if d.strip()] if dims_str else []
        is_int = dtype_short in ('i64', 'i32', 'i16', 'i8', 'u8')
        params.append({
            'name': name,
            'dtype_short': dtype_short,
            'shape': shape,
            'is_int': is_int,
        })
    return params


def parse_assignments(source: str) -> dict:
    """Parse all assignment statements from the forward body.

    Returns dict: var_name -> {
        'dtype_short': str, 'shape': list[int],
        'op': str (e.g. 'aten.embedding.default'),
        'args': list[str] (variable names or literals),
        'raw_line': str
    }
    """
    assignments = {}

    # Pattern: var_name: "dtype[shape]" = torch.ops.aten.op_name(args...)
    # Also handles: var_name: "dtype[shape]" = torch.ops.prims.op_name(args...)
    pattern = re.compile(
        r'(\w+):\s*"(\w+)\[([^\]]*)\]"\s*=\s*'
        r'torch\.ops\.(aten|prims|inductor_prims)\.(\w+)\.(\w+)\((.*?)\)'
        r'(?:\s*;.*)?$',
        re.MULTILINE,
    )

    for m in pattern.finditer(source):
        var_name = m.group(1)
        dtype_short = m.group(2)
        dims_str = m.group(3).strip()
        shape = [int(d.strip()) for d in dims_str.split(',') if d.strip()] if dims_str else []
        namespace = m.group(4)
        op_name = m.group(5)
        overload = m.group(6)
        args_str = m.group(7)

        # Parse args (simplified: split by comma but handle nested brackets)
        args = _parse_args(args_str)

        assignments[var_name] = {
            'dtype_short': dtype_short,
            'shape': shape,
            'op': f'{namespace}.{op_name}.{overload}',
            'args': args,
            'raw_line': m.group(0),
        }

    return assignments


def _parse_args(args_str: str) -> list[str]:
    """Parse argument list, handling nested brackets and keyword args."""
    args = []
    depth = 0
    current = ''
    for ch in args_str:
        if ch in ('(', '['):
            depth += 1
            current += ch
        elif ch in (')', ']'):
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            args.append(current.strip())
            current = ''
        else:
            current += ch
    if current.strip():
        args.append(current.strip())

    # Strip keyword arg syntax (e.g. "dim = 1" -> "1")
    cleaned = []
    for a in args:
        # Skip keyword args like memory_format=..., pin_memory=...
        if '=' in a and not a.startswith('[') and not a.startswith('('):
            parts = a.split('=', 1)
            key = parts[0].strip()
            val = parts[1].strip()
            # Keep kwargs that are positional-style (e.g. from named tensors)
            if key in ('memory_format', 'pin_memory', 'dtype', 'layout', 'device',
                       'requires_grad', 'correction', 'keepdim', 'start', 'step'):
                continue  # skip these kwargs entirely
            cleaned.append(val)
        else:
            cleaned.append(a)

    return cleaned


# --- Dataflow tracking ---

PASSTHROUGH_OPS = {
    'aten.reshape.default',
    'aten.view.default',
    'aten.expand.default',
    'aten.clone.default',
    'aten.unsqueeze.default',
    'aten.squeeze.default',
    'aten.squeeze.dim',
    'aten.select.int',
    'aten.slice.Tensor',
    'aten.permute.default',
    'aten.contiguous.default',
    'aten.t.default',
    'aten.transpose.int',
    'aten.lift_fresh_copy.default',
    'prims.convert_element_type.default',
    'aten.alias.default',
}

# Ops where the output carries the same index semantics as input
WHERE_OPS = {
    'aten.where.self',     # where(cond, x, y) - result derives from x/y
}

CLAMP_OPS = {
    'aten.clamp_min.default',
    'aten.clamp_max.default',
}


def build_derives_from(params: list[dict], assignments: dict) -> dict:
    """Build mapping: var_name -> set of param names it derives from.

    Only tracks integer parameters through passthrough/where/clamp ops.
    Also returns clamp_bounds: param_name -> max value from clamp_max.
    """
    int_param_names = {p['name'] for p in params if p['is_int']}

    derives_from = {}  # var_name -> set of param names
    clamp_bounds = {}  # param_name -> clamp max value

    # Initialize: each param derives from itself
    for p in params:
        if p['is_int']:
            derives_from[p['name']] = {p['name']}

    # Process assignments in source order (topological)
    # We need to process them in the order they appear in source
    # The assignments dict preserves insertion order (Python 3.7+)
    for var_name, info in assignments.items():
        op = info['op']
        args = info['args']

        if op in PASSTHROUGH_OPS:
            # First arg is the source tensor
            if args and args[0] in derives_from:
                derives_from[var_name] = set(derives_from[args[0]])

        elif op == 'aten.where.self':
            # where(cond, x, y) - result derives from x and y (args[1], args[2])
            sources = set()
            for arg in args[1:3]:
                if arg in derives_from:
                    sources.update(derives_from[arg])
            if sources:
                derives_from[var_name] = sources

        elif op == 'aten.clamp_min.default':
            # clamp_min(x, min_val) - passes through
            if args and args[0] in derives_from:
                derives_from[var_name] = set(derives_from[args[0]])

        elif op == 'aten.clamp_max.default':
            # clamp_max(x, max_val) - bound = max_val
            if args and args[0] in derives_from:
                derives_from[var_name] = set(derives_from[args[0]])
                # Record clamp bound
                if len(args) >= 2:
                    try:
                        max_val = int(args[1])
                        for param_name in derives_from[args[0]]:
                            if param_name in int_param_names:
                                old = clamp_bounds.get(param_name)
                                if old is None or max_val < old:
                                    clamp_bounds[param_name] = max_val
                    except (ValueError, TypeError):
                        pass

        elif op == 'aten.add.Tensor':
            # add(x, K) - trace through
            if args and args[0] in derives_from:
                derives_from[var_name] = set(derives_from[args[0]])
            elif len(args) >= 2 and args[1] in derives_from:
                derives_from[var_name] = set(derives_from[args[1]])

        elif op == 'aten.sub.Tensor':
            # sub(x, K) - trace through first arg
            if args and args[0] in derives_from:
                derives_from[var_name] = set(derives_from[args[0]])

        elif op == 'aten.mul.Tensor':
            # mul(x, K) - trace through
            if args and args[0] in derives_from:
                derives_from[var_name] = set(derives_from[args[0]])
            elif len(args) >= 2 and args[1] in derives_from:
                derives_from[var_name] = set(derives_from[args[1]])

        elif op == 'aten.gather.default':
            # gather(input, dim, index) - OUTPUT carries values from input
            # The index arg is the consumer (bounds come from input shape)
            # But the output values derive from input's values
            if args and args[0] in derives_from:
                derives_from[var_name] = set(derives_from[args[0]])

    return derives_from, clamp_bounds


def get_shape_for_var(var_name: str, params: list[dict], assignments: dict) -> list[int] | None:
    """Get the shape of a variable (parameter or assigned var)."""
    # Check params first
    for p in params:
        if p['name'] == var_name:
            return p['shape']
    # Check assignments
    if var_name in assignments:
        return assignments[var_name]['shape']
    return None


def infer_bounds_from_source(source: str) -> dict[str, int]:
    """Parse graph source and infer index bounds for integer inputs.

    Returns dict: param_name -> bound (exclusive upper bound for valid indices).
    """
    params = parse_forward_params(source)
    assignments = parse_assignments(source)
    derives_from, clamp_bounds = build_derives_from(params, assignments)

    int_param_names = {p['name'] for p in params if p['is_int']}
    bounds = {}  # param_name -> bound

    def _set_bound(index_var: str, bound: int):
        """Set bound for all int params that derive into index_var."""
        sources = derives_from.get(index_var, set())
        for param_name in sources & int_param_names:
            if param_name not in bounds or bounds[param_name] > bound:
                bounds[param_name] = bound

    # Scan all assignments for index-consuming ops
    for var_name, info in assignments.items():
        op = info['op']
        args = info['args']

        if op == 'aten.embedding.default':
            # embedding(weight, indices, ...) -> bound = weight.shape[0]
            if len(args) >= 2:
                weight_var = args[0]
                indices_var = args[1]
                weight_shape = get_shape_for_var(weight_var, params, assignments)
                if weight_shape and len(weight_shape) > 0:
                    # Check if indices_var derives from an int param
                    if indices_var in derives_from:
                        _set_bound(indices_var, weight_shape[0])

        elif op == 'aten.gather.default':
            # gather(input, dim, index) -> bound = input.shape[dim]
            if len(args) >= 3:
                input_var = args[0]
                dim_str = args[1]
                index_var = args[2]
                input_shape = get_shape_for_var(input_var, params, assignments)
                try:
                    dim = int(dim_str)
                    if input_shape and 0 <= dim < len(input_shape):
                        if index_var in derives_from:
                            _set_bound(index_var, input_shape[dim])
                except (ValueError, TypeError):
                    pass

        elif op in ('aten.scatter.src', 'aten.scatter.value',
                    'aten.scatter_add.default', 'aten.scatter_reduce.two'):
            # scatter(target, dim, index, src/value) -> bound = target.shape[dim]
            if len(args) >= 3:
                target_var = args[0]
                dim_str = args[1]
                index_var = args[2]
                target_shape = get_shape_for_var(target_var, params, assignments)
                try:
                    dim = int(dim_str)
                    if target_shape:
                        actual_dim = dim if dim >= 0 else dim + len(target_shape)
                        if 0 <= actual_dim < len(target_shape):
                            if index_var in derives_from:
                                _set_bound(index_var, target_shape[actual_dim])
                except (ValueError, TypeError):
                    pass

        elif op == 'aten.index_select.default':
            # index_select(input, dim, index) -> bound = input.shape[dim]
            if len(args) >= 3:
                input_var = args[0]
                dim_str = args[1]
                index_var = args[2]
                input_shape = get_shape_for_var(input_var, params, assignments)
                try:
                    dim = int(dim_str)
                    if input_shape:
                        actual_dim = dim if dim >= 0 else dim + len(input_shape)
                        if 0 <= actual_dim < len(input_shape):
                            if index_var in derives_from:
                                _set_bound(index_var, input_shape[actual_dim])
                except (ValueError, TypeError):
                    pass

        elif op == 'aten.index.Tensor':
            # index.Tensor(input, [indices...]) -> bound = input.shape[dim]
            if len(args) >= 2:
                input_var = args[0]
                indices_str = args[1]
                input_shape = get_shape_for_var(input_var, params, assignments)
                if input_shape:
                    # Parse the indices list [idx0, idx1, ...]
                    idx_list = _parse_list_arg(indices_str)
                    for dim_idx, idx_var in enumerate(idx_list):
                        if dim_idx < len(input_shape) and idx_var in derives_from:
                            _set_bound(idx_var, input_shape[dim_idx])

        elif op == 'aten.index_put.default':
            # index_put(target, [indices...], values, accumulate?)
            if len(args) >= 2:
                target_var = args[0]
                indices_str = args[1]
                target_shape = get_shape_for_var(target_var, params, assignments)
                if target_shape:
                    idx_list = _parse_list_arg(indices_str)
                    for dim_idx, idx_var in enumerate(idx_list):
                        if dim_idx < len(target_shape) and idx_var in derives_from:
                            _set_bound(idx_var, target_shape[dim_idx])

        elif op == 'aten.index_put_.default':
            # Same as index_put but in-place
            if len(args) >= 2:
                target_var = args[0]
                indices_str = args[1]
                target_shape = get_shape_for_var(target_var, params, assignments)
                if target_shape:
                    idx_list = _parse_list_arg(indices_str)
                    for dim_idx, idx_var in enumerate(idx_list):
                        if dim_idx < len(target_shape) and idx_var in derives_from:
                            _set_bound(idx_var, target_shape[dim_idx])

    # Apply clamp bounds as fallback (only if no better bound found from consumers)
    for param_name, clamp_val in clamp_bounds.items():
        # clamp_max gives an inclusive upper bound, so exclusive = clamp_val + 1
        exclusive_bound = clamp_val + 1
        if param_name not in bounds or exclusive_bound < bounds[param_name]:
            bounds[param_name] = exclusive_bound

    return bounds


def _parse_list_arg(s: str) -> list[str]:
    """Parse a bracketed list like '[var1, var2, None]' into variable names."""
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    items = []
    for item in s.split(','):
        item = item.strip()
        if item and item != 'None':
            items.append(item)
        else:
            items.append(None)
    return items


# --- Cross-entropy / nll_loss label pattern detection ---

def infer_label_bounds_from_source(source: str, params: list[dict], assignments: dict) -> dict[str, int]:
    """Detect cross-entropy loss label patterns.

    In backward graphs, labels feed into:
      reshape -> unsqueeze -> ne(-100) + where -> expand([batch, vocab]) -> eq(iota(vocab))

    Or in forward graphs, labels feed into nll_loss.

    Returns dict: param_name -> vocab_size bound
    """
    bounds = {}

    # Pattern 1: Look for eq.Tensor(expand_of_label, iota_var)
    # where iota is created with prims.iota.default(N, ...)
    # First find all iota sizes
    iota_sizes = {}  # var_name -> N
    for var_name, info in assignments.items():
        if info['op'] == 'prims.iota.default':
            args = info['args']
            if args:
                try:
                    iota_sizes[var_name] = int(args[0])
                except (ValueError, TypeError):
                    pass

    # Also find iotas that go through reshape/view
    for var_name, info in assignments.items():
        if info['op'] in PASSTHROUGH_OPS and info['args']:
            src = info['args'][0]
            if src in iota_sizes:
                iota_sizes[var_name] = iota_sizes[src]

    # Pattern 2: Look for nll_loss_forward that consumes labels
    # nll_loss_forward(input, target, ...) where target is labels
    derives_from, _ = build_derives_from(params, assignments)
    int_param_names = {p['name'] for p in params if p['is_int']}

    for var_name, info in assignments.items():
        if 'nll_loss' in info['op']:
            # nll_loss_forward(input, target, ...) -> bound = input.shape[-1]
            if len(info['args']) >= 2:
                input_var = info['args'][0]
                target_var = info['args'][1]
                input_shape = get_shape_for_var(input_var, params, assignments)
                if input_shape and len(input_shape) >= 1:
                    vocab_dim = input_shape[-1]  # last dim is usually vocab
                    if target_var in derives_from:
                        sources = derives_from[target_var] & int_param_names
                        for pn in sources:
                            if pn not in bounds or vocab_dim < bounds[pn]:
                                bounds[pn] = vocab_dim

    # Pattern 3: eq.Tensor(label_derived, iota_derived) where label has expand([batch, vocab])
    for var_name, info in assignments.items():
        if info['op'] == 'aten.eq.Tensor' and len(info['args']) >= 2:
            arg0 = info['args'][0]
            arg1 = info['args'][1]
            # Check if one arg derives from an int param and the other from an iota
            for label_arg, iota_arg in [(arg0, arg1), (arg1, arg0)]:
                if label_arg in derives_from and iota_arg in iota_sizes:
                    sources = derives_from[label_arg] & int_param_names
                    if sources:
                        vocab_size = iota_sizes[iota_arg]
                        for pn in sources:
                            if pn not in bounds or vocab_size < bounds[pn]:
                                bounds[pn] = vocab_size

    return bounds


# --- Main processing ---

def process_graph(meta_path: Path, dry_run: bool = False) -> dict:
    """Process a single meta.json file, inferring bounds for annotation_default inputs.

    Returns dict with keys: path, updated_count, updates (list of {name, old_high, new_high})
    """
    meta = json.loads(meta_path.read_text())
    graph_file = meta.get('graph')
    if not graph_file:
        return {'path': str(meta_path), 'updated_count': 0, 'updates': []}

    graph_path = meta_path.parent / graph_file
    if not graph_path.exists():
        return {'path': str(meta_path), 'updated_count': 0, 'updates': [], 'error': 'graph file not found'}

    source = graph_path.read_text()
    params = parse_forward_params(source)
    assignments = parse_assignments(source)

    # Run main bounds inference
    bounds = infer_bounds_from_source(source)

    # Run label-specific inference
    label_bounds = infer_label_bounds_from_source(source, params, assignments)

    # Merge (take tighter bound)
    for name, bound in label_bounds.items():
        if name not in bounds or bound < bounds[name]:
            bounds[name] = bound

    # Update meta.json inputs
    updates = []
    inputs = meta.get('inputs', [])
    for inp in inputs:
        name = inp.get('name', '')
        if inp.get('constraint_source') != 'annotation_default':
            continue
        if name not in bounds:
            continue

        new_high = bounds[name]
        gen = inp.get('gen') or inp.get('generator')
        old_high = gen.get('high', 0) if gen else 0

        if new_high != old_high:
            # Update the gen field
            new_gen = {'kind': 'index', 'low': 0, 'high': new_high}
            inp['gen'] = new_gen
            inp['generator'] = new_gen
            inp['constraint_source'] = 'graph_inference'
            updates.append({'name': name, 'old_high': old_high, 'new_high': new_high})

    if updates and not dry_run:
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')

    return {'path': str(meta_path), 'updated_count': len(updates), 'updates': updates}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Retroactive bounds inference for full graph metadata')
    parser.add_argument('--dry-run', action='store_true', help='Print what would change without writing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print per-file details')
    parser.add_argument('--repros-dir', type=Path, default=Path('repros/models'),
                        help='Root directory containing model repros')
    args = parser.parse_args()

    meta_files = sorted(args.repros_dir.rglob('*.meta.json'))
    print(f'Found {len(meta_files)} meta.json files')

    # Filter to only those with annotation_default int inputs
    target_files = []
    for mf in meta_files:
        data = json.loads(mf.read_text())
        for inp in data.get('inputs', []):
            if inp.get('dtype', '') in ('int64', 'int32') and inp.get('constraint_source') == 'annotation_default':
                target_files.append(mf)
                break

    print(f'Target files with annotation_default int inputs: {len(target_files)}')

    total_updated = 0
    total_inputs_fixed = 0
    files_with_updates = 0

    for mf in target_files:
        result = process_graph(mf, dry_run=args.dry_run)
        if result['updated_count'] > 0:
            files_with_updates += 1
            total_inputs_fixed += result['updated_count']
            if args.verbose:
                print(f'  {result["path"]}: {result["updated_count"]} inputs updated')
                for u in result['updates']:
                    print(f'    {u["name"]}: {u["old_high"]} -> {u["new_high"]}')

    total_updated = files_with_updates
    print(f'\nResults:')
    print(f'  Files updated: {total_updated}/{len(target_files)}')
    print(f'  Total inputs fixed: {total_inputs_fixed}')
    if args.dry_run:
        print(f'  (dry-run mode - no files written)')

    # Report remaining unfixed
    remaining = len(target_files) - total_updated
    if remaining > 0:
        print(f'\n  Remaining unfixed graphs: {remaining}')
        # Show which ones still have issues
        if args.verbose:
            for mf in target_files:
                data = json.loads(mf.read_text())
                still_default = []
                for inp in data.get('inputs', []):
                    if inp.get('constraint_source') == 'annotation_default':
                        still_default.append(inp['name'])
                if still_default:
                    print(f'    {mf}: {still_default}')


if __name__ == '__main__':
    main()
