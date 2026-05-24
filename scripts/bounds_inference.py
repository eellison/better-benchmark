"""
Reusable bounds inference for i64 tensor parameters in FX-style forward() methods.

Parses the forward() method AST to build a mini dataflow graph, then infers
gen=Index(N) / gen=Perm(N) annotations for integer tensor parameters based on
how they are consumed (gather, scatter, embedding, index_put, etc.).

Primary entry point:
    infer_bounds_for_config(repro_py, config_str) -> annotated config string
"""
import re
from pathlib import Path
from collections import defaultdict


def extract_ground_truth(config_str):
    """Extract all gen=Index(N) and gen=Perm(N) from a config string.

    Returns dict mapping parameter position (0-indexed among T()/S() entries) to (type, N).
    """
    annotations = {}
    param_idx = 0
    for match in re.finditer(r'(T\([^)]+\)|S\([^)]+\))', config_str):
        token = match.group(1)
        if token.startswith('T('):
            gen_match = re.search(r'gen=(Index|Perm)\((\d+)\)', token)
            if gen_match:
                annotations[param_idx] = (gen_match.group(1), int(gen_match.group(2)))
            param_idx += 1
        elif token.startswith('S('):
            param_idx += 1
    return annotations


def parse_forward_body(content):
    """Parse the forward() method body and build a mini dataflow graph.

    Returns:
        param_names: list of parameter names in order
        param_shapes: dict of param_name -> shape (from annotation)
        param_dtypes: dict of param_name -> dtype abbreviation
        var_shapes: dict of var_name -> shape (inferred from annotations/allocations)
        operations: list of (result_var, op_name, args_list) tuples
    """
    fwd_match = re.search(r'def forward\(self,\s*([^)]+)\)', content)
    if not fwd_match:
        return None, None, None, None, None

    params_str = fwd_match.group(1)
    param_names = []
    param_shapes = {}
    param_dtypes = {}

    for pmatch in re.finditer(r'(\w+)(?::\s*"(\w+)\[([^\]]*)\]")?', params_str):
        name = pmatch.group(1)
        param_names.append(name)
        if pmatch.group(2) is not None:
            dtype = pmatch.group(2)
            dims_str = (pmatch.group(3) or '').strip()
            if dims_str:
                shape = [int(x.strip()) for x in dims_str.split(',')]
            else:
                shape = []
            param_shapes[name] = shape
            param_dtypes[name] = dtype

    var_shapes = dict(param_shapes)

    for amatch in re.finditer(
        r'(\w+):\s*"(\w+)\[([^\]]*)\]"\s*=\s*(.+)',
        content
    ):
        var_name = amatch.group(1)
        dtype = amatch.group(2)
        dims_str = amatch.group(3).strip()
        if dims_str:
            shape = [int(x.strip()) for x in dims_str.split(',')]
        else:
            shape = []
        var_shapes[var_name] = shape
        param_dtypes[var_name] = dtype

    for amatch in re.finditer(r'(\w+):\s*"(\w+)\[\]"\s*=', content):
        var_name = amatch.group(1)
        var_shapes[var_name] = []
        param_dtypes[var_name] = amatch.group(2)

    operations = []
    for opmatch in re.finditer(
        r'(\w+)(?::\s*"[^"]*")?\s*=\s*torch\.ops\.(aten|prims)\.(\w+)\.(\w+)\((.+)',
        content
    ):
        result_var = opmatch.group(1)
        namespace = opmatch.group(2)
        op_name = opmatch.group(3)
        overload = opmatch.group(4)
        args_str = opmatch.group(5)

        args_str = re.sub(r'\)\s*;.*$', '', args_str)
        if args_str.endswith(')'):
            args_str = args_str[:-1]

        full_op = f"{namespace}.{op_name}.{overload}"
        operations.append((result_var, full_op, args_str))

    return param_names, param_shapes, param_dtypes, var_shapes, operations


def parse_args_simple(args_str):
    """Parse a simple args string to extract variable references and list literals.

    Returns list of parsed arg items: either var name (str), list of ints, list of var names,
    or None for complex/unparseable args.
    """
    args = []
    depth = 0
    current = ""
    for ch in args_str:
        if ch in '([':
            depth += 1
            current += ch
        elif ch in ')]':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current.strip())

    parsed = []
    for arg in args:
        arg = re.sub(r';.*', '', arg).strip()
        if '=' in arg and not arg.startswith('['):
            parsed.append(('kwarg', arg))
            continue
        list_match = re.match(r'\[([^\]]*)\]', arg)
        if list_match:
            inner = list_match.group(1).strip()
            if inner:
                items = [x.strip() for x in inner.split(',')]
                try:
                    parsed.append(('int_list', [int(x) for x in items]))
                except ValueError:
                    parsed.append(('var_list', items))
            else:
                parsed.append(('int_list', []))
            continue
        try:
            parsed.append(('int', int(arg)))
            continue
        except ValueError:
            pass
        try:
            parsed.append(('float', float(arg)))
            continue
        except ValueError:
            pass
        if re.match(r'\w+$', arg):
            parsed.append(('var', arg))
        else:
            parsed.append(('other', arg))

    return parsed


def _find_iota_source(var_name, operations, iota_sizes, derives_from):
    """Check if var_name derives from an iota (possibly through passthrough ops).

    Returns the iota size if found, None otherwise.
    """
    if var_name in iota_sizes:
        return iota_sizes[var_name]

    PASSTHROUGH_OPS = {
        'aten.clone.default', 'aten.reshape.default', 'aten.view.default',
        'aten.expand.default', 'aten.unsqueeze.default', 'aten.squeeze.default',
        'aten.squeeze.dim',
    }

    for result_var, op, args_str in operations:
        if result_var == var_name:
            if op in PASSTHROUGH_OPS:
                parsed = parse_args_simple(args_str)
                if parsed and parsed[0][0] == 'var':
                    return _find_iota_source(parsed[0][1], operations, iota_sizes, derives_from)
            if op == 'aten.add.Tensor':
                parsed = parse_args_simple(args_str)
                if len(parsed) >= 2 and parsed[0][0] == 'var':
                    src = parsed[0][1]
                    if parsed[1][0] == 'int' and parsed[1][1] == 0:
                        return _find_iota_source(src, operations, iota_sizes, derives_from)
                    return _find_iota_source(src, operations, iota_sizes, derives_from)
            break

    return None


def infer_bounds_from_forward(content):
    """Infer index bounds and permutation indices from the forward() method.

    Returns:
        index_bounds: dict of param_position -> bound (for Index)
        perm_bounds: dict of param_position -> bound (for Perm)
    """
    result = parse_forward_body(content)
    if result[0] is None:
        return {}, {}

    param_names, param_shapes, param_dtypes, var_shapes, operations = result

    # Build a set of int-typed params (candidates for index inference)
    int_params = set()
    for name in param_names:
        dtype = param_dtypes.get(name, "")
        if "i64" in dtype or "i32" in dtype or "i8" in dtype:
            int_params.add(name)

    # Build a "flows from" map: for passthrough ops, track which variable
    # ultimately derives from which param
    derives_from = {name: {name} for name in param_names}

    PASSTHROUGH_OPS = {
        'aten.clone.default', 'aten.reshape.default', 'aten.view.default',
        'aten.expand.default', 'aten.slice.Tensor', 'aten.unsqueeze.default',
        'aten.squeeze.default', 'aten.squeeze.dim', 'aten.select.int',
        'aten.permute.default', 'aten.contiguous.default', 'aten.t.default',
        'aten.transpose.int', 'aten.where.self',
        'aten.lift_fresh_copy.default',
        'aten.add.Tensor', 'aten.mul.Tensor',
        'prims.convert_element_type.default',
    }

    add_offset = defaultdict(int)

    # First pass: trace passthrough ops to build derives_from
    for result_var, op, args_str in operations:
        if op in PASSTHROUGH_OPS:
            parsed = parse_args_simple(args_str)
            if op == 'aten.where.self':
                for item in parsed:
                    if item[0] == 'var' and item[1] in derives_from:
                        if result_var not in derives_from:
                            derives_from[result_var] = set()
                        derives_from[result_var].update(derives_from[item[1]])
            elif op == 'aten.add.Tensor':
                if parsed and parsed[0][0] == 'var':
                    src = parsed[0][1]
                    if src in derives_from:
                        derives_from[result_var] = set(derives_from[src])
                    else:
                        derives_from[result_var] = {src}
                    offset = add_offset.get(src, 0)
                    if len(parsed) >= 2 and parsed[1][0] == 'int':
                        offset += parsed[1][1]
                    add_offset[result_var] = offset
            elif op == 'aten.mul.Tensor':
                if parsed and parsed[0][0] == 'var':
                    src = parsed[0][1]
                    if src in derives_from:
                        derives_from[result_var] = set(derives_from[src])
                    else:
                        derives_from[result_var] = {src}
                    add_offset[result_var] = 0
            else:
                if parsed and parsed[0][0] == 'var':
                    src = parsed[0][1]
                    if src in derives_from:
                        derives_from[result_var] = set(derives_from[src])
                    else:
                        derives_from[result_var] = {src}
                    if src in add_offset:
                        add_offset[result_var] = add_offset[src]
        elif op == 'aten.gather.default':
            parsed = parse_args_simple(args_str)
            if parsed and parsed[0][0] == 'var':
                src = parsed[0][1]
                if src in derives_from:
                    derives_from[result_var] = set(derives_from[src])
                else:
                    derives_from[result_var] = {src}
                if src in add_offset:
                    add_offset[result_var] = add_offset[src]

    # Second pass: handle additional passthrough-like ops
    for result_var, op, args_str in operations:
        if 'low_memory_max_pool_offsets_to_indices' in op:
            parsed = parse_args_simple(args_str)
            if parsed and parsed[0][0] == 'var':
                src = parsed[0][1]
                if src in derives_from:
                    derives_from[result_var] = set(derives_from[src])
                else:
                    derives_from[result_var] = {src}

    # Third pass: find index consumer ops and determine bounds
    index_bounds = {}
    perm_bounds = {}

    # Find iota results
    iota_sizes = {}
    for result_var, op, args_str in operations:
        if op == 'prims.iota.default':
            parsed = parse_args_simple(args_str)
            if parsed and parsed[0][0] == 'int':
                iota_sizes[result_var] = parsed[0][1]

    # Find allocations
    alloc_shapes = {}
    for result_var, op, args_str in operations:
        if op in ('aten.empty.memory_format', 'aten.full.default'):
            parsed = parse_args_simple(args_str)
            if parsed and parsed[0][0] == 'int_list':
                alloc_shapes[result_var] = parsed[0][1]

    # Handle int8 pool offsets
    for result_var, op, args_str in operations:
        if 'low_memory_max_pool_offsets_to_indices' in op:
            parsed = parse_args_simple(args_str)
            if parsed and parsed[0][0] == 'var':
                src = parsed[0][1]
                source_params = derives_from.get(src, set())
                int_sources = source_params & int_params
                if len(parsed) >= 2 and parsed[1][0] == 'int_list':
                    ks = parsed[1][1]
                    bound = 1
                    for k in ks:
                        bound *= k
                    for param in int_sources:
                        index_bounds[param] = bound

    # Heuristic for "cumsum as target" pattern
    for result_var, op, args_str in operations:
        if op == 'aten.index.Tensor':
            parsed = parse_args_simple(args_str)
            if len(parsed) >= 2:
                target_arg = parsed[0]
                indices_arg = parsed[1]
                if target_arg[0] == 'var':
                    target_var = target_arg[1]
                    source_params = derives_from.get(target_var, set())
                    int_targets = source_params & int_params
                    if int_targets and indices_arg[0] == 'var_list':
                        for param in int_targets:
                            if param in param_shapes and param_shapes[param]:
                                bound = param_shapes[param][0]
                                if param not in index_bounds:
                                    index_bounds[param] = bound

    def _set_bound(index_var, raw_bound):
        """Set bound for all int params that derive into index_var, accounting for offset."""
        source_params = derives_from.get(index_var, set())
        int_sources = source_params & int_params
        offset = add_offset.get(index_var, 0)
        bound = raw_bound - offset if offset > 0 else raw_bound
        for param in int_sources:
            if param not in index_bounds or index_bounds[param] > bound:
                index_bounds[param] = bound

    # Scan operations for index consumers
    for result_var, op, args_str in operations:
        parsed = parse_args_simple(args_str)

        if op == 'aten.gather.default':
            if len(parsed) >= 3:
                input_arg = parsed[0]
                dim_arg = parsed[1]
                index_arg = parsed[2]

                if index_arg[0] == 'var' and dim_arg[0] == 'int':
                    index_var = index_arg[1]
                    dim = dim_arg[1]
                    source_params = derives_from.get(index_var, set())
                    int_sources = source_params & int_params
                    if int_sources:
                        input_var = input_arg[1] if input_arg[0] == 'var' else None
                        if input_var and input_var in var_shapes:
                            input_shape = var_shapes[input_var]
                            if dim < len(input_shape):
                                raw_bound = input_shape[dim]
                                _set_bound(index_var, raw_bound)

        elif op == 'aten.scatter.src':
            if len(parsed) >= 4:
                target_arg = parsed[0]
                dim_arg = parsed[1]
                index_arg = parsed[2]

                if index_arg[0] == 'var' and dim_arg[0] == 'int':
                    index_var = index_arg[1]
                    dim = dim_arg[1]
                    source_params = derives_from.get(index_var, set())
                    int_sources = source_params & int_params
                    if int_sources:
                        target_var = target_arg[1] if target_arg[0] == 'var' else None
                        target_shape = None
                        if target_var:
                            target_shape = var_shapes.get(target_var) or alloc_shapes.get(target_var)
                        if target_shape:
                            actual_dim = dim if dim >= 0 else dim + len(target_shape)
                            if 0 <= actual_dim < len(target_shape):
                                _set_bound(index_var, target_shape[actual_dim])

                    # Permutation pattern
                    src_arg = parsed[3]
                    if src_arg[0] == 'var':
                        src_var = src_arg[1]
                        src_iota = _find_iota_source(src_var, operations, iota_sizes, derives_from)
                        target_var = target_arg[1] if target_arg[0] == 'var' else None
                        if src_iota is not None and target_var in alloc_shapes:
                            t_shape = alloc_shapes[target_var]
                            actual_dim = dim if dim >= 0 else dim + len(t_shape)
                            if 0 <= actual_dim < len(t_shape) and t_shape[actual_dim] == src_iota:
                                for param in int_sources:
                                    perm_bounds[param] = src_iota

        elif op == 'aten.scatter.value':
            if len(parsed) >= 3:
                target_arg = parsed[0]
                dim_arg = parsed[1]
                index_arg = parsed[2]

                if index_arg[0] == 'var' and dim_arg[0] == 'int':
                    index_var = index_arg[1]
                    dim = dim_arg[1]
                    source_params = derives_from.get(index_var, set())
                    int_sources = source_params & int_params
                    if int_sources:
                        target_var = target_arg[1] if target_arg[0] == 'var' else None
                        target_shape = None
                        if target_var:
                            target_shape = var_shapes.get(target_var) or alloc_shapes.get(target_var)
                        if target_shape:
                            actual_dim = dim if dim >= 0 else dim + len(target_shape)
                            if 0 <= actual_dim < len(target_shape):
                                _set_bound(index_var, target_shape[actual_dim])

        elif op == 'aten.scatter_add.default':
            if len(parsed) >= 3:
                target_arg = parsed[0]
                dim_arg = parsed[1]
                index_arg = parsed[2]

                if index_arg[0] == 'var' and dim_arg[0] == 'int':
                    index_var = index_arg[1]
                    dim = dim_arg[1]
                    source_params = derives_from.get(index_var, set())
                    int_sources = source_params & int_params
                    if int_sources:
                        target_var = target_arg[1] if target_arg[0] == 'var' else None
                        target_shape = None
                        if target_var:
                            target_shape = var_shapes.get(target_var) or alloc_shapes.get(target_var)
                        if target_shape:
                            actual_dim = dim if dim >= 0 else dim + len(target_shape)
                            if 0 <= actual_dim < len(target_shape):
                                _set_bound(index_var, target_shape[actual_dim])

        elif op == 'aten.index_put.default':
            if len(parsed) >= 3:
                target_arg = parsed[0]
                indices_arg = parsed[1]
                values_arg = parsed[2]

                target_var = target_arg[1] if target_arg[0] == 'var' else None
                target_shape = None
                if target_var:
                    target_shape = var_shapes.get(target_var) or alloc_shapes.get(target_var)

                if indices_arg[0] == 'var_list' and target_shape:
                    target_is_alloc = target_var in alloc_shapes
                    for dim, idx_name in enumerate(indices_arg[1]):
                        if dim < len(target_shape):
                            bound = target_shape[dim]
                            if (target_is_alloc and len(target_shape) > 1
                                    and dim > 0 and bound > target_shape[0]):
                                bound = target_shape[0]
                            _set_bound(idx_name, bound)

                # Permutation pattern
                if (indices_arg[0] == 'var_list' and values_arg[0] == 'var'
                        and target_var in alloc_shapes):
                    t_shape = alloc_shapes[target_var]
                    val_var = values_arg[1]
                    val_iota = _find_iota_source(val_var, operations, iota_sizes, derives_from)
                    if val_iota is not None and t_shape and t_shape[0] == val_iota:
                        for idx_name in indices_arg[1]:
                            source_params = derives_from.get(idx_name, set())
                            int_sources = source_params & int_params
                            for param in int_sources:
                                perm_bounds[param] = val_iota

        elif op == 'aten.index.Tensor':
            if len(parsed) >= 2:
                input_arg = parsed[0]
                indices_arg = parsed[1]

                input_var = input_arg[1] if input_arg[0] == 'var' else None
                input_shape = None
                if input_var:
                    input_shape = var_shapes.get(input_var) or alloc_shapes.get(input_var)

                if indices_arg[0] == 'var_list' and input_shape:
                    for dim, idx_name in enumerate(indices_arg[1]):
                        if dim < len(input_shape):
                            _set_bound(idx_name, input_shape[dim])

        elif op == 'aten.index_select.default':
            if len(parsed) >= 3:
                input_arg = parsed[0]
                dim_arg = parsed[1]
                index_arg = parsed[2]

                if index_arg[0] == 'var' and dim_arg[0] == 'int':
                    index_var = index_arg[1]
                    dim = dim_arg[1]
                    source_params = derives_from.get(index_var, set())
                    int_sources = source_params & int_params
                    if int_sources:
                        input_var = input_arg[1] if input_arg[0] == 'var' else None
                        if input_var and input_var in var_shapes:
                            input_shape = var_shapes[input_var]
                            actual_dim = dim if dim >= 0 else dim + len(input_shape)
                            if 0 <= actual_dim < len(input_shape):
                                _set_bound(index_var, input_shape[actual_dim])

        elif op == 'aten.embedding.default':
            if len(parsed) >= 2:
                weight_arg = parsed[0]
                indices_arg = parsed[1]

                if indices_arg[0] == 'var':
                    index_var = indices_arg[1]
                    source_params = derives_from.get(index_var, set())
                    int_sources = source_params & int_params
                    if int_sources:
                        weight_var = weight_arg[1] if weight_arg[0] == 'var' else None
                        if weight_var and weight_var in var_shapes:
                            weight_shape = var_shapes[weight_var]
                            if weight_shape:
                                _set_bound(index_var, weight_shape[0])

    # Constraint backpropagation through mul.Tensor
    for result_var, op, args_str in operations:
        if op == 'aten.mul.Tensor':
            parsed = parse_args_simple(args_str)
            if len(parsed) >= 2 and parsed[0][0] == 'var' and parsed[1][0] == 'var':
                arg1_var = parsed[0][1]
                arg2_var = parsed[1][1]

                arg1_params = derives_from.get(arg1_var, set()) & int_params
                arg1_bound = None
                for p in arg1_params:
                    if p in index_bounds:
                        b = index_bounds[p]
                        if arg1_bound is None or b < arg1_bound:
                            arg1_bound = b

                arg2_derives = derives_from.get(arg2_var, set()) if arg2_var in derives_from else {arg2_var}
                arg2_params = arg2_derives & int_params

                product_bound = arg1_bound

                if product_bound is not None and product_bound > 1 and arg2_params:
                    inferred_b2 = product_bound // (arg1_bound - 1) + 1
                    for p in arg2_params:
                        if p not in index_bounds or index_bounds[p] > inferred_b2:
                            index_bounds[p] = inferred_b2

    # Convert param names to positions
    index_bounds_by_pos = {}
    perm_bounds_by_pos = {}

    for i, name in enumerate(param_names):
        if name in perm_bounds:
            perm_bounds_by_pos[i] = perm_bounds[name]
        elif name in index_bounds:
            index_bounds_by_pos[i] = index_bounds[name]

    return index_bounds_by_pos, perm_bounds_by_pos


def annotate_config_string(config_str, index_bounds, perm_bounds):
    """Apply inferred gen=Index(N) / gen=Perm(N) annotations to a config string.

    Parameters:
        config_str: raw config string like "(T([100, 64], f32), T([32, 128], i64), ...)"
        index_bounds: dict of param_position -> bound (for Index)
        perm_bounds: dict of param_position -> bound (for Perm)

    Returns:
        annotated config string with gen= annotations inserted
    """
    if not index_bounds and not perm_bounds:
        return config_str

    # Find all T() and S() tokens with their positions
    tokens = list(re.finditer(r'(T\([^)]+\)|S\([^)]+\))', config_str))
    if not tokens:
        return config_str

    # Build result by replacing tokens that need annotations
    result_parts = []
    last_end = 0
    param_idx = 0

    for match in tokens:
        token = match.group(1)
        start = match.start()
        end = match.end()

        # Add text between tokens
        result_parts.append(config_str[last_end:start])

        if token.startswith('T('):
            # Check if this position needs an annotation
            if param_idx in perm_bounds:
                bound = perm_bounds[param_idx]
                # Only add if not already annotated
                if 'gen=' not in token:
                    # Insert gen=Perm(N) before the closing paren
                    token = token[:-1] + f', gen=Perm({bound}))'
            elif param_idx in index_bounds:
                bound = index_bounds[param_idx]
                if 'gen=' not in token:
                    token = token[:-1] + f', gen=Index({bound}))'

        result_parts.append(token)
        last_end = end
        param_idx += 1

    # Add remaining text after last token
    result_parts.append(config_str[last_end:])
    return ''.join(result_parts)


def strip_gen_annotations(config_str):
    """Remove all gen=Index(N) and gen=Perm(N) annotations from a config string."""
    # Remove ", gen=Index(N)" and ", gen=Perm(N)" patterns
    result = re.sub(r',\s*gen=(Index|Perm)\(\d+\)', '', config_str)
    return result


def infer_bounds_for_config(repro_py, config_str):
    """Given a repro.py and a raw config string (without gen= annotations),
    infer the correct gen=Index(N)/gen=Perm(N) for each i64 tensor and
    return the annotated config string.

    Parameters:
        repro_py: Path to the repro.py file (used to read forward() method)
        config_str: raw config string without gen= annotations

    Returns:
        annotated config string with gen=Index(N)/gen=Perm(N) applied
    """
    repro_py = Path(repro_py)
    content = repro_py.read_text()

    # Run inference on the forward() body
    index_bounds, perm_bounds = infer_bounds_from_forward(content)

    # Apply annotations to the config string
    return annotate_config_string(config_str, index_bounds, perm_bounds)
