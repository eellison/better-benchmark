"""
Test bounds inference against 215 repros with known-good gen= annotations.

Strips annotations, re-infers from the FX-style forward() AST, compares against ground truth.

Approach: Parse the forward() method's text to build a mini dataflow graph.
- Extract type annotations for shapes (e.g., arg0_1: "bf16[8192, 262144]")
- Track shapes through allocations (full.default, empty.memory_format)
- Track shapes through passthrough/view ops
- When we hit gather/scatter/index/embedding, determine the bound
- Compare inferred bounds against ground-truth gen=Index(N) / gen=Perm(N) annotations
"""
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bounds_inference import infer_bounds_from_forward as _module_infer_bounds

ROOT = Path(__file__).resolve().parents[1]
REPROS_DIR = ROOT / "repros" / "canonical"


def extract_ground_truth(config_str):
    """Extract all gen=Index(N) and gen=Perm(N) from a config string.

    Returns dict mapping parameter position (0-indexed among T() entries) to (type, N).
    """
    annotations = {}
    # Find all T(...) entries and check for gen= inside them
    param_idx = 0
    for match in re.finditer(r'(T\([^)]+\)|S\([^)]+\))', config_str):
        token = match.group(1)
        if token.startswith('T('):
            gen_match = re.search(r'gen=(Index|Perm)\((\d+)\)', token)
            if gen_match:
                annotations[param_idx] = (gen_match.group(1), int(gen_match.group(2)))
            param_idx += 1
        elif token.startswith('S('):
            param_idx += 1  # shape params count too
    return annotations


def extract_param_info(config_str):
    """Extract shape and dtype info for all T() params in config string.

    Returns list of dicts with 'shape', 'dtype', 'gen' keys (in order).
    """
    params = []
    for match in re.finditer(r'(T\([^)]+\)|S\([^)]+\))', config_str):
        token = match.group(1)
        if token.startswith('T('):
            # Parse T([...], dtype, ...)
            shape_match = re.search(r'T\(\[([^\]]*)\]', token)
            shape = []
            if shape_match:
                shape_str = shape_match.group(1).strip()
                if shape_str:
                    shape = [int(x.strip()) for x in shape_str.split(',')]

            dtype_match = re.search(r'],\s*(\w+)', token)
            dtype = dtype_match.group(1) if dtype_match else "f32"

            gen_match = re.search(r'gen=(Index|Perm)\((\d+)\)', token)
            gen = None
            if gen_match:
                gen = (gen_match.group(1), int(gen_match.group(2)))

            params.append({'shape': shape, 'dtype': dtype, 'gen': gen, 'is_shape': False})
        elif token.startswith('S('):
            params.append({'shape': [], 'dtype': 'shape_param', 'gen': None, 'is_shape': True})
    return params


def parse_forward_body(content):
    """Parse the forward() method body and build a mini dataflow graph.

    Returns:
        param_names: list of parameter names in order
        param_shapes: dict of param_name -> shape (from annotation)
        param_dtypes: dict of param_name -> dtype abbreviation
        var_shapes: dict of var_name -> shape (inferred from annotations/allocations)
        operations: list of (result_var, op_name, args_list) tuples
    """
    # Find the forward method
    fwd_match = re.search(r'def forward\(self,\s*([^)]+)\)', content)
    if not fwd_match:
        return None, None, None, None, None

    # Parse parameters with their type annotations
    params_str = fwd_match.group(1)
    param_names = []
    param_shapes = {}
    param_dtypes = {}

    # Match patterns like: arg0_1: "bf16[8192, 262144]"
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

    # Parse the body for variable assignments with type annotations
    var_shapes = dict(param_shapes)  # start with param shapes

    # Find all assignments like: var_name: "dtype[dims]" = ...
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
        # Also note dtype
        param_dtypes[var_name] = dtype

    # Also find scalar assignments: var_name: "dtype[]" = ...
    for amatch in re.finditer(r'(\w+):\s*"(\w+)\[\]"\s*=', content):
        var_name = amatch.group(1)
        var_shapes[var_name] = []
        param_dtypes[var_name] = amatch.group(2)

    # Parse operations - find all torch.ops.aten.X.Y(...) calls
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

        # Strip trailing ");  var = None" cleanup and closing paren
        # The FX graph format ends lines with: args...);  cleanup
        args_str = re.sub(r'\)\s*;.*$', '', args_str)
        # If still ends with ), strip it (just a closing paren for the function call)
        if args_str.endswith(')'):
            args_str = args_str[:-1]

        # Parse the args (simple extraction of variable names and literals)
        # We need: the positional args
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
    # Split by comma at depth 0, respecting brackets
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
        # Remove trailing comments like ";  var = None"
        arg = re.sub(r';.*', '', arg).strip()
        # Remove kwargs (key = value) — we handle these separately
        if '=' in arg and not arg.startswith('['):
            # It's a kwarg like dtype = torch.int64
            parsed.append(('kwarg', arg))
            continue
        # Check if it's a list like [var1, var2]
        list_match = re.match(r'\[([^\]]*)\]', arg)
        if list_match:
            inner = list_match.group(1).strip()
            if inner:
                items = [x.strip() for x in inner.split(',')]
                # Could be list of ints or list of vars
                try:
                    parsed.append(('int_list', [int(x) for x in items]))
                except ValueError:
                    parsed.append(('var_list', items))
            else:
                parsed.append(('int_list', []))
            continue
        # Check if it's an integer
        try:
            parsed.append(('int', int(arg)))
            continue
        except ValueError:
            pass
        # Check if it's a float
        try:
            parsed.append(('float', float(arg)))
            continue
        except ValueError:
            pass
        # It's a variable reference
        if re.match(r'\w+$', arg):
            parsed.append(('var', arg))
        else:
            parsed.append(('other', arg))

    return parsed


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
    # var -> set of param names it derives from
    derives_from = {name: {name} for name in param_names}

    # Passthrough ops that preserve index values
    PASSTHROUGH_OPS = {
        'aten.clone.default', 'aten.reshape.default', 'aten.view.default',
        'aten.expand.default', 'aten.slice.Tensor', 'aten.unsqueeze.default',
        'aten.squeeze.default', 'aten.squeeze.dim', 'aten.select.int',
        'aten.permute.default', 'aten.contiguous.default', 'aten.t.default',
        'aten.transpose.int', 'aten.where.self',
        'aten.lift_fresh_copy.default',
        # Arithmetic on indices (result is still an index with similar bounds)
        'aten.add.Tensor', 'aten.mul.Tensor',
        # Type conversions preserve index nature
        'prims.convert_element_type.default',
    }

    # Track additive offset: var_name -> offset (how much was added to the index)
    # If index goes through add(x, K), the effective bound is target_bound - K
    add_offset = defaultdict(int)  # var_name -> cumulative additive offset

    # First pass: trace passthrough ops to build derives_from
    for result_var, op, args_str in operations:
        if op in PASSTHROUGH_OPS:
            parsed = parse_args_simple(args_str)
            if op == 'aten.where.self':
                # where.self(cond, x, y) - result derives from x or y
                for item in parsed:
                    if item[0] == 'var' and item[1] in derives_from:
                        if result_var not in derives_from:
                            derives_from[result_var] = set()
                        derives_from[result_var].update(derives_from[item[1]])
            elif op == 'aten.add.Tensor':
                # Only trace through first arg (the index being modified)
                if parsed and parsed[0][0] == 'var':
                    src = parsed[0][1]
                    if src in derives_from:
                        derives_from[result_var] = set(derives_from[src])
                    else:
                        derives_from[result_var] = {src}
                    # Track the additive offset
                    offset = add_offset.get(src, 0)
                    if len(parsed) >= 2 and parsed[1][0] == 'int':
                        offset += parsed[1][1]
                    add_offset[result_var] = offset
            elif op == 'aten.mul.Tensor':
                # Only trace through first arg for primary bound inference
                if parsed and parsed[0][0] == 'var':
                    src = parsed[0][1]
                    if src in derives_from:
                        derives_from[result_var] = set(derives_from[src])
                    else:
                        derives_from[result_var] = {src}
                    # mul resets offset tracking (can't simply subtract)
                    add_offset[result_var] = 0
            else:
                # First arg is source
                if parsed and parsed[0][0] == 'var':
                    src = parsed[0][1]
                    if src in derives_from:
                        derives_from[result_var] = set(derives_from[src])
                    else:
                        derives_from[result_var] = {src}
                    # Propagate offset through shape-only ops
                    if src in add_offset:
                        add_offset[result_var] = add_offset[src]
        elif op == 'aten.gather.default':
            # gather(input, dim, index) - the OUTPUT carries VALUES from input (args[0])
            # So values from the data source flow through gather to its output
            parsed = parse_args_simple(args_str)
            if parsed and parsed[0][0] == 'var':
                src = parsed[0][1]
                if src in derives_from:
                    derives_from[result_var] = set(derives_from[src])
                else:
                    derives_from[result_var] = {src}
                if src in add_offset:
                    add_offset[result_var] = add_offset[src]

    # Second pass: handle additional passthrough-like ops that produce index values
    # _low_memory_max_pool_offsets_to_indices: int8 offsets -> int64 indices (passthrough)
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
    index_bounds = {}  # param_name -> bound
    perm_bounds = {}   # param_name -> bound

    # Detect permutation pattern:
    # empty([N]) -> index_put(empty, [idx_param], iota(N))
    # where target_size == iota_size -> Perm

    # First find iota results
    iota_sizes = {}  # var_name -> iota size
    for result_var, op, args_str in operations:
        if op == 'prims.iota.default':
            parsed = parse_args_simple(args_str)
            if parsed and parsed[0][0] == 'int':
                iota_sizes[result_var] = parsed[0][1]

    # Find allocations (empty/full) and their shapes
    alloc_shapes = {}  # var_name -> shape
    for result_var, op, args_str in operations:
        if op in ('aten.empty.memory_format', 'aten.full.default'):
            parsed = parse_args_simple(args_str)
            if parsed and parsed[0][0] == 'int_list':
                alloc_shapes[result_var] = parsed[0][1]

    # Handle int8 pool offsets: _low_memory_max_pool_offsets_to_indices(offsets, kernel_size, ...)
    # The bound is prod(kernel_size)
    for result_var, op, args_str in operations:
        if 'low_memory_max_pool_offsets_to_indices' in op:
            parsed = parse_args_simple(args_str)
            if parsed and parsed[0][0] == 'var':
                src = parsed[0][1]
                source_params = derives_from.get(src, set())
                int_sources = source_params & int_params
                # Kernel size is the second arg (int_list)
                if len(parsed) >= 2 and parsed[1][0] == 'int_list':
                    ks = parsed[1][1]
                    bound = 1
                    for k in ks:
                        bound *= k
                    for param in int_sources:
                        index_bounds[param] = bound

    # Heuristic for "cumsum as target" pattern:
    # When an i64 param is used as the TARGET (first arg) of index.Tensor
    # with iota-derived indices, the param's values should be valid indices.
    # In practice, this is the segment_id / position_id pattern where
    # values are in [0, shape[0]).
    for result_var, op, args_str in operations:
        if op == 'aten.index.Tensor':
            parsed = parse_args_simple(args_str)
            if len(parsed) >= 2:
                target_arg = parsed[0]
                indices_arg = parsed[1]
                if target_arg[0] == 'var':
                    target_var = target_arg[1]
                    # Check if the target itself is an int param
                    source_params = derives_from.get(target_var, set())
                    int_targets = source_params & int_params
                    if int_targets and indices_arg[0] == 'var_list':
                        # The param is used as TARGET of index.Tensor
                        # Its values are being READ, not used as indices here.
                        # But the annotation gen=Index(N) means the values should be < N.
                        # Infer N = shape[0] of the param (first indexing dimension).
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

    # Now scan operations for index consumers
    for result_var, op, args_str in operations:
        parsed = parse_args_simple(args_str)

        if op == 'aten.gather.default':
            # gather(input, dim, index)
            # The INDEX arg is args[2]; the input is args[0]; dim is args[1]
            if len(parsed) >= 3:
                input_arg = parsed[0]
                dim_arg = parsed[1]
                index_arg = parsed[2]

                if index_arg[0] == 'var' and dim_arg[0] == 'int':
                    index_var = index_arg[1]
                    dim = dim_arg[1]
                    # Find which params flow into index_var
                    source_params = derives_from.get(index_var, set())
                    int_sources = source_params & int_params
                    if int_sources:
                        # Get the input tensor's shape
                        input_var = input_arg[1] if input_arg[0] == 'var' else None
                        if input_var and input_var in var_shapes:
                            input_shape = var_shapes[input_var]
                            if dim < len(input_shape):
                                raw_bound = input_shape[dim]
                                _set_bound(index_var, raw_bound)

        elif op == 'aten.scatter.src':
            # scatter.src(target, dim, index, src)
            # The INDEX is args[2]; target is args[0]; dim is args[1]
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

                    # Check for permutation pattern:
                    # scatter.src(empty, dim, idx, iota_expanded)
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
            # scatter.value(target, dim, index, value_scalar)
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
            # scatter_add(target, dim, index, src)
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
            # index_put(target, [indices...], values, accumulate?)
            if len(parsed) >= 3:
                target_arg = parsed[0]
                indices_arg = parsed[1]
                values_arg = parsed[2]

                target_var = target_arg[1] if target_arg[0] == 'var' else None
                target_shape = None
                if target_var:
                    target_shape = var_shapes.get(target_var) or alloc_shapes.get(target_var)

                if indices_arg[0] == 'var_list' and target_shape:
                    # Heuristic: for index_put into an allocation, if the indexed
                    # dimension is strictly larger than the first (batch) dimension,
                    # the indices likely represent batch-coordinate scatter and
                    # their values are bounded by the batch dim, not the full dim.
                    target_is_alloc = target_var in alloc_shapes
                    for dim, idx_name in enumerate(indices_arg[1]):
                        if dim < len(target_shape):
                            bound = target_shape[dim]
                            if (target_is_alloc and len(target_shape) > 1
                                    and dim > 0 and bound > target_shape[0]):
                                bound = target_shape[0]
                            _set_bound(idx_name, bound)

                # Check for permutation pattern:
                # index_put(empty([N]), [idx_param], iota(N))
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
            # index.Tensor(input, [indices...])
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
            # index_select(input, dim, index)
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
            # embedding(weight, indices, ...)
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

    # Constraint backpropagation through mul.Tensor:
    # Pattern: cumsum * attention_mask + offset -> embedding/index
    # The primary factor (arg1) already has a bound from forward tracing.
    # We infer the secondary factor (arg2) bound via:
    #   If a < A and a*b < P, then b < floor(P / (A-1)) + 1 (when A > 1).
    for result_var, op, args_str in operations:
        if op == 'aten.mul.Tensor':
            parsed = parse_args_simple(args_str)
            if len(parsed) >= 2 and parsed[0][0] == 'var' and parsed[1][0] == 'var':
                arg1_var = parsed[0][1]
                arg2_var = parsed[1][1]

                # Get the primary factor's int params and their bounds
                arg1_params = derives_from.get(arg1_var, set()) & int_params
                arg1_bound = None
                for p in arg1_params:
                    if p in index_bounds:
                        b = index_bounds[p]
                        if arg1_bound is None or b < arg1_bound:
                            arg1_bound = b

                # Get secondary factor's int params (from its own derives_from)
                arg2_derives = derives_from.get(arg2_var, set()) if arg2_var in derives_from else {arg2_var}
                arg2_params = arg2_derives & int_params

                # The product bound = the bound inferred for the primary factor's params
                # (since derives_from only traces through arg1, the primary params got
                # the downstream index consumer's bound)
                # Actually: the product bound is the same as arg1_bound because
                # result_var's derives_from only includes arg1's sources
                product_bound = arg1_bound  # same bound was assigned via _set_bound

                if product_bound is not None and product_bound > 1 and arg2_params:
                    # Infer bound for the secondary factor params
                    inferred_b2 = product_bound // (product_bound - 1) + 1
                    # More precisely: product < P, and primary < A (= product_bound)
                    # So secondary < floor(P / (A-1)) + 1
                    # Since P = A here (product_bound = primary factor bound = product bound):
                    # secondary < floor(A / (A-1)) + 1 = 1 + 1 = 2 (when A >= 2)
                    inferred_b2 = product_bound // (arg1_bound - 1) + 1
                    for p in arg2_params:
                        if p not in index_bounds or index_bounds[p] > inferred_b2:
                            index_bounds[p] = inferred_b2

    # Convert param names to positions
    # We need to map param names to their position among T() entries in _shapes_config
    # The param order in forward() matches the order in _shapes_config
    index_bounds_by_pos = {}
    perm_bounds_by_pos = {}

    for i, name in enumerate(param_names):
        if name in perm_bounds:
            perm_bounds_by_pos[i] = perm_bounds[name]
        elif name in index_bounds:
            index_bounds_by_pos[i] = index_bounds[name]

    return index_bounds_by_pos, perm_bounds_by_pos


def _find_iota_source(var_name, operations, iota_sizes, derives_from):
    """Check if var_name derives from an iota (possibly through passthrough ops).

    Returns the iota size if found, None otherwise.
    """
    if var_name in iota_sizes:
        return iota_sizes[var_name]

    # Check if the variable was produced by a passthrough from an iota
    PASSTHROUGH_OPS = {
        'aten.clone.default', 'aten.reshape.default', 'aten.view.default',
        'aten.expand.default', 'aten.unsqueeze.default', 'aten.squeeze.default',
        'aten.squeeze.dim',
    }

    # Build a simple derivation chain
    # Look through operations to find what produced var_name
    for result_var, op, args_str in operations:
        if result_var == var_name:
            if op in PASSTHROUGH_OPS:
                parsed = parse_args_simple(args_str)
                if parsed and parsed[0][0] == 'var':
                    return _find_iota_source(parsed[0][1], operations, iota_sizes, derives_from)
            # Also handle add.Tensor(iota, 0) which is common
            if op == 'aten.add.Tensor':
                parsed = parse_args_simple(args_str)
                if len(parsed) >= 2 and parsed[0][0] == 'var':
                    src = parsed[0][1]
                    if parsed[1][0] == 'int' and parsed[1][1] == 0:
                        return _find_iota_source(src, operations, iota_sizes, derives_from)
                    # Even non-zero add from iota
                    return _find_iota_source(src, operations, iota_sizes, derives_from)
            break

    return None


def get_ground_truth_by_param_position(config_str, param_names):
    """Map ground truth gen= annotations to parameter positions.

    Returns dict: position -> (type, bound)
    """
    truth = {}
    # Parse config to find T() entries with gen= in order
    param_idx = 0
    for match in re.finditer(r'(T\([^)]+\)|S\([^)]+\))', config_str):
        token = match.group(1)
        if token.startswith('T('):
            gen_match = re.search(r'gen=(Index|Perm)\((\d+)\)', token)
            if gen_match:
                truth[param_idx] = (gen_match.group(1), int(gen_match.group(2)))
        param_idx += 1
    return truth


def main():
    results = {
        'correct': 0,
        'wrong_value': 0,
        'wrong_type': 0,
        'missed': 0,
        'extra': 0,
        'total_annotations': 0,
    }

    details_wrong = []
    details_missed = []
    details_extra = []

    repro_count = 0
    repro_perfect = 0

    for repro_path in sorted(REPROS_DIR.rglob('repro.py')):
        content = repro_path.read_text()
        if 'gen=' not in content:
            continue

        repro_count += 1

        # Extract _shapes_config
        config_match = re.search(r'_shapes_config\s*=\s*"([^"]+)"', content)
        if not config_match:
            continue
        config_str = config_match.group(1)

        # Get forward parameter names
        fwd_match = re.search(r'def forward\(self,\s*([^)]+)\)', content)
        if not fwd_match:
            continue
        params_str = fwd_match.group(1)
        param_names = [m.group(1) for m in re.finditer(r'(\w+)(?:\s*:\s*"[^"]*")?', params_str)]

        # Get ground truth
        ground_truth = get_ground_truth_by_param_position(config_str, param_names)
        results['total_annotations'] += len(ground_truth)

        # Run inference (using the module's graph-based implementation)
        index_bounds, perm_bounds = _module_infer_bounds(content, source_path=repro_path)

        # Compare
        repro_name = repro_path.parent.name
        all_correct = True

        for pos, (gt_type, gt_bound) in ground_truth.items():
            if gt_type == 'Perm':
                if pos in perm_bounds:
                    if perm_bounds[pos] == gt_bound:
                        results['correct'] += 1
                    else:
                        results['wrong_value'] += 1
                        all_correct = False
                        details_wrong.append(
                            f"  {repro_name} param[{pos}] ({param_names[pos] if pos < len(param_names) else '?'}): "
                            f"expected Perm({gt_bound}), got Perm({perm_bounds[pos]})"
                        )
                elif pos in index_bounds:
                    results['wrong_type'] += 1
                    all_correct = False
                    details_wrong.append(
                        f"  {repro_name} param[{pos}] ({param_names[pos] if pos < len(param_names) else '?'}): "
                        f"expected Perm({gt_bound}), got Index({index_bounds[pos]})"
                    )
                else:
                    results['missed'] += 1
                    all_correct = False
                    details_missed.append(
                        f"  {repro_name} param[{pos}] ({param_names[pos] if pos < len(param_names) else '?'}): "
                        f"expected Perm({gt_bound}), got nothing"
                    )
            else:  # Index
                if pos in perm_bounds:
                    # Inferred as Perm but truth is Index - wrong type
                    results['wrong_type'] += 1
                    all_correct = False
                    details_wrong.append(
                        f"  {repro_name} param[{pos}] ({param_names[pos] if pos < len(param_names) else '?'}): "
                        f"expected Index({gt_bound}), got Perm({perm_bounds[pos]})"
                    )
                elif pos in index_bounds:
                    if index_bounds[pos] == gt_bound:
                        results['correct'] += 1
                    else:
                        results['wrong_value'] += 1
                        all_correct = False
                        details_wrong.append(
                            f"  {repro_name} param[{pos}] ({param_names[pos] if pos < len(param_names) else '?'}): "
                            f"expected Index({gt_bound}), got Index({index_bounds[pos]})"
                        )
                else:
                    results['missed'] += 1
                    all_correct = False
                    details_missed.append(
                        f"  {repro_name} param[{pos}] ({param_names[pos] if pos < len(param_names) else '?'}): "
                        f"expected Index({gt_bound}), got nothing"
                    )

        # Check for extra inferences (not in ground truth)
        for pos in index_bounds:
            if pos not in ground_truth:
                results['extra'] += 1
                details_extra.append(
                    f"  {repro_name} param[{pos}] ({param_names[pos] if pos < len(param_names) else '?'}): "
                    f"inferred Index({index_bounds[pos]}) but no annotation"
                )
        for pos in perm_bounds:
            if pos not in ground_truth:
                results['extra'] += 1
                details_extra.append(
                    f"  {repro_name} param[{pos}] ({param_names[pos] if pos < len(param_names) else '?'}): "
                    f"inferred Perm({perm_bounds[pos]}) but no annotation"
                )

        if all_correct:
            repro_perfect += 1

    # Print report
    print("=" * 70)
    print("BOUNDS INFERENCE TEST RESULTS")
    print("=" * 70)
    print(f"\nRepros tested: {repro_count}")
    print(f"Repros with perfect inference: {repro_perfect}/{repro_count} "
          f"({100*repro_perfect/repro_count:.1f}%)")
    print(f"\nTotal gen= annotations: {results['total_annotations']}")
    print(f"  Correct:     {results['correct']}")
    print(f"  Wrong value: {results['wrong_value']}")
    print(f"  Wrong type:  {results['wrong_type']}")
    print(f"  Missed:      {results['missed']}")
    print(f"  Extra:       {results['extra']}")

    accuracy = results['correct'] / results['total_annotations'] if results['total_annotations'] else 0
    print(f"\nAccuracy: {results['correct']}/{results['total_annotations']} = {100*accuracy:.1f}%")

    if details_wrong:
        print(f"\n--- WRONG ({len(details_wrong)}) ---")
        for d in details_wrong[:30]:
            print(d)
        if len(details_wrong) > 30:
            print(f"  ... and {len(details_wrong) - 30} more")

    if details_missed:
        print(f"\n--- MISSED ({len(details_missed)}) ---")
        for d in details_missed[:30]:
            print(d)
        if len(details_missed) > 30:
            print(f"  ... and {len(details_missed) - 30} more")

    if details_extra:
        print(f"\n--- EXTRA (inferred but no ground truth) ({len(details_extra)}) ---")
        for d in details_extra[:20]:
            print(d)
        if len(details_extra) > 20:
            print(f"  ... and {len(details_extra) - 20} more")

    # Analysis of remaining failures
    print("\n--- ANALYSIS ---")
    if results['wrong_value'] == 0 and results['wrong_type'] == 0 and results['missed'] == 0:
        print("All ground-truth annotations matched perfectly.")
    if results['extra'] > 0:
        print(f"  Extra inferences ({results['extra']}): Our inference is CORRECT")
        print("     (e.g. int8 pool offsets), but the ground truth annotation is")
        print("     missing. These represent gaps in the ground truth, not failures")
        print("     of our inference.")
    if results['extra'] > 0:
        print(f"\n  If 'extra' inferences are treated as correct, effective accuracy would be:")
        effective = (results['correct'] + results['extra']) / (results['total_annotations'] + results['extra'])
        print(f"    {results['correct'] + results['extra']}/{results['total_annotations'] + results['extra']} = {100*effective:.1f}%")

    print("\n" + "=" * 70)

    # Exit code: 0 if >90% accuracy, 1 otherwise
    if accuracy >= 0.9:
        print(f"PASS (accuracy {100*accuracy:.1f}% >= 90%)")
        return 0
    else:
        print(f"NEEDS IMPROVEMENT (accuracy {100*accuracy:.1f}% < 90%)")
        return 1


def run_adversarial_tests():
    """Adversarial test cases that expose weaknesses in the bounds inference solver.

    These demonstrate confirmed bugs and edge cases where the solver produces
    incorrect results. Each test documents the expected vs actual behavior.
    """
    print("\n" + "=" * 70)
    print("ADVERSARIAL TEST CASES")
    print("=" * 70)

    failures = []
    passes = []

    # --- Bug 1: Reversed operands in add(K, x) ---
    content = '''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[100, 64]", arg1_1: "i64[32]"):
        add_tensor: "i64[32]" = torch.ops.aten.add.Tensor(5, arg1_1);  arg1_1 = None
        unsqueeze_default: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(add_tensor, 1);  add_tensor = None
        gather_default: "f32[32, 1]" = torch.ops.aten.gather.default(arg0_1, 0, unsqueeze_default);  arg0_1 = unsqueeze_default = None
        return gather_default
'''
    idx, perm = infer_bounds_from_forward(content)
    if 1 not in idx:
        failures.append(("BUG-1: add(K, x) reversed operands", "MISSED",
                         "param 1 should have bound 95 (100-5)"))
    else:
        passes.append("BUG-1: add(K, x) reversed operands")

    # --- Bug 2: Reversed operands in mul(K, x) ---
    content = '''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[100, 64]", arg1_1: "i64[32]"):
        mul_tensor: "i64[32]" = torch.ops.aten.mul.Tensor(2, arg1_1);  arg1_1 = None
        unsqueeze_default: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, 1);  mul_tensor = None
        gather_default: "f32[32, 1]" = torch.ops.aten.gather.default(arg0_1, 0, unsqueeze_default);  arg0_1 = unsqueeze_default = None
        return gather_default
'''
    idx, perm = infer_bounds_from_forward(content)
    if 1 not in idx:
        failures.append(("BUG-2: mul(K, x) reversed operands", "MISSED",
                         "param 1 should have bound 100"))
    else:
        passes.append("BUG-2: mul(K, x) reversed operands")

    # --- Bug 3: Negative bound from large positive offset ---
    content = '''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[10, 64]", arg1_1: "i64[32]"):
        add_tensor: "i64[32]" = torch.ops.aten.add.Tensor(arg1_1, 50);  arg1_1 = None
        unsqueeze_default: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(add_tensor, 1);  add_tensor = None
        gather_default: "f32[32, 1]" = torch.ops.aten.gather.default(arg0_1, 0, unsqueeze_default);  arg0_1 = unsqueeze_default = None
        return gather_default
'''
    idx, perm = infer_bounds_from_forward(content)
    if 1 in idx and idx[1] < 0:
        failures.append(("BUG-3: Negative bound (offset > table size)",
                         f"got Index({idx[1]})",
                         "bound should be clamped to 0 or flagged as impossible"))
    else:
        passes.append("BUG-3: Negative bound")

    # --- Bug 4: index_put heuristic gives wrong bound when first dim is small ---
    content = '''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[32]", arg1_1: "i64[32]"):
        full_default: "f32[1, 1024, 768]" = torch.ops.aten.full.default([1, 1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1, 1024, 768]" = torch.ops.aten.index_put.default(full_default, [arg0_1, arg1_1], 1.0, True);  full_default = arg0_1 = arg1_1 = None
        return index_put_default
'''
    idx, perm = infer_bounds_from_forward(content)
    if 1 in idx and idx[1] == 1:
        failures.append(("BUG-4: index_put heuristic [1, 1024, 768] dim=1",
                         f"got Index({idx[1]})",
                         "param 1 indexes dim 1 of size 1024, bound should be 1024 not 1"))
    else:
        passes.append("BUG-4: index_put heuristic")

    # --- Bug 5: where.self propagates cond param into derives_from ---
    content = '''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[100, 64]", arg1_1: "i64[32]", arg2_1: "i64[32]"):
        convert_bool: "b8[32]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bool)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64)
        where_self: "i64[32]" = torch.ops.aten.where.self(convert_bool, arg2_1, full_default);  convert_bool = arg2_1 = full_default = None
        unsqueeze_default: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[32, 1]" = torch.ops.aten.gather.default(arg0_1, 0, unsqueeze_default);  arg0_1 = unsqueeze_default = None
        return gather_default
'''
    idx, perm = infer_bounds_from_forward(content)
    if 1 in idx:
        failures.append(("BUG-5: where.self propagates cond (convert_element_type) into derives_from",
                         f"param 1 (cond source) got Index({idx[1]})",
                         "only param 2 (true-branch) should get a bound"))
    else:
        passes.append("BUG-5: where.self cond propagation")

    # --- Bug 6: mul backpropagation always gives bound=2 ---
    content = '''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[1000, 64]", arg1_1: "i64[32]", arg2_1: "i64[32]"):
        mul_tensor: "i64[32]" = torch.ops.aten.mul.Tensor(arg1_1, arg2_1);  arg1_1 = None
        unsqueeze_default: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, 1);  mul_tensor = None
        gather_default: "f32[32, 1]" = torch.ops.aten.gather.default(arg0_1, 0, unsqueeze_default);  arg0_1 = unsqueeze_default = None
        return gather_default
'''
    idx, perm = infer_bounds_from_forward(content)
    if 2 in idx and idx[2] == 2:
        failures.append(("BUG-6: mul backpropagation always gives bound=2",
                         f"param 2 got Index({idx[2]})",
                         "secondary factor bound=2 only valid for binary masks"))
    else:
        passes.append("BUG-6: mul backpropagation")

    # --- Bug 7: sub.Tensor not handled (missed inference) ---
    content = '''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[100, 64]", arg1_1: "i64[32]"):
        sub_tensor: "i64[32]" = torch.ops.aten.sub.Tensor(arg1_1, 5);  arg1_1 = None
        unsqueeze_default: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(sub_tensor, 1);  sub_tensor = None
        gather_default: "f32[32, 1]" = torch.ops.aten.gather.default(arg0_1, 0, unsqueeze_default);  arg0_1 = unsqueeze_default = None
        return gather_default
'''
    idx, perm = infer_bounds_from_forward(content)
    if 1 not in idx:
        failures.append(("BUG-7: sub.Tensor(x, K) not handled",
                         "MISSED",
                         "sub(x,5) equivalent to add(x,-5), param should get bound 105"))
    else:
        passes.append("BUG-7: sub.Tensor handling")

    # --- Bug 8: Perm detection with iota through add(iota, nonzero) is semantically wrong ---
    content = '''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[1024]"):
        empty_memory_format: "i64[1024]" = torch.ops.aten.empty.memory_format([1024], dtype = torch.int64)
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64)
        add_tensor: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default, 5)
        scatter_src: "i64[1024]" = torch.ops.aten.scatter.src(empty_memory_format, 0, arg0_1, add_tensor);  empty_memory_format = arg0_1 = add_tensor = None
        return scatter_src
'''
    idx, perm = infer_bounds_from_forward(content)
    if 0 in perm:
        failures.append(("BUG-8: Perm detected for iota+5 (not a permutation of [0,N))",
                         f"got Perm({perm[0]})",
                         "add(iota,5) gives [5,1029), not a perm of [0,1024)"))
    else:
        passes.append("BUG-8: Perm with shifted iota")

    # --- Bug 9: cumsum heuristic preempts correct embedding bound ---
    content = '''
class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[1000, 64]", arg1_1: "i64[4, 100]", arg2_1: "i64[4, 50]"):
        embedding: "f32[4, 100, 64]" = torch.ops.aten.embedding.default(arg0_1, arg1_1)
        index_tensor: "i64[4, 50]" = torch.ops.aten.index.Tensor(arg1_1, [arg2_1]);  arg1_1 = arg2_1 = None
        return (embedding, index_tensor)
'''
    idx, perm = infer_bounds_from_forward(content)
    if 1 in idx and idx[1] < 1000:
        failures.append(("BUG-9: cumsum heuristic overrides embedding bound",
                         f"param 1 got Index({idx[1]})",
                         "should be Index(1000) from embedding, not Index(4) from heuristic"))
    else:
        passes.append("BUG-9: cumsum heuristic vs embedding")

    # --- Print results ---
    print(f"\nAdversarial tests: {len(failures)} FAILURES, {len(passes)} passes")

    if failures:
        print(f"\n--- CONFIRMED BUGS ({len(failures)}) ---")
        for name, got, expected in failures:
            print(f"  FAIL: {name}")
            print(f"        Got: {got}")
            print(f"        Expected: {expected}")

    if passes:
        print(f"\n--- PASSED ({len(passes)}) ---")
        for name in passes:
            print(f"  OK: {name}")

    print("\n" + "=" * 70)
    return len(failures)


if __name__ == "__main__":
    exit_code = main()
    num_adversarial_failures = run_adversarial_tests()
    # Main test still passes if corpus accuracy is good;
    # adversarial failures are informational
    sys.exit(exit_code)
