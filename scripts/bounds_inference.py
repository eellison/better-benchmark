"""
Reusable bounds inference for i64 tensor parameters in FX-style forward() methods.

Uses torch.fx.symbolic_trace to build the actual FX graph, then infers
gen=Index(N) / gen=Perm(N) annotations for integer tensor parameters based on
how they are consumed (gather, scatter, embedding, index_put, etc.).

Primary entry point:
    infer_bounds_for_config(repro_py, config_str) -> annotated config string
"""
import re
import math
import importlib.util
from pathlib import Path
from collections import defaultdict

import torch
import torch.fx as fx


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


def _parse_type_annotation(type_str):
    """Parse a type annotation string like 'i64[32, 128]' into (dtype, shape).

    Returns (dtype_str, shape_list) or (None, None) if unparseable.
    """
    if type_str is None:
        return None, None
    if isinstance(type_str, str):
        m = re.match(r'(\w+)\[([^\]]*)\]', type_str)
        if m:
            dtype = m.group(1)
            dims_str = m.group(2).strip()
            shape = [int(x.strip()) for x in dims_str.split(',')] if dims_str else []
            return dtype, shape
    return None, None


def _parse_source_shapes(content):
    """Parse all variable shape annotations from the source text.

    Returns dict: var_name -> shape (list of ints)
    Also returns dict: var_name -> dtype string
    """
    var_shapes = {}
    var_dtypes = {}

    # Match patterns like: var_name: "dtype[d1, d2, ...]" = ...
    # Also matches parameters: name: "dtype[d1, d2]"
    for m in re.finditer(r'(\w+):\s*"(\w+)\[([^\]]*)\]"', content):
        vname = m.group(1)
        dtype = m.group(2)
        dims_str = m.group(3).strip()
        shape = [int(x.strip()) for x in dims_str.split(',')] if dims_str else []
        var_shapes[vname] = shape
        var_dtypes[vname] = dtype

    return var_shapes, var_dtypes


def _load_repro_graph(repro_py):
    """Load a repro module and trace it to get the FX graph.

    Returns (GraphModule, Repro instance) or raises on failure.
    """
    repro_py = Path(repro_py)
    spec = importlib.util.spec_from_file_location("repro", repro_py)
    mod = importlib.util.module_from_spec(spec)
    mod.device = torch.device
    mod.inf = math.inf
    mod.nan = math.nan
    spec.loader.exec_module(mod)
    repro = mod.Repro()
    gm = fx.symbolic_trace(repro)
    return gm, repro


def _is_iota_tensor(gm, node):
    """Check if a node is (or derives from) an iota tensor.

    Returns the iota size if found, None otherwise.
    Handles:
    - get_attr nodes pointing to tensors containing 0..N-1
    - prims.iota.default calls with static extent
    - passthrough ops (expand, view, clone, unsqueeze, etc.) from an iota
    """
    visited = set()

    def _check(n):
        if n in visited:
            return None
        visited.add(n)

        if n.op == 'get_attr':
            # Check if the tensor is an iota (values 0..N-1)
            try:
                tensor = getattr(gm, n.target)
                if tensor.dtype in (torch.int64, torch.int32) and tensor.numel() > 0:
                    flat = tensor.flatten()
                    N = flat.numel()
                    expected = torch.arange(N, dtype=tensor.dtype, device=tensor.device)
                    if torch.equal(flat, expected):
                        return N
            except (AttributeError, RuntimeError):
                pass
            return None

        if n.op == 'call_function':
            target = n.target
            # Direct iota call
            if target == torch.ops.prims.iota.default:
                extent = n.args[0] if n.args else None
                if isinstance(extent, int):
                    return extent
                return None

            # Passthrough ops
            PASSTHROUGH_TARGETS = {
                torch.ops.aten.clone.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten.view.default,
                torch.ops.aten.expand.default,
                torch.ops.aten.unsqueeze.default,
                torch.ops.aten.squeeze.default,
                torch.ops.aten.squeeze.dim,
            }
            if target in PASSTHROUGH_TARGETS:
                if n.args and isinstance(n.args[0], fx.Node):
                    return _check(n.args[0])

            # add.Tensor(iota, 0) or add.Tensor(iota, K)
            if target == torch.ops.aten.add.Tensor:
                if n.args and isinstance(n.args[0], fx.Node):
                    return _check(n.args[0])

        return None

    return _check(node)


def _get_node_shape(node, var_shapes, gm):
    """Get the shape for a node, using multiple sources.

    Priority:
    1. Source text annotations (var_shapes dict, keyed by node.name)
    2. node.type annotation (for placeholders)
    3. get_attr tensor shape
    4. Allocation args (for empty/full ops)
    """
    # Source text annotations
    if node.name in var_shapes:
        return var_shapes[node.name]

    # Placeholder type annotations
    if node.op == 'placeholder' and node.type is not None:
        _, shape = _parse_type_annotation(str(node.type))
        if shape is not None:
            return shape

    # get_attr tensor shape
    if node.op == 'get_attr':
        try:
            tensor = getattr(gm, node.target)
            if hasattr(tensor, 'shape'):
                return list(tensor.shape)
        except AttributeError:
            pass

    # Allocation ops
    if node.op == 'call_function':
        if node.target in (torch.ops.aten.empty.memory_format, torch.ops.aten.full.default):
            if node.args and isinstance(node.args[0], (list, tuple)):
                shape_arg = node.args[0]
                if all(isinstance(x, int) for x in shape_arg):
                    return list(shape_arg)

    return None


def _is_alloc_node(node):
    """Check if a node is an allocation (empty/full)."""
    if node.op == 'call_function':
        return node.target in (torch.ops.aten.empty.memory_format,
                               torch.ops.aten.full.default)
    return False


def infer_bounds_from_forward(content, source_path=None):
    """Infer index bounds and permutation indices from the forward() method.

    Uses torch.fx.symbolic_trace to get the actual FX graph, then performs
    dataflow analysis on the graph nodes.

    Parameters:
        content: source code string containing the Repro class
        source_path: optional Path to the original .py file (enables proper
                     imports for repros that reference repro_harness)

    Returns:
        index_bounds: dict of param_position -> bound (for Index)
        perm_bounds: dict of param_position -> bound (for Perm)
    """
    # Parse source shapes (needed for intermediate node shapes)
    var_shapes, var_dtypes = _parse_source_shapes(content)

    # Load and trace the module
    import tempfile
    import os

    if source_path is not None:
        # Use the actual file path (preserves sys.path resolution)
        try:
            gm, repro = _load_repro_graph(source_path)
        except Exception:
            return {}, {}
    else:
        # Write content to a temp file for tracing (unit tests)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            gm, repro = _load_repro_graph(tmp_path)
        except Exception:
            return {}, {}
        finally:
            os.unlink(tmp_path)

    return _infer_bounds_from_graph(gm, var_shapes, var_dtypes)


def _infer_bounds_from_graph(gm, var_shapes, var_dtypes):
    """Core graph-based bounds inference.

    Parameters:
        gm: fx.GraphModule (traced graph)
        var_shapes: dict of var_name -> shape (from source annotations)
        var_dtypes: dict of var_name -> dtype string

    Returns:
        index_bounds_by_pos: dict of param_position -> bound (for Index)
        perm_bounds_by_pos: dict of param_position -> bound (for Perm)
    """
    graph = gm.graph

    # Step 1: Identify placeholders and their positions
    placeholders = []  # ordered list of placeholder nodes
    int_placeholder_set = set()  # set of placeholder nodes with i64/i32/i8 dtype

    for node in graph.nodes:
        if node.op == 'placeholder':
            placeholders.append(node)
            # Check dtype from type annotation or var_dtypes
            dtype = None
            if node.type is not None:
                dtype_str, _ = _parse_type_annotation(str(node.type))
                dtype = dtype_str
            if dtype is None:
                dtype = var_dtypes.get(node.name, "")
            if dtype and ("i64" in dtype or "i32" in dtype or "i8" in dtype):
                int_placeholder_set.add(node)

    # Step 2: Build "derives from" map using graph node references
    # For each node, track which int placeholders it derives from
    # (through passthrough/view/where/add/mul ops)
    derives_from = {}  # node -> set of placeholder nodes
    add_offset = {}    # node -> cumulative additive offset

    # Initialize: each placeholder derives from itself
    for ph in placeholders:
        derives_from[ph] = {ph}
        add_offset[ph] = 0

    PASSTHROUGH_TARGETS = {
        torch.ops.aten.clone.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.select.int,
        torch.ops.aten.permute.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.lift_fresh_copy.default,
        torch.ops.prims.convert_element_type.default,
    }

    # Process nodes in topological order (guaranteed by FX graph)
    for node in graph.nodes:
        if node.op != 'call_function':
            continue
        if node in derives_from:
            continue  # already handled (shouldn't happen)

        target = node.target

        if target == torch.ops.aten.where.self:
            # where.self(cond, x, y) - result derives from x and y
            # cond is args[0], x is args[1], y is args[2]
            sources = set()
            for arg in node.args[1:]:  # skip condition
                if isinstance(arg, fx.Node) and arg in derives_from:
                    sources.update(derives_from[arg])
            if sources:
                derives_from[node] = sources
                add_offset[node] = 0

        elif target == torch.ops.aten.add.Tensor:
            # add(x, K) or add(x, y)
            if node.args and isinstance(node.args[0], fx.Node):
                src = node.args[0]
                if src in derives_from:
                    derives_from[node] = set(derives_from[src])
                    offset = add_offset.get(src, 0)
                    # Check if second arg is an integer constant
                    if len(node.args) >= 2 and isinstance(node.args[1], (int, float)):
                        offset += int(node.args[1])
                    add_offset[node] = offset

        elif target == torch.ops.aten.mul.Tensor:
            # mul(x, K) - trace through first arg
            if node.args and isinstance(node.args[0], fx.Node):
                src = node.args[0]
                if src in derives_from:
                    derives_from[node] = set(derives_from[src])
                    add_offset[node] = 0  # mul resets offset

        elif target in PASSTHROUGH_TARGETS:
            # First arg is source
            if node.args and isinstance(node.args[0], fx.Node):
                src = node.args[0]
                if src in derives_from:
                    derives_from[node] = set(derives_from[src])
                    add_offset[node] = add_offset.get(src, 0)

        elif target == torch.ops.aten.gather.default:
            # gather(input, dim, index) - OUTPUT carries values from input
            if node.args and isinstance(node.args[0], fx.Node):
                src = node.args[0]
                if src in derives_from:
                    derives_from[node] = set(derives_from[src])
                    add_offset[node] = add_offset.get(src, 0)

        else:
            # Check for _low_memory_max_pool_offsets_to_indices
            target_str = str(target)
            if 'low_memory_max_pool_offsets_to_indices' in target_str:
                if node.args and isinstance(node.args[0], fx.Node):
                    src = node.args[0]
                    if src in derives_from:
                        derives_from[node] = set(derives_from[src])
                        add_offset[node] = 0

    # Step 3: Find bounds from index consumer ops
    index_bounds = {}  # placeholder_node -> bound
    perm_bounds = {}   # placeholder_node -> bound

    def _set_bound(index_node, raw_bound):
        """Set bound for all int placeholders that derive into index_node."""
        source_phs = derives_from.get(index_node, set())
        int_sources = source_phs & int_placeholder_set
        offset = add_offset.get(index_node, 0)
        bound = raw_bound - offset if offset > 0 else raw_bound
        for ph in int_sources:
            if ph not in index_bounds or index_bounds[ph] > bound:
                index_bounds[ph] = bound

    # Handle _low_memory_max_pool_offsets_to_indices: bound = prod(kernel_size)
    for node in graph.nodes:
        if node.op != 'call_function':
            continue
        target_str = str(node.target)
        if 'low_memory_max_pool_offsets_to_indices' in target_str:
            if node.args and isinstance(node.args[0], fx.Node):
                src = node.args[0]
                source_phs = derives_from.get(src, set())
                int_sources = source_phs & int_placeholder_set
                # kernel_size is args[1]
                if len(node.args) >= 2 and isinstance(node.args[1], (list, tuple)):
                    ks = node.args[1]
                    bound = 1
                    for k in ks:
                        if isinstance(k, int):
                            bound *= k
                    for ph in int_sources:
                        index_bounds[ph] = bound

    # Heuristic: i64 param used as TARGET of index.Tensor (cumsum/position pattern)
    for node in graph.nodes:
        if node.op != 'call_function':
            continue
        if node.target == torch.ops.aten.index.Tensor:
            if len(node.args) >= 2 and isinstance(node.args[0], fx.Node):
                target_node = node.args[0]
                source_phs = derives_from.get(target_node, set())
                int_targets = source_phs & int_placeholder_set
                if int_targets and isinstance(node.args[1], (list, tuple)):
                    # The param is used as TARGET - its values are being READ
                    for ph in int_targets:
                        ph_shape = _get_node_shape(ph, var_shapes, gm)
                        if ph_shape:
                            bound = ph_shape[0]
                            if ph not in index_bounds:
                                index_bounds[ph] = bound

    # Main pass: scan for index consumer ops
    for node in graph.nodes:
        if node.op != 'call_function':
            continue

        target = node.target

        if target == torch.ops.aten.gather.default:
            # gather(input, dim, index)
            if len(node.args) >= 3:
                input_node = node.args[0]
                dim = node.args[1]
                index_node = node.args[2]

                if isinstance(index_node, fx.Node) and isinstance(dim, int):
                    source_phs = derives_from.get(index_node, set())
                    int_sources = source_phs & int_placeholder_set
                    if int_sources:
                        input_shape = _get_node_shape(input_node, var_shapes, gm) if isinstance(input_node, fx.Node) else None
                        if input_shape and dim < len(input_shape):
                            _set_bound(index_node, input_shape[dim])

        elif target == torch.ops.aten.scatter.src:
            # scatter.src(target, dim, index, src)
            if len(node.args) >= 4:
                target_node = node.args[0]
                dim = node.args[1]
                index_node = node.args[2]
                src_node = node.args[3]

                if isinstance(index_node, fx.Node) and isinstance(dim, int):
                    source_phs = derives_from.get(index_node, set())
                    int_sources = source_phs & int_placeholder_set
                    if int_sources and isinstance(target_node, fx.Node):
                        target_shape = _get_node_shape(target_node, var_shapes, gm)
                        if target_shape:
                            actual_dim = dim if dim >= 0 else dim + len(target_shape)
                            if 0 <= actual_dim < len(target_shape):
                                _set_bound(index_node, target_shape[actual_dim])

                    # Permutation pattern: scatter.src(empty, dim, idx, iota_expanded)
                    if (int_sources and isinstance(target_node, fx.Node)
                            and isinstance(src_node, fx.Node)):
                        src_iota = _is_iota_tensor(gm, src_node)
                        if src_iota is not None:
                            target_shape = _get_node_shape(target_node, var_shapes, gm)
                            is_alloc = _is_alloc_node(target_node) or target_node.op == 'get_attr'
                            if is_alloc and target_shape:
                                actual_dim = dim if dim >= 0 else dim + len(target_shape)
                                if (0 <= actual_dim < len(target_shape)
                                        and target_shape[actual_dim] == src_iota):
                                    for ph in int_sources:
                                        perm_bounds[ph] = src_iota

        elif target == torch.ops.aten.scatter.value:
            # scatter.value(target, dim, index, value_scalar)
            if len(node.args) >= 3:
                target_node = node.args[0]
                dim = node.args[1]
                index_node = node.args[2]

                if isinstance(index_node, fx.Node) and isinstance(dim, int):
                    source_phs = derives_from.get(index_node, set())
                    int_sources = source_phs & int_placeholder_set
                    if int_sources and isinstance(target_node, fx.Node):
                        target_shape = _get_node_shape(target_node, var_shapes, gm)
                        if target_shape:
                            actual_dim = dim if dim >= 0 else dim + len(target_shape)
                            if 0 <= actual_dim < len(target_shape):
                                _set_bound(index_node, target_shape[actual_dim])

        elif target == torch.ops.aten.scatter_add.default:
            # scatter_add(target, dim, index, src)
            if len(node.args) >= 3:
                target_node = node.args[0]
                dim = node.args[1]
                index_node = node.args[2]

                if isinstance(index_node, fx.Node) and isinstance(dim, int):
                    source_phs = derives_from.get(index_node, set())
                    int_sources = source_phs & int_placeholder_set
                    if int_sources and isinstance(target_node, fx.Node):
                        target_shape = _get_node_shape(target_node, var_shapes, gm)
                        if target_shape:
                            actual_dim = dim if dim >= 0 else dim + len(target_shape)
                            if 0 <= actual_dim < len(target_shape):
                                _set_bound(index_node, target_shape[actual_dim])

        elif target == torch.ops.aten.index_put.default:
            # index_put(target, [indices...], values, accumulate?)
            if len(node.args) >= 3:
                target_node = node.args[0]
                indices_arg = node.args[1]
                values_node = node.args[2]

                if isinstance(target_node, fx.Node):
                    target_shape = _get_node_shape(target_node, var_shapes, gm)
                    target_is_alloc = (_is_alloc_node(target_node)
                                       or target_node.op == 'get_attr')

                    if isinstance(indices_arg, (list, tuple)) and target_shape:
                        for dim_idx, idx_node in enumerate(indices_arg):
                            if isinstance(idx_node, fx.Node) and dim_idx < len(target_shape):
                                bound = target_shape[dim_idx]
                                # Heuristic: for index_put into an allocation,
                                # if the indexed dimension is strictly larger than
                                # the first (batch) dimension, use batch dim
                                if (target_is_alloc and len(target_shape) > 1
                                        and dim_idx > 0 and bound > target_shape[0]):
                                    bound = target_shape[0]
                                _set_bound(idx_node, bound)

                    # Permutation pattern: index_put(empty([N]), [idx], iota(N))
                    if (isinstance(indices_arg, (list, tuple))
                            and isinstance(values_node, fx.Node)
                            and target_is_alloc and target_shape):
                        val_iota = _is_iota_tensor(gm, values_node)
                        if val_iota is not None and target_shape and target_shape[0] == val_iota:
                            for idx_node in indices_arg:
                                if isinstance(idx_node, fx.Node):
                                    source_phs = derives_from.get(idx_node, set())
                                    int_sources = source_phs & int_placeholder_set
                                    for ph in int_sources:
                                        perm_bounds[ph] = val_iota

        elif target == torch.ops.aten.index.Tensor:
            # index.Tensor(input, [indices...])
            if len(node.args) >= 2:
                input_node = node.args[0]
                indices_arg = node.args[1]

                if isinstance(input_node, fx.Node) and isinstance(indices_arg, (list, tuple)):
                    input_shape = _get_node_shape(input_node, var_shapes, gm)
                    if input_shape:
                        for dim_idx, idx_node in enumerate(indices_arg):
                            if isinstance(idx_node, fx.Node) and dim_idx < len(input_shape):
                                _set_bound(idx_node, input_shape[dim_idx])

        elif target == torch.ops.aten.index_select.default:
            # index_select(input, dim, index)
            if len(node.args) >= 3:
                input_node = node.args[0]
                dim = node.args[1]
                index_node = node.args[2]

                if isinstance(index_node, fx.Node) and isinstance(dim, int):
                    source_phs = derives_from.get(index_node, set())
                    int_sources = source_phs & int_placeholder_set
                    if int_sources and isinstance(input_node, fx.Node):
                        input_shape = _get_node_shape(input_node, var_shapes, gm)
                        if input_shape:
                            actual_dim = dim if dim >= 0 else dim + len(input_shape)
                            if 0 <= actual_dim < len(input_shape):
                                _set_bound(index_node, input_shape[actual_dim])

        elif target == torch.ops.aten.embedding.default:
            # embedding(weight, indices, ...)
            if len(node.args) >= 2:
                weight_node = node.args[0]
                indices_node = node.args[1]

                if isinstance(indices_node, fx.Node):
                    source_phs = derives_from.get(indices_node, set())
                    int_sources = source_phs & int_placeholder_set
                    if int_sources and isinstance(weight_node, fx.Node):
                        weight_shape = _get_node_shape(weight_node, var_shapes, gm)
                        if weight_shape:
                            _set_bound(indices_node, weight_shape[0])

    # Constraint backpropagation through mul.Tensor
    for node in graph.nodes:
        if node.op != 'call_function':
            continue
        if node.target == torch.ops.aten.mul.Tensor:
            if (len(node.args) >= 2
                    and isinstance(node.args[0], fx.Node)
                    and isinstance(node.args[1], fx.Node)):
                arg1_node = node.args[0]
                arg2_node = node.args[1]

                # Get bound for primary factor's int placeholders
                arg1_phs = derives_from.get(arg1_node, set()) & int_placeholder_set
                arg1_bound = None
                for ph in arg1_phs:
                    if ph in index_bounds:
                        b = index_bounds[ph]
                        if arg1_bound is None or b < arg1_bound:
                            arg1_bound = b

                # Get secondary factor's int placeholders
                arg2_phs = derives_from.get(arg2_node, set()) & int_placeholder_set

                if arg1_bound is not None and arg1_bound > 1 and arg2_phs:
                    inferred_b2 = arg1_bound // (arg1_bound - 1) + 1
                    for ph in arg2_phs:
                        if ph not in index_bounds or index_bounds[ph] > inferred_b2:
                            index_bounds[ph] = inferred_b2

    # Step 4: Convert placeholder nodes to positions
    index_bounds_by_pos = {}
    perm_bounds_by_pos = {}

    for i, ph in enumerate(placeholders):
        if ph in perm_bounds:
            perm_bounds_by_pos[i] = perm_bounds[ph]
        elif ph in index_bounds:
            index_bounds_by_pos[i] = index_bounds[ph]

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

    # Parse source shapes
    var_shapes, var_dtypes = _parse_source_shapes(content)

    # Load and trace the module
    try:
        gm, repro = _load_repro_graph(repro_py)
    except Exception:
        return config_str

    # Run inference on the graph
    index_bounds, perm_bounds = _infer_bounds_from_graph(gm, var_shapes, var_dtypes)

    # Apply annotations to the config string
    return annotate_config_string(config_str, index_bounds, perm_bounds)
