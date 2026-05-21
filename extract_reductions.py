"""
Extract inductor reduction kernels as standalone FX graph repros.

Hooks into inductor's scheduler, captures reduction nodes, traces back to the
original FX graph, extracts minimal subgraphs, and serializes them as runnable
standalone Python scripts with benchmarking.
"""

import collections
import hashlib
import json
import os
import sys
import copy
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx
import torch._inductor.config as inductor_config
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)
from torch._inductor.ir import ComputedBuffer, Reduction
from torch._inductor.virtualized import V


class ReductionExtractor:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.captured: list[dict] = []
        self.seen_hashes: set[str] = set()
        self.counter = 0
        os.makedirs(output_dir, exist_ok=True)

    def _get_reduction_info(self, snode: SchedulerNode) -> dict | None:
        if not snode.is_reduction():
            return None
        node = snode.node
        if not isinstance(node, ComputedBuffer):
            return None
        data = node.data
        if not isinstance(data, Reduction):
            return None

        ranges = [str(r) for r in data.ranges]
        reduction_ranges = [str(r) for r in data.reduction_ranges]

        origins = []
        try:
            for o in node.get_origins():
                origins.append(o)
        except Exception:
            pass

        return {
            "name": snode.get_name(),
            "ranges": ranges,
            "reduction_ranges": reduction_ranges,
            "reduction_type": data.reduction_type,
            "dtype": str(data.dtype),
            "src_dtype": str(data.src_dtype),
            "device": str(data.device),
            "origins": origins,
            "group": str(snode.group),
        }

    def _generate_synthetic_repro(
        self, reduction_meta: dict
    ) -> tuple[str, dict[str, dict]]:
        """Generate a synthetic Repro module matching the reduction shapes.

        Used when the real FX subgraph contains prims ops that aren't available
        outside the inductor decomposition context.
        """
        reductions = reduction_meta["reductions"]

        # Map reduction types to aten operations
        REDUCTION_TO_OP = {
            "sum": "x.sum(dim=-1)",
            "prod": "x.prod(dim=-1)",
            "amax": "x.amax(dim=-1)",
            "amin": "x.amin(dim=-1)",
            "argmax": "x.argmax(dim=-1)",
            "argmin": "x.argmin(dim=-1)",
            "any": "x.any(dim=-1)",
            "welford_reduce": "torch.var_mean(x, dim=-1, correction=0)",
            "welford_combine": "torch.var_mean(x, dim=-1, correction=0)",
            "online_softmax_reduce": "torch.softmax(x, dim=-1)",
        }

        # Use the first reduction to determine primary shape
        red = reductions[0]
        ranges = [int(r) for r in red["ranges"] if r != "1"]
        reduction_ranges = [int(r) for r in red["reduction_ranges"]]
        full_shape = ranges + reduction_ranges
        dtype = red["dtype"].replace("torch.", "")
        red_type = red["reduction_type"]

        op = REDUCTION_TO_OP.get(red_type, f"x.sum(dim=-1)  # fallback for {red_type}")

        code = f"""class Repro(torch.nn.Module):
    def forward(self, x):
        return {op}
"""
        placeholder_info = {
            "x": {
                "shape": full_shape,
                "stride": [],
                "dtype": f"torch.{dtype}",
                "device": "cuda",
            }
        }
        return code, placeholder_info

    def _extract_subgraph(
        self, origin_nodes: list[fx.Node], gm: fx.GraphModule
    ) -> tuple[fx.GraphModule, dict[str, dict]] | None:
        """Extract a minimal subgraph containing only the origin nodes.

        Origin nodes are the FX ops that map to this fused scheduler node.
        Any dependencies they have outside the origin set become placeholders
        (inputs to the repro), NOT traced backward. This prevents pulling in
        unrelated ops like embedding lookups.
        """
        if not origin_nodes:
            return None

        # Deduplicate
        seen = set()
        unique_origins = []
        for n in origin_nodes:
            if id(n) not in seen:
                seen.add(id(n))
                unique_origins.append(n)
        origin_nodes = unique_origins

        # The "needed" set is exactly the origin nodes — no BFS backward.
        needed_nodes: set[fx.Node] = set(origin_nodes)

        # Find leaf nodes: origins not consumed by other origins.
        output_nodes = []
        for n in origin_nodes:
            has_internal_user = any(
                user in needed_nodes and user.op != "output"
                for user in n.users
            )
            if not has_internal_user:
                output_nodes.append(n)
        if not output_nodes:
            output_nodes = origin_nodes

        new_graph = fx.Graph()
        env: dict[fx.Node, fx.Node] = {}
        placeholder_info: dict[str, dict] = {}

        def _record_placeholder(name: str, meta: dict) -> None:
            val = meta.get("val", None)
            if val is not None and isinstance(val, torch.Tensor):
                placeholder_info[name] = {
                    "shape": list(val.shape),
                    "stride": list(val.stride()) if not val.is_contiguous() else [],
                    "dtype": str(val.dtype),
                    "device": str(val.device),
                }
            elif val is not None and isinstance(val, (torch.SymInt, torch.SymFloat)):
                placeholder_info[name] = {
                    "shape": [],
                    "stride": [],
                    "dtype": "symint",
                    "device": "cpu",
                }

        def _ensure_in_env(x: Any) -> Any:
            if isinstance(x, fx.Node):
                if x in env:
                    return env[x]
                ph = new_graph.placeholder(x.name)
                ph.meta = copy.copy(x.meta) if x.meta else {}
                env[x] = ph
                _record_placeholder(x.name, x.meta or {})
                return ph
            return x

        # Sort nodes in original topological order
        all_graph_nodes = list(gm.graph.nodes)
        node_order = {n: i for i, n in enumerate(all_graph_nodes)}
        sorted_needed = sorted(needed_nodes, key=lambda n: node_order.get(n, 0))

        for node in sorted_needed:
            if node.op == "placeholder":
                new_node = new_graph.placeholder(node.name)
                new_node.meta = copy.copy(node.meta) if node.meta else {}
                env[node] = new_node
                _record_placeholder(node.name, node.meta or {})
            elif node.op == "get_attr":
                new_node = new_graph.get_attr(node.target)
                new_node.meta = copy.copy(node.meta) if node.meta else {}
                env[node] = new_node
            elif node.op in ("call_function", "call_method"):
                new_args = fx.map_arg(node.args, _ensure_in_env)
                new_kwargs = fx.map_arg(node.kwargs, _ensure_in_env)
                if node.op == "call_function":
                    new_node = new_graph.call_function(
                        node.target, args=new_args, kwargs=new_kwargs
                    )
                else:
                    new_node = new_graph.call_method(
                        node.target, args=new_args, kwargs=new_kwargs
                    )
                new_node.meta = copy.copy(node.meta) if node.meta else {}
                env[node] = new_node

        mapped_outputs = [env[n] for n in output_nodes if n in env]
        if not mapped_outputs:
            return None
        if len(mapped_outputs) == 1:
            new_graph.output(mapped_outputs[0])
        else:
            new_graph.output(tuple(mapped_outputs))

        new_graph.lint()
        new_gm = fx.GraphModule(gm, new_graph)
        return new_gm, placeholder_info

    def _generate_standalone_script(
        self,
        gm: fx.GraphModule | None,
        placeholder_info: dict[str, dict],
        reduction_meta: dict,
        filename: str,
    ) -> str:
        """Generate a standalone Python script from the extracted subgraph.

        Two strategies:
        1. If the FX subgraph uses only standard aten ops, embed it directly.
        2. Otherwise, generate a synthetic model matching the reduction shapes.
        """
        use_fx_graph = False
        code = ""
        if gm is not None:
            code = gm.print_readable(print_output=False)
            code = code.replace("class GraphModule(", "class Repro(", 1)
            use_fx_graph = True

        if not use_fx_graph:
            code, placeholder_info = self._generate_synthetic_repro(reduction_meta)

        # Build input generation code
        input_lines = []
        ph_names = list(placeholder_info.keys()) if not use_fx_graph else [
            n.name for n in gm.graph.nodes if n.op == "placeholder"
        ]
        for name in ph_names:
            info = placeholder_info.get(name)
            if info and info["dtype"] != "symint":
                shape = info["shape"]
                stride = info.get("stride", [])
                dtype = info["dtype"]
                device = info.get("device", "cuda")
                if "cuda" in device:
                    device = "cuda"
                if stride and shape:
                    # Minimum storage: offset of last element + 1
                    storage_size = sum(
                        s * (d - 1) for s, d in zip(stride, shape) if d > 1
                    ) + 1
                    if "int" in dtype:
                        input_lines.append(
                            f"    torch.randint(0, 100, ({storage_size},), dtype={dtype}, device='{device}')"
                            f".as_strided({shape}, {stride}),  # {name}"
                        )
                    else:
                        input_lines.append(
                            f"    torch.randn({storage_size}, dtype={dtype}, device='{device}')"
                            f".as_strided({shape}, {stride}),  # {name}"
                        )
                else:
                    if "int" in dtype:
                        input_lines.append(
                            f"    torch.randint(0, 100, {shape}, dtype={dtype}, device='{device}'),"
                        )
                    else:
                        input_lines.append(
                            f"    torch.randn({shape}, dtype={dtype}, device='{device}'),"
                        )
            else:
                input_lines.append(f"    torch.tensor(1),  # {name} (unknown shape)")

        inputs_code = "\n".join(input_lines)

        # Metadata comment
        meta_lines = []
        for red in reduction_meta.get("reductions", []):
            meta_lines.append(
                f"#   type={red['reduction_type']}, ranges={red['ranges']}, "
                f"reduction_ranges={red['reduction_ranges']}"
            )
            origin_ops = [
                str(o.target) if hasattr(o, "target") else str(o)
                for o in red.get("origins", [])
            ]
            if origin_ops:
                meta_lines.append(f"#   origins: {origin_ops}")
        meta_comment = "\n".join(meta_lines)

        script = f'''"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
{meta_comment}
"""
import glob
import os
import torch
import torch._inductor.config as inductor_config
from math import inf
from torch import device

# The extracted FX graph subgraph:
{code}


def make_inputs():
    return [
{inputs_code}
    ]


def _count_bytes(inputs, outputs):
    """Count total read + write bytes for SOL calculation (naive)."""
    total = 0
    for t in inputs:
        if isinstance(t, torch.Tensor):
            total += t.nelement() * t.element_size()
    if isinstance(outputs, torch.Tensor):
        total += outputs.nelement() * outputs.element_size()
    elif isinstance(outputs, (tuple, list)):
        for o in outputs:
            if isinstance(o, torch.Tensor):
                total += o.nelement() * o.element_size()
    return total


def _count_bytes_adjusted(mod, inputs):
    """Count actual DRAM bytes by tracking op semantics.

    Handles: view-only pass-throughs (not counted) and
    index/gather (count output size, not full input).
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    _VIEW_OPS = {{
        torch.ops.aten.permute.default, torch.ops.aten.transpose.int,
        torch.ops.aten.t.default, torch.ops.aten.reshape.default,
        torch.ops.aten.view.default, torch.ops.aten.expand.default,
        torch.ops.aten.slice.Tensor, torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default, torch.ops.aten.squeeze.dim,
        torch.ops.aten.as_strided.default, torch.ops.aten.select.int,
    }}
    _INDEX_OPS = {{
        torch.ops.aten.index.Tensor, torch.ops.aten.gather.default,
        torch.ops.aten.index_select.default, torch.ops.aten.embedding.default,
    }}

    class _ByteCount(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self._produced = set()
            self._views_of = {{}}
            self._input_bytes = {{}}
            self._computed_inputs = set()
            self._gather_adj = {{}}

        def _root(self, tid):
            seen = set()
            while tid in self._views_of and tid not in seen:
                seen.add(tid)
                tid = self._views_of[tid]
            return tid

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {{}}
            def track(x):
                if isinstance(x, torch.Tensor) and id(x) not in self._produced:
                    if id(x) not in self._input_bytes:
                        self._input_bytes[id(x)] = x.numel() * x.element_size()
            torch.utils._pytree.tree_map(
                lambda x: track(x) if isinstance(x, torch.Tensor) else None, (args, kwargs))
            result = func(*args, **kwargs)
            def mark(x):
                if isinstance(x, torch.Tensor):
                    self._produced.add(id(x))
            torch.utils._pytree.tree_map(
                lambda x: mark(x) if isinstance(x, torch.Tensor) else None, result)
            if func in _VIEW_OPS and args and isinstance(args[0], torch.Tensor):
                if isinstance(result, torch.Tensor):
                    self._views_of[id(result)] = id(args[0])
            else:
                def mark_computed(x):
                    if isinstance(x, torch.Tensor):
                        root = self._root(id(x))
                        if root in self._input_bytes:
                            self._computed_inputs.add(root)
                torch.utils._pytree.tree_map(
                    lambda x: mark_computed(x) if isinstance(x, torch.Tensor) else None, (args, kwargs))
                if func in _INDEX_OPS and args and isinstance(args[0], torch.Tensor):
                    root = self._root(id(args[0]))
                    if root in self._input_bytes and isinstance(result, torch.Tensor):
                        gathered = result.numel() * result.element_size()
                        self._gather_adj[root] = self._gather_adj.get(root, 0) + gathered
            return result

        def total(self, outputs):
            read = sum(min(self._gather_adj.get(t, self._input_bytes[t]), self._input_bytes[t]) for t in self._computed_inputs)
            write = 0
            if isinstance(outputs, torch.Tensor):
                write = outputs.numel() * outputs.element_size()
            elif isinstance(outputs, (tuple, list)):
                for o in outputs:
                    if isinstance(o, torch.Tensor):
                        write += o.numel() * o.element_size()
            return read + write

    with _ByteCount() as bc:
        with torch.no_grad():
            out = mod(*inputs)
    return bc.total(out)


def _count_kernels(mod, inputs):
    """Compile and count how many Triton kernels Inductor generates."""
    import glob
    from torch._inductor.utils import fresh_inductor_cache
    from torch._inductor.codecache import cache_dir

    torch._dynamo.reset()
    with fresh_inductor_cache():
        compiled = torch.compile(mod)
        with torch.no_grad():
            compiled(*inputs)
            torch.cuda.synchronize()
        cd = cache_dir()
        py_files = sorted(glob.glob(os.path.join(cd, "**", "*.py"), recursive=True), key=os.path.getmtime)
        for f in reversed(py_files):
            with open(f) as fh:
                content = fh.read()
            if 'def call(' in content and '.run(' in content:
                runs = [l for l in content.split('\\n') if '.run(' in l and not l.strip().startswith('#')]
                names = []
                for r in runs:
                    name = r.strip().split('.run(')[0]
                    names.append(name)
                return len(names), names
    return 0, []


def benchmark(n_warmup=25, n_rep=200):
    from triton.testing import do_bench

    mod = Repro()
    inputs = make_inputs()

    with torch.no_grad():
        eager_out = mod(*inputs)

    total_bytes_naive = _count_bytes(inputs, eager_out)
    total_bytes = _count_bytes_adjusted(mod, inputs)
    if total_bytes_naive > total_bytes * 1.1:
        print(f"\\nBytes adjusted: {{total_bytes_naive/1e6:.1f}} MB naive -> {{total_bytes/1e6:.1f}} MB actual")

    # Count kernels generated
    n_kernels, kernel_names = _count_kernels(mod, inputs)
    print(f"\\nKernels generated: {{n_kernels}}")
    for name in kernel_names:
        print(f"  {{name}}")

    # SOL: memcopy same total bytes (copy half since copy does read+write)
    copy_elems = max(total_bytes // (2 * 4), 256)
    src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    sol_ms = do_bench(lambda: dst.copy_(src), warmup=n_warmup, rep=n_rep)
    sol_us = sol_ms * 1000
    del src, dst

    # Compiled (default heuristics)
    torch._dynamo.reset()
    compiled = torch.compile(mod)
    with torch.no_grad():
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    compiled_ms = do_bench(lambda: compiled(*inputs), warmup=n_warmup, rep=n_rep)
    compiled_us = compiled_ms * 1000

    # Compiled with coordinate descent tuning
    inductor_config.coordinate_descent_tuning = True
    torch._dynamo.reset()
    compiled_cd = torch.compile(mod)
    with torch.no_grad():
        for _ in range(3):
            compiled_cd(*inputs)
        torch.cuda.synchronize()
    cd_ms = do_bench(lambda: compiled_cd(*inputs), warmup=n_warmup, rep=n_rep)
    cd_us = cd_ms * 1000

    print(f"\\nKernel data: {{total_bytes / 1024:.1f}} KB (read+write)")
    print(f"Memcopy SOL (same size): {{sol_us:8.1f}} us")
    print(f"Compiled (default):      {{compiled_us:8.1f}} us")
    print(f"Compiled (coord desc):   {{cd_us:8.1f}} us")
    print(f"Gap (default / SOL):     {{compiled_us / sol_us:8.2f}}x")
    print(f"Gap (CD / SOL):          {{cd_us / sol_us:8.2f}}x")

    return {{
        "compiled_us": compiled_us,
        "coord_descent_us": cd_us,
        "memcopy_sol_us": sol_us,
        "total_bytes": total_bytes,
        "n_kernels": n_kernels,
        "kernel_names": kernel_names,
    }}


if __name__ == "__main__":
    benchmark()
'''

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            f.write(script)
        return filepath

    def _get_all_subnode_origins(self, node: BaseSchedulerNode) -> list[fx.Node]:
        """Get all FX origin nodes from ALL subnodes in a fused group."""
        origins = []
        seen_ids = set()
        snodes = node.get_nodes() if hasattr(node, "get_nodes") else [node]
        for snode in snodes:
            if not isinstance(snode, SchedulerNode):
                continue
            irnode = snode.node
            if irnode is None:
                continue
            try:
                for o in irnode.get_origins():
                    if id(o) not in seen_ids:
                        seen_ids.add(id(o))
                        origins.append(o)
            except Exception:
                pass
        return origins

    def post_fusion_pass(
        self, nodes: list[BaseSchedulerNode]
    ) -> list[BaseSchedulerNode]:
        """Post-fusion custom pass that captures full fused kernel subgraphs.

        For each fused node that contains at least one reduction, we capture
        the complete fused group (reductions + all fused pointwise epilogues).
        """
        try:
            gm = V.graph.orig_gm
        except Exception:
            return nodes

        for node in nodes:
            reduction_infos = []

            if isinstance(node, FusedSchedulerNode):
                for snode in node.snodes:
                    if isinstance(snode, SchedulerNode):
                        info = self._get_reduction_info(snode)
                        if info:
                            reduction_infos.append(info)
                if not reduction_infos:
                    continue
            elif isinstance(node, SchedulerNode):
                info = self._get_reduction_info(node)
                if not info:
                    continue
                reduction_infos.append(info)
            else:
                continue

            # Collect origins from ALL subnodes (not just reductions)
            all_origins = self._get_all_subnode_origins(node)

            if not all_origins:
                continue

            # Dedup by structural hash — includes all origin ops for full fused kernel
            origin_ops = sorted(
                str(o.target) for o in all_origins if hasattr(o, "target")
            )
            hash_key = hashlib.md5(
                json.dumps(
                    {
                        "reductions": [
                            {
                                "ranges": r["ranges"],
                                "reduction_ranges": r["reduction_ranges"],
                                "reduction_type": r["reduction_type"],
                                "dtype": r["dtype"],
                            }
                            for r in reduction_infos
                        ],
                        "origin_ops": origin_ops,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:16]

            if hash_key in self.seen_hashes:
                continue
            self.seen_hashes.add(hash_key)

            # Extract subgraph
            result = self._extract_subgraph(all_origins, gm)
            if result is None:
                continue
            sub_gm, placeholder_info = result

            # Count total subnodes
            n_subnodes = len(node.snodes) if isinstance(node, FusedSchedulerNode) else 1
            n_reductions = len(reduction_infos)
            n_pointwise = n_subnodes - n_reductions

            meta = {
                "node_type": type(node).__name__,
                "node_name": node.get_name(),
                "hash": hash_key,
                "reductions": reduction_infos,
                "n_subnodes": n_subnodes,
                "n_reductions": n_reductions,
                "n_pointwise": n_pointwise,
                "origin_ops": origin_ops,
            }

            red_types = "_".join(
                sorted(set(r["reduction_type"] for r in reduction_infos))
            )
            filename = f"fused_{self.counter:03d}_{red_types}_{hash_key}.py"
            self.counter += 1

            try:
                filepath = self._generate_standalone_script(
                    sub_gm, placeholder_info, meta, filename
                )
                print(f"  Extracted: {filepath}")
                self.captured.append(
                    {
                        "file": filepath,
                        "hash": hash_key,
                        "node_name": node.get_name(),
                        "reduction_types": [
                            r["reduction_type"] for r in reduction_infos
                        ],
                        "ranges": [r["ranges"] for r in reduction_infos],
                        "reduction_ranges": [
                            r["reduction_ranges"] for r in reduction_infos
                        ],
                    }
                )
            except Exception as e:
                print(f"  Failed to extract {node.get_name()}: {e}")

        return nodes


def run_extraction(model_fn, model_args_fn, output_dir, model_name="model"):
    """Run extraction on a model."""
    extractor = ReductionExtractor(output_dir)

    torch._dynamo.reset()
    inductor_config._post_fusion_custom_pass = extractor.post_fusion_pass
    inductor_config.split_reductions = False
    inductor_config.force_disable_caches = True

    print(f"Compiling {model_name} (forward)...")
    model = model_fn()
    args = model_args_fn()
    compiled = torch.compile(model)
    with torch.no_grad():
        out = compiled(*args)

    fwd_count = len(extractor.captured)

    # Also capture backward pass reductions
    print(f"Compiling {model_name} (backward)...")
    torch._dynamo.reset()
    inductor_config._post_fusion_custom_pass = extractor.post_fusion_pass
    model2 = model_fn()
    model2.train()
    args2 = model_args_fn()
    for a in args2:
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            a.requires_grad_(True)
    compiled2 = torch.compile(model2)
    try:
        out2 = compiled2(*args2)
        if isinstance(out2, torch.Tensor):
            loss = out2.sum()
        elif hasattr(out2, "loss") and out2.loss is not None:
            loss = out2.loss
        elif hasattr(out2, "logits"):
            loss = out2.logits.sum()
        elif isinstance(out2, (tuple, list)):
            loss = sum(o.sum() for o in out2 if isinstance(o, torch.Tensor))
        else:
            loss = None
        if loss is not None:
            loss.backward()
    except Exception as e:
        print(f"  Backward failed: {e}")

    bwd_count = len(extractor.captured) - fwd_count
    print(f"\nExtracted {fwd_count} forward + {bwd_count} backward = {len(extractor.captured)} total")

    # Save index
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(extractor.captured, f, indent=2)

    return extractor


# --- Test models ---


def _simple_model():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.LayerNorm(768)
            self.linear = torch.nn.Linear(768, 768)

        def forward(self, x):
            x = self.ln(x)
            x = self.linear(x)
            return x.softmax(dim=-1)

    return M().cuda()


def _rmsnorm_model():
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = 1e-6

        def forward(self, x):
            norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * norm * self.weight

    return RMSNorm(768).cuda()


def _cross_entropy_model():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(768, 32000)

        def forward(self, x, targets):
            logits = self.linear(x)
            return torch.nn.functional.cross_entropy(logits, targets)

    return M().cuda()


class _RMSNormModule(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class _LlamaLikeBlock(torch.nn.Module):
    """A single LLaMA-style transformer block with RMSNorm, GQA, SwiGLU."""

    def __init__(self, hidden, n_heads, intermediate):
        super().__init__()
        self.norm1 = _RMSNormModule(hidden)
        self.norm2 = _RMSNormModule(hidden)
        self.q_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.k_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.v_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads

    def forward(self, x):
        B, S, D = x.shape
        # Self-attention with RMSNorm
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, S, D)
        x = x + self.o_proj(attn)
        # SwiGLU FFN with RMSNorm
        h = self.norm2(x)
        x = x + self.down_proj(torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class _LlamaLikeModel(torch.nn.Module):
    def __init__(self, hidden, n_heads, n_layers, intermediate, vocab_size):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.layers = torch.nn.ModuleList(
            [_LlamaLikeBlock(hidden, n_heads, intermediate) for _ in range(n_layers)]
        )
        self.norm = _RMSNormModule(hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def _llama_like_model(hidden=4096, n_heads=32, n_layers=2, vocab_size=32000,
                       batch=4, seq_len=512):
    def make_model():
        return _LlamaLikeModel(hidden, n_heads, n_layers, hidden * 4, vocab_size).cuda().eval()

    def make_args():
        return [torch.randint(0, vocab_size, (batch, seq_len), device="cuda")]

    return make_model, make_args


class _BertLikeBlock(torch.nn.Module):
    """A BERT-style transformer block with LayerNorm and GELU."""

    def __init__(self, hidden, n_heads, intermediate):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.q_proj = torch.nn.Linear(hidden, hidden)
        self.k_proj = torch.nn.Linear(hidden, hidden)
        self.v_proj = torch.nn.Linear(hidden, hidden)
        self.o_proj = torch.nn.Linear(hidden, hidden)
        self.fc1 = torch.nn.Linear(hidden, intermediate)
        self.fc2 = torch.nn.Linear(intermediate, hidden)
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads

    def forward(self, x):
        B, S, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, S, D)
        x = x + self.o_proj(attn)
        h = self.norm2(x)
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(h)))
        return x


class _BertLikeModel(torch.nn.Module):
    def __init__(self, hidden, n_heads, n_layers, intermediate, vocab_size):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.layers = torch.nn.ModuleList(
            [_BertLikeBlock(hidden, n_heads, intermediate) for _ in range(n_layers)]
        )
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def _bert_like_model(hidden=768, n_heads=12, n_layers=2, vocab_size=30522,
                      batch=8, seq_len=512):
    def make_model():
        return _BertLikeModel(hidden, n_heads, n_layers, hidden * 4, vocab_size).cuda().eval()

    def make_args():
        return [torch.randint(0, vocab_size, (batch, seq_len), device="cuda")]

    return make_model, make_args


# --- ATen-level fusible region extraction ---


def _has_reduction(nodes: list[fx.Node]) -> bool:
    """Check if any node in the set is a reduction op using torch.Tag.reduction."""
    for n in nodes:
        if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload):
            if torch.Tag.reduction in n.target.tags:
                return True
    return False


def _merge_shared_input_reductions(components: list[list[fx.Node]]) -> list[list[fx.Node]]:
    """Merge reduction-containing partitions that share a common input.

    Models inductor's mix-order reduction: if two partitions both contain
    reductions and read from the same placeholder (or any common input outside
    both partitions), they'd be fused into one kernel. E.g., LayerNorm backward
    has inner reductions (sum -> [B,S,1]) and outer reductions (sum -> [D]) that
    both read from tangents_1.
    """
    if len(components) <= 1:
        return components

    reduction_idxs = [
        i for i, comp in enumerate(components) if _has_reduction(comp)
    ]
    if len(reduction_idxs) <= 1:
        return components

    node_to_comp = {}
    for i, comp in enumerate(components):
        for n in comp:
            node_to_comp[n] = i

    def _get_external_inputs(comp_nodes: list[fx.Node]) -> set[fx.Node]:
        node_set = set(comp_nodes)
        inputs = set()
        for n in comp_nodes:
            for inp in n.all_input_nodes:
                if inp not in node_set:
                    inputs.add(inp)
        return inputs

    # Union-Find for merging
    parent = list(range(len(components)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def _is_activation_tensor(node: fx.Node) -> bool:
        """True if the node looks like an activation (multi-dim, large).

        Rejects 1D tensors (weights/biases) and scalars, which cause
        spurious merges when weights are shared across layers (e.g. ALBERT).
        """
        val = node.meta.get("val", None)
        if isinstance(val, torch.Tensor):
            return val.dim() >= 2 and val.numel() >= 64
        return False

    inputs_by_comp = {i: _get_external_inputs(components[i]) for i in reduction_idxs}

    # Only merge when one partition is small (≤5 ops) — this captures the
    # LN backward case where a single isolated `sum(tangents, [0,1])` should
    # merge with the main LN backward partition. Avoids over-merging in
    # weight-sharing models like ALBERT.
    for i in range(len(reduction_idxs)):
        for j in range(i + 1, len(reduction_idxs)):
            ci, cj = reduction_idxs[i], reduction_idxs[j]
            smaller = min(len(components[ci]), len(components[cj]))
            if smaller > 5:
                continue
            shared = inputs_by_comp[ci] & inputs_by_comp[cj]
            if any(_is_activation_tensor(inp) for inp in shared):
                union(ci, cj)

    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(len(components)):
        groups[find(i)].append(i)

    merged = []
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(components[idxs[0]])
        else:
            combined = []
            seen = set()
            for idx in idxs:
                for n in components[idx]:
                    if id(n) not in seen:
                        seen.add(id(n))
                        combined.append(n)
            merged.append(combined)

    return merged


def _extract_regions_from_gm(
    gm: fx.GraphModule,
    extractor: "ReductionExtractor",
    gm_idx: int,
):
    """Extract reduction-containing fusion regions from a GraphModule.

    Uses CapabilityBasedPartitioner with is_fusible_node and no horizontal fusion,
    then merges reduction partitions that share common inputs (mix-order reduction).
    """
    from torch._inductor.fx_passes.fusion_regions import is_fusible_node
    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
    from torch.fx.passes.operator_support import create_op_support

    # Classify all call_function nodes as fusible or non-fusible
    non_fusible = []
    for n in gm.graph.nodes:
        if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload):
            if not is_fusible_node(n):
                non_fusible.append(n)

    def _is_supported(_submodules, node):
        return is_fusible_node(node)

    support = create_op_support(_is_supported)
    partitioner = CapabilityBasedPartitioner(
        gm, support, allows_single_node_partition=True,
    )
    partitions = partitioner.propose_partitions()
    components = [list(p.nodes.keys()) for p in partitions]

    n_before = len(components)
    components = _merge_shared_input_reductions(components)
    n_after = len(components)
    merged_msg = f" (merged {n_before} -> {n_after})" if n_before != n_after else ""

    print(f"  Graph {gm_idx}: {n_after} fusible partitions{merged_msg}")
    if non_fusible:
        from torch._inductor.lowering import fallbacks, lowerings
        from torch.utils.flop_counter import flop_registry
        nf_summary = collections.Counter()
        for n in non_fusible:
            target = n.target
            overload_packet = target.overloadpacket
            if overload_packet in flop_registry:
                reason = "compute (flop)"
            elif target in fallbacks:
                reason = "fallback"
            elif target not in lowerings:
                reason = "no lowering"
            else:
                reason = "other"
            nf_summary[f"{target.name()} ({reason})"] += 1
        print(f"  Non-fusible nodes ({len(non_fusible)}):")
        for op, count in nf_summary.most_common():
            print(f"    {op}: {count}")

    from torch._inductor.fx_passes.fusion_regions import is_view_node

    for comp in components:
        # Skip partitions that are all views/reshapes — no kernel generated
        if all(is_view_node(n) or n.op != "call_function" for n in comp):
            continue

        is_reduction = _has_reduction(comp)

        result = extractor._extract_subgraph(comp, gm)
        if result is None:
            continue
        sub_gm, placeholder_info = result

        reduction_infos = []
        for n in comp:
            if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload):
                if torch.Tag.reduction in n.target.tags:
                    val = n.meta.get("val", None)
                    shape = list(val.shape) if isinstance(val, torch.Tensor) else []
                    dtype = str(val.dtype) if isinstance(val, torch.Tensor) else "float32"
                    reduction_infos.append({
                        "reduction_type": n.target.overloadpacket.__name__,
                        "ranges": [str(s) for s in shape],
                        "reduction_ranges": [],
                        "dtype": dtype,
                        "src_dtype": dtype,
                        "origins": [n],
                    })

        origin_ops = sorted(str(n.target) for n in comp if n.op == "call_function")

        # Pattern hash: sorted op list (position-independent).
        pattern_key = hashlib.md5(
            json.dumps({"ops": origin_ops}, sort_keys=True).encode()
        ).hexdigest()[:12]

        # Shape key: input shapes determine the actual kernel config.
        input_shapes = sorted(
            f"{info.get('shape', '?')}:{info.get('dtype', '?')}"
            for info in placeholder_info.values()
        )
        shape_key = hashlib.md5(
            json.dumps(input_shapes).encode()
        ).hexdigest()[:8]

        full_key = f"{pattern_key}_{shape_key}"
        if full_key in extractor.seen_hashes:
            continue
        extractor.seen_hashes.add(full_key)

        kind = "reduction" if is_reduction else "pointwise"
        meta = {
            "node_type": "ATenPartition",
            "kind": kind,
            "node_name": f"aten_{gm_idx}_{extractor.counter}",
            "pattern_hash": pattern_key,
            "shape_hash": shape_key,
            "hash": full_key,
            "reductions": reduction_infos,
            "n_subnodes": len(comp),
            "n_reductions": len(reduction_infos),
            "n_pointwise": len(comp) - len(reduction_infos),
            "origin_ops": origin_ops,
        }

        if is_reduction:
            red_types = "_".join(sorted(set(r["reduction_type"] for r in reduction_infos)))
            filename = f"region_{extractor.counter:03d}_{red_types}_{pattern_key}_{shape_key}.py"
        else:
            filename = f"region_{extractor.counter:03d}_pointwise_{pattern_key}_{shape_key}.py"
        extractor.counter += 1

        try:
            filepath = extractor._generate_standalone_script(
                sub_gm, placeholder_info, meta, filename
            )
            print(f"    Extracted: {filepath}")
            extractor.captured.append({
                "file": filepath,
                "kind": kind,
                "pattern_hash": pattern_key,
                "shape_hash": shape_key,
                "hash": full_key,
                "node_name": meta["node_name"],
                "reduction_types": [r["reduction_type"] for r in reduction_infos],
                "n_ops": len(comp),
                "origin_ops": origin_ops,
            })
        except Exception as e:
            print(f"    Failed: {e}")


def run_aten_extraction(model_fn, model_args_fn, output_dir, model_name="model", inference_only=False, graph_dir=None):
    """Extract fusible regions (reductions + pointwise) at the ATen level.

    Captures post-decomposition FX graphs separately for fwd and bwd (skipping the
    joint fwd+bwd graph from training), then partitions with is_fusible_node (no
    horizontal fusion) to find kernel-sized fusible regions.

    Args:
        graph_dir: If set, saves full_graph_NNN.py files (complete post-grad graphs)
                   into this directory for recovery/analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    if graph_dir:
        os.makedirs(graph_dir, exist_ok=True)
    extractor = ReductionExtractor(output_dir)

    def _make_capture_hook(target_list):
        def capture_post_grad(graph_or_gm):
            if isinstance(graph_or_gm, fx.GraphModule):
                target_list.append(copy.deepcopy(graph_or_gm))
            elif isinstance(graph_or_gm, fx.Graph):
                if hasattr(graph_or_gm, 'owning_module') and graph_or_gm.owning_module is not None:
                    target_list.append(copy.deepcopy(graph_or_gm.owning_module))
            return graph_or_gm
        return capture_post_grad

    old_pass = inductor_config.post_grad_custom_pre_pass
    inductor_config.force_disable_caches = True
    inductor_config.split_reductions = False

    # Capture forward-only graph (eval, no_grad)
    fwd_gms = []
    torch._dynamo.reset()
    inductor_config.post_grad_custom_pre_pass = _make_capture_hook(fwd_gms)

    def _invoke(compiled_model, inputs):
        if len(inputs) == 1 and isinstance(inputs[0], dict):
            return compiled_model(**inputs[0])
        return compiled_model(*inputs)

    print(f"Tracing {model_name} (forward, ATen-level)...")
    model = model_fn()
    args = model_args_fn()
    compiled = torch.compile(model)
    try:
        with torch.no_grad():
            _invoke(compiled, args)
    except Exception as e:
        print(f"  Forward failed: {e}")

    bwd_gms = []
    if not inference_only:
        # Capture backward-only graph: training produces joint + bwd, keep only the last
        torch._dynamo.reset()
        inductor_config.post_grad_custom_pre_pass = _make_capture_hook(bwd_gms)

        print(f"Tracing {model_name} (backward, ATen-level)...")
        model2 = model_fn()
        model2.train()
        args2 = model_args_fn()
        for a in args2:
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                a.requires_grad_(True)
            elif isinstance(a, dict):
                for v in a.values():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        v.requires_grad_(True)
        compiled2 = torch.compile(model2)
        try:
            out2 = _invoke(compiled2, args2)
            if isinstance(out2, torch.Tensor):
                loss = out2.sum()
            elif hasattr(out2, "loss") and out2.loss is not None:
                loss = out2.loss
            elif hasattr(out2, "logits"):
                loss = out2.logits.sum()
            elif isinstance(out2, (tuple, list)):
                loss = sum(o.sum() for o in out2 if isinstance(o, torch.Tensor))
            else:
                loss = None
            if loss is not None:
                loss.backward()
        except Exception as e:
            print(f"  Backward failed: {e}")

        # Training captures [joint_fwd+bwd, bwd]. Skip the joint graph — use only the
        # last one (bwd). We already have fwd from the eval pass.
        if len(bwd_gms) >= 2:
            bwd_gms = bwd_gms[-1:]

    inductor_config.post_grad_custom_pre_pass = old_pass

    all_gms = [(gm, "fwd") for gm in fwd_gms] + [(gm, "bwd") for gm in bwd_gms]
    print(f"Captured {len(fwd_gms)} fwd + {len(bwd_gms)} bwd post-grad graphs")

    # Save full post-grad graphs for recovery
    if graph_dir:
        for g_idx, (gm, _label) in enumerate(all_gms):
            full_graph_path = os.path.join(graph_dir, f"full_graph_{g_idx:03d}.py")
            try:
                full_code = gm.print_readable(print_output=False)
                with open(full_graph_path, "w") as f:
                    f.write(full_code)
            except Exception:
                pass

    for gm_idx, (gm, label) in enumerate(all_gms):
        try:
            _extract_regions_from_gm(gm, extractor, gm_idx)
        except Exception as e:
            import traceback
            print(f"  Extraction failed on graph {gm_idx} ({label}): {e}")
            traceback.print_exc()

    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(extractor.captured, f, indent=2)

    n_red = sum(1 for c in extractor.captured if c.get("kind") == "reduction")
    n_pw = sum(1 for c in extractor.captured if c.get("kind") == "pointwise")
    print(f"\nExtracted {len(extractor.captured)} ATen-level fusion regions ({n_red} reduction, {n_pw} pointwise)")
    return extractor


# --- HuggingFace model helpers ---


def _hf_model(model_name, model_cls_name, batch=4, seq_len=512):
    """Create a model factory for a HuggingFace model."""
    def make_model():
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "use_cache"):
            config.use_cache = False
        import transformers
        model_cls = getattr(transformers, model_cls_name)
        return model_cls(config).cuda().eval()

    def make_args():
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        vocab_size = getattr(config, "vocab_size", 30522)
        return [torch.randint(0, vocab_size, (batch, seq_len), device="cuda")]

    return make_model, make_args


MODELS = {
    "simple": (
        _simple_model,
        lambda: [torch.randn(32, 512, 768, device="cuda")],
    ),
    "rmsnorm": (
        _rmsnorm_model,
        lambda: [torch.randn(32, 512, 768, device="cuda")],
    ),
    "cross_entropy": (
        _cross_entropy_model,
        lambda: [
            torch.randn(16384, 768, device="cuda"),
            torch.randint(0, 32000, (16384,), device="cuda"),
        ],
    ),
    "llama_like": _llama_like_model(),
    "llama_like_8k": _llama_like_model(hidden=4096, seq_len=8192, batch=1),
    "llama_small": _llama_like_model(hidden=768, n_heads=12, batch=32),
    "bert_like": _bert_like_model(),
}

HF_MODELS = {
    "bert": _hf_model("bert-base-uncased", "BertForMaskedLM", batch=8, seq_len=512),
    "bert_large": _hf_model("bert-large-uncased", "BertForMaskedLM", batch=4, seq_len=512),
    "gpt2": _hf_model("gpt2", "GPT2ForSequenceClassification", batch=8, seq_len=512),
    "gpt2_medium": _hf_model("gpt2-medium", "GPT2ForSequenceClassification", batch=4, seq_len=512),
    "opt": _hf_model("facebook/opt-350m", "OPTForCausalLM", batch=4, seq_len=512),
    "roberta": _hf_model("roberta-base", "RobertaForCausalLM", batch=8, seq_len=512),
    "albert": _hf_model("albert-base-v2", "AlbertForMaskedLM", batch=8, seq_len=512),
    "distilbert": _hf_model("distilbert-base-uncased", "DistilBertForMaskedLM", batch=8, seq_len=512),
    "electra": _hf_model("google/electra-base-discriminator", "ElectraForPreTraining", batch=8, seq_len=512),
    "deberta": _hf_model("microsoft/deberta-base", "DebertaForMaskedLM", batch=4, seq_len=512),
}


def update_canonical(extractor, model_name: str, canonical_dir: str):
    """Update canonical repro set with newly extracted regions.

    For each extracted region, upserts into the canonical directory:
      canonical/<kind>_<pattern_hash>/shapes.json  (adds new shape config)
      canonical/<kind>_<pattern_hash>/meta.json    (updates model list)
      canonical/<kind>_<pattern_hash>/repro.py     (written on first encounter)
    Also updates manifest.json.
    """
    from pathlib import Path
    canonical_path = Path(canonical_dir) / "canonical"
    manifest_path = Path(canonical_dir) / "manifest.json"
    canonical_path.mkdir(parents=True, exist_ok=True)

    # Load existing manifest or start fresh
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"version": 1, "models": {}}

    if model_name not in manifest["models"]:
        manifest["models"][model_name] = {"repros": []}

    for entry in extractor.captured:
        pattern_hash = entry.get("pattern_hash")
        if not pattern_hash:
            continue

        shape_hash = entry.get("shape_hash", "unknown")
        kind = entry.get("kind", "region")
        reduction_types = entry.get("reduction_types", [])
        kind_label = "_".join(reduction_types[:3]) if reduction_types else kind
        dir_name = f"{kind_label}_{pattern_hash}"
        repro_dir = canonical_path / dir_name
        repro_dir.mkdir(parents=True, exist_ok=True)

        # Update shapes.json
        shapes_path = repro_dir / "shapes.json"
        if shapes_path.exists():
            with open(shapes_path) as f:
                shapes_data = json.load(f)
        else:
            shapes_data = {"pattern_hash": pattern_hash, "configs": {}}

        config_key = f"{model_name.lower()}_{shape_hash[:8]}"
        if config_key not in shapes_data["configs"]:
            # Parse inputs from the generated repro file
            src_file = Path(entry["file"])
            if src_file.exists():
                try:
                    from canonicalize_repros import parse_make_inputs
                    input_specs = parse_make_inputs(src_file)
                except Exception:
                    input_specs = []
            else:
                input_specs = []

            shapes_data["configs"][config_key] = {
                "inputs": input_specs,
                "source_models": [model_name],
                "shape_hash": shape_hash,
            }
            with open(shapes_path, "w") as f:
                json.dump(shapes_data, f, indent=2)

        # Update meta.json
        meta_path = repro_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if model_name not in meta.get("models", []):
                meta["models"].append(model_name)
                meta["models"].sort()
                meta["n_models"] = len(meta["models"])
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        else:
            meta = {
                "pattern_hash": pattern_hash,
                "kind": kind,
                "reduction_types": reduction_types,
                "n_ops": entry.get("n_ops"),
                "origin_ops": entry.get("origin_ops", []),
                "n_models": 1,
                "models": [model_name],
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        # Write canonical repro.py if it doesn't exist yet
        repro_py = repro_dir / "repro.py"
        if not repro_py.exists():
            src_file = Path(entry["file"])
            if src_file.exists():
                try:
                    from canonicalize_repros import (
                        extract_repro_class,
                        extract_docstring,
                        extract_imports,
                        extract_make_inputs_body,
                        generate_canonical_repro,
                    )
                    repro_class = extract_repro_class(src_file)
                    docstring = extract_docstring(src_file)
                    imports = extract_imports(src_file)
                    fallback = extract_make_inputs_body(src_file)
                    if repro_class:
                        code = generate_canonical_repro(repro_class, docstring, imports, fallback)
                        repro_py.write_text(code)
                except Exception as e:
                    print(f"  Warning: could not generate canonical repro for {dir_name}: {e}")

        # Update manifest
        manifest["models"][model_name]["repros"].append({
            "pattern_hash": pattern_hash,
            "shape_config": config_key,
            "region_index": entry.get("node_name", ""),
            "original_file": Path(entry["file"]).name,
        })

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Updated canonical set: {canonical_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="simple",
                        help="Model name, 'all', or 'hf:<name>'")
    parser.add_argument("--mode", choices=["scheduler", "aten", "both"], default="scheduler",
                        help="Extraction mode: scheduler (inductor fusion), aten (is_fusible partitioning), or both")
    parser.add_argument("--inference-only", action="store_true",
                        help="Only extract inference (forward) graphs, skip backward")
    parser.add_argument("--canonical-dir", type=str, default=None,
                        help="Path to canonical repro set (e.g., repros/). If set, updates canonical structure after extraction.")
    args = parser.parse_args()

    model_name = args.model
    base_dir = os.path.dirname(__file__)

    def run_one(name, model_fn, args_fn, mode):
        if mode in ("scheduler", "both"):
            out = os.path.join(base_dir, "output", "repros", name)
            try:
                run_extraction(model_fn, args_fn, out, name)
            except Exception as e:
                import traceback
                print(f"Scheduler extraction failed on {name}: {e}")
                traceback.print_exc()
            torch._dynamo.reset()
            inductor_config._post_fusion_custom_pass = None

        if mode in ("aten", "both"):
            suffix = "_inference" if args.inference_only else ""
            out = os.path.join(base_dir, "output", "aten_repros", name + suffix)
            try:
                extractor = run_aten_extraction(model_fn, args_fn, out, name, inference_only=args.inference_only)
                if args.canonical_dir and extractor:
                    update_canonical(extractor, name + suffix, args.canonical_dir)
            except Exception as e:
                import traceback
                print(f"ATen extraction failed on {name}: {e}")
                traceback.print_exc()
            torch._dynamo.reset()

    def _dynamo_hf_model(dynamo_model_name):
        """Use benchmarks/dynamo/huggingface.py to load a model + inputs."""
        sys.path.insert(0, os.path.join(os.environ.get("PYTORCH_DIR", "/tmp/pytorch-work"), "benchmarks", "dynamo"))
        from huggingface import (
            BATCH_SIZE_KNOWN_MODELS,
            EXTRA_MODELS,
            generate_inputs_for_model,
            get_module_cls_by_model_name,
        )
        from huggingface_llm_models import HF_LLM_MODELS

        if dynamo_model_name in HF_LLM_MODELS:
            from transformers import AutoConfig, AutoModelForCausalLM, WhisperForConditionalGeneration
            config = AutoConfig.from_pretrained(dynamo_model_name)
            if hasattr(config, "use_cache"):
                config.use_cache = False
            batch_size = BATCH_SIZE_KNOWN_MODELS.get(dynamo_model_name, 8)
            is_whisper = "whisper" in dynamo_model_name.lower()

            def make_model():
                if is_whisper:
                    m = WhisperForConditionalGeneration(config)
                else:
                    m = AutoModelForCausalLM.from_config(config)
                return m.cuda().train()

            def make_args():
                if is_whisper:
                    import numpy as np
                    from transformers import WhisperProcessor
                    processor = WhisperProcessor.from_pretrained(dynamo_model_name)
                    audio = torch.randn(16000 * 30) * 0.1
                    inputs = dict(processor(audio, sampling_rate=16000, return_tensors="pt"))
                    inputs["input_features"] = inputs["input_features"].cuda()
                    inputs["decoder_input_ids"] = torch.tensor(
                        [[config.decoder_start_token_id]], device="cuda"
                    )
                    return [inputs]
                else:
                    seq_len = min(getattr(config, "max_position_embeddings", 512), 512)
                    vocab_size = config.vocab_size
                    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
                    labels = input_ids.clone()
                    return [{"input_ids": input_ids, "labels": labels}]

            return make_model, make_args

        if dynamo_model_name in EXTRA_MODELS:
            from transformers import AutoConfig
            config, model_cls = EXTRA_MODELS[dynamo_model_name]
        else:
            model_cls = get_module_cls_by_model_name(dynamo_model_name)
            config = model_cls.config_class()
            if hasattr(config, "pad_token_id") and config.pad_token_id is None:
                config.pad_token_id = 0

        if hasattr(config, "use_cache"):
            config.use_cache = False

        batch_size = BATCH_SIZE_KNOWN_MODELS.get(dynamo_model_name, 8)

        def make_model():
            if "auto" in model_cls.__module__:
                m = model_cls.from_config(config)
            else:
                m = model_cls(config)
            return m.cuda().eval()

        def make_args():
            m = make_model()
            example = generate_inputs_for_model(
                model_cls, m, dynamo_model_name, batch_size, "cuda",
                include_loss_args=True,
            )
            if isinstance(example, dict):
                return [example]
            return list(example)

        return make_model, make_args

    def _list_dynamo_hf_models():
        sys.path.insert(0, os.path.join(os.environ.get("PYTORCH_DIR", "/tmp/pytorch-work"), "benchmarks", "dynamo"))
        from huggingface import BATCH_SIZE_KNOWN_MODELS, EXTRA_MODELS
        from huggingface_llm_models import HF_LLM_MODELS
        return sorted(set(list(BATCH_SIZE_KNOWN_MODELS.keys()) + list(EXTRA_MODELS.keys()) + list(HF_LLM_MODELS.keys())))

    if model_name == "all":
        for name, entry in MODELS.items():
            model_fn, args_fn = entry if isinstance(entry, tuple) else entry
            run_one(name, model_fn, args_fn, args.mode)
    elif model_name.startswith("hf:"):
        hf_name = model_name[3:]
        if hf_name == "all":
            for name, (model_fn, args_fn) in HF_MODELS.items():
                run_one(f"hf_{name}", model_fn, args_fn, args.mode)
        elif hf_name in HF_MODELS:
            model_fn, args_fn = HF_MODELS[hf_name]
            run_one(f"hf_{hf_name}", model_fn, args_fn, args.mode)
        else:
            print(f"Unknown HF model: {hf_name}. Available: {list(HF_MODELS.keys())}")
    elif model_name.startswith("dynamo:"):
        dyn_name = model_name[7:]
        if dyn_name == "all":
            for name in _list_dynamo_hf_models():
                print(f"\n{'='*60}\n  {name}\n{'='*60}")
                try:
                    model_fn, args_fn = _dynamo_hf_model(name)
                    run_one(f"dynamo_{name}", model_fn, args_fn, args.mode)
                except Exception as e:
                    import traceback
                    print(f"SKIP {name}: {e}")
                    traceback.print_exc()
                torch._dynamo.reset()
        elif dyn_name == "list":
            for name in _list_dynamo_hf_models():
                print(name)
        else:
            model_fn, args_fn = _dynamo_hf_model(dyn_name)
            run_one(f"dynamo_{dyn_name}", model_fn, args_fn, args.mode)
    else:
        model_fn, args_fn = MODELS[model_name]
        run_one(model_name, model_fn, args_fn, args.mode)
