"""
Re-partition ALL saved full_graph_*.py files using the fixed capture hook.

The capture hook now splits independent chains into separate repros (via
connected-component analysis). This script loads each saved post-grad FX graph,
runs process_graph() to partition it, and merges the resulting regions into
repros/canonical/.

Usage:
    python scripts/repartition_from_graphs.py

    # Dry run (just test loading):
    python scripts/repartition_from_graphs.py --dry-run

    # Process a single file:
    python scripts/repartition_from_graphs.py --single repros/models/genai/RMSNormForward/full_graph_000.py
"""
import argparse
import copy
import math
import os
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")


def find_all_full_graphs(models_dir: Path) -> list[Path]:
    """Find all full_graph_*.py files under repros/models/."""
    return sorted(models_dir.rglob("full_graph_*.py"))


def infer_label_suite_mode(graph_path: Path) -> tuple[str, str, str | None]:
    """Infer label, suite, and mode from a full_graph path.

    Path patterns:
        repros/models/hf/{infer,train}/<ModelName>/full_graph_NNN.py
        repros/models/torchbench/infer/<model_name>/full_graph_NNN.py
        repros/models/timm/infer/<model_name>/full_graph_NNN.py
        repros/models/vllm/<model_name>/full_graph_NNN.py
        repros/models/genai/<OpName>/full_graph_NNN.py

    Returns:
        (label, suite, mode) where label is like "timm_resnet50_infer_000"
    """
    # Get the path relative to repros/models/
    rel = graph_path.relative_to(PROJECT_ROOT / "repros" / "models")
    parts = list(rel.parts)

    # parts[0] = suite (hf, torchbench, timm, vllm, genai)
    suite = parts[0]

    # Extract graph index from filename
    graph_idx = graph_path.stem.replace("full_graph_", "")  # e.g. "000"

    if suite in ("hf", "torchbench", "timm"):
        # Pattern: suite/mode/model_name/full_graph_NNN.py
        mode = parts[1] if len(parts) > 2 else "infer"
        model_name = parts[2] if len(parts) > 3 else parts[1]
        label = f"{suite}_{model_name}_{mode}_{graph_idx}"
    elif suite == "vllm":
        # Pattern: vllm/model_name/full_graph_NNN.py
        mode = None
        model_name = parts[1] if len(parts) > 2 else "unknown"
        label = f"vllm_{model_name}_{graph_idx}"
    elif suite == "genai":
        # Pattern: genai/OpName/full_graph_NNN.py
        mode = None
        model_name = parts[1] if len(parts) > 2 else "unknown"
        label = f"genai_{model_name}_{graph_idx}"
    else:
        mode = None
        model_name = parts[1] if len(parts) > 1 else "unknown"
        label = f"{suite}_{model_name}_{graph_idx}"

    return label, suite, mode


def _parse_input_shapes_from_annotations(content: str):
    """Parse input tensor shapes/dtypes from forward signature annotations.

    Annotations look like:
        arg0_1: "f32[768, 768]"
        primals_1: "i64[1, 512]"
        arg0_1: "bf16[1152000, 512]"
    """
    import torch

    dtype_map = {
        'f32': torch.float32,
        'f16': torch.float16,
        'bf16': torch.bfloat16,
        'i64': torch.int64,
        'i32': torch.int32,
        'i8': torch.int8,
        'u8': torch.uint8,
        'b8': torch.bool,
        'f64': torch.float64,
        'c64': torch.complex64,
        'c128': torch.complex128,
    }

    # Match the forward signature
    fwd_match = re.search(r'def forward\(self,?\s*(.*?)\):', content, re.DOTALL)
    if not fwd_match:
        return []

    sig = fwd_match.group(1)
    inputs = []
    for m in re.finditer(r'\w+:\s*"(\w+)\[([^\]]*)\]', sig):
        dtype_str, shape_str = m.group(1), m.group(2)
        dims = []
        for x in shape_str.split(','):
            x = x.strip()
            if not x:
                continue
            try:
                dims.append(int(x))
            except ValueError:
                dims.append(32)  # symbolic dim fallback
        shape = tuple(dims)
        dtype = dtype_map.get(dtype_str, torch.float32)
        inputs.append((shape, dtype))

    return inputs


def load_graph_module(graph_path: Path):
    """Load a full_graph_*.py file as an FX GraphModule.

    These files are print_readable() output: they define a class GraphModule
    that inherits from torch.nn.Module with a forward method containing
    torch.ops.aten.* calls with type annotations.

    Strategy:
    1. Prepend required imports (torch, etc.)
    2. exec() the file content
    3. Instantiate the class
    4. If it has .graph → use directly (proper GraphModule)
    5. Otherwise → trace with make_fx using inferred input shapes
    """
    import torch
    import torch.fx as fx
    import torch._inductor.inductor_prims  # noqa: F401 - registers prims.fma, inductor_seeds, etc.

    content = graph_path.read_text()

    # These files start with `class GraphModule(torch.nn.Module):` — no imports.
    # We need to provide torch in the exec globals.

    # Extract input shapes from annotations BEFORE any modifications
    input_shapes = _parse_input_shapes_from_annotations(content)

    # Build globals dict
    exec_globals = {"__builtins__": __builtins__}
    exec_globals['torch'] = torch
    exec_globals['inf'] = math.inf
    exec_globals['nan'] = float('nan')
    exec_globals['device'] = torch.device

    # Add torch.nn.* for any Module subclass references
    for attr in dir(torch.nn):
        if not attr.startswith('_'):
            exec_globals[attr] = getattr(torch.nn, attr)

    try:
        exec(compile(content, str(graph_path), "exec"), exec_globals)

        # Find the GraphModule class
        graph_cls = None
        for name, v in exec_globals.items():
            if (isinstance(v, type)
                    and issubclass(v, torch.nn.Module)
                    and v is not torch.nn.Module
                    and name not in dir(torch.nn)):
                graph_cls = v
                break

        if graph_cls is None:
            return None

        # Instantiate
        instance = graph_cls()

        # Register _frozen_param* and _tensor_constant* buffers that are
        # referenced via self.<name> in the forward body but not yet registered.
        # Parse their shapes/dtypes from the annotation lines in the source.
        _self_refs = re.findall(
            r'self\.(_(?:frozen_param|tensor_constant)\w*)',
            content
        )
        if _self_refs:
            _dtype_short_map = {
                'f32': torch.float32, 'f16': torch.float16,
                'bf16': torch.bfloat16, 'i64': torch.int64,
                'i32': torch.int32, 'b8': torch.bool,
                'f64': torch.float64, 'i8': torch.int8, 'u8': torch.uint8,
            }
            dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            for m in re.finditer(
                r'"(\w+)\[([^\]]*)\].*?=\s*self\.(_(?:frozen_param|tensor_constant)\w*)',
                content
            ):
                dtype_str, shape_str, param_name = m.group(1), m.group(2), m.group(3)
                dtype = _dtype_short_map.get(dtype_str, torch.float32)
                shape = tuple(int(x.strip()) for x in shape_str.split(',') if x.strip())
                if not hasattr(instance, param_name):
                    if dtype in (torch.int64, torch.int32, torch.int8, torch.uint8):
                        buf = torch.zeros(shape, dtype=dtype, device=dev)
                    elif dtype == torch.bool:
                        buf = torch.zeros(shape, dtype=torch.bool, device=dev)
                    else:
                        buf = torch.randn(shape, dtype=dtype, device=dev)
                    instance.register_buffer(param_name, buf)

        # If already a GraphModule (has .graph), return directly
        if hasattr(instance, 'graph') and isinstance(instance.graph, fx.Graph):
            return instance

        # Otherwise, trace with make_fx (fake mode) to produce a proper GraphModule
        # with metadata (shapes/dtypes on nodes) that process_graph needs.
        if not input_shapes:
            print(f"  No input shapes found for {graph_path.name}")
            return None

        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx

        # Use fake tensors -- no real computation, just shape propagation.
        # Use CUDA device since these are post-grad graphs from CUDA compilation.
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        with FakeTensorMode() as fake_mode:
            inputs = []
            for shape, dtype in input_shapes:
                if dtype in (torch.int64, torch.int32, torch.int8, torch.uint8):
                    t = torch.zeros(shape, dtype=dtype, device=dev)
                elif dtype == torch.bool:
                    t = torch.zeros(shape, dtype=torch.bool, device=dev)
                else:
                    t = torch.randn(shape, dtype=dtype, device=dev)
                inputs.append(fake_mode.from_tensor(t))

            gm = make_fx(instance, tracing_mode="fake")(*inputs)
        return gm

    except Exception as e:
        print(f"  Failed to load {graph_path.name}: {e}")
        return None


def process_graph_with_hook(gm, label: str, output_dir: Path, suite: str, mode: str | None,
                            validate: bool = True):
    """Run the capture hook's partitioning on a loaded GraphModule."""
    from capture_hook import _CaptureState
    from merge_captures import temporary_capture_for_merge

    with temporary_capture_for_merge(
        output_dir,
        label,
        suite=suite,
        mode=mode,
        prefix="repartition_from_graphs_",
    ) as capture:
        state = _CaptureState(str(capture.capture_dir), label=label, validate=validate)

        try:
            state.process_graph(gm)
            state.finalize()

            index_path = capture.capture_dir / "index.json"
            if not index_path.exists():
                return 0

            return capture.merge()
        except Exception as e:
            print(f"  Process failed for {label}: {e}")
            return 0


def main():
    parser = argparse.ArgumentParser(
        description="Re-partition all saved full_graph_*.py files using the fixed capture hook"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Just test loading graphs, don't process/merge")
    parser.add_argument("--single", type=Path, default=None,
                        help="Process a single graph file (for testing)")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "repros",
                        help="Output directory for canonical repros")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip eager validation of captured repros (faster)")
    parser.add_argument("--suite-filter", type=str, default=None,
                        choices=["hf", "timm", "torchbench", "vllm", "genai"],
                        help="Only process graphs from this suite")
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / "repros" / "models"

    if args.single:
        graph_files = [args.single.resolve()]
    else:
        graph_files = find_all_full_graphs(models_dir)

    if args.suite_filter:
        graph_files = [f for f in graph_files if f"/{args.suite_filter}/" in str(f)]

    print(f"Found {len(graph_files)} full_graph files to process")
    print(f"Output dir: {args.output_dir}")
    print()

    import torch
    total_regions = 0
    loaded = 0
    failed = 0

    for i, graph_path in enumerate(graph_files):
        label, suite, mode = infer_label_suite_mode(graph_path)
        rel_path = graph_path.relative_to(PROJECT_ROOT)
        print(f"[{i+1}/{len(graph_files)}] {rel_path} → {label}", end=" ", flush=True)

        gm = load_graph_module(graph_path)
        if gm is None:
            print("SKIP (load failed)")
            failed += 1
            continue
        loaded += 1

        if args.dry_run:
            n_nodes = len(list(gm.graph.nodes))
            print(f"OK ({n_nodes} nodes)")
            continue

        n = process_graph_with_hook(
            copy.deepcopy(gm), label, args.output_dir, suite, mode,
            validate=not args.skip_validation
        )
        total_regions += n
        print(f"→ {n} regions")

        # Free GPU memory periodically
        if i % 20 == 0:
            torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total graph files:    {len(graph_files)}")
    print(f"  Successfully loaded:  {loaded}")
    print(f"  Failed to load:       {failed}")
    if not args.dry_run:
        print(f"  Total regions merged: {total_regions}")
    print("=" * 60)


if __name__ == "__main__":
    main()
