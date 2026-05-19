"""
Ingest post-grad FX graphs from CI tlparse artifacts.

Downloads tlparse artifacts from a GitHub Actions run, loads the
inductor_post_grad_graph_*.txt files as GraphModules, runs the capture
hook's partitioning on each, and merges the resulting regions into the
canonical repro set.

Usage:
    # Download and ingest from a specific run:
    python scripts/ingest_tlparse.py --run-id 25956360525 --suite torchbench

    # Ingest from already-downloaded tlparse dir:
    python scripts/ingest_tlparse.py --tlparse-dir /tmp/tb_tlparse3 --suite torchbench

    # Just list what's available in a run:
    python scripts/ingest_tlparse.py --run-id 25956360525 --list-only
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def download_tlparse(run_id: int, suite: str | None, output_dir: Path) -> Path:
    """Download tlparse artifacts from a GitHub Actions run."""
    pattern = "tlparse-*"
    if suite:
        pattern = f"tlparse-*{suite}*"

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gh", "run", "download", str(run_id),
        "--repo", "pytorch/pytorch",
        "--pattern", pattern,
        "--dir", str(output_dir),
    ]
    print(f"Downloading: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_dir


def find_post_grad_graphs(tlparse_dir: Path) -> list[Path]:
    """Find all inductor_post_grad_graph_*.txt files."""
    return sorted(tlparse_dir.rglob("inductor_post_grad_graph_*.txt"))


def find_fx_graph_runnables(tlparse_dir: Path) -> list[Path]:
    """Find all fx_graph_runnable_*.txt files."""
    return sorted(tlparse_dir.rglob("fx_graph_runnable_*.txt"))


def infer_model_name(graph_path: Path) -> str:
    """Infer model name from the tlparse directory structure."""
    # Path looks like: tlparse-...-test-inductor_torchbench_perf-4-6-.../  -_0_0_0/  file.txt
    parts = str(graph_path).split("/")
    for part in parts:
        if "tlparse-" in part:
            # Extract suite and model info from directory name
            # e.g. tlparse-runattempt1-test-inductor_torchbench_perf_cuda_h100-1-9-...
            return part.split("_76")[0].replace("tlparse-runattempt1-test-", "")
    return "unknown"


def _parse_input_shapes_from_reader(content: str):
    """Parse input tensor shapes/dtypes from reader.tensor() lines in load_args.

    Lines look like:
        reader.tensor(buf0, (32, 3, 3, 3), is_leaf=True)  # primals_1
        reader.tensor(buf2, (), dtype=torch.int64, is_leaf=True)  # primals_3
    """
    import re
    import torch

    dtype_map = {
        'torch.float32': torch.float32,
        'torch.float16': torch.float16,
        'torch.bfloat16': torch.bfloat16,
        'torch.int64': torch.int64,
        'torch.int32': torch.int32,
        'torch.int8': torch.int8,
        'torch.uint8': torch.uint8,
        'torch.bool': torch.bool,
        'torch.float64': torch.float64,
        'torch.complex64': torch.complex64,
        'torch.complex128': torch.complex128,
    }

    inputs = []
    for m in re.finditer(
        r'reader\.tensor\(\w+,\s*\(([^)]*)\)(?:,\s*dtype=([^,)]+))?', content
    ):
        shape_str, dtype_str = m.group(1), m.group(2)
        # Handle symbolic shapes (e.g. 's28') by substituting a concrete value
        dims = []
        for x in shape_str.split(','):
            x = x.strip()
            if not x:
                continue
            try:
                dims.append(int(x))
            except ValueError:
                # Symbolic dim (e.g. 's28', 's0') - use 32 as reasonable batch size
                dims.append(32)
        shape = tuple(dims)
        dtype = dtype_map.get(dtype_str, torch.float32) if dtype_str else torch.float32
        inputs.append((shape, dtype))

    return inputs


def _parse_input_shapes_from_annotations(content: str):
    """Parse input tensor shapes/dtypes from forward signature annotations.

    Annotations look like:
        primals_1: "f32[768, 768][768, 1]cuda:0"
        primals_2: "i64[1, 512][512, 1]cuda:0"
    """
    import re
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
    """Load a post-grad graph txt or fx_graph_runnable as an FX GraphModule.

    Handles two formats:
    - inductor_post_grad_graph_*.txt: has `class <lambda>(torch.nn.Module):`
    - fx_graph_runnable_*.txt: has `from torch.nn import *`, class Repro,
      __init__ with submodule constructors, and `if __name__` block.

    Strategy:
    1. Exec file content with globals that include torch.nn.* exports
    2. Strip __main__ blocks to avoid running the graph
    3. Replace `class <lambda>` with `class Repro`
    4. Trace the instantiated module with make_fx to produce a GraphModule
    """
    import re
    import torch
    import torch.nn
    import torch.fx as fx

    content = graph_path.read_text()

    # Fix 1: Replace `class <lambda>` with `class Repro` (invalid Python syntax)
    content = content.replace("class <lambda>", "class Repro")

    # Fix 2: Strip `if __name__ == '__main__':` blocks to avoid running the graph.
    # These blocks compile/run the model which we don't want.
    content = re.sub(
        r'^if __name__\s*==\s*[\'"]__main__[\'"]:\s*\n(?:(?:[ \t]+.*)?\n)*',
        '', content, flags=re.MULTILINE
    )

    # Also strip `mod = Repro()` lines at module level that appear before __main__
    # (the file creates the instance itself which we'll do ourselves)
    content = re.sub(r'^mod\s*=\s*Repro\(\)\s*$', '', content, flags=re.MULTILINE)

    # Extract input shapes BEFORE exec (from the original content)
    input_shapes = _parse_input_shapes_from_reader(content)
    if not input_shapes:
        input_shapes = _parse_input_shapes_from_annotations(content)

    # Build globals dict with all needed imports pre-populated.
    # This ensures `from torch.nn import *` style usage works (Module, etc.)
    exec_globals = {"__builtins__": __builtins__}

    # Add torch.nn.* to globals (handles AdaptiveAvgPool1d, Module, etc.)
    for attr in dir(torch.nn):
        if not attr.startswith('_'):
            exec_globals[attr] = getattr(torch.nn, attr)

    # Add core imports
    exec_globals['torch'] = torch
    exec_globals['fx'] = fx
    exec_globals['inf'] = math.inf
    exec_globals['nan'] = float('nan')
    exec_globals['device'] = torch.device
    exec_globals['tensor'] = torch.tensor

    try:
        exec(compile(content, str(graph_path), "exec"), exec_globals)

        # Find the class (named 'Repro' after our lambda fix, or other nn.Module subclass)
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

        # Instantiate -- this runs __init__ which may register buffers/submodules
        instance = graph_cls()

        # For inductor_post_grad_graph files: register _frozen_param* buffers
        # These appear as: `var: "dtype[shape][strides]device" = self._frozen_paramN`
        frozen_params = re.findall(
            r'self\.(_frozen_param\w+)',
            content
        )
        if frozen_params:
            # Parse their shapes from the annotation lines
            frozen_info = {}
            for m in re.finditer(
                r'"(\w+)\[([^\]]*)\].*?=\s*self\.(_frozen_param\w+)',
                content
            ):
                dtype_str, shape_str, param_name = m.group(1), m.group(2), m.group(3)
                dtype_short_map = {
                    'f32': torch.float32, 'f16': torch.float16,
                    'bf16': torch.bfloat16, 'i64': torch.int64,
                    'i32': torch.int32, 'b8': torch.bool,
                    'f64': torch.float64, 'i8': torch.int8, 'u8': torch.uint8,
                }
                dtype = dtype_short_map.get(dtype_str, torch.float32)
                shape = tuple(int(x.strip()) for x in shape_str.split(',') if x.strip())
                frozen_info[param_name] = (shape, dtype)

            dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            for param_name, (shape, dtype) in frozen_info.items():
                if not hasattr(instance, param_name):
                    if dtype in (torch.int64, torch.int32, torch.int8, torch.uint8):
                        buf = torch.zeros(shape, dtype=dtype, device=dev)
                    elif dtype == torch.bool:
                        buf = torch.zeros(shape, dtype=torch.bool, device=dev)
                    else:
                        buf = torch.randn(shape, dtype=dtype, device=dev)
                    instance.register_buffer(param_name, buf)

        # If already a GraphModule (has .graph), return directly
        if hasattr(instance, 'graph'):
            return instance

        # Otherwise, trace with make_fx to produce a proper GraphModule
        if not input_shapes:
            print(f"  No input shapes found for {graph_path.name}")
            return None

        from torch.fx.experimental.proxy_tensor import make_fx

        # Create inputs on CUDA (forward methods typically hardcode cuda:0 device)
        dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        instance = instance.to(dev)

        inputs = []
        for shape, dtype in input_shapes:
            if dtype in (torch.int64, torch.int32, torch.int8, torch.uint8):
                t = torch.zeros(shape, dtype=dtype, device=dev)
            elif dtype == torch.bool:
                t = torch.zeros(shape, dtype=torch.bool, device=dev)
            else:
                t = torch.randn(shape, dtype=dtype, device=dev)
            inputs.append(t)

        gm = make_fx(instance, tracing_mode="real")(*inputs)
        return gm

    except Exception as e:
        print(f"  Failed to load {graph_path.name}: {e}")
        return None


def process_graph_with_hook(gm, label: str, output_dir: Path, suite: str):
    """Run the capture hook's partitioning on a loaded GraphModule."""
    import torch
    from capture_hook import _CaptureState
    from merge_captures import merge_one_capture

    cap_dir = Path(tempfile.mkdtemp())
    state = _CaptureState(str(cap_dir), label=label)

    try:
        state.process_graph(gm)
        state.finalize()

        index_path = cap_dir / "index.json"
        if not index_path.exists():
            return 0

        n = merge_one_capture(cap_dir, output_dir, label, suite=suite, mode="infer")
        return n
    except Exception as e:
        print(f"  Process failed for {label}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=None,
                        help="GitHub Actions run ID to download from")
    parser.add_argument("--tlparse-dir", type=Path, default=None,
                        help="Already-downloaded tlparse directory")
    parser.add_argument("--suite", type=str, default="torchbench",
                        choices=["torchbench", "huggingface", "timm", "all"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/tmp/scratch_space/better_benchmark/repros"))
    parser.add_argument("--list-only", action="store_true",
                        help="Just list available graphs, don't process")
    parser.add_argument("--use-runnables", action="store_true",
                        help="Use fx_graph_runnable files directly instead of post_grad_graph")
    args = parser.parse_args()

    if args.tlparse_dir:
        tlparse_dir = args.tlparse_dir
    elif args.run_id:
        tlparse_dir = Path(tempfile.mkdtemp()) / "tlparse"
        suite_filter = args.suite if args.suite != "all" else None
        download_tlparse(args.run_id, suite_filter, tlparse_dir)
    else:
        parser.error("Must provide --run-id or --tlparse-dir")

    if args.use_runnables:
        graphs = find_fx_graph_runnables(tlparse_dir)
        print(f"Found {len(graphs)} fx_graph_runnable files")
    else:
        graphs = find_post_grad_graphs(tlparse_dir)
        print(f"Found {len(graphs)} inductor_post_grad_graph files")

    if args.list_only:
        for g in graphs:
            print(f"  {g.relative_to(tlparse_dir)}")
        return

    import torch
    total_regions = 0
    for i, graph_path in enumerate(graphs):
        label = infer_model_name(graph_path) + f"_graph{i}"
        print(f"[{i+1}/{len(graphs)}] {graph_path.name} ({label})...", end=" ", flush=True)

        gm = load_graph_module(graph_path)
        if gm is None:
            print("SKIP (load failed)")
            continue

        import copy
        n = process_graph_with_hook(copy.deepcopy(gm), label, args.output_dir, args.suite)
        total_regions += n
        print(f"{n} regions")

    print(f"\nDone: {total_regions} regions from {len(graphs)} graphs")


if __name__ == "__main__":
    main()
