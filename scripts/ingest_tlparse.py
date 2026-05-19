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


def load_graph_module(graph_path: Path):
    """Load a post-grad graph txt as an FX GraphModule."""
    import torch
    import torch.fx as fx

    content = graph_path.read_text()

    # The post_grad_graph files have format:
    # class <lambda>(torch.nn.Module):
    #     def forward(self, arg0_1: "dtype[shape]", ...):
    #         ...
    # We need to exec this to get the GraphModule

    # Create a module with the graph class
    module_code = f"""
import torch
import torch.fx as fx
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

{content}
"""

    try:
        local_ns = {}
        exec(compile(module_code, str(graph_path), "exec"), local_ns)

        # Find the class (usually named '<lambda>' or 'Repro')
        graph_cls = None
        for v in local_ns.values():
            if isinstance(v, type) and issubclass(v, torch.nn.Module) and v is not torch.nn.Module:
                graph_cls = v
                break

        if graph_cls is None:
            return None

        # Instantiate and get the graph
        instance = graph_cls()
        if hasattr(instance, 'graph'):
            return instance
        return None

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
