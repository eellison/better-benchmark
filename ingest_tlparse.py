"""
Ingest fx_graph_runnable files from tlparse CI extraction into the canonical repro set.

These files are full post-AOT FX graphs from PyTorch CI perf runs. We run each through
torch.compile with our capture hook installed, which partitions them into fused kernel
regions and extracts individual repros.

Usage:
    # Ingest all fx_graph_runnables from an extracted tlparse dir
    python ingest_tlparse.py /path/to/tlparse_h100_extracted/

    # Ingest a single file
    python ingest_tlparse.py /path/to/fx_graph_runnable_5.txt

    # Dry run (just count what would be processed)
    python ingest_tlparse.py /path/to/tlparse_h100_extracted/ --dry-run
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def find_fx_graph_runnables(search_path: Path) -> list[Path]:
    if search_path.is_file():
        return [search_path]
    return sorted(search_path.rglob("*fx_graph_runnable*"))


def model_label_from_path(fx_path: Path) -> str:
    """Extract a model label from the tlparse directory structure."""
    parts = str(fx_path)
    # Pattern: tlparse-test-inductor_{suite}_perf_cuda_{hw}-{shard}-...
    m = re.search(r'tlparse-test-(inductor_\w+_perf_cuda_\w+)-(\d+)-', parts)
    if m:
        suite = m.group(1).replace("inductor_", "").replace("_perf_cuda_h100", "")
        shard = m.group(2)
        graph_num = re.search(r'fx_graph_runnable_(\d+)', fx_path.name)
        num = graph_num.group(1) if graph_num else "0"
        return f"tlparse_{suite}_s{shard}_g{num}"
    return f"tlparse_{fx_path.stem}"


def run_capture(fx_path: Path, output_dir: Path, label: str) -> bool:
    """Run an fx_graph_runnable through compile with capture hook.

    Strategy: exec the file to define Repro + load_args, install our capture
    hook as post_grad_custom_pre_pass, then just compile and run once. The
    post-grad pass fires during compilation and captures all fused regions.
    """
    script = f"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

import torch
import torch._inductor.config as inductor_config
from torch import tensor, device
from math import inf
from torch._dynamo.testing import rand_strided

# Execute the fx_graph_runnable to define Repro, load_args, and set configs.
_src = open('{fx_path}').read()
_src = _src.replace("if __name__ == '__main__':", "if False:  # suppressed")
exec(_src)

# Disable expensive passes that slow compilation for capture purposes
inductor_config.pre_grad_fusion_options = {{}}
inductor_config.post_grad_fusion_options = {{}}

# Install capture hook (post_grad_custom_pre_pass)
sys.path.insert(0, '{ROOT}')
from capture_hook import install_capture_hook, uninstall_capture_hook
install_capture_hook('{output_dir}', label='{label}')

# run_repro command='run' does torch.compile + run internally,
# which triggers our post_grad pass during compilation
from torch._dynamo.repro.after_aot import run_repro
with torch.no_grad():
    run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)

uninstall_capture_hook()
"""
    env = os.environ.copy()
    env["TORCH_NATIVE_SKIP_VERSION_CHECK"] = "1"

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120, env=env,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT (120s)")
        return False
    if proc.returncode != 0:
        stderr_tail = proc.stderr[-1000:] if proc.stderr else "(no stderr)"
        stdout_tail = proc.stdout[-1000:] if proc.stdout else ""
        print(f"    FAILED (rc={proc.returncode}):")
        if stdout_tail:
            print(f"    stdout: {stdout_tail}")
        print(f"    stderr: {stderr_tail}")
        return False

    index = output_dir / "index.json"
    return index.exists()


def main():
    parser = argparse.ArgumentParser(description="Ingest tlparse fx_graph_runnables into canonical repros")
    parser.add_argument("path", type=Path,
                        help="Path to tlparse extracted dir or single fx_graph_runnable file")
    parser.add_argument("--canonical-dir", type=Path, default=ROOT / "repros",
                        help="Output canonical repro directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just list files that would be processed")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="Skip graphs larger than this (default: no limit)")
    parser.add_argument("--min-lines", type=int, default=90,
                        help="Skip graphs smaller than this (trivial)")
    args = parser.parse_args()

    fx_files = find_fx_graph_runnables(args.path)
    print(f"Found {len(fx_files)} fx_graph_runnable files")

    # Filter by size
    filtered = []
    for f in fx_files:
        n_lines = sum(1 for _ in open(f))
        if args.max_lines and n_lines > args.max_lines:
            print(f"  SKIP (too large, {n_lines} lines): {f.name}")
            continue
        if n_lines < args.min_lines:
            print(f"  SKIP (too small, {n_lines} lines): {f.name}")
            continue
        filtered.append((f, n_lines))

    print(f"\nProcessing {len(filtered)} files (skipped {len(fx_files) - len(filtered)})")

    if args.dry_run:
        for f, n in filtered:
            label = model_label_from_path(f)
            print(f"  {n:5d} lines  {label}")
        return

    success = 0
    failed = 0
    total_regions = 0

    for i, (fx_path, n_lines) in enumerate(filtered, 1):
        label = model_label_from_path(fx_path)
        print(f"\n[{i}/{len(filtered)}] {label} ({n_lines} lines)...")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            if run_capture(fx_path, tmp_path, label):
                # Count captured regions
                import json
                index = json.loads((tmp_path / "index.json").read_text())
                n_regions = len(index)
                total_regions += n_regions
                print(f"    OK: {n_regions} regions captured")

                # Merge into canonical set
                from merge_captures import merge_one_capture
                merge_one_capture(tmp_path, args.canonical_dir, label)
                success += 1
            else:
                failed += 1

    print(f"\n{'='*60}")
    print(f"Done: {success} graphs processed, {failed} failed")
    print(f"Total kernel regions captured: {total_regions}")
    print(f"Canonical repros at: {args.canonical_dir}/")


if __name__ == "__main__":
    main()
