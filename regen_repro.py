"""
Regenerate a canonical repro by running it through torch.compile with the capture hook.

This re-extracts the FX graph from scratch, picking up any changes to decompositions,
lowerings, or fusion behavior in the current PyTorch checkout.

Usage:
    # Regenerate one repro (uses its default inputs)
    python regen_repro.py repros/canonical/var_mean_a7cbd072693b/

    # Regenerate and merge back into the canonical set
    python regen_repro.py repros/canonical/var_mean_a7cbd072693b/ --merge

    # Regenerate all canonical repros
    python regen_repro.py repros/canonical/ --all --merge
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def regen_one(repro_dir: Path, output_dir: Path) -> bool:
    """Regenerate a single repro by compiling it with the capture hook."""
    repro_py = repro_dir / "repro.py"
    if not repro_py.exists():
        print(f"  No repro.py in {repro_dir}")
        return False

    # Run in subprocess with capture hook installed
    script = f"""
import sys, os
sys.path.insert(0, '{Path(__file__).resolve().parent}')
from capture_hook import install_capture_hook
install_capture_hook('{output_dir}', label='regen')

# Load and run the repro through compile
import importlib.util
spec = importlib.util.spec_from_file_location('repro_mod', '{repro_py.resolve()}')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import torch
repro = mod.Repro()
inputs = mod.make_inputs()

compiled = torch.compile(repro)
with torch.no_grad():
    compiled(*inputs)
    torch.cuda.synchronize()

from capture_hook import uninstall_capture_hook
uninstall_capture_hook()
"""
    env = os.environ.copy()
    env["TORCH_NATIVE_SKIP_VERSION_CHECK"] = "1"

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120, env=env
    )
    if proc.returncode != 0:
        print(f"  FAILED: {proc.stderr[-300:]}")
        return False

    # Check output
    index = output_dir / "index.json"
    if not index.exists():
        print(f"  No captures produced")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Regenerate canonical repros")
    parser.add_argument("path", type=Path,
                        help="Path to a canonical repro dir, or parent dir with --all")
    parser.add_argument("--all", action="store_true",
                        help="Regenerate all repros under the given path")
    parser.add_argument("--merge", action="store_true",
                        help="Merge regenerated captures back into canonical set")
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros"),
                        help="Canonical set root (for --merge)")
    args = parser.parse_args()

    if args.all:
        repro_dirs = sorted(d for d in args.path.iterdir() if d.is_dir() and (d / "repro.py").exists())
    else:
        repro_dirs = [args.path]

    print(f"Regenerating {len(repro_dirs)} repro(s)...\n")

    success = 0
    failed = 0
    for repro_dir in repro_dirs:
        print(f"  {repro_dir.name}...", end=" ")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            if regen_one(repro_dir, tmp_path):
                print("OK")
                success += 1
                if args.merge:
                    from merge_captures import merge_one_capture
                    model_name = f"regen_{repro_dir.name}"
                    merge_one_capture(tmp_path, args.canonical_dir, model_name)
            else:
                failed += 1

    print(f"\nDone: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
