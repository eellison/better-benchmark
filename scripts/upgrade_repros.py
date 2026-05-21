"""
Upgrade repros to current format by re-compiling through the capture hook.

Each repro is already a valid FX graph. torch.compile(Repro()) with the
capture hook installed re-captures it with the current template — no regex,
no parsing, just fresh generation.

Usage:
    python scripts/upgrade_repros.py                    # all outdated
    python scripts/upgrade_repros.py --repro var_mean_2096009f00b8  # one
    python scripts/upgrade_repros.py --dry-run          # show what would change
    python scripts/upgrade_repros.py --version 2        # upgrade to version 2
"""
import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CURRENT_VERSION = 2


def get_repro_version(repro_path: Path) -> int:
    """Read _repro_version from a repro file. Returns 0 if not present."""
    content = repro_path.read_text()
    match = re.search(r'^_repro_version\s*=\s*(\d+)', content, re.MULTILINE)
    return int(match.group(1)) if match else 0


def upgrade_one(repro_dir: Path, dry_run: bool = False) -> bool:
    """Upgrade one repro by re-compiling through the capture hook.

    Returns True if upgraded, False if skipped/failed.
    """
    repro_py = repro_dir / "repro.py"
    if not repro_py.exists():
        return False

    version = get_repro_version(repro_py)
    if version >= CURRENT_VERSION:
        return False  # already current

    if dry_run:
        print(f"  WOULD UPGRADE {repro_dir.name} (v{version} -> v{CURRENT_VERSION})")
        return True

    # Run in subprocess to isolate CUDA state
    result = subprocess.run(
        [sys.executable, "-c", f"""
import sys, os, importlib.util, math, tempfile, shutil, json
sys.path.insert(0, "{Path(__file__).resolve().parents[1]}")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch._dynamo
from capture_hook import install_capture_hook, uninstall_capture_hook

# Load the existing repro
spec = importlib.util.spec_from_file_location("repro", "{repro_py}")
mod = importlib.util.module_from_spec(spec)
mod.device = torch.device
mod.inf = math.inf
mod.nan = float("nan")
spec.loader.exec_module(mod)

repro = mod.Repro()
inputs = mod._default_make_inputs()

# Re-capture through the hook
cap_dir = tempfile.mkdtemp()
torch._dynamo.reset()
install_capture_hook(cap_dir, label="{repro_dir.name}")
try:
    compiled = torch.compile(repro)
    with torch.no_grad():
        compiled(*inputs)
        torch.cuda.synchronize()
except Exception as e:
    print(f"COMPILE_FAIL: {{e}}")
    sys.exit(1)
finally:
    uninstall_capture_hook()

# Check if capture produced a repro
index_path = os.path.join(cap_dir, "index.json")
if not os.path.exists(index_path):
    print("NO_CAPTURE")
    sys.exit(1)

index = json.loads(open(index_path).read())
if not index:
    print("EMPTY_CAPTURE")
    sys.exit(1)

# Copy the new repro over the old one (keep meta.json and shapes.txt)
new_repro = index[0]["file"]
print(f"OK: {{new_repro}}")
"""],
        capture_output=True, text=True, timeout=120,
    )

    if "OK:" in result.stdout:
        new_repro_path = result.stdout.strip().split("OK: ")[1]
        # Copy new repro.py over old, preserve meta.json and shapes.txt
        shutil.copy2(new_repro_path, repro_py)
        print(f"  UPGRADED {repro_dir.name}")
        return True
    else:
        err = result.stderr.strip().split("\n")[-1][:80] if result.stderr.strip() else result.stdout.strip()[:80]
        print(f"  FAILED {repro_dir.name}: {err}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repro", type=str, default=None)
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros/canonical"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--version", type=int, default=CURRENT_VERSION)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    pass  # uses module-level CURRENT_VERSION

    if args.repro:
        dirs = [args.canonical_dir / args.repro]
    else:
        dirs = sorted(args.canonical_dir.iterdir())

    outdated = []
    current = 0
    for d in dirs:
        repro = d / "repro.py"
        if not repro.exists():
            continue
        v = get_repro_version(repro)
        if v < CURRENT_VERSION:
            outdated.append(d)
        else:
            current += 1

    print(f"Current (v{CURRENT_VERSION}): {current}")
    print(f"Outdated: {len(outdated)}")

    if not outdated:
        print("Nothing to upgrade.")
        return

    upgraded = 0
    failed = 0
    for d in outdated:
        if upgrade_one(d, dry_run=args.dry_run):
            upgraded += 1
        else:
            failed += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Upgraded: {upgraded}, Failed: {failed}")


if __name__ == "__main__":
    main()
