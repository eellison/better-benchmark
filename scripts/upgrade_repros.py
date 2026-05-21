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
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repro_harness import CURRENT_REPRO_VERSION, read_repro_version

CURRENT_VERSION = CURRENT_REPRO_VERSION


def get_repro_version(repro_path: Path) -> int:
    """Read _repro_version from a repro file. Returns 0 if not present."""
    return read_repro_version(repro_path)


def validate_target_version(target_version: int) -> None:
    """Validate that the requested target is one this generator can produce."""
    if target_version != CURRENT_REPRO_VERSION:
        raise ValueError(
            f"target version must be v{CURRENT_REPRO_VERSION}; "
            f"current capture generation emits v{CURRENT_REPRO_VERSION}"
        )


def _ok_capture_path(stdout: str) -> Path | None:
    for line in stdout.splitlines():
        if line.startswith("OK: "):
            return Path(line.removeprefix("OK: ").strip())
    return None


def select_single_capture_file(index: object) -> str:
    """Return the only captured repro path, or fail closed on split captures."""
    if not isinstance(index, list):
        raise ValueError("index.json is not a list")
    if len(index) != 1:
        raise ValueError(f"expected exactly one captured region, got {len(index)}")
    entry = index[0]
    if not isinstance(entry, dict):
        raise ValueError("captured region entry is not an object")
    file_name = entry.get("file")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("captured region is missing file path")
    return file_name


def copy_validated_repro(
    new_repro_path: Path,
    repro_py: Path,
    target_version: int,
) -> None:
    """Copy a captured repro only after confirming it has the target marker."""
    new_version = get_repro_version(new_repro_path)
    if new_version != target_version:
        raise ValueError(
            f"captured repro has version v{new_version}, expected v{target_version}"
        )
    shutil.copy2(new_repro_path, repro_py)


def upgrade_one(
    repro_dir: Path,
    dry_run: bool = False,
    target_version: int = CURRENT_REPRO_VERSION,
) -> bool:
    """Upgrade one repro by re-compiling through the capture hook.

    Returns True if upgraded, False if skipped/failed.
    """
    validate_target_version(target_version)

    repro_py = repro_dir / "repro.py"
    if not repro_py.exists():
        return False

    try:
        version = get_repro_version(repro_py)
    except ValueError as e:
        print(f"  FAILED {repro_dir.name}: invalid version marker: {e}")
        return False

    if version == target_version:
        return False  # already current
    if version > target_version:
        print(
            f"  FAILED {repro_dir.name}: repro version v{version} "
            f"is newer than target v{target_version}"
        )
        return False

    if dry_run:
        print(f"  WOULD UPGRADE {repro_dir.name} (v{version} -> v{target_version})")
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
sys.path.insert(0, "{Path(__file__).resolve().parent}")
from upgrade_repros import select_single_capture_file

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

try:
    new_repro = select_single_capture_file(index)
except ValueError as e:
    print(f"BAD_CAPTURE: {{e}}")
    sys.exit(1)
print(f"OK: {{new_repro}}")
"""],
        capture_output=True, text=True, timeout=120,
    )

    new_repro_path = _ok_capture_path(result.stdout)
    if new_repro_path is None:
        err = (
            result.stderr.strip().split("\n")[-1][:80]
            if result.stderr.strip()
            else result.stdout.strip()[:80]
        )
        print(f"  FAILED {repro_dir.name}: {err}")
        return False

    try:
        copy_validated_repro(new_repro_path, repro_py, target_version)
    except (OSError, ValueError) as e:
        print(f"  FAILED {repro_dir.name}: {e}")
        return False

    print(f"  UPGRADED {repro_dir.name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repro", type=str, default=None)
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros/canonical"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--version", type=int, default=CURRENT_VERSION)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    try:
        validate_target_version(args.version)
    except ValueError as e:
        parser.error(str(e))

    if args.repro:
        dirs = [args.canonical_dir / args.repro]
    else:
        dirs = sorted(args.canonical_dir.iterdir())

    outdated = []
    current = 0
    future = []
    invalid = []
    for d in dirs:
        repro = d / "repro.py"
        if not repro.exists():
            continue
        try:
            v = get_repro_version(repro)
        except ValueError as e:
            invalid.append((d.name, str(e)))
            continue
        if v < args.version:
            outdated.append(d)
        elif v == args.version:
            current += 1
        else:
            future.append((d.name, v))

    print(f"Target version: v{args.version}")
    print(f"Current (v{args.version}): {current}")
    print(f"Outdated: {len(outdated)}")

    if future:
        print(f"Newer than target: {len(future)}")
        for name, version in future[:20]:
            print(f"  {name}: v{version}")
        if len(future) > 20:
            print(f"  ... and {len(future) - 20} more")
        sys.exit(1)

    if invalid:
        print(f"Invalid version markers: {len(invalid)}")
        for name, err in invalid[:20]:
            print(f"  {name}: {err}")
        if len(invalid) > 20:
            print(f"  ... and {len(invalid) - 20} more")
        sys.exit(1)

    if not outdated:
        print("Nothing to upgrade.")
        return

    upgraded = 0
    failed = 0
    for d in outdated:
        if upgrade_one(d, dry_run=args.dry_run, target_version=args.version):
            upgraded += 1
        else:
            failed += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Upgraded: {upgraded}, Failed: {failed}")


if __name__ == "__main__":
    main()
