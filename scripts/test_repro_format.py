"""
Test repro format consistency.

Checks that all repros follow the canonical format:
1. Has _shapes_config string
2. _default_make_inputs uses parse_shapes_config (no raw torch.randn/randint)
3. _shapes_config produces correct number of inputs for forward() signature
4. shapes.txt lines parse and produce correct input count

Usage:
    python scripts/test_repro_format.py              # all repros
    python scripts/test_repro_format.py --quick      # format checks only (no GPU)
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_has_shapes_config(content: str) -> str | None:
    if '_shapes_config' not in content:
        return "missing _shapes_config"
    return None


def check_no_raw_inputs(content: str) -> str | None:
    """Verify _default_make_inputs uses parse_shapes_config, not raw torch calls."""
    match = re.search(r'def _default_make_inputs\(\):(.*?)(?=\ndef |\Z)', content, re.DOTALL)
    if not match:
        return "missing _default_make_inputs"
    body = match.group(1)
    if 'torch.randn' in body or 'torch.randint' in body or 'torch.randperm' in body:
        return "raw torch factory in _default_make_inputs (should use parse_shapes_config)"
    if 'parse_shapes_config' not in body:
        return "_default_make_inputs doesn't call parse_shapes_config"
    return None


def check_input_count(content: str) -> str | None:
    """Check _shapes_config produces right number of inputs for forward()."""
    # Count forward args
    fwd_match = re.search(r'def forward\(self,([^)]*)\)', content)
    if not fwd_match:
        return "no forward() found"
    fwd_args = [a.strip() for a in fwd_match.group(1).split(',') if a.strip()]
    n_fwd = len(fwd_args)

    # Count items in _shapes_config
    config_match = re.search(r'^_shapes_config\s*=\s*"(.+)"', content, re.MULTILINE)
    if not config_match:
        return None  # already caught by check_has_shapes_config
    config = config_match.group(1)

    # Count T() and S() occurrences
    n_items = len(re.findall(r'(?:T|S)\(', config))

    if n_fwd != n_items:
        return f"forward expects {n_fwd} args, _shapes_config has {n_items} items"
    return None


def check_shapes_txt(repro_dir: Path, content: str) -> list[str]:
    """Check shapes.txt lines parse and have correct input count."""
    shapes_txt = repro_dir / "shapes.txt"
    if not shapes_txt.exists():
        return []

    fwd_match = re.search(r'def forward\(self,([^)]*)\)', content)
    if not fwd_match:
        return []
    n_fwd = len([a.strip() for a in fwd_match.group(1).split(',') if a.strip()])

    errors = []
    for i, line in enumerate(shapes_txt.read_text().splitlines()):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        colon = line.find(':')
        if colon < 0:
            errors.append(f"shapes.txt line {i+1}: no colon")
            continue
        expr = line[colon+1:].strip()
        n_items = len(re.findall(r'(?:T|S)\(', expr))
        if n_items != n_fwd:
            label = line[:colon].strip()
            errors.append(f"shapes.txt [{label}]: {n_items} items vs forward expects {n_fwd}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Format checks only")
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros/canonical"))
    args = parser.parse_args()

    dirs = sorted(d for d in args.canonical_dir.iterdir() if (d / "repro.py").exists())
    print(f"Checking {len(dirs)} repros...")

    errors = []
    for d in dirs:
        content = (d / "repro.py").read_text()
        name = d.name

        err = check_has_shapes_config(content)
        if err:
            errors.append((name, err))
            continue

        err = check_no_raw_inputs(content)
        if err:
            errors.append((name, err))

        err = check_input_count(content)
        if err:
            errors.append((name, err))

        if not args.quick:
            for err in check_shapes_txt(d, content):
                errors.append((name, err))

    if errors:
        print(f"\nFAILED: {len(errors)} issues")
        for name, err in errors[:20]:
            print(f"  {name}: {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more")
        sys.exit(1)
    else:
        print(f"\nPASSED: all {len(dirs)} repros follow canonical format")


if __name__ == "__main__":
    main()
