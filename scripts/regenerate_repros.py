"""
Regenerate repro.py files with the current template format.

Preserves: class Repro (forward method) + _shapes_config string.
Regenerates: imports, _default_make_inputs, make_inputs, __main__ block.

Usage:
    python scripts/regenerate_repros.py                    # all repros
    python scripts/regenerate_repros.py --repro var_mean_2096009f00b8  # one
    python scripts/regenerate_repros.py --dry-run          # show what would change
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEMPLATE = '''\
"""
Standalone repro captured via capture_hook.
{docstring_extra}"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "{shapes_config}"

{repro_class}


def _default_make_inputs():
    def T(shape, dtype, stride=None):
        return {{"kind": "tensor", "shape": shape, "dtype": dtype, "stride": list(stride) if stride else None, "device": "cuda"}}
    def S(dims):
        return {{"kind": "shape", "dims": dims}}
    _ns = {{"T": T, "S": S, "f32": "f32", "f16": "f16", "bf16": "bf16", "f64": "f64",
            "i64": "i64", "i32": "i32", "i16": "i16", "i8": "i8", "b8": "b8", "u8": "u8"}}
    inputs = eval(_shapes_config, {{"__builtins__": {{}}}}, _ns)
    if isinstance(inputs, dict):
        inputs = [inputs]
    elif isinstance(inputs, tuple):
        inputs = list(inputs)
    return make_inputs_from_config({{"inputs": inputs}})


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
'''


def extract_repro_class(content: str) -> str | None:
    """Extract the class Repro(...): ... block."""
    match = re.search(
        r'^(class Repro\(.*?\):.*?)(?=\n(?:def |_shapes_config|if __name__|$))',
        content, re.MULTILINE | re.DOTALL
    )
    return match.group(1).rstrip() if match else None


def extract_shapes_config(content: str) -> str | None:
    """Extract the _shapes_config string value."""
    match = re.search(r'^_shapes_config\s*=\s*"(.+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def extract_docstring_extra(content: str) -> str:
    """Extract label/hash info from the docstring."""
    lines = []
    match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    if match:
        for line in match.group(1).strip().splitlines():
            line = line.strip()
            if line.startswith("Label:") or line.startswith("Pattern hash:") or line.startswith("Shape hash:"):
                lines.append(line)
    return "\n".join(lines) + "\n" if lines else ""


def regenerate_one(repro_path: Path, dry_run: bool = False) -> bool:
    """Regenerate one repro.py. Returns True if changed."""
    content = repro_path.read_text()

    repro_class = extract_repro_class(content)
    shapes_config = extract_shapes_config(content)

    if not repro_class:
        print(f"  SKIP {repro_path.parent.name}: no class Repro found")
        return False
    if not shapes_config:
        print(f"  SKIP {repro_path.parent.name}: no _shapes_config found")
        return False

    docstring_extra = extract_docstring_extra(content)

    new_content = TEMPLATE.format(
        docstring_extra=docstring_extra,
        shapes_config=shapes_config,
        repro_class=repro_class,
    )

    if new_content == content:
        return False

    if dry_run:
        print(f"  WOULD UPDATE {repro_path.parent.name}")
    else:
        repro_path.write_text(new_content)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repro", type=str, default=None)
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros/canonical"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repro:
        dirs = [args.canonical_dir / args.repro]
    else:
        dirs = sorted(args.canonical_dir.iterdir())

    updated = 0
    skipped = 0
    unchanged = 0

    for d in dirs:
        repro = d / "repro.py"
        if not repro.exists():
            continue
        content = repro.read_text()
        if "_shapes_config" not in content:
            skipped += 1
            continue
        if regenerate_one(repro, dry_run=args.dry_run):
            updated += 1
        else:
            unchanged += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Updated: {updated}, Unchanged: {unchanged}, Skipped (no _shapes_config): {skipped}")


if __name__ == "__main__":
    main()
