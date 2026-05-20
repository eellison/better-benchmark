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
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


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
    # Find start of class Repro
    class_match = re.search(r'^class Repro\(', content, re.MULTILINE)
    if not class_match:
        return None

    # Find the end: next top-level def or _shapes_config or if __name__
    # (i.e., a line at column 0 that starts with def/if/__name__/_shapes_config)
    rest = content[class_match.start():]
    end_match = re.search(
        r'\n(?=def |_shapes_config|if __name__)',
        rest[1:]  # skip first char to avoid matching at the very start
    )
    if end_match:
        class_block = rest[:end_match.start() + 1]
    else:
        class_block = rest
    return class_block.rstrip()


def extract_shapes_config(content: str) -> str | None:
    """Extract the _shapes_config string value."""
    match = re.search(r'^_shapes_config\s*=\s*"(.+)"', content, re.MULTILINE)
    return match.group(1) if match else None


def _extract_randint_bounds(source: str) -> dict[int, int]:
    """Parse source of _default_make_inputs() to extract randint upper bounds.

    Returns a dict mapping positional index (0-based) of each item in the return
    list to its upper bound value. Only entries with torch.randint are included.
    """
    # Find _default_make_inputs body - extract everything between "return [" and
    # the closing "]" of that function
    match = re.search(
        r'def _default_make_inputs\(\):\s*\n\s*return \[',
        source
    )
    if not match:
        return {}

    # Get the text after "return ["
    start = match.end()
    # Find end by counting brackets
    depth = 1
    pos = start
    while pos < len(source) and depth > 0:
        if source[pos] == '[':
            depth += 1
        elif source[pos] == ']':
            depth -= 1
        pos += 1
    body = source[start:pos - 1]

    # Parse line by line. Each non-empty, non-comment line is one item in the list.
    bounds = {}
    idx = 0
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Remove trailing comma
        line_clean = line.rstrip(',').strip()
        if not line_clean:
            continue

        # Check if this is a torch.randint line
        randint_match = re.match(r'torch\.randint\(\s*(\d+)\s*,\s*(\d+)\s*,', line_clean)
        if randint_match:
            upper_bound = int(randint_match.group(2))
            bounds[idx] = upper_bound

        idx += 1

    return bounds


def _derive_from_source(source: str) -> str | None:
    """Parse _default_make_inputs source to derive shapes config without executing.

    This handles cases where exec fails (e.g. torch.randn with dtype=torch.bool).
    """
    _DTYPE_SHORT = {
        "torch.float32": "f32", "torch.float16": "f16",
        "torch.bfloat16": "bf16", "torch.float64": "f64",
        "torch.int64": "i64", "torch.int32": "i32",
        "torch.int16": "i16", "torch.int8": "i8",
        "torch.bool": "b8", "torch.uint8": "u8",
    }

    # Find _default_make_inputs body
    match = re.search(r'def _default_make_inputs\(\):\s*\n\s*return \[', source)
    if not match:
        return None

    start = match.end()
    depth = 1
    pos = start
    while pos < len(source) and depth > 0:
        if source[pos] == '[':
            depth += 1
        elif source[pos] == ']':
            depth -= 1
        pos += 1
    body = source[start:pos - 1]

    parts = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        line_clean = line.rstrip(',').strip()
        if not line_clean:
            continue

        # Shape param: list literal like [8, 512, 1536]
        shape_match = re.match(r'^\[([^\]]*)\]', line_clean)
        if shape_match and 'torch.' not in line_clean:
            # It's a shape param
            parts.append(f"S([{shape_match.group(1)}])")
            continue

        # torch.randn(...).as_strided(shape, stride)
        strided_match = re.match(
            r'torch\.(?:randn|randint)\(.*?dtype\s*=\s*torch\.(\w+).*?\)\.as_strided\(\s*\[([^\]]*)\]\s*,\s*\[([^\]]*)\]',
            line_clean
        )
        if strided_match:
            dtype_str = strided_match.group(1)
            shape_str = strided_match.group(2)
            stride_str = strided_match.group(3)
            dt = _DTYPE_SHORT.get(f"torch.{dtype_str}", dtype_str)
            shape = [int(x.strip()) for x in shape_str.split(',')]
            stride = tuple(int(x.strip()) for x in stride_str.split(','))
            parts.append(f"T({shape}, {dt}, stride={stride})")
            continue

        # torch.randint(low, high, shape, dtype=...)
        randint_match = re.match(
            r'torch\.randint\(\s*(\d+)\s*,\s*(\d+)\s*,\s*\[([^\]]*)\]\s*,\s*dtype\s*=\s*torch\.(\w+)',
            line_clean
        )
        if randint_match:
            upper = int(randint_match.group(2))
            shape_str = randint_match.group(3)
            dtype_str = randint_match.group(4)
            dt = _DTYPE_SHORT.get(f"torch.{dtype_str}", dtype_str)
            shape = [int(x.strip()) for x in shape_str.split(',')]
            if dt == "b8":
                parts.append(f"T({shape}, {dt})")
            else:
                parts.append(f"T({shape}, {dt}, max={upper})")
            continue

        # torch.randn(shape, dtype=...)  or  torch.randn([shape], dtype=...)
        randn_match = re.match(
            r'torch\.randn\(\s*\[([^\]]*)\]\s*,\s*dtype\s*=\s*torch\.(\w+)',
            line_clean
        )
        if randn_match:
            shape_str = randn_match.group(1)
            dtype_str = randn_match.group(2)
            dt = _DTYPE_SHORT.get(f"torch.{dtype_str}", dtype_str)
            shape = [int(x.strip()) for x in shape_str.split(',')]
            parts.append(f"T({shape}, {dt})")
            continue

        # torch.randn(N, dtype=...) - scalar-ish (single int size)
        randn_scalar_match = re.match(
            r'torch\.randn\(\s*(\d+)\s*,\s*dtype\s*=\s*torch\.(\w+)',
            line_clean
        )
        if randn_scalar_match:
            size = int(randn_scalar_match.group(1))
            dtype_str = randn_scalar_match.group(2)
            dt = _DTYPE_SHORT.get(f"torch.{dtype_str}", dtype_str)
            parts.append(f"T([{size}], {dt})")
            continue

        # If we can't parse this line, bail
        return None

    if not parts:
        return None
    return f"({', '.join(parts)})"


def derive_shapes_config(repro_path: Path) -> str | None:
    """Derive _shapes_config by exec'ing _default_make_inputs() and inspecting results.

    Also parses source code to extract randint upper bounds for max= annotations.
    Falls back to source-only parsing if exec fails.
    """
    import importlib.util
    import math
    import torch

    source = repro_path.read_text()

    # Extract randint bounds from source before exec
    randint_bounds = _extract_randint_bounds(source)

    exec_failed = False
    try:
        spec = importlib.util.spec_from_file_location("repro_derive", str(repro_path))
        mod = importlib.util.module_from_spec(spec)
        mod.device = torch.device
        mod.inf = math.inf
        mod.nan = math.nan
        spec.loader.exec_module(mod)

        if hasattr(mod, '_default_make_inputs'):
            inputs = mod._default_make_inputs()
        elif hasattr(mod, 'make_inputs'):
            inputs = mod.make_inputs()
        else:
            exec_failed = True
    except Exception:
        exec_failed = True

    if exec_failed:
        # Fall back to source-only parsing
        return _derive_from_source(source)

    _DTYPE_SHORT = {
        "torch.float32": "f32", "torch.float16": "f16",
        "torch.bfloat16": "bf16", "torch.float64": "f64",
        "torch.int64": "i64", "torch.int32": "i32",
        "torch.int16": "i16", "torch.int8": "i8",
        "torch.bool": "b8", "torch.uint8": "u8",
    }

    parts = []
    for idx, item in enumerate(inputs):
        if isinstance(item, list):
            parts.append(f"S({item})")
        elif isinstance(item, int):
            parts.append(f"S([{item}])")
        elif isinstance(item, torch.Tensor):
            dt = _DTYPE_SHORT.get(str(item.dtype), str(item.dtype))
            shape = list(item.shape)
            stride = list(item.stride()) if not item.is_contiguous() else None
            max_val = randint_bounds.get(idx)
            extras = []
            if stride:
                extras.append(f"stride={tuple(stride)}")
            if max_val is not None and dt not in ("b8",):
                extras.append(f"max={max_val}")
            if extras:
                parts.append(f"T({shape}, {dt}, {', '.join(extras)})")
            else:
                parts.append(f"T({shape}, {dt})")
        else:
            return None

    return f"({', '.join(parts)})"


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


def regenerate_one(repro_path: Path, dry_run: bool = False, shapes_config_override: str | None = None) -> bool:
    """Regenerate one repro.py. Returns True if changed."""
    content = repro_path.read_text()

    repro_class = extract_repro_class(content)
    shapes_config = shapes_config_override or extract_shapes_config(content)

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
        config_override = None
        if "_shapes_config" not in content:
            # Derive _shapes_config from existing _default_make_inputs
            config_override = derive_shapes_config(repro)
            if config_override is None:
                skipped += 1
                if skipped <= 5:
                    print(f"  SKIP {d.name}: could not derive _shapes_config")
                continue
        if regenerate_one(repro, dry_run=args.dry_run, shapes_config_override=config_override):
            updated += 1
        else:
            unchanged += 1

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Updated: {updated}, Unchanged: {unchanged}, Skipped (no _shapes_config): {skipped}")


if __name__ == "__main__":
    main()
