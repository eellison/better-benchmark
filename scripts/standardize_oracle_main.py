#!/usr/bin/env python3
"""Standardize oracle main() functions to use bench_oracle from oracle_harness.

This script:
1. Updates oracle_harness imports to include bench_oracle, bench_oracle_all_shapes,
   get_hardware_info, and has_stochastic_ops.
2. Replaces custom main() functions with the standard pattern that uses bench_oracle.
3. Preserves all kernel code, oracle_forward, and helper functions above main().
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Standard main() template
STANDARD_MAIN = '''\
def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result[\'repro_id\']} (ratio={result[\'ratio\']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result[\'ratio\']:.3f}x)")


if __name__ == "__main__":
    main()
'''

# Required imports from oracle_harness
REQUIRED_IMPORTS = {
    "bench_oracle",
    "bench_oracle_all_shapes",
    "check_oracle",
    "get_hardware_info",
    "get_inputs",
    "get_repro_instance",
    "has_stochastic_ops",
}


def fix_imports(content: str) -> str:
    """Ensure the oracle_harness import block includes all required imports."""
    # Find the existing import block
    import_pattern = re.compile(
        r'from oracle_harness import \((.*?)\)',
        re.DOTALL,
    )
    match = import_pattern.search(content)
    if not match:
        # Try single-line import
        import_pattern_single = re.compile(
            r'from oracle_harness import (.+?)(?:\s*#[^\n]*)?\n'
        )
        match = import_pattern_single.search(content)

    if not match:
        print("  WARNING: Could not find oracle_harness import block!")
        return content

    # Parse existing imports
    existing_text = match.group(1)
    # Extract import names (handle aliases like "get_inputs as _harness_get_inputs")
    existing_imports = {}
    for item in re.findall(r'(\w+)(?:\s+as\s+(\w+))?', existing_text):
        name = item[0]
        alias = item[1] if item[1] else None
        if name in ('noqa', 'E402'):  # skip comments
            continue
        existing_imports[name] = alias

    # Determine what to add
    # Map standard aliases
    STANDARD_ALIASES = {
        "get_inputs": "_harness_get_inputs",
        "get_repro_instance": "_harness_get_repro_instance",
    }

    needs_update = False
    for imp in REQUIRED_IMPORTS:
        if imp not in existing_imports:
            needs_update = True
            existing_imports[imp] = STANDARD_ALIASES.get(imp)

    if not needs_update:
        return content

    # Build new import block
    import_lines = []
    for name in sorted(existing_imports.keys()):
        alias = existing_imports[name]
        if alias:
            import_lines.append(f"    {name} as {alias},")
        else:
            import_lines.append(f"    {name},")

    new_import = "from oracle_harness import (\n" + "\n".join(import_lines) + "\n)"

    # Replace the old import
    content = content[:match.start()] + new_import + content[match.end():]
    return content


def ensure_repro_path(content: str) -> str:
    """Ensure REPRO_PATH is defined (needed for has_stochastic_ops)."""
    if "REPRO_PATH" not in content:
        # Add after REPRO_DIR definition
        repro_dir_match = re.search(r'(REPRO_DIR\s*=.+?\n)', content)
        if repro_dir_match:
            insert_pos = repro_dir_match.end()
            content = content[:insert_pos] + 'REPRO_PATH = REPRO_DIR / "repro.py"\n' + content[insert_pos:]
    return content


def ensure_harness_aliases(content: str) -> str:
    """Ensure _harness_get_inputs and _harness_get_repro_instance are available.

    Some files use different aliases or direct names.
    """
    # Check if we have the standard aliases in the import
    has_harness_get_inputs = "_harness_get_inputs" in content
    has_harness_get_repro = "_harness_get_repro_instance" in content

    if not has_harness_get_inputs:
        # Check for other aliases
        if "get_inputs as _get_inputs" in content:
            content = content.replace("get_inputs as _get_inputs", "get_inputs as _harness_get_inputs")
        elif re.search(r'from oracle_harness.*\bget_inputs\b(?!\s+as)', content):
            # Imported without alias - add an alias line after imports
            pass  # The import fix should have handled this

    if not has_harness_get_repro:
        if "get_repro_instance as " in content:
            # Replace whatever alias with standard one
            content = re.sub(
                r'get_repro_instance as \w+',
                'get_repro_instance as _harness_get_repro_instance',
                content,
            )

    return content


def replace_main(content: str) -> str:
    """Replace main() and everything after it (including __name__ block) with standard."""
    # Find the main() function definition
    main_match = re.search(r'^def main\(\).*?:\n', content, re.MULTILINE)
    if not main_match:
        print("  WARNING: Could not find main() function!")
        return content

    # Replace everything from main() onward
    content = content[:main_match.start()] + STANDARD_MAIN
    return content


def process_file(filepath: str) -> bool:
    """Process a single oracle file. Returns True if modified."""
    path = Path(filepath)
    original = path.read_text()
    content = original

    # Step 1: Fix imports
    content = fix_imports(content)

    # Step 2: Ensure REPRO_PATH exists
    content = ensure_repro_path(content)

    # Step 3: Ensure standard aliases
    content = ensure_harness_aliases(content)

    # Step 4: Replace main()
    content = replace_main(content)

    if content != original:
        path.write_text(content)
        return True
    return False


def main():
    files_path = Path("/tmp/non_standard_oracles.txt")
    files = [line.strip() for line in files_path.read_text().splitlines() if line.strip()]

    modified = 0
    for filepath in files:
        print(f"Processing: {filepath}")
        try:
            if process_file(filepath):
                modified += 1
                print(f"  MODIFIED")
            else:
                print(f"  NO CHANGE")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nTotal modified: {modified}/{len(files)}")


if __name__ == "__main__":
    main()
