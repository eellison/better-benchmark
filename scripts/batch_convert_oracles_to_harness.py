#!/usr/bin/env python3
"""
Batch-convert non-standard oracle kernels to use the oracle_harness infrastructure.

Problem: Oracles with their own inline benchmarking (not using CUDAGraph capture)
produce fake gaps due to dynamo dispatch overhead. For fast kernels (~5-20us), this
overhead is ~10us and creates apparent 2x gaps that don't exist.

Solution: All oracles must use `from oracle_harness import bench_oracle` which does
CUDAGraph + GPU lock + interleaved timing.

This script performs the conversion in three tiers:

  Tier 1 (--tier1): oracle_forward(inputs) already exists, just need to:
    - Add oracle_harness imports
    - Replace inline bench/check logic with harness calls in main()

  Tier 2 (--tier2): oracle_full/oracle_fused/oracle_triton(*inputs) exists, need to:
    - Add a thin oracle_forward(inputs) wrapper
    - Then apply Tier 1 logic

  Tier 3 (--tier3): named-parameter oracle functions that need a wrapper like:
    def oracle_forward(inputs):
        return oracle_XXXX(*inputs)

Usage:
    # Dry-run: show what would be converted
    python scripts/batch_convert_oracles_to_harness.py --dry-run

    # Convert tier 1 only (safest)
    python scripts/batch_convert_oracles_to_harness.py --tier1

    # Convert all tiers
    python scripts/batch_convert_oracles_to_harness.py --tier1 --tier2 --tier3

    # Convert a specific file
    python scripts/batch_convert_oracles_to_harness.py --file repros/canonical/xxx/oracle_yyy.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = REPO_ROOT / "repros" / "canonical"


def find_non_standard_oracles() -> list[Path]:
    """Find all oracle files not using oracle_harness."""
    result = subprocess.run(
        ["find", str(CANONICAL_DIR), "-name", "oracle_*.py",
         "-exec", "grep", "-L", "from oracle_harness", "{}", ";"],
        capture_output=True, text=True
    )
    return sorted(Path(p) for p in result.stdout.strip().split("\n") if p)


def classify_oracle(path: Path) -> str:
    """Classify an oracle into tier1, tier2, tier3, or skip."""
    content = path.read_text()

    # Skip thin wrappers that import from another oracle
    if re.search(r"from oracle_\w+ import", content):
        return "skip"

    # Tier 1: has oracle_forward(inputs)
    if re.search(r"def oracle_forward\s*\(\s*inputs", content):
        return "tier1"

    # Tier 2: has oracle_full/fused/triton(*inputs) or oracle_XYZ(*inputs)
    if re.search(r"def oracle_(full|fused|triton|torch)\s*\(\s*\*inputs", content):
        return "tier2"

    # Tier 3: has a named oracle function called with *inputs somewhere
    oracle_funcs = re.findall(r"def (oracle_\w+|triton_\w+_oracle)\s*\(", content)
    if oracle_funcs:
        for func_name in oracle_funcs:
            if func_name in ("oracle_forward", "oracle_forward_with_options"):
                continue
            # Check if called with *inputs
            if re.search(rf"{func_name}\s*\(\s*\*inputs", content):
                return "tier3"
        # Has an oracle function but doesn't call with *inputs - still tier3
        # just needs more manual review
        return "tier3"

    return "unknown"


def get_oracle_function_name(content: str) -> str | None:
    """Find the primary oracle function name in the content."""
    # Priority: oracle_forward > oracle_full > oracle_fused > oracle_triton > others
    for pattern in [
        r"def (oracle_forward)\s*\(",
        r"def (oracle_full)\s*\(",
        r"def (oracle_fused)\s*\(",
        r"def (oracle_triton)\s*\(",
        r"def (oracle_torch)\s*\(",
        r"def (triton_fused_oracle)\s*\(",
    ]:
        match = re.search(pattern, content)
        if match:
            return match.group(1)

    # Find any oracle_ function
    match = re.search(r"def (oracle_\w+)\s*\(", content)
    if match and match.group(1) not in ("oracle_forward_with_options",):
        return match.group(1)

    return None


def analyze_file(path: Path) -> dict:
    """Analyze an oracle file and return info about it."""
    content = path.read_text()
    tier = classify_oracle(path)
    oracle_func = get_oracle_function_name(content)

    has_cuda_graph = bool(re.search(r"CUDAGraph|cuda\.graph", content))
    has_do_bench = "do_bench" in content
    has_perf_counter = "perf_counter" in content
    has_gpu_lock = bool(re.search(r"gpu_lock|exclusive_lock", content))

    if has_cuda_graph:
        bench_method = "CUDAGraph"
    elif has_do_bench:
        bench_method = "do_bench"
    elif has_perf_counter:
        bench_method = "perf_counter"
    else:
        bench_method = "none/delegated"

    return {
        "path": path,
        "tier": tier,
        "oracle_func": oracle_func,
        "bench_method": bench_method,
        "has_cuda_graph": has_cuda_graph,
        "has_gpu_lock": has_gpu_lock,
    }


def print_summary(analyses: list[dict]) -> None:
    """Print categorized summary."""
    from collections import Counter

    tier_counts = Counter(a["tier"] for a in analyses)
    bench_counts = Counter(a["bench_method"] for a in analyses)

    print("=" * 70)
    print("ORACLE HARNESS MIGRATION AUDIT")
    print("=" * 70)
    print(f"\nTotal non-standard oracles: {len(analyses)}")
    print("\n--- Tier Classification ---")
    print(f"  Tier 1 (oracle_forward(inputs), just swap bench): {tier_counts.get('tier1', 0)}")
    print(f"  Tier 2 (oracle_xyz(*inputs), add wrapper + swap): {tier_counts.get('tier2', 0)}")
    print(f"  Tier 3 (named params, need wrapper): {tier_counts.get('tier3', 0)}")
    print(f"  Skip (thin wrappers/delegates): {tier_counts.get('skip', 0)}")
    print(f"  Unknown: {tier_counts.get('unknown', 0)}")
    print("\n--- Current Benchmarking Method ---")
    print(f"  CUDAGraph (partial correctness): {bench_counts.get('CUDAGraph', 0)}")
    print(f"  do_bench only (NO CUDAGraph): {bench_counts.get('do_bench', 0)}")
    print(f"  perf_counter only (WORST - raw wall time): {bench_counts.get('perf_counter', 0)}")
    print(f"  none/delegated: {bench_counts.get('none/delegated', 0)}")
    print("\n--- Critical Deficiencies ---")
    no_lock = sum(1 for a in analyses if not a["has_gpu_lock"])
    no_graph = sum(1 for a in analyses if not a["has_cuda_graph"])
    print(f"  Missing GPU lock: {no_lock}/{len(analyses)} (100%!)")
    print(f"  Missing CUDAGraph: {no_graph}/{len(analyses)}")
    print("\n--- Impact Assessment ---")
    fast_kernel_risk = bench_counts.get("do_bench", 0) + bench_counts.get("perf_counter", 0)
    print(f"  At risk of fake gaps (no CUDAGraph): {fast_kernel_risk} oracles")
    print("  These produce ~10us dynamo overhead on every call")
    print("  For kernels <20us, this creates 2x+ apparent gaps that DON'T EXIST")


# ============================================================================
# Tier 2 conversion implementation
# ============================================================================


def _find_cut_point(content: str, oracle_func_name: str) -> int:
    """Find the char position right after the main oracle function ends.

    We locate oracle_func_name (e.g. oracle_full), find where that function body
    ends, and return the offset. Everything after this is bench/check boilerplate.
    """
    lines = content.split('\n')

    # Find the oracle function definition line
    oracle_start_line = None
    for i, line in enumerate(lines):
        if re.match(rf'^def {re.escape(oracle_func_name)}\s*\(', line):
            oracle_start_line = i
            break

    if oracle_start_line is None:
        # Fallback: find the last oracle_ function
        for i, line in enumerate(lines):
            if re.match(r'^def oracle_\w+\s*\(', line):
                oracle_start_line = i

    if oracle_start_line is None:
        return len(content)

    # First, skip past the function signature (may be multi-line)
    # We need to find the colon that ends the 'def' statement
    sig_end_line = oracle_start_line
    paren_depth = 0
    found_sig_end = False
    for i in range(oracle_start_line, len(lines)):
        for ch in lines[i]:
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
        # Signature ends when parens are balanced and we see ':'
        if paren_depth == 0 and ':' in lines[i]:
            sig_end_line = i
            found_sig_end = True
            break

    if not found_sig_end:
        sig_end_line = oracle_start_line

    # Now find end of function body: next top-level def/class/if-name after the signature
    func_end_line = sig_end_line + 1
    while func_end_line < len(lines):
        line = lines[func_end_line]
        # Skip blank lines and indented lines (function body)
        if not line or line[0].isspace():
            func_end_line += 1
            continue
        # Skip decorators
        if line.startswith('@'):
            func_end_line += 1
            continue
        # Skip comment lines at module level
        if line.startswith('#'):
            func_end_line += 1
            continue
        # This is a new top-level statement - function ended before here
        break

    # func_end_line points to the first line after oracle_full's body
    return sum(len(lines[i]) + 1 for i in range(func_end_line))


def _add_harness_import(content: str) -> str:
    """Add oracle_harness import block after the last import/try-except."""
    if 'from oracle_harness import' in content:
        return content

    harness_import = (
        "\nfrom oracle_harness import (\n"
        "    bench_oracle,\n"
        "    bench_oracle_all_shapes,\n"
        "    check_oracle,\n"
        "    get_hardware_info,\n"
        "    get_inputs as _harness_get_inputs,\n"
        "    get_repro_instance as _harness_get_repro_instance,\n"
        "    has_stochastic_ops,\n"
        ")\n"
    )

    # Find insertion point: after the triton try/except block or last import
    triton_block = re.search(
        r'^try:\s*\n\s+import triton.*?^except[^\n]*:.*?\n(?:\s+[^\n]*\n)*',
        content, re.MULTILINE | re.DOTALL
    )

    if triton_block:
        insert_pos = triton_block.end()
    else:
        # After last top-level import
        insert_pos = 0
        for m in re.finditer(r'^(?:import |from )[^\n]*\n', content, re.MULTILINE):
            insert_pos = m.end()

    content = content[:insert_pos] + harness_import + content[insert_pos:]
    return content


def _ensure_path_import(content: str) -> str:
    """Ensure 'from pathlib import Path' is present."""
    if 'from pathlib import Path' in content or 'import pathlib' in content:
        return content
    # Insert after 'import torch' or 'import sys'
    for anchor in ['\nimport torch\n', '\nimport sys\n']:
        if anchor in content:
            content = content.replace(anchor, anchor + 'from pathlib import Path\n', 1)
            return content
    # Fallback: after first import
    m = re.search(r'^import \w+\n', content, re.MULTILINE)
    if m:
        content = content[:m.end()] + 'from pathlib import Path\n' + content[m.end():]
    return content


def _ensure_constants(content: str, path: Path) -> str:
    """Ensure REPRO_ID, REPRO_DIR, REPRO_PATH constants exist."""
    repro_id = path.parent.name
    m = re.search(r'^REPRO_ID\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if m:
        repro_id = m.group(1)

    has_repro_id = bool(re.search(r'^REPRO_ID\s*=', content, re.MULTILINE))
    has_repro_dir = bool(re.search(r'^REPRO_DIR\s*=', content, re.MULTILINE))
    has_repro_path = bool(re.search(r'^REPRO_PATH\s*=', content, re.MULTILINE))

    if has_repro_id and has_repro_dir and has_repro_path:
        return content

    additions = []
    if not has_repro_dir:
        additions.append('REPRO_DIR = Path(__file__).resolve().parent')
    if not has_repro_path:
        additions.append('REPRO_PATH = REPRO_DIR / "repro.py"')

    if not additions:
        return content

    addition_text = '\n'.join(additions) + '\n'

    if has_repro_id:
        m = re.search(r'^REPRO_ID\s*=[^\n]*\n', content, re.MULTILINE)
        if m:
            content = content[:m.end()] + addition_text + content[m.end():]
    else:
        # Insert after oracle_harness import block
        m = re.search(r'^from oracle_harness import.*?\)\n', content, re.MULTILINE | re.DOTALL)
        if m:
            insert_text = f'\nREPRO_ID = "{repro_id}"\n' + addition_text
            content = content[:m.end()] + insert_text + content[m.end():]

    return content


def _remove_unused_imports(content: str) -> str:
    """Remove imports no longer needed after conversion."""
    if 'import time' in content:
        rest = content.replace('import time', '', 1)
        if 'time.' not in rest:
            content = re.sub(r'^import time\n', '', content, count=1, flags=re.MULTILINE)
    if 'from typing import Callable' in content:
        rest = content.replace('from typing import Callable', '', 1)
        if 'Callable' not in rest:
            content = re.sub(r'^from typing import Callable\n', '', content, count=1, flags=re.MULTILINE)
    if 'import math' in content:
        rest = content.replace('import math', '', 1)
        if 'math.' not in rest:
            content = re.sub(r'^import math\n', '', content, count=1, flags=re.MULTILINE)
    return content


def _build_tail(oracle_func: str) -> str:
    """Build the file tail: oracle_forward wrapper + standard main."""
    return f'''

def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return {oracle_func}(*inputs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {{REPRO_ID}}",
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
        print(f"NOTE: {{REPRO_ID}} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {{REPRO_ID}}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        print(f"Correctness: {{'PASS' if ok else 'FAIL'}}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {{REPRO_ID}}...")
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
                          f"for {{result['repro_id']}} (ratio={{result['ratio']:.3f}}x)")
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
                      f"(ratio={{result['ratio']:.3f}}x)")


if __name__ == "__main__":
    main()
'''


def convert_tier2(path: Path) -> bool:
    """Convert a tier 2 oracle: has oracle_xyz(*inputs), add wrapper + replace bench.

    Strategy:
    1. Find where boilerplate starts (make_inputs, _as_tuple, synchronize, etc.)
    2. Cut everything from that point to EOF
    3. Add oracle_harness imports and constants to the kept portion
    4. Append oracle_forward wrapper + standard main template
    """
    content = path.read_text()
    oracle_func = get_oracle_function_name(content)
    if not oracle_func:
        print(f"  [TIER2] SKIP (no oracle func found): {path.relative_to(REPO_ROOT)}")
        return False

    rel_path = path.relative_to(REPO_ROOT)
    print(f"  [TIER2] Converting: {rel_path}")
    print(f"          Adding: def oracle_forward(inputs): return {oracle_func}(*inputs)")

    # Step 1: Find cut point and keep only oracle implementation
    cut_pos = _find_cut_point(content, oracle_func)
    kept = content[:cut_pos].rstrip() + '\n'

    # Step 2: Add oracle_harness imports
    kept = _add_harness_import(kept)

    # Step 3: Ensure pathlib.Path is imported
    kept = _ensure_path_import(kept)

    # Step 4: Ensure REPRO_DIR/REPRO_PATH constants
    kept = _ensure_constants(kept, path)

    # Step 5: Remove unused imports from the kept portion
    kept = _remove_unused_imports(kept)

    # Step 6: Append the standard tail (oracle_forward + main)
    tail = _build_tail(oracle_func)
    result = kept.rstrip() + '\n' + tail

    # Clean up excessive blank lines
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    result = result.rstrip() + '\n'

    path.write_text(result)
    return True


def _build_main_only() -> str:
    """Build just the standard main (no oracle_forward wrapper - tier1 already has it)."""
    return '''


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


def _find_boilerplate_start(content: str) -> int:
    """Find the character position where bench/check boilerplate begins.

    For tier 1 oracles, we want to keep ALL oracle code (including helpers called
    by oracle_forward that may be defined after it), and only cut the boilerplate:
    _max_diff, run_check, run_bench, _bench_cuda_graph, _compile_with_config, main, etc.

    Strategy: find the FIRST boilerplate function that appears AFTER the last
    oracle_* function definition. This ensures we keep oracle_full, oracle_triton_prepared,
    etc. even if they're defined after oracle_forward.
    """
    lines = content.split('\n')

    # Find the line number of the LAST oracle_* function (or oracle_forward_with_options)
    last_oracle_line = 0
    for i, line in enumerate(lines):
        if re.match(r'^def oracle_\w+\s*\(', line):
            last_oracle_line = i

    # Functions that are pure bench/check boilerplate (never called by oracle kernels)
    boilerplate_funcs = {
        '_max_diff', 'run_check', 'run_bench', 'benchmark', 'main',
        '_bench_cuda_graph', '_compile_with_config', '_check_one', '_do_bench',
        '_compile_repro', '_bench_one', '_shape_names', '_named_inputs',
        '_repro_forward', '_default_inputs', '_bench_oracle_variant',
        '_chunked_reference_check', 'synchronize',
    }

    # Find the first boilerplate function AFTER last_oracle_line
    first_boilerplate_line = None
    for i in range(last_oracle_line + 1, len(lines)):
        line = lines[i]
        func_match = re.match(r'^def (\w+)\s*\(', line)
        if func_match and func_match.group(1) in boilerplate_funcs:
            first_boilerplate_line = i
            break
        if re.match(r'^if __name__\s*==', line):
            first_boilerplate_line = i
            break

    if first_boilerplate_line is None:
        return len(content)

    # Back up past blank lines before the boilerplate
    cut_line = first_boilerplate_line
    while cut_line > 0 and lines[cut_line - 1].strip() == '':
        cut_line -= 1

    return sum(len(lines[i]) + 1 for i in range(cut_line))


def convert_tier1(path: Path) -> bool:
    """Convert a tier 1 oracle: has oracle_forward(inputs), just replace bench infra.

    Same strategy as tier 2 but without adding an oracle_forward wrapper.
    """
    content = path.read_text()

    if 'from oracle_harness import' in content:
        print(f"  [TIER1] SKIP (already converted): {path.relative_to(REPO_ROOT)}")
        return True

    rel_path = path.relative_to(REPO_ROOT)
    print(f"  [TIER1] Converting: {rel_path}")

    # Step 1: Find cut point. For tier 1, we cut at the first boilerplate function
    # (not at oracle_forward itself, since oracle_forward may call helpers defined
    # AFTER it in the file, like oracle_full, oracle_triton_prepared, etc.)
    cut_pos = _find_boilerplate_start(content)
    kept = content[:cut_pos].rstrip() + '\n'

    # Step 2: Add oracle_harness imports
    kept = _add_harness_import(kept)

    # Step 3: Ensure pathlib.Path is imported
    kept = _ensure_path_import(kept)

    # Step 4: Ensure REPRO_DIR/REPRO_PATH constants
    kept = _ensure_constants(kept, path)

    # Step 5: Ensure argparse and sys imports for main
    if 'import argparse' not in kept:
        # Add after 'from __future__' or at top of imports
        if 'from __future__' in kept:
            kept = re.sub(
                r'(from __future__ import[^\n]*\n)',
                r'\1\nimport argparse\n',
                kept, count=1
            )
        else:
            kept = 'import argparse\n' + kept
    if 'import sys' not in kept:
        kept = re.sub(
            r'(import argparse\n)',
            r'\1import sys\n',
            kept, count=1
        )

    # Step 6: Remove unused imports from the kept portion
    kept = _remove_unused_imports(kept)

    # Step 7: Append the standard main (no oracle_forward wrapper needed)
    tail = _build_main_only()
    result_content = kept.rstrip() + '\n' + tail

    # Clean up excessive blank lines
    result_content = re.sub(r'\n{4,}', '\n\n\n', result_content)
    result_content = result_content.rstrip() + '\n'

    path.write_text(result_content)
    return True


def convert_tier3(path: Path) -> bool:
    """Convert a tier 3 oracle: named params, need to add oracle_forward wrapper."""
    content = path.read_text()
    oracle_func = get_oracle_function_name(content)
    if not oracle_func:
        print(f"  [TIER3] SKIP (no oracle func found): {path.relative_to(REPO_ROOT)}")
        return False
    print(f"  [TIER3] Would convert: {path.relative_to(REPO_ROOT)}")
    print(f"          Add: def oracle_forward(inputs): return {oracle_func}(*inputs)")
    return True


# ============================================================================
# Main entry point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be converted without making changes")
    parser.add_argument("--tier1", action="store_true",
                        help="Convert tier 1 oracles (oracle_forward exists)")
    parser.add_argument("--tier2", action="store_true",
                        help="Convert tier 2 oracles (oracle_xyz(*inputs) exists)")
    parser.add_argument("--tier3", action="store_true",
                        help="Convert tier 3 oracles (named params, need wrapper)")
    parser.add_argument("--file", type=Path, default=None,
                        help="Convert a specific file only")
    parser.add_argument("--list", action="store_true",
                        help="List all non-standard oracles with their tier")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary statistics only")
    args = parser.parse_args()

    if args.file:
        oracles = [args.file.resolve()]
    else:
        oracles = find_non_standard_oracles()

    analyses = [analyze_file(p) for p in oracles]

    if args.summary or args.dry_run:
        print_summary(analyses)

    if args.list:
        print(f"\n{'Tier':<8} {'Bench':<12} {'Oracle Func':<30} Path")
        print("-" * 100)
        for a in sorted(analyses, key=lambda x: (x["tier"], str(x["path"]))):
            rel = a["path"].relative_to(REPO_ROOT)
            print(f"{a['tier']:<8} {a['bench_method']:<12} {(a['oracle_func'] or 'N/A'):<30} {rel}")

    if args.dry_run:
        print("\n--- Conversion Plan ---")
        for tier_name in ("tier1", "tier2", "tier3"):
            tier_files = [a for a in analyses if a["tier"] == tier_name]
            if tier_files:
                print(f"\n  {tier_name.upper()} ({len(tier_files)} files):")
                for a in tier_files[:5]:
                    print(f"    {a['path'].relative_to(REPO_ROOT)}")
                if len(tier_files) > 5:
                    print(f"    ... and {len(tier_files) - 5} more")
        return

    if not (args.tier1 or args.tier2 or args.tier3):
        if not (args.list or args.summary or args.dry_run):
            parser.error("Specify --tier1, --tier2, --tier3, --dry-run, --list, or --summary")
        return

    # Actual conversion
    converted = 0
    failed = 0
    for a in analyses:
        if a["tier"] == "tier1" and args.tier1:
            if convert_tier1(a["path"]):
                converted += 1
            else:
                failed += 1
        elif a["tier"] == "tier2" and args.tier2:
            if convert_tier2(a["path"]):
                converted += 1
            else:
                failed += 1
        elif a["tier"] == "tier3" and args.tier3:
            if convert_tier3(a["path"]):
                converted += 1
            else:
                failed += 1

    print(f"\nConversion complete: {converted} converted, {failed} failed")


if __name__ == "__main__":
    main()
