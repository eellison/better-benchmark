#!/usr/bin/env python3
"""Convert remaining 11 non-standard oracles to use oracle_harness.

These are the last holdouts from the harness migration. Each has a named-parameter
oracle function that needs an oracle_forward(inputs) wrapper and the standard
main() that delegates to oracle_harness for CUDAGraph-captured timing.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def find_function_end(content: str, func_name: str) -> int:
    """Find the character position after the end of a function definition."""
    lines = content.split('\n')
    func_start = None
    for i, line in enumerate(lines):
        if re.match(rf'^def {re.escape(func_name)}\s*\(', line):
            func_start = i
            break
    if func_start is None:
        return -1

    # Find end of function: next unindented line that isn't blank/comment/decorator
    func_end = func_start + 1
    # Skip signature (may be multi-line)
    paren_depth = 0
    for i in range(func_start, len(lines)):
        for ch in lines[i]:
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
        if paren_depth <= 0 and ':' in lines[i]:
            func_end = i + 1
            break

    # Now skip function body
    while func_end < len(lines):
        line = lines[func_end]
        if not line or line[0].isspace() or line.startswith('#'):
            func_end += 1
            continue
        if line.startswith('@'):
            func_end += 1
            continue
        break

    return sum(len(lines[i]) + 1 for i in range(func_end))


def find_boilerplate_start(content: str, oracle_func: str) -> int:
    """Find where boilerplate begins (after the oracle function and its helpers)."""
    lines = content.split('\n')

    # Boilerplate functions that indicate we should cut
    boilerplate_funcs = {
        'make_inputs', '_load_repro_module', 'synchronize', 'benchmark',
        'max_abs_diff', 'allclose_check', 'allclose', 'get_git_commit',
        'load_baseline_row', 'append_csv', 'parse_args', 'main',
        'run_check', 'run_bench', '_bench_cuda', '_compile_with_config',
        'reference_outputs', 'check_correctness', 'reference_forward',
        '_max_diff', 'mean_abs_diff', 'eager_reference', 'verify_correctness',
        'benchmark_fn', 'benchmark_all', 'assert_close', 'append_result',
        'allclose_with_nan', '_named_inputs',
    }

    # Find the oracle function
    oracle_line = None
    for i, line in enumerate(lines):
        if re.match(rf'^def {re.escape(oracle_func)}\s*\(', line):
            oracle_line = i
            # Don't break - keep looking for later oracle functions

    if oracle_line is None:
        return len(content)

    # Find the first boilerplate function after the oracle
    # But we need to keep helper functions that the oracle calls
    # Strategy: find the first function after oracle_line that's in boilerplate_funcs
    for i in range(oracle_line + 1, len(lines)):
        line = lines[i]
        m = re.match(r'^def (\w+)\s*\(', line)
        if m and m.group(1) in boilerplate_funcs:
            # Back up past blank lines
            cut_line = i
            while cut_line > 0 and lines[cut_line - 1].strip() == '':
                cut_line -= 1
            return sum(len(lines[j]) + 1 for j in range(cut_line))

    # Also check for `if __name__` as cut point
    for i in range(oracle_line + 1, len(lines)):
        if re.match(r'^if __name__\s*==', lines[i]):
            cut_line = i
            while cut_line > 0 and lines[cut_line - 1].strip() == '':
                cut_line -= 1
            return sum(len(lines[j]) + 1 for j in range(cut_line))

    return len(content)


def add_harness_import(content: str) -> str:
    """Add oracle_harness import after the last import block."""
    if 'from oracle_harness import' in content:
        return content

    harness_block = (
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

    # Find insertion point: after try/except triton block or last import
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

    return content[:insert_pos] + harness_block + content[insert_pos:]


def ensure_constants(content: str, repro_id: str) -> str:
    """Ensure REPRO_ID, REPRO_DIR, REPRO_PATH constants are present."""
    has_id = bool(re.search(r'^REPRO_ID\s*=', content, re.MULTILINE))
    has_dir = bool(re.search(r'^REPRO_DIR\s*=', content, re.MULTILINE))
    has_path = bool(re.search(r'^REPRO_PATH\s*=', content, re.MULTILINE))

    if has_id and has_dir and has_path:
        return content

    additions = []
    if not has_id:
        additions.append(f'REPRO_ID = "{repro_id}"')
    if not has_dir:
        additions.append('REPRO_DIR = Path(__file__).resolve().parent')
    if not has_path:
        additions.append('REPRO_PATH = REPRO_DIR / "repro.py"')

    if not additions:
        return content

    # Insert after oracle_harness import block
    m = re.search(r'^from oracle_harness import.*?\)\n', content, re.MULTILINE | re.DOTALL)
    if m:
        insert_text = '\n' + '\n'.join(additions) + '\n'
        content = content[:m.end()] + insert_text + content[m.end():]
    return content


def ensure_imports(content: str) -> str:
    """Ensure argparse, sys, pathlib imports."""
    if 'from pathlib import Path' not in content and 'import pathlib' not in content:
        m = re.search(r'^import torch\n', content, re.MULTILINE)
        if m:
            content = content[:m.end()] + 'from pathlib import Path\n' + content[m.end():]
    if 'import argparse' not in content:
        m = re.search(r'^(from __future__[^\n]*\n)', content, re.MULTILINE)
        if m:
            content = content[:m.end()] + '\nimport argparse\n' + content[m.end():]
        else:
            content = 'import argparse\n' + content
    if 'import sys' not in content:
        m = re.search(r'^import argparse\n', content, re.MULTILINE)
        if m:
            content = content[:m.end()] + 'import sys\n' + content[m.end():]
    return content


def build_tail(wrapper_call: str) -> str:
    """Build oracle_forward wrapper + standard main."""
    return f'''

def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return {wrapper_call}


def main():
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


def convert_file(rel_path: str, oracle_func: str, wrapper_call: str,
                 rename_to: str | None = None) -> bool:
    """Convert a single oracle file."""
    path = REPO_ROOT / rel_path
    if not path.exists():
        print(f"  SKIP (not found): {rel_path}")
        return False

    content = path.read_text()
    repro_id = path.parent.name

    print(f"  Converting: {rel_path}")

    # If oracle function needs renaming (e.g., oracle_forward -> _oracle_impl)
    if rename_to:
        content = re.sub(
            rf'^def {re.escape(oracle_func)}\(',
            f'def {rename_to}(',
            content, count=1, flags=re.MULTILINE
        )
        # Also rename any self-referential calls
        content = content.replace(f'{oracle_func}(', f'{rename_to}(')
        oracle_func = rename_to

    # Find cut point (where boilerplate starts)
    cut_pos = find_boilerplate_start(content, oracle_func)
    kept = content[:cut_pos].rstrip() + '\n'

    # Add oracle_harness imports
    kept = add_harness_import(kept)

    # Ensure standard imports
    kept = ensure_imports(kept)

    # Ensure REPRO_ID/DIR/PATH constants
    kept = ensure_constants(kept, repro_id)

    # Remove unused imports
    for unused in ['import time', 'import csv', 'import subprocess',
                   'import math', 'from typing import Callable',
                   'from datetime import datetime, timezone']:
        if unused in kept:
            rest = kept.replace(unused, '', 1)
            # Check if the imported name is still used
            module_name = unused.split()[-1] if 'from' not in unused else unused.split('import ')[-1].split(',')[0].strip()
            if module_name + '.' not in rest and module_name + '(' not in rest:
                kept = re.sub(re.escape(unused) + r'\n', '', kept, count=1)

    # Append oracle_forward + standard main
    tail = build_tail(wrapper_call)
    result = kept.rstrip() + '\n' + tail

    # Clean up excessive blank lines
    result = re.sub(r'\n{4,}', '\n\n\n', result)
    result = result.rstrip() + '\n'

    # Validate syntax
    try:
        ast.parse(result)
    except SyntaxError as e:
        print(f"    ERROR: Syntax error after conversion: {e}")
        return False

    path.write_text(result)
    print(f"    OK (syntax valid)")
    return True


def main():
    configs = [
        ('repros/canonical/sum_sum_mean_1dfbbe76c078/oracle_rmsnorm_bwd.py',
         'oracle_forward', '_oracle_impl(*inputs)', '_oracle_impl'),
        ('repros/canonical/pointwise_71e3a6c09140/oracle_stencil_fusion.py',
         'triton_fused_bn_relu_maxpool', 'triton_fused_bn_relu_maxpool(*inputs)', None),
        ('repros/canonical/pointwise_cf3acd87ba9e/oracle_stencil_fusion.py',
         'triton_fused_channel_shuffle', 'triton_fused_channel_shuffle(*inputs)', None),
        ('repros/canonical/sum_3ee4028cab37/oracle_structured_scatter_reduce.py',
         'torch_fused_scatter_reduce', 'torch_fused_scatter_reduce(*inputs)', None),
        ('repros/canonical/sum_b699ca824f8e/oracle_avgpool_backward_reduce.py',
         'triton_direct_avgpool_backward_reduce', 'triton_direct_avgpool_backward_reduce(*inputs)', None),
        ('repros/canonical/sum_sum_230671a8764d/oracle_rmsnorm_rope_bwd.py',
         'structured_oracle', 'structured_oracle(*inputs)', None),
        ('repros/canonical/sum_sum_8bcd6e12dcd4/oracle_structured_scatter_reduce.py',
         'torch_fused_scatter_reduce', 'torch_fused_scatter_reduce(*inputs)', None),
        ('repros/canonical/sum_sum_sum_45f02142ecfd/oracle_structured_scatter_reduce.py',
         'torch_fused_bilinear_scatter_reduce', 'torch_fused_bilinear_scatter_reduce(*inputs)', None),
        ('repros/canonical/sum_sum_sum_512dcf4a167b/oracle_multi_output_reduction.py',
         'oracle_fused', 'oracle_fused(*prepare_oracle_inputs(*inputs))', None),
        ('repros/canonical/sum_sum_sum_70d71fcb0d68/oracle_cross_dim_reduction.py',
         'cross_dim_oracle', 'cross_dim_oracle(*inputs)', None),
        ('repros/canonical/var_mean_765fb8f2c85e/oracle_bn_training_forward.py',
         'triton_bn_relu', 'triton_bn_relu(*inputs)', None),
    ]

    converted = 0
    failed = 0
    for rel_path, oracle_func, wrapper_call, rename_to in configs:
        if convert_file(rel_path, oracle_func, wrapper_call, rename_to):
            converted += 1
        else:
            failed += 1

    print(f"\nDone: {converted} converted, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
