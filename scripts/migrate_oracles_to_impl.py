"""Mass-migrate oracle files to oracle_impl registration.

Adds to each repros/canonical/<id>/oracle_*.py:
  - `oracle_impl` in the oracle_harness import block
  - `@oracle_impl(hardware="H100", shapes="<_shapes_config>")` above
    def oracle_forward

Idempotent: files already containing "oracle_impl(" are skipped.

Usage:
    python scripts/migrate_oracles_to_impl.py            # all
    python scripts/migrate_oracles_to_impl.py --prefix sum_,var_mean_
    python scripts/migrate_oracles_to_impl.py --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORWARD_RE = re.compile(r"^def oracle_forward\(", re.MULTILINE)
SIG_RE = re.compile(r'_shapes_config\s*=\s*"(.+?)"\s*$', re.MULTILINE)
IMPORT_BLOCK_RE = re.compile(r"from oracle_harness import \(")


def migrate_file(oracle: Path, sig: str, dry_run: bool = False) -> str:
    text = oracle.read_text()
    if "oracle_impl(" in text:
        return "already"
    m = FORWARD_RE.search(text)
    if not m:
        # annotated signature variant: def oracle_forward(inputs: ...)
        m = re.search(r"^def oracle_forward\b", text, re.MULTILINE)
        if not m:
            return "no_oracle_forward"
    if '"' in sig:
        return "sig_has_quote"

    # 1. import
    if IMPORT_BLOCK_RE.search(text):
        text = IMPORT_BLOCK_RE.sub("from oracle_harness import (\n    oracle_impl,", text, count=1)
    elif "from oracle_harness import" in text:
        text = text.replace("from oracle_harness import",
                            "from oracle_harness import oracle_impl,", 1)
    elif "import oracle_harness" in text:
        text = text.replace("import oracle_harness",
                            "import oracle_harness\nfrom oracle_harness import oracle_impl", 1)
    else:
        return "no_harness_import"

    # 2. decorator above the FIRST def oracle_forward
    decorator = f'@oracle_impl(hardware="H100", shapes="{sig}")\n'
    m = re.search(r"^def oracle_forward\b[^\n]*", text, re.MULTILINE)
    text = text[: m.start()] + decorator + text[m.start():]

    if not dry_run:
        oracle.write_text(text)
        # syntax check
        import py_compile
        try:
            py_compile.compile(str(oracle), doraise=True)
        except py_compile.PyCompileError as e:
            return f"compile_fail: {e}"
    return "migrated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None,
                    help="comma-separated repro_id prefixes (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    prefixes = tuple(args.prefix.split(",")) if args.prefix else None

    stats = {}
    failures = []
    for repro_py in sorted((ROOT / "repros/canonical").glob("*/repro.py")):
        rid = repro_py.parent.name
        if prefixes and not rid.startswith(prefixes):
            continue
        sig_m = SIG_RE.search(repro_py.read_text())
        if not sig_m:
            stats["no_shapes_config"] = stats.get("no_shapes_config", 0) + 1
            failures.append((rid, "no_shapes_config"))
            continue
        sig = sig_m.group(1)
        for oracle in sorted(repro_py.parent.glob("oracle_*.py")):
            result = migrate_file(oracle, sig, dry_run=args.dry_run)
            stats[result.split(":")[0]] = stats.get(result.split(":")[0], 0) + 1
            if result not in ("migrated", "already"):
                failures.append((f"{rid}/{oracle.name}", result))

    print(f"Stats: {stats}")
    if failures:
        print(f"Failures ({len(failures)}):")
        for name, why in failures[:30]:
            print(f"  {name}: {why}")
    return 1 if any(w.startswith("compile_fail") for _, w in failures) else 0


if __name__ == "__main__":
    sys.exit(main())
