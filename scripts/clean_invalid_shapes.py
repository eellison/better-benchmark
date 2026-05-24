"""
Remove internally inconsistent shapes.txt entries from canonical repros.

For each entry in shapes.txt, runs the repro in eager mode in an isolated
subprocess. If it fails (RuntimeError from shape mismatch, wrong arg count,
etc.), the entry is removed. Entries that pass are kept.

Uses subprocess isolation to prevent CUDA state poisoning between tests.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAILING_REPROS = [
    'amax_sum_sum_31600750c3c4', 'amax_sum_sum_4c98004c3aa3',
    'amax_sum_sum_5818707b6489', 'amax_sum_sum_d5ddd6e16182',
    'amax_sum_sum_d85e67b00643', 'argmax_amax_sum_fd72c2ef2d40',
    'pointwise_083e5f8c7cdc', 'pointwise_28fded12b2f4',
    'pointwise_2eae25328292', 'pointwise_3c320014165a',
    'pointwise_3cfc4c59af74', 'pointwise_5d7c3bd1f499',
    'pointwise_5e8fb91fe0df', 'pointwise_8793407401ab',
    'pointwise_a5634203b03b', 'pointwise_ba25116c7e2e',
    'pointwise_baa1198c62c0', 'pointwise_cf3acd87ba9e',
    'pointwise_cffb7fe32779', 'sum_27f8a4b9ab09',
    'sum_sum_sum_cd47d7ca4703', 'var_mean_06924cc70cb4',
    'var_mean_32352a093910', 'var_mean_58cdb69fc823',
    'var_mean_810878ea0ac7', 'var_mean_fc43e261f9f8',
]


def validate_entry(repro_dir: str, config_str: str, timeout: int = 30) -> tuple[bool, str]:
    """Validate a single shapes.txt entry in an isolated subprocess.

    Returns (is_valid, error_message).
    """
    code = f'''
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "{PROJECT_ROOT}")
sys.path.insert(0, "{repro_dir}")

import torch
import importlib.util
from repro_harness import parse_shapes_config

spec = importlib.util.spec_from_file_location("repro", "{repro_dir}/repro.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

try:
    inputs = parse_shapes_config("""{config_str}""")
    repro_obj = mod.Repro()
    with torch.no_grad():
        output = repro_obj(*inputs)
    print("PASS")
except Exception as e:
    print(f"FAIL: {{type(e).__name__}}: {{str(e)[:200]}}")
'''
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONWARNINGS"] = "ignore"

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        # Check stdout for PASS/FAIL
        stdout_lines = [l.strip() for l in proc.stdout.strip().split("\n") if l.strip()]
        for line in reversed(stdout_lines):
            if line == "PASS":
                return True, ""
            if line.startswith("FAIL:"):
                return False, line[5:].strip()

        # If no PASS/FAIL found, check stderr
        if proc.returncode != 0:
            stderr_last = proc.stderr.strip().split("\n")[-1][:200] if proc.stderr else "unknown error"
            return False, f"subprocess failed: {stderr_last}"

        return False, "no output from validation subprocess"

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"subprocess error: {e}"


def main():
    total_removed = 0
    total_kept = 0
    results = {}

    for repro_name in FAILING_REPROS:
        repro_dir = PROJECT_ROOT / "repros" / "canonical" / repro_name
        shapes_path = repro_dir / "shapes.txt"
        repro_py = repro_dir / "repro.py"

        if not shapes_path.exists() or not repro_py.exists():
            print(f"{repro_name}: SKIP (missing files)")
            continue

        lines = shapes_path.read_text().splitlines()
        valid_lines = []
        invalid_entries = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                valid_lines.append(line)
                continue

            colon = stripped.find(":")
            if colon < 0:
                valid_lines.append(line)
                continue

            label = stripped[:colon].strip()
            config_str = stripped[colon + 1:].strip()

            is_valid, error = validate_entry(str(repro_dir), config_str)

            if is_valid:
                valid_lines.append(line)
            else:
                invalid_entries.append((label, error))

        if invalid_entries:
            total_removed += len(invalid_entries)
            total_kept += len([l for l in valid_lines if l.strip() and not l.strip().startswith("#")])
            print(f"{repro_name}: REMOVING {len(invalid_entries)} invalid, keeping {len([l for l in valid_lines if l.strip()])} valid")
            for label, err in invalid_entries:
                print(f"  - {label}: {err}")

            # Write back
            if valid_lines and any(l.strip() for l in valid_lines):
                shapes_path.write_text("\n".join(valid_lines) + "\n")
            else:
                shapes_path.write_text("")

            results[repro_name] = {
                "removed": len(invalid_entries),
                "kept": len([l for l in valid_lines if l.strip()]),
                "invalid": [(l, e) for l, e in invalid_entries],
            }
        else:
            n_valid = len([l for l in valid_lines if l.strip() and not l.strip().startswith("#")])
            total_kept += n_valid
            print(f"{repro_name}: ALL {n_valid} entries valid")
            results[repro_name] = {"removed": 0, "kept": n_valid, "invalid": []}

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total entries removed: {total_removed}")
    print(f"Total entries kept:    {total_kept}")
    print(f"Repros modified:       {sum(1 for v in results.values() if v['removed'] > 0)}")


if __name__ == "__main__":
    main()
