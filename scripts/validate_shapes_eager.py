"""
Validate that (repro, shape_config) pairs actually work end-to-end in eager mode.

For each canonical repro:
1. Load shapes (from shapes.txt, shapes.json, or embedded _shapes_config)
2. For each shape config, generate inputs via parse_shapes_config / make_inputs_from_config
3. Run Repro()(*inputs) in eager mode
4. Check output is a valid tensor (not NaN/Inf)

Uses subprocess isolation to avoid CUDA state poisoning between repros.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = REPO_ROOT / "repros" / "canonical"


def find_repros(limit=None):
    """Find all canonical repro directories."""
    repro_dirs = sorted(CANONICAL_DIR.iterdir())
    repro_dirs = [d for d in repro_dirs if d.is_dir() and (d / "repro.py").exists()]
    if limit:
        repro_dirs = repro_dirs[:limit]
    return repro_dirs


def validate_single_repro(repro_dir: str, gpu_id: str = "0", timeout: int = 60) -> dict:
    """Validate a single repro in a subprocess. Returns result dict."""
    repro_dir = Path(repro_dir)
    repro_name = repro_dir.name

    # Build the validation script to run in subprocess
    validation_code = f'''
import sys, os, json, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
sys.path.insert(0, "{REPO_ROOT}")

import torch

repro_dir = "{repro_dir}"
repro_name = "{repro_name}"
results = []

try:
    # Import repro module
    sys.path.insert(0, repro_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("repro", os.path.join(repro_dir, "repro.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from repro_harness import (
        parse_shapes_config, make_inputs_from_config, load_shape_configs,
        make_inputs_safely
    )

    # 1. Validate default _shapes_config
    shapes_config = getattr(mod, "_shapes_config", None)
    if shapes_config:
        try:
            inputs = parse_shapes_config(shapes_config)
            repro_obj = mod.Repro()
            with torch.no_grad():
                output = repro_obj(*inputs)

            # Check output validity
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                valid = not has_nan and not has_inf
                results.append({{
                    "config": "default/_shapes_config",
                    "status": "pass" if valid else "output_invalid",
                    "details": f"nan={{has_nan}}, inf={{has_inf}}" if not valid else None,
                    "n_inputs": len(inputs),
                    "output_shape": list(output.shape),
                }})
            elif isinstance(output, (tuple, list)):
                # Multiple outputs
                all_valid = True
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        if torch.isnan(o).any().item() or torch.isinf(o).any().item():
                            all_valid = False
                            break
                results.append({{
                    "config": "default/_shapes_config",
                    "status": "pass" if all_valid else "output_invalid",
                    "n_inputs": len(inputs),
                }})
            else:
                results.append({{
                    "config": "default/_shapes_config",
                    "status": "pass",
                    "n_inputs": len(inputs),
                }})
        except Exception as e:
            tb = traceback.format_exc()
            err_type = type(e).__name__
            err_msg = str(e)[:200]
            results.append({{
                "config": "default/_shapes_config",
                "status": "fail",
                "error_type": err_type,
                "error": err_msg,
                "traceback": tb[-500:],
            }})

    # 2. Validate shapes.txt / shapes.json configs
    configs = load_shape_configs(os.path.join(repro_dir, "repro.py"))
    for config_name, config_data in configs.items():
        try:
            inputs = make_inputs_from_config(config_data)

            # Check input count matches forward() signature
            repro_obj = mod.Repro()

            # Try to get expected arg count from forward signature
            import inspect
            sig = inspect.signature(repro_obj.forward)
            expected_args = len([
                p for p in sig.parameters.values()
                if p.name != "self"
            ])

            if len(inputs) < expected_args:
                # Try merging with default inputs (like the harness does)
                default_inputs = parse_shapes_config(shapes_config) if shapes_config else []
                if len(default_inputs) == expected_args:
                    merged = []
                    config_idx = 0
                    for di in default_inputs:
                        if isinstance(di, (list, int)) and not isinstance(di, torch.Tensor):
                            merged.append(di)
                        else:
                            if config_idx < len(inputs):
                                merged.append(inputs[config_idx])
                                config_idx += 1
                            else:
                                merged.append(di)
                    inputs = merged

            with torch.no_grad():
                output = repro_obj(*inputs)

            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                valid = not has_nan and not has_inf
                results.append({{
                    "config": config_name,
                    "status": "pass" if valid else "output_invalid",
                    "details": f"nan={{has_nan}}, inf={{has_inf}}" if not valid else None,
                    "n_inputs": len(inputs),
                }})
            else:
                results.append({{
                    "config": config_name,
                    "status": "pass",
                    "n_inputs": len(inputs),
                }})
        except Exception as e:
            tb = traceback.format_exc()
            err_type = type(e).__name__
            err_msg = str(e)[:200]
            results.append({{
                "config": config_name,
                "status": "fail",
                "error_type": err_type,
                "error": err_msg,
                "traceback": tb[-500:],
            }})

    if not results:
        results.append({{
            "config": "none",
            "status": "skip",
            "details": "no shapes config found",
        }})

except Exception as e:
    tb = traceback.format_exc()
    results = [{{
        "config": "module_load",
        "status": "fail",
        "error_type": type(e).__name__,
        "error": str(e)[:200],
        "traceback": tb[-500:],
    }}]

print(json.dumps({{"repro": repro_name, "results": results}}))
'''

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    # Suppress torch warnings
    env["PYTHONWARNINGS"] = "ignore"

    try:
        result = subprocess.run(
            [sys.executable, "-c", validation_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
            env=env,
        )

        if result.returncode != 0:
            # Try to parse any JSON from stdout first
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        pass

            return {
                "repro": repro_name,
                "results": [{
                    "config": "subprocess",
                    "status": "fail",
                    "error_type": "SubprocessError",
                    "error": result.stderr[-300:] if result.stderr else "unknown",
                }],
            }

        # Parse JSON output
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass

        return {
            "repro": repro_name,
            "results": [{
                "config": "subprocess",
                "status": "fail",
                "error_type": "ParseError",
                "error": f"No JSON in output: {result.stdout[-200:]}",
            }],
        }

    except subprocess.TimeoutExpired:
        return {
            "repro": repro_name,
            "results": [{
                "config": "subprocess",
                "status": "fail",
                "error_type": "Timeout",
                "error": f"Timed out after {timeout}s",
            }],
        }
    except Exception as e:
        return {
            "repro": repro_name,
            "results": [{
                "config": "subprocess",
                "status": "fail",
                "error_type": type(e).__name__,
                "error": str(e)[:200],
            }],
        }


def main():
    parser = argparse.ArgumentParser(description="Validate shapes configs for canonical repros")
    parser.add_argument("--limit", type=int, default=None, help="Max number of repros to validate")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID")
    parser.add_argument("--timeout", type=int, default=60, help="Per-repro timeout in seconds")
    parser.add_argument("--output", type=str, default=None, help="Write full results JSON to file")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Print each result as it comes")
    args = parser.parse_args()

    repro_dirs = find_repros(args.limit)
    print(f"Found {len(repro_dirs)} repros to validate")
    print(f"GPU: {args.gpu}, timeout: {args.timeout}s, workers: {args.workers}")
    print("-" * 60)

    all_results = []
    total_configs = 0
    pass_count = 0
    fail_count = 0
    skip_count = 0
    invalid_count = 0
    error_categories = {}

    start_time = time.time()

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(validate_single_repro, str(d), args.gpu, args.timeout): d
                for d in repro_dirs
            }
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                for r in result.get("results", []):
                    total_configs += 1
                    status = r.get("status", "unknown")
                    if status == "pass":
                        pass_count += 1
                    elif status == "fail":
                        fail_count += 1
                        err_type = r.get("error_type", "Unknown")
                        error_categories[err_type] = error_categories.get(err_type, 0) + 1
                    elif status == "output_invalid":
                        invalid_count += 1
                    else:
                        skip_count += 1
                if args.verbose:
                    repro_name = result.get("repro", "?")
                    statuses = [r.get("status") for r in result.get("results", [])]
                    print(f"  {repro_name}: {statuses}")
    else:
        for i, repro_dir in enumerate(repro_dirs):
            result = validate_single_repro(str(repro_dir), args.gpu, args.timeout)
            all_results.append(result)

            for r in result.get("results", []):
                total_configs += 1
                status = r.get("status", "unknown")
                if status == "pass":
                    pass_count += 1
                elif status == "fail":
                    fail_count += 1
                    err_type = r.get("error_type", "Unknown")
                    error_categories[err_type] = error_categories.get(err_type, 0) + 1
                elif status == "output_invalid":
                    invalid_count += 1
                else:
                    skip_count += 1

            if args.verbose or (i + 1) % 10 == 0:
                repro_name = result.get("repro", "?")
                statuses = [r.get("status") for r in result.get("results", [])]
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(repro_dirs)}] {repro_name}: {statuses} ({rate:.1f} repros/s)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY")
    print(f"=" * 60)
    print(f"Repros validated:  {len(all_results)}")
    print(f"Total configs:     {total_configs}")
    print(f"  PASS:            {pass_count}")
    print(f"  FAIL:            {fail_count}")
    print(f"  INVALID OUTPUT:  {invalid_count}")
    print(f"  SKIP:            {skip_count}")
    print(f"Pass rate:         {pass_count/max(total_configs,1)*100:.1f}%")
    print(f"Time:              {elapsed:.1f}s")
    print()

    if error_categories:
        print("FAILURE CATEGORIES:")
        for err_type, count in sorted(error_categories.items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count}")
        print()

    # Print sample failures for each category
    if fail_count > 0:
        print("SAMPLE FAILURES (up to 3 per category):")
        shown = {}
        for result in all_results:
            for r in result.get("results", []):
                if r.get("status") == "fail":
                    err_type = r.get("error_type", "Unknown")
                    if err_type not in shown:
                        shown[err_type] = []
                    if len(shown[err_type]) < 3:
                        shown[err_type].append({
                            "repro": result.get("repro"),
                            "config": r.get("config"),
                            "error": r.get("error"),
                            "traceback": r.get("traceback", "")[-300:],
                        })
        for err_type, samples in sorted(shown.items()):
            print(f"\n  [{err_type}]")
            for s in samples:
                print(f"    repro: {s['repro']}")
                print(f"    config: {s['config']}")
                print(f"    error: {s['error']}")
                if s.get('traceback'):
                    # Show last 2 lines of traceback
                    tb_lines = [l for l in s['traceback'].split('\n') if l.strip()]
                    for tl in tb_lines[-2:]:
                        print(f"      {tl}")
                print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "summary": {
                    "total_repros": len(all_results),
                    "total_configs": total_configs,
                    "pass": pass_count,
                    "fail": fail_count,
                    "invalid_output": invalid_count,
                    "skip": skip_count,
                    "pass_rate": pass_count / max(total_configs, 1),
                    "error_categories": error_categories,
                    "elapsed_s": elapsed,
                },
                "results": all_results,
            }, f, indent=2)
        print(f"\nFull results written to: {args.output}")


if __name__ == "__main__":
    main()
