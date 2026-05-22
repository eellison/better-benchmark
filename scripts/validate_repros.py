"""
Validate canonical repros for correctness.

Checks:
1. repro.py parses (syntax OK)
2. _shapes_config arg count matches forward() signature
3. Default shape runs without crashing
4. Each shapes.txt/shapes.json config runs without crashing

Usage:
    python scripts/validate_repros.py                    # validate all
    python scripts/validate_repros.py --quick            # syntax + arg count only (no GPU)
    python scripts/validate_repros.py --repro var_mean_a7cbd072693b  # one repro
"""
import argparse
import ast
import importlib.util
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def validate_syntax(repro_path: Path) -> str | None:
    """Check repro.py parses."""
    import ast
    try:
        ast.parse(repro_path.read_text())
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e}"


def _forward_arg_count(content: str) -> int | None:
    tree = ast.parse(content)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Repro":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    return len(item.args.args) - 1
    return None


def _shapes_config(content: str) -> str | None:
    tree = ast.parse(content)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "_shapes_config"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
        return None
    return None


def _shape_expr_input_count(expr: str) -> int:
    tree = ast.parse(expr, mode="eval")
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"T", "S"}
    )


def validate_arg_count(repro_path: Path) -> str | None:
    """Check _shapes_config describes the right number of args for forward()."""
    content = repro_path.read_text()

    n_fwd = _forward_arg_count(content)
    if n_fwd is None:
        return "No forward() found"

    config = _shapes_config(content)
    if config is None:
        return "Missing _shapes_config"

    try:
        n_inputs = _shape_expr_input_count(config)
    except SyntaxError as e:
        return f"_shapes_config parse failed: {e}"

    if n_fwd != n_inputs:
        return f"Arg mismatch: forward expects {n_fwd}, _shapes_config has {n_inputs}"
    return None


def validate_default_runs(repro_path: Path) -> str | None:
    """Check default shape runs without crash."""
    import torch
    try:
        spec = importlib.util.spec_from_file_location("repro", str(repro_path))
        mod = importlib.util.module_from_spec(spec)
        mod.device = torch.device
        mod.inf = math.inf
        mod.nan = math.nan
        spec.loader.exec_module(mod)

        repro = mod.Repro()
        from repro_harness import make_inputs_safely

        make_inputs = mod.make_inputs if hasattr(mod, "make_inputs") else mod._default_make_inputs
        inputs = make_inputs_safely(make_inputs)

        with torch.no_grad():
            repro(*inputs)
        return None
    except Exception as e:
        return f"Default run failed: {str(e)[:100]}"


def validate_shapes_configs(repro_path: Path) -> list[str]:
    """Check each shapes.txt/shapes.json config runs."""
    import torch
    from repro_harness import load_shape_configs, make_inputs_from_config

    errors = []
    configs = load_shape_configs(str(repro_path))
    if not configs:
        return []

    try:
        spec = importlib.util.spec_from_file_location("repro", str(repro_path))
        mod = importlib.util.module_from_spec(spec)
        mod.device = torch.device
        mod.inf = math.inf
        mod.nan = math.nan
        spec.loader.exec_module(mod)
        repro = mod.Repro()
    except Exception as e:
        return [f"Load failed: {e}"]

    for key, config in configs.items():
        try:
            inputs = make_inputs_from_config(config)
            with torch.no_grad():
                repro(*inputs)
        except Exception as e:
            errors.append(f"Config '{key}': {str(e)[:80]}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Syntax + arg count only (no GPU)")
    parser.add_argument("--repro", type=str, default=None, help="Validate one repro by name")
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros/canonical"))
    args = parser.parse_args()

    if args.repro:
        dirs = [args.canonical_dir / args.repro]
    else:
        dirs = sorted(args.canonical_dir.iterdir())

    total = 0
    syntax_fail = 0
    arg_fail = 0
    run_fail = 0
    shape_fail = 0
    shape_configs_tested = 0
    shape_configs_failed = 0

    for d in dirs:
        repro = d / "repro.py"
        if not repro.exists():
            continue
        total += 1

        # 1. Syntax
        err = validate_syntax(repro)
        if err:
            print(f"SYNTAX FAIL: {d.name}: {err}")
            syntax_fail += 1
            continue

        # 2. Arg count
        err = validate_arg_count(repro)
        if err:
            print(f"ARG FAIL: {d.name}: {err}")
            arg_fail += 1
            continue

        if args.quick:
            continue

        # 3. Default shape runs
        err = validate_default_runs(repro)
        if err:
            print(f"RUN FAIL: {d.name}: {err}")
            run_fail += 1
            continue

        # 4. Shape configs
        errors = validate_shapes_configs(repro)
        if errors:
            shape_fail += 1
            shape_configs_failed += len(errors)
            for e in errors[:3]:
                print(f"SHAPE FAIL: {d.name}: {e}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more")

        from repro_harness import load_shape_configs

        shape_configs_tested += len(load_shape_configs(str(repro)))

    print(f"\n{'='*60}")
    print(f"Validated {total} repros")
    if args.quick:
        print(f"  Syntax errors: {syntax_fail}")
        print(f"  Arg count mismatches: {arg_fail}")
    else:
        print(f"  Syntax errors: {syntax_fail}")
        print(f"  Arg count mismatches: {arg_fail}")
        print(f"  Default run failures: {run_fail}")
        print(f"  Shape config failures: {shape_fail} repros ({shape_configs_failed}/{shape_configs_tested} configs)")
        print(f"  All pass: {total - syntax_fail - arg_fail - run_fail - shape_fail}")


if __name__ == "__main__":
    main()
