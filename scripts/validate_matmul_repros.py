"""Reload every generated repro/shape point and run FakeTensor execution."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repro_harness import load_shape_configs, make_inputs_from_config
from roundtrip_validation import fake_tensor_ctx
from scripts.extract_matmul_corpus import write_json


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "mm_repro")
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    start = time.perf_counter()
    failures, points, files = [], 0, 0
    for path in sorted((output / "canonical").glob("*/repro.py")):
        try:
            spec = importlib.util.spec_from_file_location(
                "mm_validation_" + path.parent.name, path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            configs = load_shape_configs(str(path))
            if not configs:
                raise ValueError("repro has no shape points")
            with fake_tensor_ctx():
                model = module.Repro()
                for label, config in configs.items():
                    try:
                        model(*make_inputs_from_config(config))
                        points += 1
                    except Exception:  # noqa: BLE001 -- record every failed point
                        failures.append(
                            {
                                "repro": str(path),
                                "shape": label,
                                "error": traceback.format_exc(),
                            }
                        )
            files += 1
        except Exception:  # noqa: BLE001 -- record every failed artifact
            failures.append({"repro": str(path), "error": traceback.format_exc()})
        if files % 100 == 0:
            print(
                f"Reloaded {files} repros / {points} shape points; failures={len(failures)}",
                flush=True,
            )
    result = {
        "repros": files,
        "shape_points": points,
        "failures": failures,
        "seconds": round(time.perf_counter() - start, 3),
    }
    write_json(output / "validation.json", result)
    if files != report["unique_repros"] or points != report["shape_points"]:
        failures.append({"error": "reloaded file/shape counts differ from report.json"})
        write_json(output / "validation.json", result)
    report["fake_tensor_validation"] = result
    write_json(report_path, report)
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
