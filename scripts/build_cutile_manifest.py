"""Build the ordered work list of Triton oracles awaiting cuTile ports.

Writes a JSON manifest with one entry per canonical dir:
  {
    "name": "pointwise_000209e1748d",
    "kind": "pointwise",   # first token of name
    "triton_oracle_path": "repros/canonical/.../oracle.py",
    "cutile_dir": "repros_cutile/canonical/...",
    "has_cutile_oracle": bool,
    "triton_oracle_lines": int,
    "shapes": [{"shape_hash": "...", "num_models": N, ...}, ...]
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
TRITON_ROOT = _REPO_ROOT / "repros" / "canonical"
CUTILE_ROOT = _REPO_ROOT / "repros_cutile" / "canonical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "scripts" / "cutile_manifest.json")
    args = ap.parse_args()

    entries = []
    for d in sorted(TRITON_ROOT.iterdir()):
        if not d.is_dir():
            continue
        oracle = d / "oracle.py"
        if not oracle.exists():
            continue
        cutile_dir = CUTILE_ROOT / d.name
        cutile_oracle = cutile_dir / "oracle.py"
        try:
            shapes = json.loads((d / "shapes.json").read_text())
        except FileNotFoundError:
            shapes = {"points": []}
        try:
            n_lines = sum(1 for _ in oracle.open())
        except OSError:
            n_lines = 0
        entries.append({
            "name": d.name,
            "kind": d.name.split("_")[0],
            "triton_oracle_path": str(oracle.relative_to(_REPO_ROOT)),
            "cutile_dir": str(cutile_dir.relative_to(_REPO_ROOT)),
            "has_cutile_oracle": cutile_oracle.exists(),
            "triton_oracle_lines": n_lines,
            "shape_points": [
                {
                    "shape_hash": p.get("shape_hash"),
                    "n_models": len(p.get("models", {})),
                    "n_inputs": len(p.get("inputs", [])),
                }
                for p in shapes.get("points", [])
            ],
        })
    args.out.write_text(json.dumps(entries, indent=2))
    print(f"wrote {len(entries)} entries -> {args.out}")


if __name__ == "__main__":
    main()
