"""Recapture canonical repros from saved full_graph_*.py artifacts.

This reuses the same graph loading and region extraction path as tlparse
ingestion, but infers suite/mode/model labels from repros/models paths instead
of rerunning source benchmarks.
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_tlparse import load_graph_module


DEFAULT_MODELS_ROOT = Path("repros/models")
DEFAULT_CANONICAL_ROOT = Path("/tmp/scratch_space/better_benchmark/repros")
MODE_NAMES = {"infer", "train"}


@dataclass(frozen=True)
class FullGraphTarget:
    path: Path
    suite: str
    mode: str | None
    model: str
    graph_name: str

    @property
    def label(self) -> str:
        parts = [self.suite]
        if self.mode:
            parts.append(self.mode)
        parts.extend([self.model.replace("/", "__"), self.graph_name])
        return "_".join(parts)


@dataclass(frozen=True)
class RecaptureResult:
    target: FullGraphTarget
    regions: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _as_resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _check_under_models_root(path: Path, models_root: Path) -> None:
    try:
        _as_resolved(path).relative_to(_as_resolved(models_root))
    except ValueError as exc:
        raise ValueError(f"{path} is not under models root {models_root}") from exc


def infer_target(path: Path, models_root: Path = DEFAULT_MODELS_ROOT) -> FullGraphTarget:
    """Infer suite/mode/model labels from a full_graph path under models_root."""
    if not path.name.startswith("full_graph_") or path.suffix != ".py":
        raise ValueError(f"{path} is not a full_graph_*.py file")

    _check_under_models_root(path, models_root)
    rel = _as_resolved(path).relative_to(_as_resolved(models_root))
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(
            f"{path} must look like {models_root}/<suite>/<model>/full_graph_*.py "
            f"or {models_root}/<suite>/<mode>/<model>/full_graph_*.py"
        )

    suite = parts[0]
    label_parts = list(parts[1:-1])
    mode = None
    if label_parts and label_parts[0] in MODE_NAMES:
        mode = label_parts.pop(0)
    if not label_parts:
        raise ValueError(f"{path} does not include a model directory")

    return FullGraphTarget(
        path=path,
        suite=suite,
        mode=mode,
        model=Path(*label_parts).as_posix(),
        graph_name=path.stem,
    )


def find_full_graphs(
    paths: Sequence[Path] | None = None,
    models_root: Path = DEFAULT_MODELS_ROOT,
) -> list[Path]:
    """Find selected full_graph_*.py files under models_root."""
    selected = list(paths) if paths else [models_root]
    found: list[Path] = []
    seen: set[Path] = set()

    for raw_path in selected:
        path = raw_path.expanduser()
        _check_under_models_root(path, models_root)
        if path.is_file():
            if not path.name.startswith("full_graph_") or path.suffix != ".py":
                raise ValueError(f"{path} is not a full_graph_*.py file")
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("full_graph_*.py"))
        else:
            raise FileNotFoundError(f"selection path does not exist: {path}")

        for candidate in candidates:
            if not candidate.is_file():
                continue
            if not candidate.name.startswith("full_graph_") or candidate.suffix != ".py":
                continue
            resolved = _as_resolved(candidate)
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(candidate)

    return found


def discover_targets(
    paths: Sequence[Path] | None = None,
    models_root: Path = DEFAULT_MODELS_ROOT,
) -> list[FullGraphTarget]:
    return [
        infer_target(path, models_root=models_root)
        for path in find_full_graphs(paths, models_root=models_root)
    ]


def process_graph_for_target(
    gm,
    target: FullGraphTarget,
    canonical_root: Path,
    *,
    validate: bool = True,
) -> int:
    """Partition one loaded full graph and merge regions into canonical_root."""
    from capture_hook import _CaptureState
    from merge_captures import merge_one_capture

    with tempfile.TemporaryDirectory(prefix="recapture_full_graph_") as tmp:
        cap_dir = Path(tmp)
        state = _CaptureState(str(cap_dir), label=target.label, validate=validate)
        state.process_graph(gm)
        state.finalize()

        index_path = cap_dir / "index.json"
        if not index_path.exists():
            return 0

        return merge_one_capture(
            cap_dir,
            canonical_root,
            target.model,
            suite=target.suite,
            mode=target.mode,
        )


LoadGraphFn = Callable[[Path], object | None]
ProcessGraphFn = Callable[[object, FullGraphTarget, Path], int]


def recapture_targets(
    targets: Iterable[FullGraphTarget],
    canonical_root: Path,
    *,
    dry_run: bool = False,
    validate: bool = True,
    fail_fast: bool = False,
    load_fn: LoadGraphFn = load_graph_module,
    process_fn: ProcessGraphFn | None = None,
) -> list[RecaptureResult]:
    """Recapture all targets, returning one result per selected full graph."""
    targets = list(targets)
    if dry_run:
        return [RecaptureResult(target) for target in targets]

    if process_fn is None:
        def process_fn(gm, target, canonical_root):
            return process_graph_for_target(
                copy.deepcopy(gm),
                target,
                canonical_root,
                validate=validate,
            )

    results: list[RecaptureResult] = []
    for target in targets:
        try:
            gm = load_fn(target.path)
            if gm is None:
                raise RuntimeError("graph load returned None")
            regions = process_fn(gm, target, canonical_root)
            results.append(RecaptureResult(target, regions=regions))
        except Exception as exc:
            results.append(RecaptureResult(target, error=str(exc)))
            if fail_fast:
                break
    return results


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def format_target(target: FullGraphTarget) -> str:
    mode = target.mode if target.mode is not None else "-"
    return (
        f"{_display_path(target.path)} "
        f"suite={target.suite} mode={mode} model={target.model}"
    )


def print_targets(targets: Sequence[FullGraphTarget]) -> None:
    for target in targets:
        print(format_target(target))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recapture canonical repros from saved repros/models full_graph_*.py files.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "full_graph_*.py files or directories under repros/models "
            "(default: all under repros/models)"
        ),
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help="Root containing saved model graph artifacts.",
    )
    parser.add_argument(
        "--canonical-root",
        "--output-dir",
        dest="canonical_root",
        type=Path,
        default=DEFAULT_CANONICAL_ROOT,
        help="Canonical repro root to write, containing canonical/ and models/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected graphs and inferred labels without loading or writing.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Alias for --dry-run.",
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        default=True,
        help="Run eager validation for generated region repros (default).",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip eager validation for generated region repros.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first load or recapture failure.",
    )
    args = parser.parse_args(argv)

    targets = discover_targets(args.paths, models_root=args.models_root)
    if args.dry_run or args.list_only:
        print_targets(targets)
        print(f"\nFound {len(targets)} full graph(s). No files written.")
        return 0

    print(f"Recapturing {len(targets)} full graph(s) into {args.canonical_root}")
    total_regions = 0
    failures = 0
    results = recapture_targets(
        targets,
        args.canonical_root,
        validate=args.validate,
        fail_fast=args.fail_fast,
    )
    for i, result in enumerate(results, start=1):
        target = result.target
        prefix = f"[{i}/{len(targets)}] {format_target(target)}"
        if result.ok:
            total_regions += result.regions
            print(f"{prefix} -> {result.regions} region(s)")
        else:
            failures += 1
            print(f"{prefix} -> FAILED: {result.error}")

    attempted = len(results)
    successes = attempted - failures
    print(
        f"\nDone: {total_regions} region(s) merged from "
        f"{successes}/{attempted} attempted graph(s)."
    )
    if attempted != len(targets):
        print(f"Stopped before {len(targets) - attempted} graph(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
