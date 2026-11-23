"""Exercise the built package without tests/conftest.py exposing the checkout."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def test_runtime_modules_import_from_wheel(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    for path in [root / "pyproject.toml", *root.glob("*.py")]:
        shutil.copy2(path, source / path.name)

    build = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('dist')",
        ],
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheel, = (source / "dist").glob("*.whl")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)

    # -I and the temporary cwd prevent the checkout/PYTHONPATH from hiding
    # missing modules. Check origins too, in case an editable copy is installed.
    check = subprocess.run(
        [sys.executable, "-I", "-c", """
import importlib
import sys
from pathlib import Path

installed = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installed))
for name in (
    'repro_harness', 'oracle_harness', 'full_graph_harness',
    'capture_hook', 'merge_captures', 'roundtrip_validation',
    'dynamic_shape_replay', 'dynamic_benchmark',
    'canonicalize_repros', 'fusion_partitioner',
):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(installed), (
        name, module.__file__
    )
""", str(installed)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr
