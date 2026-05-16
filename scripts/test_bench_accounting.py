import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench import _count_bytes, _resolve_input_path


def test_count_bytes_deduplicates_output_aliases():
    inp = torch.empty(4, dtype=torch.float32)
    out = torch.empty(16, dtype=torch.float32)

    assert _count_bytes([inp], (out, out.view(4, 4), out.reshape(2, 8))) == (
        inp.nelement() + out.nelement()
    ) * inp.element_size()


def test_count_bytes_counts_distinct_outputs():
    inp = torch.empty(4, dtype=torch.float32)
    out0 = torch.empty(16, dtype=torch.float32)
    out1 = torch.empty(16, dtype=torch.float32)

    assert _count_bytes([inp], (out0, out1)) == (
        inp.nelement() + out0.nelement() + out1.nelement()
    ) * inp.element_size()


def test_resolve_input_path_strips_accidental_whitespace():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repro = tmp_path / "repro.py"
        repro.write_text("# repro\n")
        assert _resolve_input_path(f" {repro}") == repro


def test_resolve_input_path_rejects_missing_path():
    missing = " models/does/not/exist/repro.py"
    try:
        _resolve_input_path(missing)
    except FileNotFoundError as exc:
        assert "Benchmark path does not exist" in str(exc)
    else:
        raise AssertionError("missing benchmark path should fail")


def test_resolve_input_path_rejects_whitespace_only_path():
    try:
        _resolve_input_path(" ")
    except FileNotFoundError as exc:
        assert "Benchmark path does not exist" in str(exc)
    else:
        raise AssertionError("whitespace-only benchmark path should fail")


if __name__ == "__main__":
    test_count_bytes_deduplicates_output_aliases()
    test_count_bytes_counts_distinct_outputs()
    test_resolve_input_path_strips_accidental_whitespace()
    test_resolve_input_path_rejects_missing_path()
    test_resolve_input_path_rejects_whitespace_only_path()
    print("All tests passed.")
