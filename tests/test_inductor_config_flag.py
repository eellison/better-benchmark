"""Unit tests for the --inductor-config parse/validate helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_parallel import _parse_inductor_config, _unknown_inductor_config_names


def test_parse_literals():
    got = _parse_inductor_config([
        "combo_kernels=True",
        "combo_kernel_max_num_nodes=16",
        "some_ratio=1.5",
        "some_opt=None",
        'name="hello"',
    ])
    assert got == {
        "combo_kernels": True,
        "combo_kernel_max_num_nodes": 16,
        "some_ratio": 1.5,
        "some_opt": None,
        "name": "hello",
    }


def test_parse_raw_string_fallback():
    # Not a valid python literal -> kept as the raw string.
    assert _parse_inductor_config(["mode=max-autotune"]) == {"mode": "max-autotune"}


def test_parse_dotted_name():
    assert _parse_inductor_config(["triton.multi_kernel=1"]) == {"triton.multi_kernel": 1}


def test_parse_value_keeps_later_equals():
    # Only the first '=' splits name from value.
    assert _parse_inductor_config(["k=a=b"]) == {"k": "a=b"}


@pytest.mark.parametrize("bad", ["noequals", "=1", "1bad=2", "a b=1"])
def test_parse_errors(bad):
    with pytest.raises(ValueError):
        _parse_inductor_config([bad])


def test_unknown_names_flagged():
    # A real knob is accepted; a bogus leaf and a bogus dotted parent are flagged.
    unknown = _unknown_inductor_config_names(
        {"combo_kernels": True, "definitely_not_a_knob": 1, "trion.multi_kernel": 1}
    )
    assert "combo_kernels" not in unknown
    assert set(unknown) == {"definitely_not_a_knob", "trion.multi_kernel"}
