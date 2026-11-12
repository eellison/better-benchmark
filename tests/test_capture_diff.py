"""Static vs dynamic capture of the same function, diffed at the hint.

A dynamic family's static rows bench the dynamic graph at the captured
binding. These pin down where that matches a static capture and where it
doesn't (scripts/capture_diff.py reports the same for whole models).
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

from capture_diff import diff_captures  # noqa: E402
from test_dynamic_roundtrip import MLP, _capture, _mlp_args  # noqa: E402


def _diff(tmp_path, fn, args):
    static, dynamic = tmp_path / "static", tmp_path / "dynamic"
    _capture(fn, [args], static, name="m", dynamic=False)
    _capture(fn, [args], dynamic, name="m", dynamic=True)
    (report,) = diff_captures(static, dynamic).values()
    assert not report["unmatched_static"] and not report["unmatched_dynamic"], report
    return report["pairs"]


def _identical(pair):
    return not any(pair[k] for k in ("ops_only_static", "ops_only_dynamic",
                                     "inputs_only_static", "inputs_only_dynamic"))


def test_mlp_regions_match_a_static_capture(tmp_path):
    torch.manual_seed(0)
    pairs = _diff(tmp_path, MLP().eval(), _mlp_args(8))
    assert len(pairs) == 3 and all(map(_identical, pairs)), pairs


def test_permutation_generator_matches_a_static_capture(tmp_path):
    def f(perm, x):
        n = perm.shape[0]
        out = torch.empty(n, dtype=torch.int64).index_put_((perm,), torch.arange(n))
        return x[out] * 2

    pairs = _diff(tmp_path, f, (torch.randperm(16), torch.randn(16, 8)))
    assert all(map(_identical, pairs)), pairs


def test_scale_from_a_size_is_computed_at_runtime(tmp_path):
    """math.sqrt(q.shape[-2]) is a constant in a static capture and runtime
    ops on a SymInt in a dynamic one: same inputs, extra ops."""
    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(32, 32)

        def forward(self, x):
            q = self.lin(x)
            return ((q @ q.transpose(-1, -2)) / math.sqrt(q.shape[-2])).softmax(-1) @ q

    torch.manual_seed(0)
    (pair,) = _diff(tmp_path, Attn().eval(), (torch.randn(4, 24, 32),))
    assert not pair["inputs_only_static"] and not pair["inputs_only_dynamic"], pair
    assert not pair["ops_only_static"], pair
    runtime_scale = {"aten.sqrt.default", "aten.scalar_tensor.default"}
    assert runtime_scale <= set(pair["ops_only_dynamic"]), pair
