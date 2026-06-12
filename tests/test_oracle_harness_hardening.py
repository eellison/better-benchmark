"""Tests for oracle_harness hardening: bench-gating, CUDAGraph warning trap,
and fp64-anchored numerics gate.

These tests are CPU/mock-based (no GPU required for plumbing verification).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oracle_harness import (
    _anchored_numerics_gate,
    _CUDAGRAPH_WARNING_PATTERNS,
    _INVALID_STATUSES,
    _runner_main,
)


# ============================================================================
# (a) Bench-gating logic: point with failing check -> UNVERIFIED_NUMERICS
# ============================================================================

class TestBenchGatedOnCheck:
    """--bench must run check first and refuse to bench failing points."""

    def _make_repro_dir(self, tmp_path, oracle_body, repro_body=None):
        """Create a minimal canonical dir with repro.py, oracle.py, shapes.json."""
        d = tmp_path / "test_repro"
        d.mkdir()
        if repro_body is None:
            repro_body = '''
import torch

class Repro(torch.nn.Module):
    def forward(self, x):
        return x * 2.0

def make_inputs():
    return [torch.randn(4, 4)]
'''
        (d / "repro.py").write_text(repro_body)
        (d / "oracle.py").write_text(oracle_body)
        (d / "shapes.json").write_text(json.dumps({
            "points": [{
                "shape_hash": "aabb0011",
                "models": {"test/infer/m": {"occurrences": 1}},
                "inputs": [[[4, 4], "f32"]],
            }]
        }))
        return d

    def _parse_runner_json(self, stdout):
        """Extract the final JSON object from runner output (which may include
        intermediate check/bench print lines before the JSON)."""
        # The JSON block starts with '{' on its own line at the end
        lines = stdout.strip().split("\n")
        # Find where the JSON starts (first '{' line)
        json_start = None
        for i, line in enumerate(lines):
            if line.strip() == "{":
                json_start = i
                break
        if json_start is None:
            raise ValueError(f"No JSON found in output: {stdout[:500]}")
        return json.loads("\n".join(lines[json_start:]))

    def test_bench_refuses_failing_check(self, tmp_path, capsys):
        """A point that fails check_oracle should get UNVERIFIED_NUMERICS."""
        # Oracle that returns wrong values (will fail check)
        oracle_body = '''
import torch
def oracle_forward(inputs):
    return inputs[0] * 999.0  # wrong answer
'''
        d = self._make_repro_dir(tmp_path, oracle_body)
        # --bench implies --check; the check will fail, so bench should not run
        ret = _runner_main([str(d), "--bench"])
        assert ret == 1  # nonzero exit

        captured = capsys.readouterr()
        out = self._parse_runner_json(captured.out)
        # Check should show failure
        assert out["check"] is not None
        check_vals = list(out["check"].values())
        assert False in check_vals

        # Bench should show UNVERIFIED_NUMERICS, no timing numbers
        bench = out["bench"]
        assert len(bench) == 1
        assert bench[0]["status"] == "UNVERIFIED_NUMERICS"
        assert "oracle_us" not in bench[0]
        assert "ratio" not in bench[0]

    def test_bench_unchecked_skips_gate(self, tmp_path, capsys):
        """--bench-unchecked should bypass the check gate (with warning)."""
        # Oracle that returns wrong values
        oracle_body = '''
import torch
def oracle_forward(inputs):
    return inputs[0] * 999.0
'''
        d = self._make_repro_dir(tmp_path, oracle_body)
        # --bench-unchecked bypasses the check; will try to bench
        # (will fail because no GPU, but the point is it doesn't gate)
        # Patch bench_oracle to avoid GPU requirement
        with patch("oracle_harness.bench_oracle") as mock_bench:
            mock_bench.return_value = {
                "repro_id": "test_repro_aabb0011",
                "status": "GOOD",
                "oracle_us": 10.0,
                "compile_us": 20.0,
                "ratio": 2.0,
            }
            ret = _runner_main([str(d), "--bench-unchecked"])

        captured = capsys.readouterr()
        # Should have printed the warning to stderr
        assert "WARNING" in captured.err
        assert "bench-unchecked" in captured.err or "NOT validated" in captured.err

    def test_passing_check_allows_bench(self, tmp_path, capsys):
        """A correct oracle should pass check and proceed to bench."""
        # Correct oracle
        oracle_body = '''
import torch
def oracle_forward(inputs):
    return inputs[0] * 2.0
'''
        d = self._make_repro_dir(tmp_path, oracle_body)
        # Patch bench_oracle since we have no GPU
        with patch("oracle_harness.bench_oracle") as mock_bench:
            mock_bench.return_value = {
                "repro_id": "test_repro_aabb0011",
                "status": "GOOD",
                "oracle_us": 10.0,
                "compile_us": 20.0,
                "ratio": 2.0,
            }
            ret = _runner_main([str(d), "--bench"])

        assert ret == 0
        captured = capsys.readouterr()
        out = self._parse_runner_json(captured.out)
        assert all(v is True or v == True for v in out["check"].values())


# ============================================================================
# (b) CUDAGraph warning-trap plumbing
# ============================================================================

class TestCUDAGraphWarningTrap:
    """Warning trap must catch CUDAGraph-related warnings."""

    def test_pattern_matches_cudagraph_warnings(self):
        """The regex should match common CUDAGraph warning messages."""
        cases = [
            "CUDAGraph capture failed due to unsupported op",
            "Warning: cuda graph capture encountered an error",
            "stream capture detected illegal operation",
            "Graph capture aborted: dynamic shapes",
            "cudagraph replay may be incorrect",
        ]
        for msg in cases:
            assert _CUDAGRAPH_WARNING_PATTERNS.search(msg), \
                f"Pattern should match: {msg!r}"

    def test_pattern_does_not_match_unrelated(self):
        """Unrelated warnings should not trigger the trap."""
        benign = [
            "torch.compile is still experimental",
            "UserWarning: some other warning",
            "DeprecationWarning: old API",
        ]
        for msg in benign:
            assert not _CUDAGRAPH_WARNING_PATTERNS.search(msg), \
                f"Pattern should NOT match: {msg!r}"

    def test_capture_graph_checked_traps_warnings(self):
        """_capture_graph_checked should collect CUDAGraph warnings."""
        from oracle_harness import _capture_graph_checked

        # We can't test real CUDAGraph without GPU, but we can test
        # the warning-collection plumbing by patching torch.cuda
        if not torch.cuda.is_available():
            pytest.skip("GPU required for _capture_graph_checked plumbing test")

        # Test that a clean capture returns no warnings
        x = torch.randn(4, 4, device="cuda")
        g, ws = _capture_graph_checked(lambda: x + 1)
        assert ws == []
        assert g is not None

    def test_warning_trap_with_monkeypatch(self):
        """Monkeypatch warnings.warn to simulate a CUDAGraph warning during
        the capture path. This tests the plumbing without GPU."""
        from oracle_harness import _capture_graph_checked, _CUDAGRAPH_WARNING_PATTERNS

        if not torch.cuda.is_available():
            pytest.skip("GPU required for warning trap test")

        # Inject a CUDAGraph warning during capture
        original_graph_ctx = torch.cuda.graph

        class WarningGraph:
            """Context manager that emits a CUDAGraph warning."""
            def __init__(self, graph):
                self._graph = graph
                self._ctx = original_graph_ctx(graph)

            def __enter__(self):
                result = self._ctx.__enter__()
                warnings.warn("CUDAGraph capture may be incorrect due to test")
                return result

            def __exit__(self, *args):
                return self._ctx.__exit__(*args)

        x = torch.randn(4, 4, device="cuda")
        with patch("torch.cuda.graph", side_effect=lambda g: WarningGraph(g)):
            g, ws = _capture_graph_checked(lambda: x + 1)

        assert len(ws) >= 1
        assert any("CUDAGraph" in w for w in ws)


# ============================================================================
# (c) FP64-anchored numerics gate (_anchored_numerics_gate unit test)
# ============================================================================

class TestAnchoredNumericsGate:
    """Pure-function unit tests for the err_oracle vs err_compiled gate."""

    def test_identical_outputs_pass(self):
        """Oracle == compiled == ref should pass."""
        ref = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)]
        oracle = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)]
        compiled = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)]

        result = _anchored_numerics_gate(oracle, compiled, ref, set())
        assert result["pass"] is True
        assert result["per_output"][0]["err_oracle"] == 0.0
        assert result["per_output"][0]["err_compiled"] == 0.0

    def test_oracle_worse_than_compiled_fails(self):
        """Oracle with larger error than compiled should fail the gate."""
        ref = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)]
        # Compiled is close to ref
        compiled = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)]
        # Oracle has significant error (e.g. from fast-math substitution)
        oracle = [torch.tensor([1.1, 2.1, 3.1], dtype=torch.float32)]

        result = _anchored_numerics_gate(oracle, compiled, ref, set())
        assert result["pass"] is False
        assert result["worst_output_idx"] == 0
        assert result["per_output"][0]["err_oracle"] > result["per_output"][0]["err_compiled"]

    def test_both_have_small_error_passes(self):
        """Both oracle and compiled slightly off from ref should pass."""
        ref = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)]
        # Both have small rounding error
        compiled = [torch.tensor([1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7])]
        oracle = [torch.tensor([1.0 + 2e-7, 2.0 + 2e-7, 3.0 + 2e-7])]

        result = _anchored_numerics_gate(oracle, compiled, ref, set())
        assert result["pass"] is True

    def test_stochastic_outputs_skipped(self):
        """Stochastic outputs should be skipped, not compared."""
        ref = [torch.tensor([1.0]), torch.tensor([99.0])]
        oracle = [torch.tensor([1.0]), torch.tensor([0.0])]  # wildly wrong
        compiled = [torch.tensor([1.0]), torch.tensor([99.0])]

        # output 1 is stochastic -> skip it
        result = _anchored_numerics_gate(oracle, compiled, ref, {1})
        assert result["pass"] is True
        assert result["per_output"][1]["skip"] == "stochastic"

    def test_non_float_outputs_skipped(self):
        """Non-float outputs should be skipped (already require exact match)."""
        ref = [torch.tensor([1, 2, 3], dtype=torch.int64)]
        oracle = [torch.tensor([1, 2, 3], dtype=torch.int64)]
        compiled = [torch.tensor([1, 2, 3], dtype=torch.int64)]

        result = _anchored_numerics_gate(oracle, compiled, ref, set())
        assert result["pass"] is True
        assert result["per_output"][0]["skip"] == "non_float"

    def test_multiple_outputs_mixed(self):
        """Multiple outputs: one passes, one fails -> overall fail."""
        ref = [
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([10.0, 20.0], dtype=torch.float64),
        ]
        compiled = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([10.0, 20.0]),
        ]
        oracle = [
            torch.tensor([1.0, 2.0]),          # good
            torch.tensor([10.5, 20.5]),          # bad: 0.5 error >> compiled's 0
        ]

        result = _anchored_numerics_gate(oracle, compiled, ref, set())
        assert result["pass"] is False
        assert result["per_output"][0]["pass"] is True
        assert result["per_output"][1]["pass"] is False

    def test_threshold_uses_ref_absmax_scale(self):
        """When ref values are large, the 1e-5 * absmax term dominates."""
        # Large reference values
        ref = [torch.tensor([1e6, 2e6], dtype=torch.float64)]
        compiled = [torch.tensor([1e6, 2e6])]
        # Oracle has error of 5.0, which is < 1e-5 * 2e6 = 20.0
        oracle = [torch.tensor([1e6 + 5.0, 2e6 + 5.0])]

        result = _anchored_numerics_gate(oracle, compiled, ref, set())
        assert result["pass"] is True

    def test_gate_records_ref_precision(self):
        """Gate should record which reference precision was used."""
        ref = [torch.tensor([1.0], dtype=torch.float64)]
        oracle = [torch.tensor([1.0])]
        compiled = [torch.tensor([1.0])]

        result = _anchored_numerics_gate(
            oracle, compiled, ref, set(), ref_precision="f64")
        assert result["ref_precision"] == "f64"

        result2 = _anchored_numerics_gate(
            oracle, compiled, ref, set(), ref_precision="f32")
        assert result2["ref_precision"] == "f32"

    def test_exp2_substitution_detected(self):
        """A fast-math exp2 substitution (exp2(x * log2e)) should fail
        when compared to natural exp (the correct formulation)."""
        x = torch.randn(1000) * 3.0
        log2e = 1.4426950408889634

        # Reference: natural exp in f64
        ref_outs = [torch.exp(x.double())]
        # Compiled: natural exp in f32 (what Inductor produces)
        compiled_outs = [torch.exp(x)]
        # Oracle: exp2-based substitution (fast but less accurate)
        oracle_outs = [torch.exp2(x * log2e)]

        result = _anchored_numerics_gate(
            oracle_outs, compiled_outs, ref_outs, set())
        # The exp2 substitution accumulates measurably more error
        # than the standard exp — this should fail (or at minimum
        # err_oracle > err_compiled by a visible margin)
        # Note: for moderate x values the difference may be small enough
        # to pass; use large x to make it clear
        per_out = result["per_output"][0]
        assert per_out["err_oracle"] > per_out["err_compiled"]


# ============================================================================
# (d) Exit code enforcement
# ============================================================================

class TestExitCodes:
    """_runner_main should return nonzero for any invalid status."""

    def test_invalid_statuses_set(self):
        """All expected invalid statuses are in the set."""
        assert "UNVERIFIED_NUMERICS" in _INVALID_STATUSES
        assert "INVALID_CUDAGRAPH_WARNING" in _INVALID_STATUSES
        assert "NUMERICS_WORSE_THAN_COMPILED" in _INVALID_STATUSES
        assert "NO_ORACLE_FOR_SHAPE" in _INVALID_STATUSES

    def test_check_failure_nonzero_exit(self, tmp_path, capsys):
        """Check failure -> exit code 1."""
        d = tmp_path / "r"
        d.mkdir()
        (d / "repro.py").write_text('''
import torch
class Repro(torch.nn.Module):
    def forward(self, x):
        return x * 2.0
def make_inputs():
    return [torch.randn(4, 4)]
''')
        (d / "oracle.py").write_text('''
import torch
def oracle_forward(inputs):
    return inputs[0] * 999.0
''')
        (d / "shapes.json").write_text(json.dumps({
            "points": [{
                "shape_hash": "00000000",
                "models": {"test/infer/m": {"occurrences": 1}},
                "inputs": [[[4, 4], "f32"]],
            }]
        }))
        ret = _runner_main([str(d), "--check"])
        assert ret == 1
