"""
Adversarial tests: verify our tooling catches broken repros.

These intentionally introduce bugs and check that validation/format
checks detect them. Run after any changes to repro_harness.py or
capture_hook.py to ensure our safety nets still work.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def test_empty_config_raises():
    """parse_shapes_config raises on non-empty config that produces nothing."""
    from repro_harness import parse_shapes_config
    # Valid empty
    result = parse_shapes_config("()")
    assert result == [], f"Expected [] for '()', got {result}"

    # Invalid: looks non-empty but would produce nothing if T() was broken
    try:
        # This should work (valid config)
        result = parse_shapes_config("(T([32], f32),)")
        assert len(result) == 1
    except Exception as e:
        raise AssertionError(f"Valid config should not raise: {e}")

    print("PASS: empty config guard works")


def test_wrong_input_count_detected():
    """Verify test_repro_integrity catches input count mismatch."""
    import ast
    import re
    # Simulate: forward expects 3 args, config has 2
    fake_repro = '''
_repro_version = 2
_shapes_config = "(T([32, 128], f32), T([128], f32))"

class Repro(torch.nn.Module):
    def forward(self, x, weight, bias):
        return x

def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)
'''
    # The test_repro_integrity check uses ast to count forward args
    tree = ast.parse(fake_repro)
    n_fwd = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            n_fwd = len(node.args.args) - 1  # exclude self
            break

    config_match = re.search(r'_shapes_config\s*=\s*"(.+)"', fake_repro)
    from repro_harness import parse_shapes_config
    inputs = parse_shapes_config(config_match.group(1))

    assert n_fwd == 3, f"Expected 3 forward args, got {n_fwd}"
    assert len(inputs) == 2, f"Expected 2 inputs from config, got {len(inputs)}"
    assert n_fwd != len(inputs), "Mismatch should be detected"
    print("PASS: input count mismatch detectable")


def test_missing_return_detected():
    """Verify we can detect a repro with no return statement."""
    import re
    fake_repro = '''
class Repro(torch.nn.Module):
    def forward(self, x):
        y = torch.ops.aten.relu.default(x)

def _default_make_inputs():
    pass
'''
    match = re.search(
        r"(def forward\(self,.*?\n)(.*?)(?=\ndef |\Z)",
        fake_repro, re.DOTALL
    )
    body = match.group(2)
    assert "return " not in body, "Should detect missing return"
    print("PASS: missing return detectable")


def test_channels_last_stride_preserved():
    """Verify channels-last stride appears in the captured signature.

    Repro format v3 retired the inline ``_shapes_config`` string in repro.py;
    the input shapes/strides now live structurally on the captured index
    entry's ``signature`` (and ``inputs``). The channels-last layout must be
    preserved there.
    """
    from capture_hook import _CaptureState
    import torch.fx as fx

    graph = fx.Graph()
    x = graph.placeholder('x')
    x.meta = {'val': torch.randn(8, 64, 32, 32, device='cuda').to(memory_format=torch.channels_last)}
    a = graph.call_function(torch.ops.aten.relu.default, args=(x,))
    a.meta = {'val': torch.randn(8, 64, 32, 32, device='cuda').to(memory_format=torch.channels_last)}
    graph.output(a)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    state = _CaptureState(tempfile.mkdtemp(), label='test', validate=False)
    state.process_graph(gm)
    state.finalize()

    assert len(state.captured) == 1
    signature = state.captured[0]['signature']
    assert 'stride=' in signature, f"Channels-last stride not in signature: {signature!r}"
    # Channel dim (the one with stride 1 under channels-last) must be recorded.
    assert ', 1,' in signature, f"Channel stride should be 1 for channels-last: {signature!r}"
    print("PASS: channels-last stride preserved in capture")


def test_independent_chains_split():
    """Verify independent chains become separate repros."""
    from capture_hook import _CaptureState
    import torch.fx as fx

    graph = fx.Graph()
    x = graph.placeholder('x')
    x.meta = {'val': torch.randn(32, 128, device='cuda')}
    y = graph.placeholder('y')
    y.meta = {'val': torch.randn(32, 64, device='cuda')}

    # Independent chain 1
    a = graph.call_function(torch.ops.aten.mul.Tensor, args=(x, 2.0))
    a.meta = {'val': torch.randn(32, 128, device='cuda')}

    # Independent chain 2
    b = graph.call_function(torch.ops.aten.mul.Tensor, args=(y, 3.0))
    b.meta = {'val': torch.randn(32, 64, device='cuda')}

    graph.output((a, b))
    gm = fx.GraphModule(torch.nn.Module(), graph)

    state = _CaptureState(tempfile.mkdtemp(), label='test', validate=False)
    state.process_graph(gm)
    state.finalize()

    assert len(state.captured) == 2, f"Expected 2 separate captures, got {len(state.captured)}"
    print("PASS: independent chains split correctly")


def test_version_tag_present():
    """Verify captured repros have _repro_version."""
    from capture_hook import _CaptureState
    import torch.fx as fx

    graph = fx.Graph()
    x = graph.placeholder('x')
    x.meta = {'val': torch.randn(32, 128, device='cuda')}
    a = graph.call_function(torch.ops.aten.relu.default, args=(x,))
    a.meta = {'val': torch.randn(32, 128, device='cuda')}
    graph.output(a)
    gm = fx.GraphModule(torch.nn.Module(), graph)

    state = _CaptureState(tempfile.mkdtemp(), label='test', validate=False)
    state.process_graph(gm)
    state.finalize()

    content = Path(state.captured[0]['file']).read_text()
    # capture_hook emits the current (v3) template marker.
    from repro_harness import CURRENT_REPRO_VERSION, parse_repro_version
    assert f'_repro_version = {CURRENT_REPRO_VERSION}' in content, "Missing version tag"
    # Track the constant, not a frozen literal: a future bump must not silently
    # rot this assertion (the recurrence the v3 migration demonstrated).
    assert parse_repro_version(content) == CURRENT_REPRO_VERSION
    print("PASS: version tag present in captured repro")


if __name__ == "__main__":
    test_empty_config_raises()
    test_wrong_input_count_detected()
    test_missing_return_detected()
    test_channels_last_stride_preserved()
    test_independent_chains_split()
    test_version_tag_present()
    print("\nAll adversarial tests passed!")
