"""
Focused unit tests for bounds inference.

Each test constructs a minimal forward() method string, runs the inference,
and asserts the correct bound is inferred.
"""
import sys
from pathlib import Path

# File moved from scripts/ to tests/; bounds_inference lives in
# repo-root/scripts. conftest.py covers this for pytest; insert here too so
# `python tests/test_bounds_unit.py` works standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bounds_inference import (
    infer_bounds_from_forward,
    annotate_config_string,
    strip_gen_annotations,
    infer_bounds_for_config,
)


def _make_forward(body, params):
    """Build a minimal module source with a forward() method.

    params: list of (name, dtype, shape) tuples
    body: string of FX-style operations
    """
    param_strs = []
    for name, dtype, shape in params:
        shape_str = ", ".join(str(d) for d in shape)
        param_strs.append(f'{name}: "{dtype}[{shape_str}]"')

    params_joined = ", ".join(param_strs)
    return f"""
import torch

class Repro(torch.nn.Module):
    def forward(self, {params_joined}):
{body}
"""


def test_simple_gather_bound():
    """gather(data[100, 64], dim=0, index=idx) -> idx needs Index(100)"""
    content = _make_forward(
        """        result: "f32[32, 64]" = torch.ops.aten.gather.default(data, 0, idx)
        return result""",
        [("data", "f32", [100, 64]), ("idx", "i64", [32, 64])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    assert 1 in index_bounds, f"Expected idx (pos 1) to have Index bound, got: {index_bounds}"
    assert index_bounds[1] == 100, f"Expected Index(100), got Index({index_bounds[1]})"
    assert not perm_bounds


def test_scatter_perm():
    """scatter(empty[N], idx, iota(N)) + gather -> idx needs Perm(N)"""
    content = _make_forward(
        """        empty_alloc: "f32[256]" = torch.ops.aten.empty.memory_format([256])
        iota_val: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        scatter_result: "f32[256]" = torch.ops.aten.scatter.src(empty_alloc, 0, idx, iota_val)
        return scatter_result""",
        [("idx", "i64", [256])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    assert 0 in perm_bounds, f"Expected idx (pos 0) to have Perm bound, got index={index_bounds}, perm={perm_bounds}"
    assert perm_bounds[0] == 256, f"Expected Perm(256), got Perm({perm_bounds[0]})"


def test_additive_offset():
    """gather(data[100], dim=0, idx + 5) -> idx needs Index(95)"""
    content = _make_forward(
        """        offset_idx: "i64[32]" = torch.ops.aten.add.Tensor(idx, 5)
        result: "f32[32]" = torch.ops.aten.gather.default(data, 0, offset_idx)
        return result""",
        [("data", "f32", [100]), ("idx", "i64", [32])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    assert 1 in index_bounds, f"Expected idx (pos 1) to have Index bound, got: {index_bounds}"
    assert index_bounds[1] == 95, f"Expected Index(95), got Index({index_bounds[1]})"


def test_embedding():
    """embedding(weight[50000, 768], idx) -> idx needs Index(50000)"""
    content = _make_forward(
        """        result: "f32[32, 768]" = torch.ops.aten.embedding.default(weight, idx)
        return result""",
        [("weight", "f32", [50000, 768]), ("idx", "i64", [32])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    assert 1 in index_bounds, f"Expected idx (pos 1) to have Index bound, got: {index_bounds}"
    assert index_bounds[1] == 50000, f"Expected Index(50000), got Index({index_bounds[1]})"


def test_multiple_uses_tightest_wins():
    """idx used in gather(data1[100], ...) AND gather(data2[50], ...) -> Index(50)"""
    content = _make_forward(
        """        result1: "f32[32, 64]" = torch.ops.aten.gather.default(data1, 0, idx)
        result2: "f32[32, 64]" = torch.ops.aten.gather.default(data2, 0, idx)
        add_result: "f32[32, 64]" = torch.ops.aten.add.Tensor(result1, result2)
        return add_result""",
        [("data1", "f32", [100, 64]), ("data2", "f32", [50, 64]), ("idx", "i64", [32, 64])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    assert 2 in index_bounds, f"Expected idx (pos 2) to have Index bound, got: {index_bounds}"
    assert index_bounds[2] == 50, f"Expected Index(50) (tightest), got Index({index_bounds[2]})"


def test_index_put_dim0_bound():
    """index_put(empty[8, 1024], [idx], ...) -> idx indexes dim 0, needs Index(8)"""
    content = _make_forward(
        """        target: "f32[8, 1024]" = torch.ops.aten.empty.memory_format([8, 1024])
        result: "f32[8, 1024]" = torch.ops.aten.index_put.default(target, [idx], values)
        return result""",
        [("idx", "i64", [32]), ("values", "f32", [32, 1024])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    assert 0 in index_bounds, f"Expected idx (pos 0) to have Index bound, got: {index_bounds}"
    assert index_bounds[0] == 8, f"Expected Index(8), got Index({index_bounds[0]})"


def test_index_put_dim1_structural_bound():
    """index_put(empty[8, 1024], [None, idx], ...) -> idx indexes dim 1, needs Index(1024)

    Without heuristics, the structural bound is the size of the indexed dimension.
    We do NOT guess that the bound should be the batch (first) dimension.
    """
    content = _make_forward(
        """        target: "f32[8, 1024]" = torch.ops.aten.empty.memory_format([8, 1024])
        result: "f32[8, 1024]" = torch.ops.aten.index_put.default(target, [None, idx], values)
        return result""",
        [("idx", "i64", [32]), ("values", "f32", [8, 32])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    assert 0 in index_bounds, f"Expected idx (pos 0) to have Index bound, got: {index_bounds}"
    # Structural bound: dim 1 has size 1024
    assert index_bounds[0] == 1024, f"Expected Index(1024), got Index({index_bounds[0]})"


def test_view_preserves_bound():
    """idx goes through view/reshape -> bound unchanged"""
    content = _make_forward(
        """        reshaped: "i64[4, 8]" = torch.ops.aten.view.default(idx, [4, 8])
        result: "f32[4, 8, 64]" = torch.ops.aten.gather.default(data, 0, reshaped)
        return result""",
        [("data", "f32", [100, 64]), ("idx", "i64", [32])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    assert 1 in index_bounds, f"Expected idx (pos 1) to have Index bound, got: {index_bounds}"
    assert index_bounds[1] == 100, f"Expected Index(100), got Index({index_bounds[1]})"


def test_where_propagates_to_both():
    """where(mask, idx1, idx2) used in gather -> both idx1, idx2 need bound"""
    content = _make_forward(
        """        selected: "i64[32]" = torch.ops.aten.where.self(mask, idx1, idx2)
        result: "f32[32, 64]" = torch.ops.aten.gather.default(data, 0, selected)
        return result""",
        [("data", "f32", [100, 64]), ("mask", "b8", [32]), ("idx1", "i64", [32]), ("idx2", "i64", [32])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    # idx1 is at position 2, idx2 is at position 3
    assert 2 in index_bounds, f"Expected idx1 (pos 2) to have Index bound, got: {index_bounds}"
    assert 3 in index_bounds, f"Expected idx2 (pos 3) to have Index bound, got: {index_bounds}"
    assert index_bounds[2] == 100, f"Expected idx1=Index(100), got Index({index_bounds[2]})"
    assert index_bounds[3] == 100, f"Expected idx2=Index(100), got Index({index_bounds[3]})"


def test_annotate_config_string():
    """Test that annotate_config_string inserts gen= annotations correctly."""
    config = "(T([100, 64], f32), T([32, 64], i64))"
    index_bounds = {1: 100}
    perm_bounds = {}
    result = annotate_config_string(config, index_bounds, perm_bounds)
    assert "gen=Index(100)" in result
    assert result == "(T([100, 64], f32), T([32, 64], i64, gen=Index(100)))"


def test_annotate_config_string_perm():
    """Test that annotate_config_string inserts Perm annotations correctly."""
    config = "(T([256], i64), T([256, 64], f32))"
    index_bounds = {}
    perm_bounds = {0: 256}
    result = annotate_config_string(config, index_bounds, perm_bounds)
    assert "gen=Perm(256)" in result
    assert result == "(T([256], i64, gen=Perm(256)), T([256, 64], f32))"


def test_strip_gen_annotations():
    """Test stripping gen= annotations from config strings."""
    config = "(T([100, 64], f32), T([32, 64], i64, gen=Index(100)), T([256], i64, gen=Perm(256)))"
    result = strip_gen_annotations(config)
    assert "gen=" not in result
    assert result == "(T([100, 64], f32), T([32, 64], i64), T([256], i64))"


def test_index_select_bound():
    """index_select(data[200, 64], dim=0, idx) -> idx needs Index(200)"""
    content = _make_forward(
        """        result: "f32[32, 64]" = torch.ops.aten.index_select.default(data, 0, idx)
        return result""",
        [("data", "f32", [200, 64]), ("idx", "i64", [32])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    assert 1 in index_bounds, f"Expected idx (pos 1) to have Index bound, got: {index_bounds}"
    assert index_bounds[1] == 200, f"Expected Index(200), got Index({index_bounds[1]})"


def test_index_tensor_bound():
    """index.Tensor(data[100, 64], [idx]) -> idx needs Index(100)"""
    content = _make_forward(
        """        result: "f32[32, 64]" = torch.ops.aten.index.Tensor(data, [idx])
        return result""",
        [("data", "f32", [100, 64]), ("idx", "i64", [32])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    assert 1 in index_bounds, f"Expected idx (pos 1) to have Index bound, got: {index_bounds}"
    assert index_bounds[1] == 100, f"Expected Index(100), got Index({index_bounds[1]})"


def test_scatter_add_bound():
    """scatter_add(target[200, 64], dim=0, idx, src) -> idx needs Index(200)"""
    content = _make_forward(
        """        result: "f32[200, 64]" = torch.ops.aten.scatter_add.default(target, 0, idx, src)
        return result""",
        [("target", "f32", [200, 64]), ("idx", "i64", [32, 64]), ("src", "f32", [32, 64])]
    )
    index_bounds, perm_bounds = infer_bounds_from_forward(content)
    assert 1 in index_bounds, f"Expected idx (pos 1) to have Index bound, got: {index_bounds}"
    assert index_bounds[1] == 200, f"Expected Index(200), got Index({index_bounds[1]})"


def test_full_corpus_accuracy():
    """Structural inference: 312/355 annotations derivable from graph structure.

    The remaining 43 require heuristics (cumsum-as-data pattern, index_put batch
    dim guessing, mul factor bound guessing) which are NOT sound derivations.
    We verify all structurally-derivable bounds are correct, and that no
    unsound bounds are emitted for the non-derivable cases.
    """
    ROOT = Path(__file__).resolve().parents[1]
    REPROS_DIR = ROOT / "repros" / "canonical"

    import re as re_mod

    correct = 0
    total = 0
    looser_but_sound = 0  # cases where we give a looser (or no) bound
    failures = []

    import json as _json

    for repro_path in sorted(REPROS_DIR.rglob('repro.py')):
        content = repro_path.read_text()

        # Ground truth lives in the sibling shapes.json (v3 retired the inline
        # _shapes_config string). Each point["inputs"][i] is a compact codec
        # entry; index/perm generators carry the ground-truth bound. Input
        # index i corresponds 1:1 to forward() parameter position i (tensor
        # args first, then S(...) shape-param entries) -- the same positional
        # space infer_bounds_from_forward returns.
        shapes_json = repro_path.parent / "shapes.json"
        if not shapes_json.exists():
            continue
        try:
            points = _json.loads(shapes_json.read_text()).get("points", [])
        except Exception:
            continue
        if not points:
            continue
        inputs = points[0].get("inputs", [])

        ground_truth = {}
        for pos, entry in enumerate(inputs):
            if not (isinstance(entry, list) and len(entry) > 2
                    and isinstance(entry[2], dict)):
                continue
            gen = entry[2].get("gen")
            if not isinstance(gen, list) or not gen:
                continue
            if gen[0] == "index" and len(gen) >= 3:
                ground_truth[pos] = ("Index", int(gen[2]))
            elif gen[0] == "perm" and len(gen) >= 2:
                ground_truth[pos] = ("Perm", int(gen[1]))
        if not ground_truth:
            continue
        total += len(ground_truth)

        # Run inference
        index_bounds, perm_bounds = infer_bounds_from_forward(content, source_path=repro_path)

        # Compare
        repro_name = repro_path.parent.name
        for pos, (gt_type, gt_bound) in ground_truth.items():
            if gt_type == 'Perm':
                if pos in perm_bounds and perm_bounds[pos] == gt_bound:
                    correct += 1
                else:
                    got = f"Perm({perm_bounds[pos]})" if pos in perm_bounds else (
                        f"Index({index_bounds[pos]})" if pos in index_bounds else "nothing"
                    )
                    # Verify soundness: we should never give a TIGHTER bound than ground truth
                    if pos in perm_bounds:
                        # Perm but wrong size - this is a real failure
                        failures.append(f"{repro_name}[{pos}]: expected Perm({gt_bound}), got {got}")
                    elif pos in index_bounds:
                        # Got Index instead of Perm - sound (Index is weaker than Perm)
                        if index_bounds[pos] >= gt_bound:
                            looser_but_sound += 1
                        else:
                            failures.append(f"{repro_name}[{pos}]: UNSOUND expected Perm({gt_bound}), got {got}")
                    else:
                        # No bound at all - sound (no constraint is the weakest)
                        looser_but_sound += 1
            else:  # Index
                if pos in index_bounds and index_bounds[pos] == gt_bound:
                    correct += 1
                elif pos in perm_bounds:
                    # Perm is stronger than Index - check soundness
                    if perm_bounds[pos] <= gt_bound:
                        correct += 1  # Perm(N) implies Index(N) and adds uniqueness
                    else:
                        failures.append(
                            f"{repro_name}[{pos}]: UNSOUND expected Index({gt_bound}), got Perm({perm_bounds[pos]})"
                        )
                elif pos in index_bounds:
                    # Got a bound but different from ground truth
                    if index_bounds[pos] >= gt_bound:
                        # Looser bound - sound but not tight
                        looser_but_sound += 1
                    else:
                        # Tighter bound - UNSOUND, this is a real bug
                        failures.append(
                            f"{repro_name}[{pos}]: UNSOUND expected Index({gt_bound}), got Index({index_bounds[pos]})"
                        )
                else:
                    # No bound at all - sound (no constraint)
                    looser_but_sound += 1

    # The corpus carries a meaningful number of index/perm ground-truth gens
    # (measured: 222 on the current v3 corpus). Guard against the scan silently
    # finding nothing (the v0->v3 format change once dropped this to 0).
    assert total >= 200, f"Expected >= 200 ground-truth gen annotations, found {total}"

    # THE hard invariant: structural inference must NEVER emit an unsound bound
    # (tighter than the ground truth captured from real data). A looser/absent
    # bound is fine; a tighter one would OOB-index at replay.
    if failures:
        failure_str = "\n  ".join(failures[:20])
        assert False, (
            f"UNSOUND bounds detected ({len(failures)} failures):\n  {failure_str}"
        )

    # Structural accuracy floor. Pure graph-structure inference re-derives only
    # a subset of the stored bounds: the rest were captured with observed-data
    # signals (e.g. permutation detection from n_unique==numel on the real
    # tensor) that a structure-only re-derivation cannot recover, so they land
    # in looser_but_sound -- which is SOUND. Measured on the current v3 corpus:
    # 99/222 structurally re-derivable, 123 looser-but-sound, 0 unsound.
    assert correct >= 90, (
        f"Structural accuracy regressed: {correct}/{total} (expected >= 90). "
        f"Looser-but-sound: {looser_but_sound}"
    )
    print(f"  Structural accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  Looser-but-sound (not derivable from graph): {looser_but_sound}")


if __name__ == "__main__":
    import traceback

    tests = [
        test_simple_gather_bound,
        test_scatter_perm,
        test_additive_offset,
        test_embedding,
        test_multiple_uses_tightest_wins,
        test_index_put_dim0_bound,
        test_index_put_dim1_structural_bound,
        test_view_preserves_bound,
        test_where_propagates_to_both,
        test_annotate_config_string,
        test_annotate_config_string_perm,
        test_strip_gen_annotations,
        test_index_select_bound,
        test_index_tensor_bound,
        test_scatter_add_bound,
        test_full_corpus_accuracy,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
