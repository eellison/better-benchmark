import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernel_sensitivity import (
    ARMS,
    INDEX_ARMS,
    index_profile,
    make_indices,
    JaggedInput,
    resolve_tier,
    structure_lengths,
    to_tensor,
    verdict,
)


KV = JaggedInput("kv", "offsets", 1024, 17_629_228, 40_000, torch.int64)
Q = JaggedInput("q", "offsets", 1024, 199_424, 480, torch.int32, pair_with="kv")


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_keeps_count_total_and_cap(arm):
    lengths = structure_lengths([KV, Q], arm)
    for spec in (KV, Q):
        got = lengths[spec.name]
        assert got.numel() == spec.count
        assert int(got.sum()) == spec.total
        assert int(got.max()) <= spec.max_len
        assert int(got.min()) >= (0 if arm == "uniform" else 1)


def test_arms_are_reproducible_and_differ():
    a = structure_lengths([KV], "lognormal2", seed=3)["kv"]
    assert torch.equal(a, structure_lengths([KV], "lognormal2", seed=3)["kv"])
    assert not torch.equal(a, structure_lengths([KV], "lognormal2", seed=4)["kv"])
    skew = structure_lengths([KV], "max_skew")["kv"]
    assert int((skew == KV.max_len).sum()) == (KV.total - KV.count) // (KV.max_len - 1)
    assert int(structure_lengths([KV], "balanced")["kv"].max()) - int(
        structure_lengths([KV], "balanced")["kv"].min()
    ) <= 1


def test_comono_pairs_rank_for_rank():
    lengths = structure_lengths([KV, Q], "lognormal1", "comono")
    order = torch.argsort(lengths["kv"], stable=True)
    assert bool((lengths["q"][order].diff() >= 0).all())
    indep = structure_lengths([KV, Q], "lognormal1", "indep")
    assert torch.equal(torch.sort(indep["q"]).values, torch.sort(lengths["q"]).values)


def test_offsets_tensor():
    lengths = structure_lengths([Q], "powerlaw")["q"]
    offsets = to_tensor("offsets", lengths, Q.dtype, "cpu")
    assert offsets.dtype == torch.int32 and offsets.numel() == Q.count + 1
    assert int(offsets[0]) == 0 and int(offsets[-1]) == Q.total


def test_verdict_threshold_and_needs():
    flat = {"balanced": [1.0, 1.01, 1.0], "uniform": [1.02, 1.02, 1.03]}
    assert not verdict(flat)["qualifies"]
    marginal = {"balanced": [1.0] * 3, "uniform": [1.12] * 3}
    result = verdict(marginal)
    assert result["qualifies"] and result["needs"] == ["marginal histogram"]
    noisy = {"balanced": [1.0, 1.2, 1.0], "uniform": [1.12] * 3}
    assert not verdict(noisy)["qualifies"]
    paired = {"u|indep": [1.0] * 3, "u|comono": [1.3] * 3, "b|indep": [1.01] * 3, "b|comono": [1.01] * 3}
    result = verdict(paired, {"u": ("u|indep", "u|comono"), "b": ("b|indep", "b|comono")})
    assert result["needs"] == ["joint histogram of the pair"]
    assert result["pairing_effect_pct"]["u"] == pytest.approx(30.0)


def test_tiers_fall_back_to_synthetic(tmp_path):
    assert resolve_tier("synthetic", None) == ("synthetic", [])
    tier, notes = resolve_tier("real", None)
    assert tier == "synthetic" and len(notes) == 2 and "not available yet" in notes[0]
    stats = tmp_path / "stats.json"
    stats.write_text('{"schema_version": 1, "inputs": {}}')
    tier, notes = resolve_tier("stats", stats)
    assert tier == "synthetic" and "parsed" in notes[0]
    stats.write_text('{"schema_version": 9, "inputs": {}}')
    with pytest.raises(ValueError):
        resolve_tier("stats", stats)


@pytest.mark.parametrize("arm", INDEX_ARMS)
def test_index_arms_stay_in_bounds(arm):
    ids = make_indices(10_000, 5, 1_005, arm)
    assert ids.numel() == 10_000 and int(ids.min()) >= 5 and int(ids.max()) < 1_005
    assert torch.equal(ids, make_indices(10_000, 5, 1_005, arm))


def test_index_profiles_differ_as_named():
    def p(arm):
        return index_profile(make_indices(4_096, 0, 1 << 20, arm))

    assert p("unique")["unique_fraction"] == 1.0
    assert p("single")["unique_fraction"] == 1 / 4_096 and p("single")["top1pct_mass"] == 1.0
    assert p("zipf_sorted")["sorted_fraction"] == 1.0 and p("zipf")["sorted_fraction"] < 0.6
    assert p("zipf")["top1pct_mass"] > 5 * p("uniform")["top1pct_mass"]
    assert p("runs8")["mean_run_length"] == pytest.approx(8.0, rel=0.05)


def test_verdict_ignores_differences_below_timer_resolution():
    result = verdict({"balanced": [0.015] * 3, "uniform": [0.017] * 3})
    assert not result["qualifies"] and result["spread_pct"] > 10


def test_verdict_names_index_profile():
    result = verdict({"index:uniform": [1.0] * 3, "index:single": [4.0] * 3})
    assert result["qualifies"] and result["needs"] == ["index profile"]


def _config():
    from input_codec import compact_from_spec, spec_from_compact

    specs = [
        {"kind": "tensor", "name": "kv", "shape": [65], "dtype": "int64", "device": "cpu",
         "gen": {"kind": "jagged_offsets", "total": {"value": 5_000}, "max_len": 400,
                 "distribution": "uniform"}},
        {"kind": "tensor", "name": "q", "shape": [65], "dtype": "int32", "device": "cpu",
         "gen": {"kind": "jagged_offsets", "total": {"value": 900}, "max_len": 40,
                 "distribution": {"family": "uniform", "pair_with": "kv"}}},
        {"kind": "tensor", "name": "ids", "shape": [2, 50], "dtype": "int32", "device": "cpu",
         "gen": {"kind": "index", "low": 0, "high": 7}},
        {"kind": "tensor", "name": "scale", "shape": [], "dtype": "float32", "device": "cpu",
         "gen": {"kind": "constant", "value": 1.5}},
    ]
    # Round-trip through shapes.json's compact form, as a repro loads it.
    return {"inputs": [spec_from_compact(compact_from_spec(s, include_name=True)) for s in specs]}


def test_repro_inputs_follow_arms():
    from repro_harness import make_inputs_from_config

    kv, q, ids, scale = make_inputs_from_config(_config(), {"length": "lognormal2", "pairing": "comono",
                                                            "index": "single"})
    assert kv.dtype == torch.int64 and int(kv[0]) == 0 and int(kv[-1]) == 5_000
    assert q.dtype == torch.int32 and int(q[-1]) == 900 and int(q.diff().max()) <= 40
    order = torch.argsort(kv.diff(), stable=True)
    assert bool((q.diff()[order].diff() >= 0).all())
    assert ids.shape == (2, 50) and ids.unique().numel() == 1
    assert float(scale) == 1.5
    kv0, *_ = make_inputs_from_config(_config())
    assert int(kv0[-1]) == 5_000 and not torch.equal(kv0, kv)


def test_custom_kernel_meta(tmp_path):
    from repro_harness import custom_kernel_meta

    repro = tmp_path / "repro.py"
    assert custom_kernel_meta(repro) is None
    (tmp_path / "meta.json").write_text('{"kind": "reduction"}')
    assert custom_kernel_meta(repro) is None
    (tmp_path / "meta.json").write_text('{"kind": "custom_kernel", "kernel": "k"}')
    assert custom_kernel_meta(repro)["kernel"] == "k"


def test_resolve_bundle_refuses_dev_runs(monkeypatch):
    import user_triton

    class Kernel:
        pass

    marked = user_triton.mark_dev_run(Kernel(), "edited_kernel", "differs from the capture")
    bundle = {user_triton.BUNDLE_KEY: {"schema_version": user_triton.BUNDLE_SCHEMA_VERSION,
                                       "kernels": {"0": {"name": "k"}}, "constant_args": {}}}
    monkeypatch.setattr(user_triton, "_resolver", lambda: lambda entry: marked)
    assert user_triton.resolve_bundle(bundle) is None
    assert "edited_kernel" in user_triton.dev_runs()
    monkeypatch.setattr(user_triton, "_resolver", lambda: lambda entry: Kernel())
    assert user_triton.resolve_bundle(bundle) is not None
