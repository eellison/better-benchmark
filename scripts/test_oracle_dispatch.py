"""Self-test for OracleRegistry (hardware, shape, configs) dispatch.

Registers ONE shared implementation under 3 (hardware, shape, configs) combos
plus a distinct unconstrained default, then verifies:
  - each dispatch tier (1-5) matches correctly
  - configs are passed through as kwargs (configs=None -> plain call)
  - the decorator returns the function unchanged (re-registrable)
  - last_dispatch_info reports tier / fallback / configs
  - lambda shape predicates are rejected
  - nearest-shape uses sum of |log(actual/registered)| with same rank required

No GPU work: hardware detection is monkeypatched. Run:
    python scripts/test_oracle_dispatch.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import oracle_harness as oh
from oracle_harness import OracleRegistry


CALLS = []  # (fn_label, configs_dict_or_None)


def _shared_impl(inputs, *, BLOCK=0, num_warps=0):
    """One kernel body, registered 3x with different configs."""
    CALLS.append(("shared", {"BLOCK": BLOCK, "num_warps": num_warps}))
    return inputs[0]


def _default_impl(inputs):
    """Distinct unconstrained default; called plain (configs=None)."""
    CALLS.append(("default", None))
    return inputs[0]


def make_registry():
    registry = OracleRegistry()
    r1 = registry.register(hardware="H100", shape=(32768, 1024),
                           configs={"BLOCK": 1024, "num_warps": 4})(_shared_impl)
    r2 = registry.register(hardware="B200", shape=(32768, 1024),
                           configs={"BLOCK": 2048, "num_warps": 8})(_shared_impl)
    r3 = registry.register(hardware="B200", shape=(8192, 262144),
                           configs={"BLOCK": 4096, "num_warps": 16})(_shared_impl)
    # Decorator must return the function unchanged so it can be re-registered.
    assert r1 is _shared_impl and r2 is _shared_impl and r3 is _shared_impl, \
        "register() must return the function unchanged"
    registry.register(description="unconstrained default")(_default_impl)
    return registry


def fake_inputs(*shape):
    return [torch.empty(shape, device="cpu", dtype=torch.float16)]


def run_case(registry, hw, shape, *, want_tier, want_fn, want_configs,
             want_dist=None, label=""):
    oh.get_gpu_kind = lambda: hw  # monkeypatch hardware detection
    CALLS.clear()
    registry.dispatch(fake_inputs(*shape))
    info = registry.last_dispatch_info

    assert len(CALLS) == 1, f"{label}: expected exactly one call, got {CALLS}"
    got_fn, got_configs = CALLS[0]
    assert got_fn == want_fn, f"{label}: dispatched to {got_fn}, want {want_fn}"
    assert got_configs == want_configs, \
        f"{label}: configs {got_configs}, want {want_configs}"
    assert info["tier"] == want_tier, \
        f"{label}: tier {info['tier']} ({info['tier_name']}), want {want_tier}"
    assert info["fallback"] == (want_tier > 1), f"{label}: fallback flag wrong"
    assert info["current_hardware"] == hw
    if want_dist is not None:
        assert info["distance"] is not None and \
            math.isclose(info["distance"], want_dist, rel_tol=1e-9), \
            f"{label}: distance {info['distance']}, want {want_dist}"
    print(f"  PASS [{label}] hw={hw} shape={shape} -> tier {info['tier']} "
          f"({info['tier_name']}) fn={info['fn_name']} configs={info['configs']} "
          f"dist={info['distance']}")


def main():
    orig_get_gpu_kind = oh.get_gpu_kind
    try:
        registry = make_registry()

        print("Tier 1: exact (hardware, shape) match")
        run_case(registry, "B200", (32768, 1024),
                 want_tier=1, want_fn="shared",
                 want_configs={"BLOCK": 2048, "num_warps": 8},
                 want_dist=0.0, label="tier1-B200")
        run_case(registry, "H100", (32768, 1024),
                 want_tier=1, want_fn="shared",
                 want_configs={"BLOCK": 1024, "num_warps": 4},
                 want_dist=0.0, label="tier1-H100")
        run_case(registry, "B200", (8192, 262144),
                 want_tier=1, want_fn="shared",
                 want_configs={"BLOCK": 4096, "num_warps": 16},
                 want_dist=0.0, label="tier1-B200-big")

        print("Tier 2: same shape, any hardware (unregistered GPU)")
        # H200 not registered; (32768, 1024) matches H100 entry (first in
        # registration order among the two distance-0 entries).
        run_case(registry, "H200", (32768, 1024),
                 want_tier=2, want_fn="shared",
                 want_configs={"BLOCK": 1024, "num_warps": 4},
                 want_dist=0.0, label="tier2")

        print("Tier 3: same hardware, nearest shape")
        # B200 with unseen shape (16384, 1024). Distances:
        #   to (32768, 1024):  |log(16384/32768)| = log(2)
        #   to (8192, 262144): |log(2)| + |log(1024/262144)| = log(2)+log(256)
        # -> nearest is the B200 (32768, 1024) registration.
        run_case(registry, "B200", (16384, 1024),
                 want_tier=3, want_fn="shared",
                 want_configs={"BLOCK": 2048, "num_warps": 8},
                 want_dist=math.log(2), label="tier3-near")
        # Nearest-shape must pick the OTHER B200 entry when actual is closer
        # to (8192, 262144).
        run_case(registry, "B200", (8192, 131072),
                 want_tier=3, want_fn="shared",
                 want_configs={"BLOCK": 4096, "num_warps": 16},
                 want_dist=math.log(2), label="tier3-other-entry")

        print("Tier 4: any registration, nearest shape (wrong hw AND shape)")
        # H200 + unseen shape: nearest is (32768, 1024); H100 entry wins the
        # distance tie by registration order.
        run_case(registry, "H200", (16384, 1024),
                 want_tier=4, want_fn="shared",
                 want_configs={"BLOCK": 1024, "num_warps": 4},
                 want_dist=math.log(2), label="tier4")

        print("Tier 5: default (rank mismatch -> all shape distances inf)")
        run_case(registry, "H200", (64, 64, 64),
                 want_tier=5, want_fn="default", want_configs=None,
                 label="tier5-rank")
        run_case(registry, "A100", (4, 4, 4, 4),
                 want_tier=5, want_fn="default", want_configs=None,
                 label="tier5-rank4")

        print("Shape spec: tuple-of-tuples (all input shapes)")
        multi = OracleRegistry()
        multi.register(hardware="B200", shape=((128, 256), (256,)),
                       configs={"BLOCK": 64})(_shared_impl)
        multi.register(description="default")(_default_impl)
        oh.get_gpu_kind = lambda: "B200"
        CALLS.clear()
        multi.dispatch([torch.empty(128, 256), torch.empty(256)])
        assert CALLS == [("shared", {"BLOCK": 64, "num_warps": 0})], \
            f"full-signature exact match failed: {CALLS}"
        assert multi.last_dispatch_info["tier"] == 1
        # Wrong arity (one input vs two registered) -> falls to default.
        CALLS.clear()
        multi.dispatch([torch.empty(128, 256)])
        assert CALLS == [("default", None)], f"arity mismatch fallback failed: {CALLS}"
        assert multi.last_dispatch_info["tier"] == 5
        print("  PASS [full-signature] exact match + arity-mismatch fallback")

        print("Error handling")
        try:
            OracleRegistry().register(shape=lambda inputs: True)
            raise AssertionError("lambda shape should raise TypeError")
        except TypeError:
            print("  PASS [reject-lambda] callable shape raises TypeError")
        try:
            OracleRegistry().register(shape=(2, 2), configs=[1, 2])
            raise AssertionError("non-dict configs should raise TypeError")
        except TypeError:
            print("  PASS [reject-configs] non-dict configs raises TypeError")
        empty = OracleRegistry()
        try:
            empty.dispatch(fake_inputs(8, 8))
            raise AssertionError("empty registry should raise RuntimeError")
        except RuntimeError:
            print("  PASS [empty-registry] dispatch raises RuntimeError")

        print("Introspection: list_entries")
        entries = registry.list_entries()
        assert len(entries) == 4
        assert entries[0] == {
            "hardware": "H100", "shape": (32768, 1024), "full_signature": False,
            "fn_name": "_shared_impl", "configs": {"BLOCK": 1024, "num_warps": 4},
            "description": "_shared_impl",
        }, f"unexpected entry: {entries[0]}"
        print("  PASS [list_entries] 4 entries, shared fn registered 3x")

        print("\nALL TESTS PASSED")
    finally:
        oh.get_gpu_kind = orig_get_gpu_kind


if __name__ == "__main__":
    main()
