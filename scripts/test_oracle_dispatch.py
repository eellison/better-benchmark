"""Self-test for oracle_impl signature-based dispatch (no GPU required).

Covers the settled design:
  - shapes = FULL input signature string (T()/S() format, same as
    _shapes_config); S() entries skipped, dtypes recorded but not matched
  - exact=True (default): shape-specific, matches declared signature only,
    never selected by nearest-shape fallback
  - exact=False: shape-general, eligible for tier 3/4 nearest-shape dispatch
  - no match -> OracleDispatchError (loud failure, no garbage floors)
  - configs passthrough: one kernel body, N (hardware, shapes, configs) points
  - unmigrated modules (no registrations) are untouched by resolve_oracle

Run: python scripts/test_oracle_dispatch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import oracle_harness
from oracle_harness import (
    OracleDispatchError,
    oracle_impl,
    parse_shapes_signature,
    reset_oracle_registry,
    resolve_oracle,
)

MOD = "test_oracle_dispatch_fake"
CALLS = []


def fake_inputs(*shapes, dtype=torch.float32):
    return [torch.empty(s, dtype=dtype, device="meta") for s in shapes]


def fresh():
    reset_oracle_registry(MOD)
    CALLS.clear()


def make(name, expect_configs=False):
    if expect_configs:
        def fn(inputs, **cfg):
            CALLS.append((name, cfg))
            return name
    else:
        def fn(inputs):
            CALLS.append((name, None))
            return name
    fn.__module__ = MOD
    fn.__name__ = name
    return fn


def run_tests():
    # Deterministic hardware for tests.
    oracle_harness.get_gpu_kind._cached = "B200"

    # --- parse_shapes_signature -------------------------------------------
    sig = parse_shapes_signature(
        "(T([512, 256, 13, 13], f32), T([512, 128, 27, 27], b8), S([131072, 169]))"
    )
    assert sig == ((512, 256, 13, 13), (512, 128, 27, 27)), sig  # S() skipped
    assert parse_shapes_signature(None) is None
    assert parse_shapes_signature(((8, 4), (4,))) == ((8, 4), (4,))
    sig_gen = parse_shapes_signature("(T([8, 512], i64, gen=Index(32128)),)")
    assert sig_gen == ((8, 512),), sig_gen
    print("PASS parse_shapes_signature")

    # --- tier 1: exact (hardware, signature) -------------------------------
    fresh()
    oracle_impl(hardware="B200", shapes="(T([32768, 1024], bf16),)")(make("b200"))
    oracle_impl(hardware="H100", shapes="(T([32768, 1024], bf16),)")(make("h100"))
    _, info = resolve_oracle(make("entry"), fake_inputs((32768, 1024)))
    assert info["tier"] == 1 and info["fn_name"] == "b200" and not info["fallback"]
    print("PASS tier1 exact hw+sig")

    # --- tier 2: same signature, other hardware (H100 oracle on B200) ------
    fresh()
    oracle_impl(hardware="H100", shapes="(T([32768, 1024], bf16),)")(make("h100"))
    _, info = resolve_oracle(make("entry"), fake_inputs((32768, 1024)))
    assert info["tier"] == 2 and info["fallback"] and info["fn_name"] == "h100"
    print("PASS tier2 H100-on-B200 fallback")

    # --- dtype is NOT a match requirement (corpus dedupes across dtypes) ----
    _, info = resolve_oracle(
        make("entry"), fake_inputs((32768, 1024), dtype=torch.float16))
    assert info["tier"] == 2, info
    print("PASS dtype-agnostic match")

    # --- exact=True (default) does NOT shape-fallback: raises --------------
    fresh()
    oracle_impl(hardware="B200", shapes="(T([32768, 1024], bf16),)")(make("strict"))
    try:
        resolve_oracle(make("entry"), fake_inputs((32768, 4096)))
        raise AssertionError("expected OracleDispatchError")
    except OracleDispatchError:
        pass
    print("PASS exact=True no fallback -> raises")

    # --- exact=False IS eligible for nearest-shape dispatch ----------------
    fresh()
    oracle_impl(hardware="B200", shapes="(T([32768, 1024], bf16),)",
                exact=False)(make("general"))
    _, info = resolve_oracle(make("entry"), fake_inputs((32768, 4096)))
    assert info["tier"] == 3 and info["fn_name"] == "general" and info["fallback"]
    assert info["exact"] is False
    print("PASS exact=False nearest-shape tier3")

    # --- nearest picks the closer of two general impls ---------------------
    fresh()
    oracle_impl(hardware="B200", shapes="(T([32768, 1024], bf16),)",
                exact=False)(make("near"))
    oracle_impl(hardware="B200", shapes="(T([32768, 65536], bf16),)",
                exact=False)(make("far"))
    _, info = resolve_oracle(make("entry"), fake_inputs((32768, 2048)))
    assert info["fn_name"] == "near", info
    print("PASS nearest-shape distance")

    # --- mixed: exact=True sibling never steals nearest dispatch ------------
    fresh()
    oracle_impl(hardware="B200", shapes="(T([32768, 2048], bf16),)")(make("strict_near"))
    oracle_impl(hardware="B200", shapes="(T([32768, 65536], bf16),)",
                exact=False)(make("general_far"))
    _, info = resolve_oracle(make("entry"), fake_inputs((32768, 4096)))
    assert info["fn_name"] == "general_far", info  # strict_near is closer but exact
    print("PASS exact=True excluded from nearest even when closer")

    # --- structure mismatch (tensor count) never matches --------------------
    fresh()
    oracle_impl(hardware="B200",
                shapes="(T([1024, 64], f32), T([64], f32))",
                exact=False)(make("two_tensor"))
    try:
        resolve_oracle(make("entry"), fake_inputs((1024, 64)))
        raise AssertionError("expected OracleDispatchError")
    except OracleDispatchError:
        pass
    print("PASS structure mismatch raises")

    # --- configs passthrough: one body, two hardware points -----------------
    fresh()
    shared = make("shared", expect_configs=True)
    oracle_impl(hardware="H100", shapes="(T([32768, 1024], bf16),)",
                configs={"BLOCK": 1024, "num_warps": 4})(shared)
    oracle_impl(hardware="B200", shapes="(T([32768, 1024], bf16),)",
                configs={"BLOCK": 2048, "num_warps": 8})(shared)
    fn, info = resolve_oracle(make("entry"), fake_inputs((32768, 1024)))
    assert info["tier"] == 1
    fn(fake_inputs((32768, 1024)))
    assert CALLS[-1] == ("shared", {"BLOCK": 2048, "num_warps": 8}), CALLS[-1]
    print("PASS configs passthrough (B200 configs win on B200)")

    # --- unconstrained default (tier 5) -------------------------------------
    fresh()
    oracle_impl(hardware="B200", shapes="(T([32768, 1024], bf16),)")(make("strict"))
    oracle_impl()(make("default"))
    _, info = resolve_oracle(make("entry"), fake_inputs((7, 7)))
    assert info["tier"] == 5 and info["fn_name"] == "default"
    print("PASS tier5 default")

    # --- unmigrated module: resolve is a no-op ------------------------------
    def plain_oracle(inputs):
        return "plain"
    plain_oracle.__module__ = "some_unmigrated_module"
    fn, info = resolve_oracle(plain_oracle, fake_inputs((3, 3)))
    assert fn is plain_oracle and info is None
    print("PASS unmigrated module unaffected")

    reset_oracle_registry(MOD)
    print("\nALL DISPATCH TESTS PASS")


if __name__ == "__main__":
    run_tests()
