"""Self-test for oracle_impl signature-based dispatch (no GPU required).

Settled design:
  - shapes = FULL input signature string (T()/S() format, same as
    _shapes_config); S() entries skipped, dtypes recorded but not matched
  - matching is EXACT-ONLY: an impl either exactly matches the runtime
    signature, or it declared shapes=None (shape-general). No fuzzy
    nearest-shape matching — each repro has a small finite set of shapes.
  - match order: "hardware+shape" > "shape" > "hardware" (shape-general,
    hw match) > "any" (shape-general, unconstrained) > OracleDispatchError
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
    # no-trailing-comma single tensor
    assert parse_shapes_signature("(T([8192, 262144], bf16))") == ((8192, 262144),)
    sig_gen = parse_shapes_signature("(T([8, 512], i64, gen=Index(32128)),)")
    assert sig_gen == ((8, 512),), sig_gen
    print("PASS parse_shapes_signature")

    # --- matched: hardware+shape --------------------------------------------
    fresh()
    oracle_impl(hardware="B200", shapes="(T([32768, 1024], bf16),)")(make("b200"))
    oracle_impl(hardware="H100", shapes="(T([32768, 1024], bf16),)")(make("h100"))
    _, info = resolve_oracle(make("entry"), fake_inputs((32768, 1024)))
    assert info["matched"] == "hardware+shape" and info["fn_name"] == "b200"
    assert not info["fallback"]
    print("PASS matched=hardware+shape")

    # --- matched: shape (H100 oracle measured on B200) ----------------------
    fresh()
    oracle_impl(hardware="H100", shapes="(T([32768, 1024], bf16),)")(make("h100"))
    _, info = resolve_oracle(make("entry"), fake_inputs((32768, 1024)))
    assert info["matched"] == "shape" and info["fallback"]
    assert info["tuned_hardware"] == "H100"
    print("PASS matched=shape (H100-on-B200 fallback)")

    # --- dtype is NOT a match requirement (corpus dedupes across dtypes) ----
    _, info = resolve_oracle(
        make("entry"), fake_inputs((32768, 1024), dtype=torch.float16))
    assert info["matched"] == "shape", info
    print("PASS dtype-agnostic match")

    # --- shape-specific impl never runs at a different shape ----------------
    fresh()
    oracle_impl(hardware="B200", shapes="(T([32768, 1024], bf16),)")(make("strict"))
    try:
        resolve_oracle(make("entry"), fake_inputs((32768, 4096)))
        raise AssertionError("expected OracleDispatchError")
    except OracleDispatchError:
        pass
    print("PASS wrong shape raises (no fuzzy matching)")

    # --- shapes=None means shape-general: matches any shape -----------------
    fresh()
    oracle_impl(hardware="B200")(make("general_b200"))
    _, info = resolve_oracle(make("entry"), fake_inputs((32768, 4096)))
    assert info["matched"] == "hardware" and info["fn_name"] == "general_b200"
    assert info["fallback"]
    print("PASS matched=hardware (shape-general)")

    # --- exact shape match beats shape-general ------------------------------
    fresh()
    oracle_impl(hardware="B200")(make("general"))
    oracle_impl(hardware="H100", shapes="(T([32768, 1024], bf16),)")(make("tuned_h100"))
    _, info = resolve_oracle(make("entry"), fake_inputs((32768, 1024)))
    assert info["fn_name"] == "tuned_h100" and info["matched"] == "shape", info
    print("PASS exact shape beats shape-general")

    # --- multi-shape: same body registered at 3 shapes.txt lines ------------
    fresh()
    shared = make("shared", expect_configs=True)
    oracle_impl(hardware="B200", shapes="(T([32768, 256], bf16),)",
                BLOCK=256)(shared)
    oracle_impl(hardware="B200", shapes="(T([32768, 1024], bf16),)",
                BLOCK=1024)(shared)
    oracle_impl(hardware="B200", shapes="(T([8192, 262144], bf16),)",
                BLOCK=4096)(shared)
    fn, info = resolve_oracle(make("entry"), fake_inputs((8192, 262144)))
    assert info["matched"] == "hardware+shape"
    fn(fake_inputs((8192, 262144)))
    assert CALLS[-1] == ("shared", {"BLOCK": 4096}), CALLS[-1]
    fn, _ = resolve_oracle(make("entry"), fake_inputs((32768, 256)))
    fn(fake_inputs((32768, 256)))
    assert CALLS[-1] == ("shared", {"BLOCK": 256}), CALLS[-1]
    print("PASS multi-shape: per-shape configs on one body")

    # --- multi-tensor signatures --------------------------------------------
    fresh()
    oracle_impl(hardware="B200",
                shapes="(T([1024, 64], f32), T([64], f32))")(make("two_tensor"))
    _, info = resolve_oracle(make("entry"), fake_inputs((1024, 64), (64,)))
    assert info["matched"] == "hardware+shape"
    try:  # tensor-count mismatch
        resolve_oracle(make("entry"), fake_inputs((1024, 64)))
        raise AssertionError("expected OracleDispatchError")
    except OracleDispatchError:
        pass
    print("PASS multi-tensor signature + count mismatch raises")

    # --- configs passthrough across hardware on one body --------------------
    fresh()
    shared = make("shared_hw", expect_configs=True)
    SIG = "(T([32768, 1024], bf16),)"
    oracle_impl(hardware="H100", shapes=SIG,
                BLOCK=1024, num_warps=4)(shared)
    oracle_impl(hardware="B200", shapes=SIG,
                BLOCK=2048, num_warps=8)(shared)
    fn, info = resolve_oracle(make("entry"), fake_inputs((32768, 1024)))
    assert info["matched"] == "hardware+shape"
    fn(fake_inputs((32768, 1024)))
    assert CALLS[-1] == ("shared_hw", {"BLOCK": 2048, "num_warps": 8}), CALLS[-1]
    print("PASS configs passthrough (B200 configs win on B200)")

    # --- unconstrained shape-general default ("any") -------------------------
    fresh()
    oracle_impl(hardware="H100", shapes=SIG)(make("strict"))
    oracle_impl()(make("default"))
    _, info = resolve_oracle(make("entry"), fake_inputs((7, 7)))
    assert info["matched"] == "any" and info["fn_name"] == "default"
    print("PASS matched=any (unconstrained default)")

    # --- dtype honesty: bf16-tuned kernel serving f32 inputs is flagged -----
    fresh()
    oracle_impl(hardware="B200", shapes="(T([32768, 1024], bf16),)")(make("bf16_tuned"))
    _, info = resolve_oracle(make("entry"), fake_inputs((32768, 1024), dtype=torch.float32))
    assert info.get("dtypes_differ") is True, info
    assert info["tuned_dtypes"] == ("bf16",) and info["actual_dtypes"] == ("f32",)
    _, info = resolve_oracle(make("entry"), fake_inputs((32768, 1024), dtype=torch.bfloat16))
    assert "dtypes_differ" not in info, info
    print("PASS dtype honesty flag")

    # --- strategy kwargs: looped vs persistent on one body ------------------
    fresh()
    body = make("reduction", expect_configs=True)
    oracle_impl(hardware="H100", shapes="(T([4096, 512], f32),)",
                persistent=True, RBLOCK=512)(body)
    oracle_impl(hardware="B200", shapes="(T([4096, 512], f32),)",
                persistent=False, RBLOCK=128)(body)
    fn, _ = resolve_oracle(make("entry"), fake_inputs((4096, 512)))
    fn(fake_inputs((4096, 512)))
    assert CALLS[-1] == ("reduction", {"persistent": False, "RBLOCK": 128}), CALLS[-1]
    print("PASS strategy kwargs (looped vs persistent)")

    # --- empty signature: repro with zero tensor inputs (29 in corpus) ------
    fresh()
    oracle_impl(hardware="H100", shapes="()")(make("no_tensors"))
    _, info = resolve_oracle(make("entry"), [512, 3.0])  # scalars only
    assert info["matched"] == "shape" and info["fn_name"] == "no_tensors", info
    fresh()
    oracle_impl(hardware="H100", shapes="(S([1, -1, 512, 512]))")(make("s_only"))
    _, info = resolve_oracle(make("entry"), [512])
    assert info["matched"] == "shape" and info["fn_name"] == "s_only", info
    print("PASS empty / S()-only signatures match scalar-only inputs")

    # --- unmigrated module: resolve is a no-op ------------------------------
    def plain_oracle(inputs):
        return "plain"
    plain_oracle.__module__ = "some_unmigrated_module"
    fn, info = resolve_oracle(plain_oracle, fake_inputs((3, 3)))
    assert fn is plain_oracle and info is None
    print("PASS unmigrated module unaffected")

    # --- point= dispatch: DYNAMIC point routes by HASH, not shape -----------
    # A dynamic point's resolved shape is symbolic (64,64,s0,s1) and would
    # never == concrete runtime shapes; oracle_impl(point=) registers it
    # shape-less and select(point=hash) routes by hash. One registration
    # covers the whole family (every binding) — dispatch is identity, the
    # kernel handles concrete dims internally.
    _run_point_dispatch_tests()

    reset_oracle_registry(MOD)
    print("\nALL DISPATCH TESTS PASS")


def _run_point_dispatch_tests():
    import json
    import tempfile
    import types
    from pathlib import Path

    PMOD = "test_oracle_dispatch_point_fake"
    reset_oracle_registry(PMOD)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # A shapes.json with ONE dynamic family, TWO binding points (distinct
        # shape_hashes), shapes symbolic over s0/s1. (As produced by capture +
        # canonical symbol naming.)
        shapes = {
            "symbols": {"s0": {"hint": 8, "range": [2, None]},
                        "s1": {"hint": 8, "range": [2, None]}},
            "guards": [],
            "points": [
                {"shape_hash": "aaaa1111", "captured_dynamic": True,
                 "bindings": {"s0": 8, "s1": 8},
                 "inputs": [[[64, 64, "s0", "s1"], "f32"]]},
                {"shape_hash": "bbbb2222", "captured_dynamic": True,
                 "bindings": {"s0": 24, "s1": 24},
                 "inputs": [[[64, 64, "s0", "s1"], "f32"]]},
            ],
        }
        (d / "shapes.json").write_text(json.dumps(shapes))

        # A module whose __file__ sits in the dir (oracle_impl resolves the
        # sibling shapes.json off the registering fn's module file).
        mod = types.ModuleType(PMOD)
        mod.__file__ = str(d / "oracle.py")
        sys.modules[PMOD] = mod

        def pmake(name):
            def fn(inputs):
                return name
            fn.__module__ = PMOD
            fn.__name__ = name
            return fn

        # Register ONE kernel per point hash (shape-less; hash is the key).
        oracle_impl(point="aaaa1111")(pmake("k_8"))
        oracle_impl(point="bbbb2222")(pmake("k_24"))

        # The dynamic point registered shape-less (symbolic shape dropped).
        from oracle_harness import get_module_registry
        reg = get_module_registry(PMOD)
        assert all(e["shape"] is None for e in reg._entries), \
            "dynamic point should register shape-less"
        assert {e["point"] for e in reg._entries} == {"aaaa1111", "bbbb2222"}

        # Dispatch by hash routes to the right kernel — and the SAME hash
        # matches REGARDLESS of the concrete runtime shape (family coverage).
        e, info = reg.select(fake_inputs((64, 64, 8, 8)), point="aaaa1111")
        assert e["fn"].__name__ == "k_8" and info["matched"] in (
            "point", "point+hardware"), info
        e, _ = reg.select(fake_inputs((64, 64, 24, 24)), point="bbbb2222")
        assert e["fn"].__name__ == "k_24"
        # the 8-point kernel also matches a DIFFERENT concrete shape under its
        # hash (proves it's hash-keyed, not shape-keyed):
        e, _ = reg.select(fake_inputs((64, 64, 99, 7)), point="aaaa1111")
        assert e["fn"].__name__ == "k_8"
        # no hash threaded -> shape-less entries are NOT shape-general here
        # (they carry a point), so a bare shape lookup finds nothing -> loud.
        try:
            reg.select(fake_inputs((64, 64, 8, 8)))
            raise AssertionError("expected RuntimeError (no hash, no shape match)")
        except RuntimeError:
            pass
        # unknown hash -> loud (no silent wrong-kernel)
        try:
            reg.select(fake_inputs((64, 64, 8, 8)), point="zzzz9999")
            raise AssertionError("expected RuntimeError (unknown hash)")
        except RuntimeError:
            pass

        reset_oracle_registry(PMOD)
        del sys.modules[PMOD]
    print("PASS point= dispatch (dynamic point routes by hash, family coverage)")


if __name__ == "__main__":
    run_tests()
