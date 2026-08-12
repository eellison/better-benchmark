"""Partition output-selection invariants (capture fidelity).

A repro partition must constrain the computation exactly as the model does:
  1. ESCAPING values (any use outside the partition, incl. graph outputs)
     are partition outputs — even if they also have internal users. The
     pre-fix code only exported leaves, so interior escaping values (e.g.
     squeezenet's where_self also feeding convolution_backward) were free
     to be eliminated in the repro but not the model; gaps didn't compose.
  2. Purely-internal values stay internal.
  3. MUTATING ops (copy_, index_put_, ...) are kept even with zero FX
     users — their effect is the buffer mutation, not the return value.

Run: python scripts/test_partition_outputs.py  (CPU only)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import extract_partition_subgraph


def _outputs(sub_gm):
    out = [n for n in sub_gm.graph.nodes if n.op == "output"][0].args[0]
    return [a.name for a in (out if isinstance(out, (tuple, list)) else [out])]


def test_escaping_interior_value_is_output():
    def f(mask, x):
        w = torch.where(mask, torch.zeros((), dtype=x.dtype), x)
        return w.sum(dim=0), w * 2.0  # mul is the external consumer

    gm = make_fx(f, tracing_mode="fake")(
        torch.ones(4, 8, dtype=torch.bool), torch.randn(4, 8))
    part = [n for n in gm.graph.nodes if n.op == "call_function"
            and any(k in n.name for k in ("where", "sum", "zeros"))]
    names = _outputs(extract_partition_subgraph(part, gm)[0])
    assert any("where" in n for n in names), names
    assert any("sum" in n for n in names), names
    print("PASS escaping interior value is output")


def test_purely_internal_stays_internal():
    def g(mask, x):
        w = torch.where(mask, torch.zeros((), dtype=x.dtype), x)
        return w.sum(dim=0)

    gm = make_fx(g, tracing_mode="fake")(
        torch.ones(4, 8, dtype=torch.bool), torch.randn(4, 8))
    part = [n for n in gm.graph.nodes if n.op == "call_function"
            and any(k in n.name for k in ("where", "sum", "zeros"))]
    names = _outputs(extract_partition_subgraph(part, gm)[0])
    assert not any("where" in n for n in names), names
    print("PASS purely-internal value stays internal")


def test_zero_user_mutating_op_kept():
    def h(x, buf):
        s = x.sum(dim=0)
        buf.copy_(s)  # zero FX users — pure side effect
        return x * 1.0

    gm = make_fx(h, tracing_mode="fake")(torch.randn(4, 8), torch.randn(8))
    part = [n for n in gm.graph.nodes if n.op == "call_function"
            and ("sum" in n.name or "copy" in n.name)]
    names = _outputs(extract_partition_subgraph(part, gm)[0])
    assert any("copy" in n for n in names), names
    print("PASS zero-user mutating op kept as output")


if __name__ == "__main__":
    test_escaping_interior_value_is_output()
    test_purely_internal_stays_internal()
    test_zero_user_mutating_op_kept()
    print("\nALL PARTITION-OUTPUT TESTS PASS")
