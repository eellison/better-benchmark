"""Input synthesis for captured regions: values the region's ops can consume.

  1. fp8 inputs are drawn in fp32 and rounded (randn has no fp8 kernels).
  2. index_put values are cast to the destination dtype, as Inductor's lowering
     does, so a post-grad region with mismatched dtypes runs in eager.

Run: python tests/test_input_synthesis.py  (CPU only)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import cast_index_put_values
from repro_harness import make_inputs_from_config


def test_fp8_inputs_are_generated():
    (x, y) = make_inputs_from_config({"inputs": [
        {"kind": "tensor", "shape": [4, 8], "dtype": "torch.float8_e4m3fn",
         "device": "cpu"},
        {"kind": "tensor", "shape": [4, 8], "stride": [16, 1],
         "dtype": "torch.float8_e5m2", "device": "cpu"},
    ]})
    assert x.dtype == torch.float8_e4m3fn and y.dtype == torch.float8_e5m2
    assert x.float().std() > 0.1
    print("PASS fp8 inputs are generated")


def test_index_put_values_cast_to_destination():
    def f(dst, idx, v):
        return torch.ops.aten.index_put.default(dst, [idx], v.to(dst.dtype))

    gm = make_fx(f, tracing_mode="fake")(
        torch.zeros(4, 3, dtype=torch.bfloat16), torch.tensor([1]),
        torch.randn(1, 3))
    (put,) = [n for n in gm.graph.nodes
              if n.target == torch.ops.aten.index_put.default]
    put.update_arg(2, put.args[2].args[0])  # post-grad graphs drop the cast
    cast_index_put_values(gm)
    out = gm(torch.zeros(4, 3, dtype=torch.bfloat16), torch.tensor([1]),
             torch.randn(1, 3))
    assert out.dtype == torch.bfloat16
    print("PASS index_put values are cast to the destination dtype")


if __name__ == "__main__":
    test_fp8_inputs_are_generated()
    test_index_put_values_cast_to_destination()
