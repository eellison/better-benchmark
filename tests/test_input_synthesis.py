"""Input synthesis for captured regions: values the region's ops can consume.

  1. fp8 inputs are drawn in fp32 and rounded (randn has no fp8 kernels).

Run: python tests/test_input_synthesis.py  (CPU only)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

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


if __name__ == "__main__":
    test_fp8_inputs_are_generated()
