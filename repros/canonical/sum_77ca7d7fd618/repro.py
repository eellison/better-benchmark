"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g174
Pattern hash: 77ca7d7fd618
Shape hash: 74a3da4f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_41: "f16[48, 476, 476]", arg41_1: "f32[4, 12, 476, 476]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 12, 476, 476]" = torch.ops.aten.reshape.default(bmm_41, [4, 12, 476, 476]);  bmm_41 = None
        convert_element_type_default: "f32[4, 12, 476, 476]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[4, 12, 476, 476]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg41_1);  convert_element_type_default = None
        sum_dim_int_list: "f32[4, 12, 476, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[4, 12, 476, 476]" = torch.ops.aten.neg.default(arg41_1);  arg41_1 = None
        fma_default: "f32[4, 12, 476, 476]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_1: "f16[4, 12, 476, 476]" = torch.ops.prims.convert_element_type.default(fma_default, torch.float16);  fma_default = None
        div_tensor: "f16[4, 12, 476, 476]" = torch.ops.aten.div.Tensor(convert_element_type_default_1, 8.0);  convert_element_type_default_1 = None
        reshape_default_1: "f16[48, 476, 476]" = torch.ops.aten.reshape.default(div_tensor, [48, 476, 476]);  div_tensor = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randn([48, 476, 476], dtype=torch.float16, device='cuda'),
    torch.randn(10876400, dtype=torch.float32, device='cuda').as_strided([4, 12, 476, 476], [2719104, 226592, 476, 1]),  # arg41_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
