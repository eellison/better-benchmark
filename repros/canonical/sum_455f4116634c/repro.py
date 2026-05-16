"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: 455f4116634c
Shape hash: 62d93a50
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_41: "f16[32, 1024, 1024]", arg140_1: "b8[4, 8, 1024, 1024]", arg139_1: "f32[4, 8, 1024, 1024]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_41, [4, 8, 1024, 1024]);  bmm_41 = None
        convert_element_type_default: "f16[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.float16);  arg140_1 = None
        mul_tensor: "f16[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f16[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None
        convert_element_type_default_1: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float32);  mul_tensor_1 = None
        mul_tensor_2: "f32[4, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, arg139_1);  convert_element_type_default_1 = None
        sum_dim_int_list: "f32[4, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[4, 8, 1024, 1024]" = torch.ops.aten.neg.default(arg139_1);  arg139_1 = None
        fma_default: "f32[4, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        convert_element_type_default_2: "f16[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_default, torch.float16);  fma_default = None
        reshape_default_1: "f16[32, 1024, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default_2, [32, 1024, 1024]);  convert_element_type_default_2 = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randn([32, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [4, 8, 1024, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
