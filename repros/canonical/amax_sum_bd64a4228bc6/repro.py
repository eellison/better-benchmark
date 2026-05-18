"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g21
Pattern hash: bd64a4228bc6
Shape hash: 7ff4da31
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
    def forward(self, arg192_1: "f32[768]", arg191_1: "f32[768, 768]", bmm_22: "f16[12, 512, 512]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg192_1, torch.float16);  arg192_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg191_1, torch.float16);  arg191_1 = None
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        reshape_default: "f16[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [1, 12, 512, 512]);  bmm_22 = None
        convert_element_type_default_2: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        amax_default: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        exp_default: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_3: "f16[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        native_dropout_default = torch.ops.aten.native_dropout.default(convert_element_type_default_3, 0.1, True);  convert_element_type_default_3 = None
        getitem: "f16[1, 12, 512, 512]" = native_dropout_default[0]
        getitem_1: "b8[1, 12, 512, 512]" = native_dropout_default[1];  native_dropout_default = None
        return (convert_element_type_default, permute_default, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
