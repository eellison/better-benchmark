"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g62
Pattern hash: 7f6d501ade3d
Shape hash: 233e2ca5
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 476]", addmm_2: "f16[1904, 768]", bmm: "f16[48, 476, 476]"):
        # No stacktrace found for following nodes
        unsqueeze_default: "i64[4, 1, 476]" = torch.ops.aten.unsqueeze.default(arg0_1, 1);  arg0_1 = None
        unsqueeze_default_1: "i64[4, 1, 1, 476]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        convert_element_type_default: "f32[4, 1, 1, 476]" = torch.ops.prims.convert_element_type.default(unsqueeze_default_1, torch.float32);  unsqueeze_default_1 = None
        sub_tensor: "f32[4, 1, 1, 476]" = torch.ops.aten.sub.Tensor(1.0, convert_element_type_default);  convert_element_type_default = None
        mul_tensor: "f32[4, 1, 1, 476]" = torch.ops.aten.mul.Tensor(sub_tensor, -10000.0);  sub_tensor = None
        reshape_default: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(addmm_2, [4, 476, 768]);  addmm_2 = None
        reshape_default_1: "f16[4, 476, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, [4, 476, 12, 64]);  reshape_default = None
        permute_default: "f16[4, 12, 476, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        reshape_default_2: "f16[4, 12, 476, 476]" = torch.ops.aten.reshape.default(bmm, [4, 12, 476, 476]);  bmm = None
        div_tensor: "f16[4, 12, 476, 476]" = torch.ops.aten.div.Tensor(reshape_default_2, 8.0);  reshape_default_2 = None
        add_tensor: "f32[4, 12, 476, 476]" = torch.ops.aten.add.Tensor(div_tensor, mul_tensor);  div_tensor = mul_tensor = None
        amax_default: "f32[4, 12, 476, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor_1: "f32[4, 12, 476, 476]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[4, 12, 476, 476]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[4, 12, 476, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[4, 12, 476, 476]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "f16[4, 12, 476, 476]" = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.float16);  div_tensor_1 = None
        expand_default: "f16[4, 12, 476, 476]" = torch.ops.aten.expand.default(convert_element_type_default_1, [4, 12, 476, 476]);  convert_element_type_default_1 = None
        reshape_default_3: "f16[48, 476, 476]" = torch.ops.aten.reshape.default(expand_default, [48, 476, 476]);  expand_default = None
        expand_default_1: "f16[4, 12, 476, 64]" = torch.ops.aten.expand.default(permute_default, [4, 12, 476, 64]);  permute_default = None
        clone_default: "f16[4, 12, 476, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f16[48, 476, 64]" = torch.ops.aten.reshape.default(clone_default, [48, 476, 64]);  clone_default = None
        return (reshape_default_3, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [4, 476], dtype=torch.int64, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([48, 476, 476], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
