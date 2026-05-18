"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g62
Pattern hash: 5cbe925f22c7
Shape hash: 4c649a8a
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
    def forward(self, addmm_68: "f16[1904, 768]", bmm_22: "f16[48, 476, 476]", mul: "f32[4, 1, 1, 476]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(addmm_68, _shape_param_0);  addmm_68 = _shape_param_0 = None
        reshape_default_1: "f16[4, 476, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[4, 12, 476, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        reshape_default_2: "f16[4, 12, 476, 476]" = torch.ops.aten.reshape.default(bmm_22, _shape_param_2);  bmm_22 = _shape_param_2 = None
        div_tensor: "f16[4, 12, 476, 476]" = torch.ops.aten.div.Tensor(reshape_default_2, 8.0);  reshape_default_2 = None
        add_tensor: "f32[4, 12, 476, 476]" = torch.ops.aten.add.Tensor(div_tensor, mul);  div_tensor = mul = None
        amax_default: "f32[4, 12, 476, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[4, 12, 476, 476]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[4, 12, 476, 476]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[4, 12, 476, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[4, 12, 476, 476]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default: "f16[4, 12, 476, 476]" = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.float16);  div_tensor_1 = None
        expand_default: "f16[4, 12, 476, 476]" = torch.ops.aten.expand.default(convert_element_type_default, _shape_param_3);  convert_element_type_default = _shape_param_3 = None
        reshape_default_3: "f16[48, 476, 476]" = torch.ops.aten.reshape.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None
        expand_default_1: "f16[4, 12, 476, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f16[4, 12, 476, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f16[48, 476, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        return (reshape_default_3, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([48, 476, 476], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1, 1, 476], dtype=torch.float32, device='cuda'),
    [4, 476, 768],  # _shape_param_0
    [4, 476, 12, 64],  # _shape_param_1
    [4, 12, 476, 476],  # _shape_param_2
    [4, 12, 476, 476],  # _shape_param_3
    [48, 476, 476],  # _shape_param_4
    [4, 12, 476, 64],  # _shape_param_5
    [48, 476, 64],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
