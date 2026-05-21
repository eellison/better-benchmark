"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train_001
Pattern hash: ad5c5ecccffb
Shape hash: eec381ba
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([16384, 768], f32), T([128, 768], f32), T([128, 128, 768], b8), T([128, 128, 768], b8), S([128, 128, 768]), S([16384, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[16384, 768]", mm_2: "f32[128, 768]", arg283_1: "b8[128, 128, 768]", arg282_1: "b8[128, 128, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[128, 128, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        full_default: "f32[128, 128, 768]" = torch.ops.aten.full.default([128, 128, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[128, 128, 768]" = torch.ops.aten.select_scatter.default(full_default, mm_2, 1, 0);  full_default = mm_2 = None
        add_tensor: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_default, select_scatter_default);  view_default = select_scatter_default = None
        convert_element_type_default: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(arg283_1, torch.float32);  arg283_1 = None
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor);  add_tensor = mul_tensor = None
        convert_element_type_default_1: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.float32);  arg282_1 = None
        mul_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        view_default_1: "f32[16384, 768]" = torch.ops.aten.view.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        permute_default: "f32[768, 16384]" = torch.ops.aten.permute.default(view_default_1, [1, 0])
        sum_dim_int_list: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True);  view_default_1 = None
        view_default_2: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_2);  sum_dim_int_list = _shape_param_2 = None
        return (permute_default, view_default_2)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
