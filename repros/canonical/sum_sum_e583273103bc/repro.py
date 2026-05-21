"""
Standalone repro captured via capture_hook.
Label: timm_timm_vit_base_patch16_siglip_256_train_train_001
Pattern hash: e583273103bc
Shape hash: 4df072a7
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
_shapes_config = "(T([128, 768], f32), T([768], f32), T([128, 768], f32), T([128, 1, 1], f32), T([128, 1, 1], f32), T([128, 1, 768], f32), S([128, 1, 768]), S([128, 1, 768]), S([128, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_2: "f32[128, 768]", arg80_1: "f32[768]", arg250_1: "f32[128, 768]", arg251_1: "f32[128, 1, 1]", arg252_1: "f32[128, 1, 1]", select_scatter: "f32[128, 1, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[128, 1, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_0);  mm_2 = _shape_param_0 = None
        mul_tensor: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(view_default, arg80_1);  view_default = arg80_1 = None
        mul_tensor_1: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        view_default_1: "f32[128, 1, 768]" = torch.ops.aten.view.default(arg250_1, _shape_param_1);  arg250_1 = _shape_param_1 = None
        sub_tensor: "f32[128, 1, 768]" = torch.ops.aten.sub.Tensor(view_default_1, arg251_1);  view_default_1 = arg251_1 = None
        mul_tensor_2: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, arg252_1);  sub_tensor = None
        mul_tensor_3: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 1, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 1, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(arg252_1, 768);  arg252_1 = None
        mul_tensor_5: "f32[128, 1, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor: "f32[128, 1, 768]" = torch.ops.aten.add.Tensor(select_scatter, mul_tensor_5);  select_scatter = mul_tensor_5 = None
        view_default_2: "f32[128, 768]" = torch.ops.aten.view.default(add_tensor, _shape_param_2);  add_tensor = _shape_param_2 = None
        return view_default_2



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
