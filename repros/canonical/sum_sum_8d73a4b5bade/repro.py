"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train_001
Pattern hash: 8d73a4b5bade
Shape hash: a17bdef9
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([16, 3, 3, 3], f32), T([16, 3, 3, 3], f32), T([1, 16, 1], f32), T([16], f32), T([16, 1, 1, 1], f32), S([1, 16, 27]), S([1, 16, 27]), S([16, 1, 1, 1]), S([16, 3, 3, 3]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_241: "f32[16, 3, 3, 3]", arg0_1: "f32[16, 3, 3, 3]", arg478_1: "f32[1, 16, 1]", arg152_1: "f32[16]", arg1_1: "f32[16, 1, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        clone_default: "f32[16, 3, 3, 3]" = torch.ops.aten.clone.default(getitem_241, memory_format = torch.contiguous_format);  getitem_241 = None
        view_default: "f32[1, 16, 27]" = torch.ops.aten.view.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        sum_dim_int_list: "f32[16]" = torch.ops.aten.sum.dim_IntList(view_default, [0, 2])
        clone_default_1: "f32[16, 3, 3, 3]" = torch.ops.aten.clone.default(arg0_1, memory_format = torch.contiguous_format);  arg0_1 = None
        view_default_1: "f32[1, 16, 27]" = torch.ops.aten.view.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        sub_tensor: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view_default_1, arg478_1);  view_default_1 = arg478_1 = None
        mul_tensor: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(view_default, sub_tensor)
        sum_dim_int_list_1: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2]);  mul_tensor = None
        mul_tensor_1: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.037037037037037035);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        mul_tensor_2: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.037037037037037035)
        mul_tensor_3: "f32[16]" = torch.ops.aten.mul.Tensor(arg152_1, arg152_1)
        mul_tensor_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_2: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_3: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None
        mul_tensor_5: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg1_1, 0.19245008972987526);  arg1_1 = None
        view_default_2: "f32[16]" = torch.ops.aten.view.default(mul_tensor_5, [-1]);  mul_tensor_5 = None
        mul_tensor_6: "f32[16]" = torch.ops.aten.mul.Tensor(arg152_1, view_default_2);  view_default_2 = None
        unsqueeze_default_4: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_5: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        mul_tensor_7: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        sub_tensor_1: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(view_default, mul_tensor_7);  view_default = mul_tensor_7 = None
        sub_tensor_2: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_1);  sub_tensor_1 = unsqueeze_default_1 = None
        mul_tensor_8: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_5);  sub_tensor_2 = unsqueeze_default_5 = None
        mul_tensor_9: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg152_1);  sum_dim_int_list_1 = arg152_1 = None
        view_default_3: "f32[16, 1, 1, 1]" = torch.ops.aten.view.default(mul_tensor_9, _shape_param_2);  mul_tensor_9 = _shape_param_2 = None
        mul_tensor_10: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(view_default_3, 0.19245008972987526);  view_default_3 = None
        view_default_4: "f32[16, 3, 3, 3]" = torch.ops.aten.view.default(mul_tensor_8, _shape_param_3);  mul_tensor_8 = _shape_param_3 = None
        return (mul_tensor_10, view_default_4)

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
