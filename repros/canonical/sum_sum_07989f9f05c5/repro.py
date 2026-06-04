"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train_001
Pattern hash: 07989f9f05c5
Shape hash: 7a85eacb
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 512], f32), T([4096, 512], f32), T([512], f32), T([32, 128, 512], f32), T([32, 128, 1], f32), T([32, 128, 512], f32), T([32, 128, 512], b8), S([32, 128, 512]), S([32, 128, 512]), S([512]), S([32, 128, 512]), S([4096, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_279: "f32[4096, 512]", mm_281: "f32[4096, 512]", arg7_1: "f32[512]", arg202_1: "f32[32, 128, 512]", arg203_1: "f32[32, 128, 1]", add_213: "f32[32, 128, 512]", arg201_1: "b8[32, 128, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_279, _shape_param_0);  mm_279 = _shape_param_0 = None
        view_default_1: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_281, _shape_param_1);  mm_281 = _shape_param_1 = None
        add_tensor: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor, arg7_1);  arg7_1 = None
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg202_1, arg203_1)
        mul_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_1);  add_tensor = mul_tensor_1 = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True);  mul_tensor_2 = None
        view_default_2: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_2);  sum_dim_int_list = _shape_param_2 = None
        mul_tensor_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg202_1)
        mul_tensor_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg203_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        add_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_213, mul_tensor_4);  add_213 = mul_tensor_4 = None
        pow_tensor_scalar: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg203_1, 3);  arg203_1 = None
        mul_scalar: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_5: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_3);  mul_tensor_5 = _shape_param_3 = None
        div_scalar: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg202_1, 1.0);  arg202_1 = None
        mul_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_6);  add_tensor_1 = mul_tensor_6 = None
        convert_element_type_default: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg201_1, torch.float32);  arg201_1 = None
        mul_tensor_7: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_8: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_7);  add_tensor_2 = mul_tensor_7 = None
        view_default_3: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_8, _shape_param_4);  mul_tensor_8 = _shape_param_4 = None
        permute_default: "f32[512, 4096]" = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
        return (view_default_2, permute_default)

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
