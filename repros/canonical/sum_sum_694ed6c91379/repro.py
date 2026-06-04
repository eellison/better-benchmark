"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_001
Pattern hash: 694ed6c91379
Shape hash: a884d836
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 1024], bf16), T([2048, 1024], bf16), T([2048, 1024], bf16), T([1024], bf16), T([4, 512, 1024], bf16), T([4, 512, 1], f32), T([4, 512, 1024], bf16), S([4, 512, 1024]), S([4, 512, 1024]), S([4, 512, 1024]), S([1024]), S([4, 512, 1024]), S([2048, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_375: "bf16[2048, 1024]", mm_377: "bf16[2048, 1024]", mm_379: "bf16[2048, 1024]", arg14_1: "bf16[1024]", arg333_1: "bf16[4, 512, 1024]", arg334_1: "f32[4, 512, 1]", add_397: "bf16[4, 512, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(mm_375, _shape_param_0);  mm_375 = _shape_param_0 = None
        view_default_1: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(mm_377, _shape_param_1);  mm_377 = _shape_param_1 = None
        add_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(mm_379, _shape_param_2);  mm_379 = _shape_param_2 = None
        add_tensor_1: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        mul_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg14_1);  arg14_1 = None
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.float32);  arg333_1 = None
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg334_1)
        convert_element_type_default_1: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, convert_element_type_default_1);  add_tensor_1 = convert_element_type_default_1 = None
        sum_dim_int_list: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True);  mul_tensor_2 = None
        view_default_3: "bf16[1024]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        convert_element_type_default_2: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        mul_tensor_3: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default)
        mul_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, arg334_1);  convert_element_type_default_2 = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg334_1, 3);  arg334_1 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_5: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 512, 1024]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_4);  mul_tensor_5 = _shape_param_4 = None
        div_scalar: "f32[4, 512, 1024]" = torch.ops.aten.div.Scalar(expand_default, 1024);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None
        convert_element_type_default_3: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        add_tensor_3: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_397, convert_element_type_default_3);  add_397 = convert_element_type_default_3 = None
        view_default_4: "bf16[2048, 1024]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_5);  add_tensor_3 = _shape_param_5 = None
        permute_default: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_default_4, [1, 0]);  view_default_4 = None
        return (view_default_3, permute_default)

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
