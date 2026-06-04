"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train_001
Pattern hash: a2d8f03e6a24
Shape hash: 9e167a03
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 1024], f32), T([32, 128, 1024], b8), T([4096, 1024], f32), T([4096, 1024], f32), S([32, 128, 1024]), S([32, 128, 1024]), S([32, 128, 1024]), S([4096, 1024]), S([4096, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_277: "f32[4096, 1024]", arg207_1: "b8[32, 128, 1024]", arg205_1: "f32[4096, 1024]", arg206_1: "f32[4096, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128, 1024]" = torch.ops.aten.view.default(mm_277, _shape_param_0);  mm_277 = _shape_param_0 = None
        convert_element_type_default: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg207_1, torch.float32);  arg207_1 = None
        mul_tensor: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  view_default = mul_tensor = None
        view_default_1: "f32[32, 128, 1024]" = torch.ops.aten.view.default(arg205_1, _shape_param_1);  arg205_1 = _shape_param_1 = None
        mul_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_default_1, 0.5)
        pow_tensor_scalar: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_default_1, 3.0)
        mul_tensor_3: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_default_1, mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_5: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1)
        mul_tensor_6: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_5 = None
        view_default_2: "f32[32, 128, 1024]" = torch.ops.aten.view.default(arg206_1, _shape_param_2);  arg206_1 = _shape_param_2 = None
        mul_tensor_7: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, view_default_2);  mul_tensor_1 = view_default_2 = None
        view_default_3: "f32[4096, 1024]" = torch.ops.aten.view.default(mul_tensor_6, _shape_param_3);  mul_tensor_6 = _shape_param_3 = None
        permute_default: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
        mul_tensor_8: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_9: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_7, add_tensor_1);  mul_tensor_7 = add_tensor_1 = None
        mul_tensor_10: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_tensor_10);  mul_tensor_10 = None
        mul_tensor_11: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_8, sub_tensor);  mul_tensor_8 = sub_tensor = None
        mul_tensor_12: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_11, 0.7978845608028654);  mul_tensor_11 = None
        mul_tensor_13: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.044715)
        pow_tensor_scalar_1: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_default_1, 2.0);  view_default_1 = None
        mul_scalar: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_14: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_13, mul_scalar);  mul_tensor_13 = mul_scalar = None
        add_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_12, mul_tensor_14);  mul_tensor_12 = mul_tensor_14 = None
        mul_tensor_15: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_9, 0.5);  mul_tensor_9 = None
        add_tensor_3: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_15);  add_tensor_2 = mul_tensor_15 = None
        view_default_4: "f32[4096, 1024]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_4);  add_tensor_3 = _shape_param_4 = None
        permute_default_1: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_default_4, [1, 0]);  view_default_4 = None
        return (permute_default, permute_default_1)

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
