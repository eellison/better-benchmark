"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train_001
Pattern hash: e106b4fbe172
Shape hash: 268accaf
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 512], f32), T([32, 128, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([4096, 512], f32), T([32, 128, 512], b8), T([512], f32), T([32, 128, 512], f32), T([32, 128, 1], f32), T([32, 128, 512], b8), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([32, 128, 512]), S([512]), S([32, 128, 512]), S([4096, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_11: "f32[4096, 512]", arg602_1: "f32[32, 128, 512]", mm_13: "f32[4096, 512]", mm_33: "f32[4096, 512]", mm_35: "f32[4096, 512]", mm_55: "f32[4096, 512]", mm_57: "f32[4096, 512]", mm_77: "f32[4096, 512]", mm_79: "f32[4096, 512]", mm_99: "f32[4096, 512]", mm_101: "f32[4096, 512]", mm_121: "f32[4096, 512]", mm_123: "f32[4096, 512]", mm_143: "f32[4096, 512]", mm_145: "f32[4096, 512]", mm_165: "f32[4096, 512]", mm_167: "f32[4096, 512]", arg317_1: "b8[32, 128, 512]", arg74_1: "f32[512]", arg315_1: "f32[32, 128, 512]", arg316_1: "f32[32, 128, 1]", arg314_1: "b8[32, 128, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_11, _shape_param_0);  mm_11 = _shape_param_0 = None
        add_tensor: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(arg602_1, view_default);  arg602_1 = view_default = None
        view_default_1: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_13, _shape_param_1);  mm_13 = _shape_param_1 = None
        add_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, view_default_1);  add_tensor = view_default_1 = None
        view_default_2: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_33, _shape_param_2);  mm_33 = _shape_param_2 = None
        add_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_2);  add_tensor_1 = view_default_2 = None
        view_default_3: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_35, _shape_param_3);  mm_35 = _shape_param_3 = None
        add_tensor_3: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_2, view_default_3);  add_tensor_2 = view_default_3 = None
        view_default_4: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_55, _shape_param_4);  mm_55 = _shape_param_4 = None
        add_tensor_4: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_3, view_default_4);  add_tensor_3 = view_default_4 = None
        view_default_5: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_57, _shape_param_5);  mm_57 = _shape_param_5 = None
        add_tensor_5: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_4, view_default_5);  add_tensor_4 = view_default_5 = None
        view_default_6: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_77, _shape_param_6);  mm_77 = _shape_param_6 = None
        add_tensor_6: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_5, view_default_6);  add_tensor_5 = view_default_6 = None
        view_default_7: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_79, _shape_param_7);  mm_79 = _shape_param_7 = None
        add_tensor_7: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_6, view_default_7);  add_tensor_6 = view_default_7 = None
        view_default_8: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_99, _shape_param_8);  mm_99 = _shape_param_8 = None
        add_tensor_8: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_7, view_default_8);  add_tensor_7 = view_default_8 = None
        view_default_9: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_101, _shape_param_9);  mm_101 = _shape_param_9 = None
        add_tensor_9: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_8, view_default_9);  add_tensor_8 = view_default_9 = None
        view_default_10: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_121, _shape_param_10);  mm_121 = _shape_param_10 = None
        add_tensor_10: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_9, view_default_10);  add_tensor_9 = view_default_10 = None
        view_default_11: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_123, _shape_param_11);  mm_123 = _shape_param_11 = None
        add_tensor_11: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_10, view_default_11);  add_tensor_10 = view_default_11 = None
        view_default_12: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_143, _shape_param_12);  mm_143 = _shape_param_12 = None
        add_tensor_12: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_11, view_default_12);  add_tensor_11 = view_default_12 = None
        view_default_13: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_145, _shape_param_13);  mm_145 = _shape_param_13 = None
        add_tensor_13: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_12, view_default_13);  add_tensor_12 = view_default_13 = None
        view_default_14: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_165, _shape_param_14);  mm_165 = _shape_param_14 = None
        add_tensor_14: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_13, view_default_14);  add_tensor_13 = view_default_14 = None
        view_default_15: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_167, _shape_param_15);  mm_167 = _shape_param_15 = None
        add_tensor_15: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor_14, view_default_15);  add_tensor_14 = view_default_15 = None
        convert_element_type_default: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.float32);  arg317_1 = None
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_15, mul_tensor);  add_tensor_15 = mul_tensor = None
        mul_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg74_1);  arg74_1 = None
        mul_tensor_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg315_1, arg316_1)
        mul_tensor_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = mul_tensor_3 = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True);  mul_tensor_4 = None
        view_default_16: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_16);  sum_dim_int_list = _shape_param_16 = None
        mul_tensor_5: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg315_1)
        mul_tensor_6: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg316_1);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        pow_tensor_scalar: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg316_1, 3);  arg316_1 = None
        mul_scalar: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_7: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_tensor_7, _shape_param_17);  mul_tensor_7 = _shape_param_17 = None
        div_scalar: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg315_1, 1.0);  arg315_1 = None
        mul_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_8: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_16: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_8);  mul_tensor_6 = mul_tensor_8 = None
        convert_element_type_default_1: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg314_1, torch.float32);  arg314_1 = None
        mul_tensor_9: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_10: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_16, mul_tensor_9);  add_tensor_16 = mul_tensor_9 = None
        view_default_17: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_10, _shape_param_18);  mul_tensor_10 = _shape_param_18 = None
        permute_default: "f32[512, 4096]" = torch.ops.aten.permute.default(view_default_17, [1, 0]);  view_default_17 = None
        return (view_default_16, permute_default)

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
