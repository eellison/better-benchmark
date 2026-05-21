"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_train_001
Pattern hash: fdefa5604cda
Shape hash: 345db9d9
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
_shapes_config = "(T([8192, 512], f32), T([8, 1024, 512], f32), T([8192, 512], f32), T([8192, 512], f32), T([8192, 512], f32), T([8192, 512], f32), T([8192, 512], f32), T([8192, 512], f32), T([8192, 512], f32), T([8192, 512], f32), T([8192, 512], f32), T([8192, 512], f32), T([8192, 512], f32), T([8, 1024, 512], b8), T([512], f32), T([8, 1024, 512], f32), T([8, 1024, 1], f32), T([8, 1024, 512], b8), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([8, 1024, 512]), S([512]), S([8, 1024, 512]), S([8192, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_9: "f32[8192, 512]", arg436_1: "f32[8, 1024, 512]", mm_11: "f32[8192, 512]", mm_29: "f32[8192, 512]", mm_31: "f32[8192, 512]", mm_49: "f32[8192, 512]", mm_51: "f32[8192, 512]", mm_69: "f32[8192, 512]", mm_71: "f32[8192, 512]", mm_89: "f32[8192, 512]", mm_91: "f32[8192, 512]", mm_109: "f32[8192, 512]", mm_111: "f32[8192, 512]", arg217_1: "b8[8, 1024, 512]", arg50_1: "f32[512]", arg215_1: "f32[8, 1024, 512]", arg216_1: "f32[8, 1024, 1]", arg214_1: "b8[8, 1024, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_9, _shape_param_0);  mm_9 = _shape_param_0 = None
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(arg436_1, view_default);  arg436_1 = view_default = None
        view_default_1: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_11, _shape_param_1);  mm_11 = _shape_param_1 = None
        add_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor, view_default_1);  add_tensor = view_default_1 = None
        view_default_2: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_29, _shape_param_2);  mm_29 = _shape_param_2 = None
        add_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_2);  add_tensor_1 = view_default_2 = None
        view_default_3: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_31, _shape_param_3);  mm_31 = _shape_param_3 = None
        add_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_2, view_default_3);  add_tensor_2 = view_default_3 = None
        view_default_4: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_49, _shape_param_4);  mm_49 = _shape_param_4 = None
        add_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_3, view_default_4);  add_tensor_3 = view_default_4 = None
        view_default_5: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_51, _shape_param_5);  mm_51 = _shape_param_5 = None
        add_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_4, view_default_5);  add_tensor_4 = view_default_5 = None
        view_default_6: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_69, _shape_param_6);  mm_69 = _shape_param_6 = None
        add_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_5, view_default_6);  add_tensor_5 = view_default_6 = None
        view_default_7: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_71, _shape_param_7);  mm_71 = _shape_param_7 = None
        add_tensor_7: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_6, view_default_7);  add_tensor_6 = view_default_7 = None
        view_default_8: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_89, _shape_param_8);  mm_89 = _shape_param_8 = None
        add_tensor_8: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_7, view_default_8);  add_tensor_7 = view_default_8 = None
        view_default_9: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_91, _shape_param_9);  mm_91 = _shape_param_9 = None
        add_tensor_9: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_8, view_default_9);  add_tensor_8 = view_default_9 = None
        view_default_10: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_109, _shape_param_10);  mm_109 = _shape_param_10 = None
        add_tensor_10: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_9, view_default_10);  add_tensor_9 = view_default_10 = None
        view_default_11: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_111, _shape_param_11);  mm_111 = _shape_param_11 = None
        add_tensor_11: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_10, view_default_11);  add_tensor_10 = view_default_11 = None
        convert_element_type_default: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg217_1, torch.float32);  arg217_1 = None
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_11, mul_tensor);  add_tensor_11 = mul_tensor = None
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg50_1);  arg50_1 = None
        mul_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg215_1, arg216_1)
        mul_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = mul_tensor_3 = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True);  mul_tensor_4 = None
        view_default_12: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_12);  sum_dim_int_list = _shape_param_12 = None
        mul_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg215_1)
        mul_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg216_1);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        pow_tensor_scalar: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg216_1, 3);  arg216_1 = None
        mul_scalar: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_7: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_7, _shape_param_13);  mul_tensor_7 = _shape_param_13 = None
        div_scalar: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg215_1, 1.0);  arg215_1 = None
        mul_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_8: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_12: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_8);  mul_tensor_6 = mul_tensor_8 = None
        convert_element_type_default_1: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg214_1, torch.float32);  arg214_1 = None
        mul_tensor_9: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_10: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_12, mul_tensor_9);  add_tensor_12 = mul_tensor_9 = None
        view_default_13: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_10, _shape_param_14);  mul_tensor_10 = _shape_param_14 = None
        permute_default: "f32[512, 8192]" = torch.ops.aten.permute.default(view_default_13, [1, 0]);  view_default_13 = None
        return (view_default_12, permute_default)



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
