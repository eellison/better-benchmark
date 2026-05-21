"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train_001
Pattern hash: 6793fecde620
Shape hash: 54b7bd15
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
_shapes_config = "(T([4096, 512], f32), T([32, 128, 512], b8), T([512], f32), T([32, 128, 512], f32), T([32, 128, 1], f32), T([32, 128, 512], b8), S([32, 128, 512]), S([32, 128, 512]), S([4096, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_1: "f32[4096, 512]", arg498_1: "b8[32, 128, 512]", arg188_1: "f32[512]", arg496_1: "f32[32, 128, 512]", arg497_1: "f32[32, 128, 1]", arg495_1: "b8[32, 128, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128, 512]" = torch.ops.aten.view.default(mm_1, _shape_param_0);  mm_1 = _shape_param_0 = None
        convert_element_type_default: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg498_1, torch.float32);  arg498_1 = None
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_default, mul_tensor);  view_default = mul_tensor = None
        mul_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg188_1);  mul_tensor_1 = arg188_1 = None
        mul_tensor_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg496_1)
        mul_tensor_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg497_1);  mul_tensor_2 = None
        sum_dim_int_list: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg497_1, 3);  arg497_1 = None
        mul_scalar: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_5: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_1);  mul_tensor_5 = _shape_param_1 = None
        div_scalar: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg496_1, 1.0);  arg496_1 = None
        mul_scalar_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None
        convert_element_type_default_1: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg495_1, torch.float32);  arg495_1 = None
        mul_tensor_7: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_8: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_7);  add_tensor = mul_tensor_7 = None
        view_default_1: "f32[4096, 512]" = torch.ops.aten.view.default(mul_tensor_8, _shape_param_2);  mul_tensor_8 = _shape_param_2 = None
        return view_default_1



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
