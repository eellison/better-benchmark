"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_001
Pattern hash: b79adf915966
Shape hash: c9aa1d29
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
_shapes_config = "(T([2048, 1024], bf16), T([1024], bf16), T([4, 512, 1024], bf16), T([4, 512, 1], f32), S([4, 512, 1024]), S([4, 512, 1024]), S([2048, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_1: "bf16[2048, 1024]", arg311_1: "bf16[1024]", arg873_1: "bf16[4, 512, 1024]", arg874_1: "f32[4, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(mm_1, _shape_param_0);  mm_1 = _shape_param_0 = None
        mul_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_default, arg311_1);  view_default = arg311_1 = None
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg873_1, torch.float32);  arg873_1 = None
        convert_element_type_default_1: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default)
        mul_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, arg874_1);  convert_element_type_default_1 = None
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg874_1, 3);  arg874_1 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 512, 1024]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        div_scalar: "f32[4, 512, 1024]" = torch.ops.aten.div.Scalar(expand_default, 1024);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None
        convert_element_type_default_2: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        view_default_1: "bf16[2048, 1024]" = torch.ops.aten.view.default(convert_element_type_default_2, _shape_param_2);  convert_element_type_default_2 = _shape_param_2 = None
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
