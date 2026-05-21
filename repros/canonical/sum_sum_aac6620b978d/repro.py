"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_001
Pattern hash: aac6620b978d
Shape hash: 432917d8
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
_shapes_config = "(T([2048, 1024], bf16), T([2048, 1024], bf16), T([1024], bf16), T([4, 512, 1024], bf16), T([4, 512, 1], f32), T([4, 512, 1024], bf16), S([4, 512, 1024]), S([4, 512, 1024]), S([1024]), S([4, 512, 1024]), S([2048, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_369: "bf16[2048, 1024]", mm_371: "bf16[2048, 1024]", arg21_1: "bf16[1024]", arg347_1: "bf16[4, 512, 1024]", arg348_1: "f32[4, 512, 1]", add_392: "bf16[4, 512, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(mm_369, _shape_param_0);  mm_369 = _shape_param_0 = None
        view_default_1: "bf16[4, 512, 1024]" = torch.ops.aten.view.default(mm_371, _shape_param_1);  mm_371 = _shape_param_1 = None
        add_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        mul_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, arg21_1);  arg21_1 = None
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg347_1, torch.float32);  arg347_1 = None
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg348_1)
        convert_element_type_default_1: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, convert_element_type_default_1);  add_tensor = convert_element_type_default_1 = None
        sum_dim_int_list: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True);  mul_tensor_2 = None
        view_default_2: "bf16[1024]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_2);  sum_dim_int_list = _shape_param_2 = None
        convert_element_type_default_2: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        mul_tensor_3: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default)
        mul_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, arg348_1);  convert_element_type_default_2 = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg348_1, 3);  arg348_1 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_5: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 512, 1024]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_3);  mul_tensor_5 = _shape_param_3 = None
        div_scalar: "f32[4, 512, 1024]" = torch.ops.aten.div.Scalar(expand_default, 1024);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None
        convert_element_type_default_3: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        add_tensor_2: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_392, convert_element_type_default_3);  add_392 = convert_element_type_default_3 = None
        view_default_3: "bf16[2048, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_4);  add_tensor_2 = _shape_param_4 = None
        permute_default: "bf16[1024, 2048]" = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
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
