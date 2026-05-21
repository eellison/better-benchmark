"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-30B-A3B_001
Pattern hash: 3d7b75da73d0
Shape hash: 3b4ca287
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
_shapes_config = "(T([16384, 1], b8), T([], bf16), T([16384, 2048], bf16), T([16384], i64, gen=Index(2048)), T([2048, 2048], bf16), T([2048], bf16), T([4, 512, 2048], bf16), T([4, 512, 1], f32), T([4, 512, 2048], bf16), S([4, 512, 2048]), S([4, 512, 2048]), S([2048, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, arg171_1: "b8[16384, 1]", full_3: "bf16[]", _grouped_mm_3: "bf16[16384, 2048]", arg169_1: "i64[16384]", mm_3: "bf16[2048, 2048]", arg42_1: "bf16[2048]", arg159_1: "bf16[4, 512, 2048]", arg160_1: "f32[4, 512, 1]", convert_element_type_5: "bf16[4, 512, 2048]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        where_self: "bf16[16384, 2048]" = torch.ops.aten.where.self(arg171_1, full_3, _grouped_mm_3);  arg171_1 = full_3 = _grouped_mm_3 = None
        full_default: "bf16[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "bf16[2048, 2048]" = torch.ops.aten.index_put.default(full_default, [arg169_1], where_self, True);  full_default = arg169_1 = where_self = None
        add_tensor: "bf16[2048, 2048]" = torch.ops.aten.add.Tensor(index_put_default, mm_3);  index_put_default = mm_3 = None
        view_default: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(add_tensor, _shape_param_0);  add_tensor = _shape_param_0 = None
        mul_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_default, arg42_1);  view_default = arg42_1 = None
        convert_element_type_default: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(arg159_1, torch.float32);  arg159_1 = None
        convert_element_type_default_1: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        mul_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default)
        mul_tensor_2: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, arg160_1);  convert_element_type_default_1 = None
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg160_1, 3);  arg160_1 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        div_scalar: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_default, 2048);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None
        convert_element_type_default_2: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        add_tensor_2: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_5, convert_element_type_default_2);  convert_element_type_5 = convert_element_type_default_2 = None
        view_default_1: "bf16[2048, 2048]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
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
