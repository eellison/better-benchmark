"""
Standalone repro captured via capture_hook.
Label: vllm_mistralai_Mistral-7B-Instruct-v0.3_001
Pattern hash: 17a11b547fed
Shape hash: 1264c06c
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
_shapes_config = "(T([2048, 4096], bf16), T([2048, 4096], bf16), T([2048, 4096], bf16), T([4096], bf16), T([4, 512, 4096], bf16), T([4, 512, 1], f32), T([4, 512, 4096], bf16), T([4, 512], i64, gen=Index(32768)), T([], f32), S([4, 512, 4096]), S([4, 512, 4096]), S([4, 512, 4096]), S([4096]), S([4, 512, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, mm_109: "bf16[2048, 4096]", mm_111: "bf16[2048, 4096]", mm_113: "bf16[2048, 4096]", arg2_1: "bf16[4096]", arg76_1: "bf16[4, 512, 4096]", arg77_1: "f32[4, 512, 1]", add_99: "bf16[4, 512, 4096]", arg0_1: "i64[4, 512]", full_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 4096]" = torch.ops.aten.view.default(mm_109, _shape_param_0);  mm_109 = _shape_param_0 = None
        view_default_1: "bf16[4, 512, 4096]" = torch.ops.aten.view.default(mm_111, _shape_param_1);  mm_111 = _shape_param_1 = None
        add_tensor: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "bf16[4, 512, 4096]" = torch.ops.aten.view.default(mm_113, _shape_param_2);  mm_113 = _shape_param_2 = None
        add_tensor_1: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        mul_tensor: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg2_1);  arg2_1 = None
        convert_element_type_default: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(arg76_1, torch.float32);  arg76_1 = None
        mul_tensor_1: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg77_1)
        convert_element_type_default_1: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2: "bf16[4, 512, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_1, convert_element_type_default_1);  add_tensor_1 = convert_element_type_default_1 = None
        sum_dim_int_list: "bf16[1, 1, 4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True);  mul_tensor_2 = None
        view_default_3: "bf16[4096]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None
        convert_element_type_default_2: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        mul_tensor_3: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default)
        mul_tensor_4: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, arg77_1);  convert_element_type_default_2 = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg77_1, 3);  arg77_1 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_5: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 512, 4096]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_4);  mul_tensor_5 = _shape_param_4 = None
        div_scalar: "f32[4, 512, 4096]" = torch.ops.aten.div.Scalar(expand_default, 4096);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 4096]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None
        convert_element_type_default_3: "bf16[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        add_tensor_3: "bf16[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_99, convert_element_type_default_3);  add_99 = convert_element_type_default_3 = None
        convert_element_type_default_4: "f32[4, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32);  add_tensor_3 = None
        eq_scalar: "b8[4, 512]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[4, 512, 4096]" = torch.ops.aten.where.self(unsqueeze_default, full_1, convert_element_type_default_4);  unsqueeze_default = full_1 = convert_element_type_default_4 = None
        full_default: "f32[32768, 4096]" = torch.ops.aten.full.default([32768, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32768, 4096]" = torch.ops.aten.index_put.default(full_default, [arg0_1], where_self, True);  full_default = arg0_1 = where_self = None
        convert_element_type_default_5: "bf16[32768, 4096]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.bfloat16);  index_put_default = None
        return (view_default_3, convert_element_type_default_5)



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
