"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-30B-A3B_001
Pattern hash: 3457474b535c
Shape hash: 7eddf7fa
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
_shapes_config = "(T([2048, 2048], bf16), T([2048], bf16), T([4, 512, 2048], bf16), T([4, 512, 1], f32), T([16384], i64, gen=Index(16384)), T([16384, 1], b8), T([2048, 8], f32), T([16384], i64, gen=Index(16384)), S([4, 512, 2048]), S([4, 512, 2048]), S([2048, 2048]), S([2048, 8, 2048]), S([16384, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, mm_1: "bf16[2048, 2048]", arg46_1: "bf16[2048]", arg177_1: "bf16[4, 512, 2048]", arg178_1: "f32[4, 512, 1]", arg176_1: "i64[16384]", arg171_1: "b8[16384, 1]", arg167_1: "f32[2048, 8]", arg168_1: "i64[16384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(mm_1, _shape_param_0);  mm_1 = _shape_param_0 = None
        mul_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_default, arg46_1);  view_default = arg46_1 = None
        convert_element_type_default: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.float32);  arg177_1 = None
        convert_element_type_default_1: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        mul_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default)
        mul_tensor_2: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, arg178_1);  convert_element_type_default_1 = None
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg178_1, 3);  arg178_1 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        div_scalar: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_default, 2048);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None
        convert_element_type_default_2: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        view_default_1: "bf16[2048, 2048]" = torch.ops.aten.view.default(convert_element_type_default_2, _shape_param_2);  convert_element_type_default_2 = _shape_param_2 = None
        unsqueeze_default: "bf16[2048, 1, 2048]" = torch.ops.aten.unsqueeze.default(view_default_1, 1);  view_default_1 = None
        expand_default_1: "bf16[2048, 8, 2048]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_3);  unsqueeze_default = _shape_param_3 = None
        clone_default: "bf16[2048, 8, 2048]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_2: "bf16[16384, 2048]" = torch.ops.aten.view.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        full_default: "bf16[16384, 2048]" = torch.ops.aten.full.default([16384, 2048], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "bf16[16384, 2048]" = torch.ops.aten.index_put.default(full_default, [arg176_1], view_default_2, True);  full_default = arg176_1 = view_default_2 = None
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[16384, 2048]" = torch.ops.aten.where.self(arg171_1, full_default_1, index_put_default);  arg171_1 = full_default_1 = index_put_default = None
        convert_element_type_default_3: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.bfloat16);  arg167_1 = None
        view_default_3: "bf16[16384]" = torch.ops.aten.view.default(convert_element_type_default_3, [-1]);  convert_element_type_default_3 = None
        index_tensor: "bf16[16384]" = torch.ops.aten.index.Tensor(view_default_3, [arg168_1]);  view_default_3 = arg168_1 = None
        unsqueeze_default_1: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_tensor, -1);  index_tensor = None
        mul_tensor_5: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_self, unsqueeze_default_1);  where_self = unsqueeze_default_1 = None
        return mul_tensor_5



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
