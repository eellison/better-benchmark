"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-30B-A3B_001
Pattern hash: 7a1a63f6113c
Shape hash: 2d1fa603
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
_shapes_config = "(T([2048, 2048], bf16), T([2048, 2048], bf16), T([2048, 2048], bf16), T([2048], bf16), T([4, 512, 2048], bf16), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([16384, 2048], bf16), T([16384], i64, gen=Index(16384)), T([16384, 1], b8), T([], bf16), T([2048, 8], f32), T([16384], i64, gen=Index(16384)), S([4, 512, 2048]), S([4, 512, 2048]), S([4, 512, 2048]), S([4, 512, 2048]), S([2048, 2048]), S([2048, 8, 2048]), S([16384, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, mm_27: "bf16[2048, 2048]", mm_29: "bf16[2048, 2048]", mm_31: "bf16[2048, 2048]", arg13_1: "bf16[2048]", arg81_1: "bf16[4, 512, 2048]", arg82_1: "f32[4, 512, 1]", add_40: "bf16[4, 512, 2048]", full_2: "bf16[16384, 2048]", arg80_1: "i64[16384]", arg75_1: "b8[16384, 1]", full_3: "bf16[]", arg71_1: "f32[2048, 8]", arg72_1: "i64[16384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(mm_27, _shape_param_0);  mm_27 = _shape_param_0 = None
        view_default_1: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(mm_29, _shape_param_1);  mm_29 = _shape_param_1 = None
        add_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(mm_31, _shape_param_2);  mm_31 = _shape_param_2 = None
        add_tensor_1: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        mul_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg13_1);  add_tensor_1 = arg13_1 = None
        convert_element_type_default: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(arg81_1, torch.float32);  arg81_1 = None
        convert_element_type_default_1: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        mul_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default)
        mul_tensor_2: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, arg82_1);  convert_element_type_default_1 = None
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg82_1, 3);  arg82_1 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_3);  mul_tensor_3 = _shape_param_3 = None
        div_scalar: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_default, 2048);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None
        convert_element_type_default_2: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        add_tensor_3: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_40, convert_element_type_default_2);  add_40 = convert_element_type_default_2 = None
        view_default_3: "bf16[2048, 2048]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_4);  add_tensor_3 = _shape_param_4 = None
        unsqueeze_default: "bf16[2048, 1, 2048]" = torch.ops.aten.unsqueeze.default(view_default_3, 1);  view_default_3 = None
        expand_default_1: "bf16[2048, 8, 2048]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_5);  unsqueeze_default = _shape_param_5 = None
        clone_default: "bf16[2048, 8, 2048]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_4: "bf16[16384, 2048]" = torch.ops.aten.view.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        index_put_default: "bf16[16384, 2048]" = torch.ops.aten.index_put.default(full_2, [arg80_1], view_default_4, True);  full_2 = arg80_1 = view_default_4 = None
        where_self: "bf16[16384, 2048]" = torch.ops.aten.where.self(arg75_1, full_3, index_put_default);  arg75_1 = full_3 = index_put_default = None
        convert_element_type_default_3: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(arg71_1, torch.bfloat16);  arg71_1 = None
        view_default_5: "bf16[16384]" = torch.ops.aten.view.default(convert_element_type_default_3, [-1]);  convert_element_type_default_3 = None
        index_tensor: "bf16[16384]" = torch.ops.aten.index.Tensor(view_default_5, [arg72_1]);  view_default_5 = arg72_1 = None
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
