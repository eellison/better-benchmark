"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-0.6B_001
Pattern hash: 230671a8764d
Shape hash: f1707ced
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
_shapes_config = "(T([4, 16, 512, 128], bf16), T([1, 1, 512, 128], bf16), T([4, 16, 512, 128], bf16), T([1, 1, 512, 128], bf16), T([128], bf16), T([2048, 2048], bf16), T([4, 512, 16, 1], f32), S([4, 512, 2048]), S([4, 512, -1, 128]), S([128]), S([4, 512, 16, 128]), S([4, 512, 2048]), S([2048, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_81: "bf16[4, 16, 512, 128]", unsqueeze_6: "bf16[1, 1, 512, 128]", full_3: "bf16[4, 16, 512, 128]", unsqueeze_7: "bf16[1, 1, 512, 128]", arg5_1: "bf16[128]", arg315_1: "bf16[2048, 2048]", arg316_1: "f32[4, 512, 16, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mul_tensor: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_81, unsqueeze_6);  unsqueeze_6 = None
        slice_tensor: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor, 3, 0, 64)
        slice_tensor_1: "bf16[4, 16, 512, 64]" = torch.ops.aten.slice.Tensor(mul_tensor, 3, 64, 128);  mul_tensor = None
        neg_default: "bf16[4, 16, 512, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None
        slice_scatter_default: "bf16[4, 16, 512, 128]" = torch.ops.aten.slice_scatter.default(full_3, neg_default, 3, 64, 9223372036854775807);  neg_default = None
        slice_scatter_default_1: "bf16[4, 16, 512, 128]" = torch.ops.aten.slice_scatter.default(full_3, slice_tensor_1, 3, 0, 64);  full_3 = slice_tensor_1 = None
        add_tensor: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1);  slice_scatter_default = slice_scatter_default_1 = None
        mul_tensor_1: "bf16[4, 16, 512, 128]" = torch.ops.aten.mul.Tensor(getitem_81, unsqueeze_7);  getitem_81 = unsqueeze_7 = None
        add_tensor_1: "bf16[4, 16, 512, 128]" = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_1);  add_tensor = mul_tensor_1 = None
        permute_default: "bf16[4, 512, 16, 128]" = torch.ops.aten.permute.default(add_tensor_1, [0, 2, 1, 3]);  add_tensor_1 = None
        mul_tensor_2: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_default, arg5_1);  arg5_1 = None
        view_default: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(arg315_1, _shape_param_0);  arg315_1 = _shape_param_0 = None
        view_default_1: "bf16[4, 512, 16, 128]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        convert_element_type_default: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        mul_tensor_3: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg316_1)
        convert_element_type_default_1: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        mul_tensor_4: "bf16[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(permute_default, convert_element_type_default_1);  permute_default = convert_element_type_default_1 = None
        sum_dim_int_list: "bf16[1, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1, 2], True);  mul_tensor_4 = None
        view_default_2: "bf16[128]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_2);  sum_dim_int_list = _shape_param_2 = None
        convert_element_type_default_2: "f32[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float32);  mul_tensor_2 = None
        mul_tensor_5: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default)
        mul_tensor_6: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, arg316_1);  convert_element_type_default_2 = None
        sum_dim_int_list_1: "f32[4, 512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [3], True);  mul_tensor_5 = None
        pow_tensor_scalar: "f32[4, 512, 16, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg316_1, 3);  arg316_1 = None
        mul_scalar: "f32[4, 512, 16, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_7: "f32[4, 512, 16, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 512, 16, 128]" = torch.ops.aten.expand.default(mul_tensor_7, _shape_param_3);  mul_tensor_7 = _shape_param_3 = None
        div_scalar: "f32[4, 512, 16, 128]" = torch.ops.aten.div.Scalar(expand_default, 128);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 16, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_8: "f32[4, 512, 16, 128]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 16, 128]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_8);  mul_tensor_6 = mul_tensor_8 = None
        convert_element_type_default_3: "bf16[4, 512, 16, 128]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        clone_default: "bf16[4, 512, 16, 128]" = torch.ops.aten.clone.default(convert_element_type_default_3, memory_format = torch.contiguous_format);  convert_element_type_default_3 = None
        view_default_3: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        view_default_4: "bf16[2048, 2048]" = torch.ops.aten.view.default(view_default_3, _shape_param_5);  view_default_3 = _shape_param_5 = None
        permute_default_1: "bf16[2048, 2048]" = torch.ops.aten.permute.default(view_default_4, [1, 0]);  view_default_4 = None
        return (view_default_2, permute_default_1)



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
