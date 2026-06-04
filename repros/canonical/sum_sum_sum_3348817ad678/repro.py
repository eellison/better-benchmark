"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-30B-A3B_001
Pattern hash: 3348817ad678
Shape hash: 362b8c66
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 2048], bf16), T([2048], bf16), T([4, 512, 2048], bf16), T([4, 512, 1], f32), T([16384], i64, gen=Index(16384)), T([16384, 1], b8), T([16384, 2048], bf16), T([2048, 8], f32), T([16384], i64, gen=Index(16384)), T([2048, 1], f32), T([2048, 8], i64, gen=Index(128)), T([2048, 128], bf16), T([2048, 1], f32), T([2048, 1], f32), S([4, 512, 2048]), S([2048]), S([4, 512, 2048]), S([2048, 2048]), S([2048, 8, 2048]), S([16384, 2048]), S([2048, 8]), S([2048, 8]))"

class Repro(torch.nn.Module):
    def forward(self, mm_1: "bf16[2048, 2048]", arg46_1: "bf16[2048]", arg177_1: "bf16[4, 512, 2048]", arg178_1: "f32[4, 512, 1]", arg176_1: "i64[16384]", arg171_1: "b8[16384, 1]", arg175_1: "bf16[16384, 2048]", arg167_1: "f32[2048, 8]", arg168_1: "i64[16384]", arg166_1: "f32[2048, 1]", arg165_1: "i64[2048, 8]", arg162_1: "bf16[2048, 128]", arg163_1: "f32[2048, 1]", arg164_1: "f32[2048, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        view_default: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(mm_1, _shape_param_0);  mm_1 = _shape_param_0 = None
        mul_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_default, arg46_1);  arg46_1 = None
        convert_element_type_default: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.float32);  arg177_1 = None
        mul_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg178_1)
        convert_element_type_default_1: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(view_default, convert_element_type_default_1);  view_default = convert_element_type_default_1 = None
        sum_dim_int_list: "bf16[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True);  mul_tensor_2 = None
        view_default_1: "bf16[2048]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        convert_element_type_default_2: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        mul_tensor_3: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default)
        mul_tensor_4: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, arg178_1);  convert_element_type_default_2 = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg178_1, 3);  arg178_1 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_5: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_2);  mul_tensor_5 = _shape_param_2 = None
        div_scalar: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_default, 2048);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None
        convert_element_type_default_3: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        view_default_2: "bf16[2048, 2048]" = torch.ops.aten.view.default(convert_element_type_default_3, _shape_param_3);  convert_element_type_default_3 = _shape_param_3 = None
        unsqueeze_default: "bf16[2048, 1, 2048]" = torch.ops.aten.unsqueeze.default(view_default_2, 1);  view_default_2 = None
        expand_default_1: "bf16[2048, 8, 2048]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_4);  unsqueeze_default = _shape_param_4 = None
        clone_default: "bf16[2048, 8, 2048]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_3: "bf16[16384, 2048]" = torch.ops.aten.view.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        full_default: "bf16[16384, 2048]" = torch.ops.aten.full.default([16384, 2048], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "bf16[16384, 2048]" = torch.ops.aten.index_put.default(full_default, [arg176_1], view_default_3, True);  full_default = arg176_1 = view_default_3 = None
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[16384, 2048]" = torch.ops.aten.where.self(arg171_1, full_default_1, index_put_default);  arg171_1 = full_default_1 = index_put_default = None
        mul_tensor_7: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_self, arg175_1);  arg175_1 = None
        convert_element_type_default_4: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.bfloat16)
        view_default_4: "bf16[16384]" = torch.ops.aten.view.default(convert_element_type_default_4, [-1]);  convert_element_type_default_4 = None
        index_tensor: "bf16[16384]" = torch.ops.aten.index.Tensor(view_default_4, [arg168_1]);  view_default_4 = None
        unsqueeze_default_1: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_tensor, -1);  index_tensor = None
        mul_tensor_8: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(where_self, unsqueeze_default_1);  where_self = unsqueeze_default_1 = None
        sum_dim_int_list_2: "bf16[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [1], True);  mul_tensor_7 = None
        squeeze_dim: "bf16[16384]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_2, -1);  sum_dim_int_list_2 = None
        permute_default: "bf16[2048, 16384]" = torch.ops.aten.permute.default(mul_tensor_8, [1, 0]);  mul_tensor_8 = None
        full_default_2: "bf16[16384]" = torch.ops.aten.full.default([16384], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "bf16[16384]" = torch.ops.aten.index_put.default(full_default_2, [arg168_1], squeeze_dim, True);  full_default_2 = arg168_1 = squeeze_dim = None
        view_default_5: "bf16[2048, 8]" = torch.ops.aten.view.default(index_put_default_1, _shape_param_6);  index_put_default_1 = _shape_param_6 = None
        convert_element_type_default_5: "f32[2048, 8]" = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        div_tensor: "f32[2048, 8]" = torch.ops.aten.div.Tensor(arg167_1, arg166_1);  arg167_1 = None
        neg_default: "f32[2048, 8]" = torch.ops.aten.neg.default(convert_element_type_default_5)
        mul_tensor_9: "f32[2048, 8]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor);  neg_default = div_tensor = None
        div_tensor_1: "f32[2048, 8]" = torch.ops.aten.div.Tensor(convert_element_type_default_5, arg166_1);  convert_element_type_default_5 = arg166_1 = None
        sum_dim_int_list_3: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [1], True);  mul_tensor_9 = None
        expand_default_2: "f32[2048, 8]" = torch.ops.aten.expand.default(sum_dim_int_list_3, _shape_param_7);  sum_dim_int_list_3 = _shape_param_7 = None
        add_tensor_1: "f32[2048, 8]" = torch.ops.aten.add.Tensor(div_tensor_1, expand_default_2);  div_tensor_1 = expand_default_2 = None
        full_default_3: "f32[2048, 128]" = torch.ops.aten.full.default([2048, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_src: "f32[2048, 128]" = torch.ops.aten.scatter.src(full_default_3, -1, arg165_1, add_tensor_1);  full_default_3 = arg165_1 = add_tensor_1 = None
        convert_element_type_default_6: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.float32);  arg162_1 = None
        sub_tensor: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default_6, arg163_1);  convert_element_type_default_6 = arg163_1 = None
        exp_default: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor_2: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp_default, arg164_1);  exp_default = arg164_1 = None
        mul_tensor_10: "f32[2048, 128]" = torch.ops.aten.mul.Tensor(scatter_src, div_tensor_2);  scatter_src = None
        sum_dim_int_list_4: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [-1], True)
        neg_default_1: "f32[2048, 128]" = torch.ops.aten.neg.default(div_tensor_2);  div_tensor_2 = None
        fma_default: "f32[2048, 128]" = torch.ops.prims.fma.default(neg_default_1, sum_dim_int_list_4, mul_tensor_10);  neg_default_1 = sum_dim_int_list_4 = mul_tensor_10 = None
        convert_element_type_default_7: "bf16[2048, 128]" = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        permute_default_1: "bf16[128, 2048]" = torch.ops.aten.permute.default(convert_element_type_default_7, [1, 0]);  convert_element_type_default_7 = None
        return (view_default_1, permute_default, permute_default_1)

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
