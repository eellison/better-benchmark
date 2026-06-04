"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-30B-A3B_000
Pattern hash: 9af96955f8cc
Shape hash: 8f7ea893
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 8], f32), T([16384], i64, gen=Perm(16384)), T([16384, 2048], bf16), T([16384, 1], b8), T([4, 512, 2048], bf16), T([2048], bf16), S([2048, 8, 2048]), S([4, 512, 2048]), S([2048, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_54: "f32[2048, 8]", getitem_57: "i64[16384]", _grouped_mm_7: "bf16[16384, 2048]", unsqueeze_36: "b8[16384, 1]", add_35: "bf16[4, 512, 2048]", arg47_1: "bf16[2048]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sum_dim_int_list: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(getitem_54, [-1], True)
        div_tensor: "f32[2048, 8]" = torch.ops.aten.div.Tensor(getitem_54, sum_dim_int_list);  getitem_54 = sum_dim_int_list = None
        convert_element_type_default: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        view_default: "bf16[16384]" = torch.ops.aten.view.default(convert_element_type_default, [-1]);  convert_element_type_default = None
        index_tensor: "bf16[16384]" = torch.ops.aten.index.Tensor(view_default, [getitem_57]);  view_default = None
        unsqueeze_default: "bf16[16384, 1]" = torch.ops.aten.unsqueeze.default(index_tensor, -1);  index_tensor = None
        mul_tensor: "bf16[16384, 2048]" = torch.ops.aten.mul.Tensor(_grouped_mm_7, unsqueeze_default);  _grouped_mm_7 = unsqueeze_default = None
        where_self: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_36, full_default, mul_tensor);  unsqueeze_36 = full_default = mul_tensor = None
        empty_memory_format: "i64[16384]" = torch.ops.aten.empty.memory_format([16384], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_default: "i64[16384]" = torch.ops.prims.iota.default(16384, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put_default: "i64[16384]" = torch.ops.aten.index_put.default(empty_memory_format, [getitem_57], iota_default);  empty_memory_format = getitem_57 = iota_default = None
        index_tensor_1: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(where_self, [index_put_default]);  where_self = index_put_default = None
        view_default_1: "bf16[2048, 8, 2048]" = torch.ops.aten.view.default(index_tensor_1, _shape_param_0);  index_tensor_1 = _shape_param_0 = None
        sum_dim_int_list_1: "bf16[2048, 2048]" = torch.ops.aten.sum.dim_IntList(view_default_1, [1]);  view_default_1 = None
        view_default_2: "bf16[4, 512, 2048]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        add_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_35, view_default_2);  add_35 = view_default_2 = None
        convert_element_type_default_1: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        pow_tensor_scalar: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_1, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_default);  convert_element_type_default_1 = rsqrt_default = None
        convert_element_type_default_2: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(arg47_1, convert_element_type_default_2);  arg47_1 = convert_element_type_default_2 = None
        view_default_3: "bf16[2048, 2048]" = torch.ops.aten.view.default(mul_tensor_2, _shape_param_2);  mul_tensor_2 = _shape_param_2 = None
        return view_default_3

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
