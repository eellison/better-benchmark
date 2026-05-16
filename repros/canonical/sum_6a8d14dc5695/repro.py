"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_training
Pattern hash: 6a8d14dc5695
Shape hash: 5955fc4f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_3: "f32[128, 128, 128]", _shape_param_0, gt: "b8[8, 16, 128, 128]", primals_10: "b8[8, 1, 128, 128]", full_default_1: "f32[]", bmm: "f32[128, 128, 128]", _shape_param_1, amax: "f32[8, 16, 128, 1]", sum_1: "f32[8, 16, 128, 1]", logical_not_1: "b8[8, 16, 128, 1]", _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[8, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_3, _shape_param_0);  bmm_3 = _shape_param_0 = None
        convert_element_type_default: "f32[8, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1, 128, 128]" = torch.ops.aten.where.self(primals_10, full_default_1, full_default);  primals_10 = full_default_1 = full_default = None
        reshape_default_1: "f32[8, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None
        add_tensor: "f32[8, 16, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default_1, where_self);  reshape_default_1 = where_self = None
        sub_tensor: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax);  add_tensor = amax = None
        exp_default: "f32[8, 16, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor: "f32[8, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_1);  exp_default = sum_1 = None
        full_default_2: "f32[8, 16, 128, 128]" = torch.ops.aten.full.default([8, 16, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8, 16, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div_tensor);  logical_not_1 = full_default_2 = div_tensor = None
        mul_tensor_2: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, where_self_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[8, 16, 128, 128]" = torch.ops.aten.neg.default(where_self_1);  where_self_1 = None
        fma_default: "f32[8, 16, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        reshape_default_2: "f32[128, 128, 128]" = torch.ops.aten.reshape.default(fma_default, _shape_param_2);  fma_default = _shape_param_2 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([128, 128, 128], dtype=torch.float32, device='cuda'),
    [8, 16, 128, 128],  # _shape_param_0
    torch.randint(0, 2, [8, 16, 128, 128], dtype=torch.bool, device='cuda'),
    torch.randn(128, dtype=torch.bool, device='cuda').as_strided([8, 1, 128, 128], [0, 128, 1, 0]),  # primals_10
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([128, 128, 128], dtype=torch.float32, device='cuda'),
    [8, 16, 128, 128],  # _shape_param_1
    torch.randn([8, 16, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 16, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 16, 128, 1], dtype=torch.bool, device='cuda'),
    [128, 128, 128],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
