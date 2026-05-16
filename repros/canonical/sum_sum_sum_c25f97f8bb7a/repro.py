"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['32', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['32', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_default_1: "f32[16384, 768]", primals_203: "f32[768]", addmm_72: "f32[16384, 768]", getitem_51: "f32[32, 512, 1]", rsqrt_25: "f32[32, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_default_1, [32, 512, 768]);  mm_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_203);  primals_203 = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_72, [32, 512, 768]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.5)
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default_1, 0.7071067811865476)
        erf_default: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_4, getitem_51);  mul_tensor_4 = getitem_51 = None
        mul_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_25);  sub_tensor = None
        mul_tensor_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_5);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2], True);  mul_tensor_6 = None
        mul_tensor_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_5, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_7);  sub_tensor_1 = mul_tensor_7 = None
        div_tensor: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
        mul_tensor_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor_5);  mul_tensor_5 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default_1, reshape_default_1)
        mul_tensor_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_11, -0.5);  mul_tensor_11 = None
        exp_default: "f32[32, 512, 768]" = torch.ops.aten.exp.default(mul_tensor_12);  mul_tensor_12 = None
        mul_tensor_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default_1, mul_tensor_13);  reshape_default_1 = mul_tensor_13 = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_14);  mul_tensor_10 = mul_tensor_14 = None
        mul_tensor_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_8, add_tensor_1);  mul_tensor_8 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_2: "f32[16384, 768]" = torch.ops.aten.reshape.default(mul_tensor_15, [16384, 768]);  mul_tensor_15 = None
        permute_default: "f32[768, 16384]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0])
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(reshape_default_2, [0], True);  reshape_default_2 = None
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [768]);  sum_dim_int_list_4 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
