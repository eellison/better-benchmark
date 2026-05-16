"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['8', '1024', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '1024', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1024', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_97: "f32[8192, 768]", primals_4: "f32[768]", embedding: "f32[8, 1024, 768]", embedding_1: "f32[1, 1024, 768]", gt: "b8[8, 1024, 768]", getitem_1: "f32[8, 1024, 1]", rsqrt: "f32[8, 1024, 1]", add_147: "f32[8, 1024, 768]", unsqueeze: "i64[1, 1024]", full_default_2: "f32[]", primals_1: "i64[8, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_97, [8, 1024, 768]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:412 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_4);  primals_4 = None
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:866 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_tensor: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:909 in forward, code: hidden_states = self.drop(hidden_states)
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, add_tensor);  add_tensor = None
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:412 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_3, getitem_1);  mul_tensor_3 = getitem_1 = None
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_6);  sub_tensor_1 = mul_tensor_6 = None
        div_tensor: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None
        add_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_147, mul_tensor_7);  add_147 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:909 in forward, code: hidden_states = self.drop(hidden_states)
        convert_element_type_default: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_9);  add_tensor_1 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:866 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        sum_dim_int_list_4: "f32[1, 1024, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:865 in forward, code: position_embeds = self.wpe(position_ids)
        eq_scalar: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(unsqueeze, -1)
        unsqueeze_default: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_2, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default, [unsqueeze], where_self, True);  full_default = unsqueeze = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:855 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_scalar_1: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default_1: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_2, mul_tensor_10);  unsqueeze_default_1 = full_default_2 = mul_tensor_10 = None
        full_default_3: "f32[50257, 768]" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[50257, 768]" = torch.ops.aten.index_put.default(full_default_3, [primals_1], where_self_1, True);  full_default_3 = primals_1 = where_self_1 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, index_put_default_1)


def _default_make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 768], dtype=torch.bool, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
