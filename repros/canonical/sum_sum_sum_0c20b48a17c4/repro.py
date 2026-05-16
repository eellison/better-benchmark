"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['8', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '512', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_142: "f32[4096, 768]", mul_451: "f32[8, 512, 768]", mm_144: "f32[4096, 768]", mm_146: "f32[4096, 768]", gt: "b8[8, 512, 768]", primals_7: "f32[768]", mul: "f32[8, 512, 768]", div_25: "f32[8, 512, 1]", primals_3: "i64[1, 512]", primals_2: "i64[1, 512]", primals_1: "i64[8, 512]", mm_1: "f32[30522, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        reshape_default: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_142, [8, 512, 768]);  mm_142 = None
        add_tensor: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_451, reshape_default);  mul_451 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        reshape_default_1: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_144, [8, 512, 768]);  mm_144 = None
        add_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        reshape_default_2: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_146, [8, 512, 768]);  mm_146 = None
        add_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:187 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[8, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor);  add_tensor_2 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:186 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_7);  primals_7 = None
        mul_tensor_3: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 768)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        mul_tensor_4: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_tensor_1);  div_25 = sub_tensor_1 = None
        mul_tensor_7: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul);  mul = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:185 in forward, code: embeddings += position_embeddings
        sum_dim_int_list_4: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:184 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_3, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default_1: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default_1, [primals_3], where_self, True);  full_default_1 = primals_3 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:931 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[8, 512]" = torch.ops.aten.expand.default(primals_2, [8, 512]);  primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_scalar_1: "b8[8, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default_1: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_6);  unsqueeze_default_1 = None
        full_default_2: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[2, 768]" = torch.ops.aten.index_put.default(full_default_2, [expand_default], where_self_1, True);  full_default_2 = expand_default = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_2: "b8[8, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_2: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[8, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default, mul_tensor_6);  unsqueeze_default_2 = full_default = mul_tensor_6 = None
        full_default_3: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[30522, 768]" = torch.ops.aten.index_put.default(full_default_3, [primals_1], where_self_2, True);  full_default_3 = primals_1 = where_self_2 = None
        add_tensor_3: "f32[30522, 768]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, index_put_default_1, add_tensor_3)


def _default_make_inputs():
    return [
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 512, 768], dtype=torch.bool, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randn([30522, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
