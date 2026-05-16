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
    def forward(self, mm_default: "f32[30524, 768]", mm_142: "f32[16384, 768]", mul_534: "f32[32, 512, 768]", mm_144: "f32[16384, 768]", mm_146: "f32[16384, 768]", gt: "b8[32, 512, 768]", primals_10: "f32[768]", mul_1: "f32[32, 512, 768]", div_39: "f32[32, 512, 1]", full_default_4: "f32[]", full_default: "i64[32, 512]", sub_2: "i64[32, 512]", sub_1: "i64[32, 512]", select_3: "i64[32, 512]", select_2: "i64[32, 512]", select_1: "i64[32, 512]", select: "i64[32, 512]", primals_3: "i64[1, 512]", primals_1: "i64[32, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:445 in forward, code: hidden_states = self.decoder(hidden_states)
        slice_tensor: "f32[30522, 768]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -2);  mm_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:185 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_142, [32, 512, 768]);  mm_142 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_534, reshape_default);  mul_534 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:184 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_144, [32, 512, 768]);  mm_144 = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:183 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_146, [32, 512, 768]);  mm_146 = None
        add_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:117 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor);  add_tensor_2 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:116 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_10);  primals_10 = None
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 768)
        sum_dim_int_list: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        mul_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_1);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_tensor_1);  div_39 = sub_tensor_1 = None
        mul_tensor_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_1);  mul_1 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:106 in forward, code: words_embeddings
        sum_dim_int_list_4: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:103 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        full_default: "b8[32, 512, 1]" = torch.ops.aten.full.default([32, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 512, 768]" = torch.ops.aten.where.self(full_default, full_default_4, mul_tensor_6);  full_default = None
        full_default_5: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6 = full_default
        index_put_default: "f32[2, 768]" = torch.ops.aten.index_put.default(full_default_5, [full_default_6], where_self, True);  full_default_5 = full_default_6 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:102 in forward, code: w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        eq_scalar: "b8[32, 512]" = torch.ops.aten.eq.Scalar(sub_2, -1)
        unsqueeze_default: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self_1: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_4, mul_tensor_6);  unsqueeze_default = None
        full_default_7: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_7, [sub_2], where_self_1, True);  sub_2 = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:101 in forward, code: h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        eq_scalar_1: "b8[32, 512]" = torch.ops.aten.eq.Scalar(sub_1, -1)
        unsqueeze_default_1: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_2: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_4, mul_tensor_6);  unsqueeze_default_1 = None
        index_put_default_2: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_7, [sub_1], where_self_2, True);  sub_1 = where_self_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:97 in forward, code: lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        eq_scalar_2: "b8[32, 512]" = torch.ops.aten.eq.Scalar(select_3, -1)
        unsqueeze_default_2: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_3: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default_4, mul_tensor_6);  unsqueeze_default_2 = None
        index_put_default_3: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_7, [select_3], where_self_3, True);  select_3 = where_self_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:96 in forward, code: right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        eq_scalar_3: "b8[32, 512]" = torch.ops.aten.eq.Scalar(select_2, -1)
        unsqueeze_default_3: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_3, -1);  eq_scalar_3 = None
        where_self_4: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_3, full_default_4, mul_tensor_6);  unsqueeze_default_3 = None
        index_put_default_4: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_7, [select_2], where_self_4, True);  select_2 = where_self_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:95 in forward, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        eq_scalar_4: "b8[32, 512]" = torch.ops.aten.eq.Scalar(select_1, -1)
        unsqueeze_default_4: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_4, -1);  eq_scalar_4 = None
        where_self_5: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_4, full_default_4, mul_tensor_6);  unsqueeze_default_4 = None
        index_put_default_5: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_7, [select_1], where_self_5, True);  select_1 = where_self_5 = None
        add_tensor_3: "f32[1024, 768]" = torch.ops.aten.add.Tensor(index_put_default_3, index_put_default_5);  index_put_default_3 = index_put_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:94 in forward, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        eq_scalar_5: "b8[32, 512]" = torch.ops.aten.eq.Scalar(select, -1)
        unsqueeze_default_5: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_5, -1);  eq_scalar_5 = None
        where_self_6: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_5, full_default_4, mul_tensor_6);  unsqueeze_default_5 = None
        index_put_default_6: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_7, [select], where_self_6, True);  full_default_7 = select = where_self_6 = None
        add_tensor_4: "f32[1024, 768]" = torch.ops.aten.add.Tensor(index_put_default_4, index_put_default_6);  index_put_default_4 = index_put_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:92 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar_6: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_3, -1)
        unsqueeze_default_6: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_6, -1);  eq_scalar_6 = None
        where_self_7: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_6, full_default_4, sum_dim_int_list_4);  unsqueeze_default_6 = sum_dim_int_list_4 = None
        full_default_8: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_7: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default_8, [primals_3], where_self_7, True);  full_default_8 = primals_3 = where_self_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:89 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_7: "b8[32, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_7: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_7, -1);  eq_scalar_7 = None
        where_self_8: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_7, full_default_4, mul_tensor_6);  unsqueeze_default_7 = full_default_4 = mul_tensor_6 = None
        full_default_9: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_8: "f32[30522, 768]" = torch.ops.aten.index_put.default(full_default_9, [primals_1], where_self_8, True);  full_default_9 = primals_1 = where_self_8 = None
        add_tensor_5: "f32[30522, 768]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_default_8);  slice_tensor = index_put_default_8 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, index_put_default_1, index_put_default_2, add_tensor_3, add_tensor_4, index_put_default_7, add_tensor_5)


def _default_make_inputs():
    return [
    torch.randn([30524, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 512, 768], dtype=torch.bool, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.tensor(1),  # full_default_6 (unknown shape)
    torch.randint(0, 2, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [32, 512], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
