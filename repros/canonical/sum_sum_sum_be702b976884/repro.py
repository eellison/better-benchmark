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
    def forward(self, mm_default: "f32[30524, 768]", mm_198: "f32[16384, 768]", mul_567: "f32[32, 512, 768]", mm_202: "f32[16384, 768]", getitem_121: "f32[32, 768, 512]", mm_204: "f32[16384, 768]", mm_206: "f32[16384, 768]", gt: "b8[32, 512, 768]", primals_7: "f32[768]", mul_1: "f32[32, 512, 768]", div_75: "f32[32, 512, 1]", primals_2: "i64[1, 512]", full_default_2: "f32[]", primals_4: "i64[1, 512]", primals_1: "i64[32, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:938 in forward, code: prediction_scores = self.generator_lm_head(prediction_scores)
        slice_tensor: "f32[30522, 768]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -2);  mm_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:365 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_198, [32, 512, 768]);  mm_198 = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_567, reshape_default);  mul_567 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: mixed_query_layer = self.query(hidden_states)
        reshape_default_1: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_202, [32, 512, 768]);  mm_202 = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:346 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_default: "f32[32, 512, 768]" = torch.ops.aten.permute.default(getitem_121, [0, 2, 1]);  getitem_121 = None
        add_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, permute_default);  add_tensor_1 = permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:344 in forward, code: mixed_value_layer = self.value(hidden_states)
        reshape_default_2: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_204, [32, 512, 768]);  mm_204 = None
        add_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_2);  add_tensor_2 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:343 in forward, code: mixed_key_layer = self.key(hidden_states)
        reshape_default_3: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_206, [32, 512, 768]);  mm_206 = None
        add_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_3);  add_tensor_3 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:228 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_4, mul_tensor);  add_tensor_4 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_7);  primals_7 = None
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 768)
        sum_dim_int_list: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        mul_tensor_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_1);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_75, sub_tensor_1);  div_75 = sub_tensor_1 = None
        mul_tensor_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_1);  mul_1 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:226 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        sum_dim_int_list_4: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:837 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[32, 512]" = torch.ops.aten.expand.default(primals_2, [32, 512]);  primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_scalar: "b8[32, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_2, mul_tensor_6);  unsqueeze_default = None
        full_default: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2, 768]" = torch.ops.aten.index_put.default(full_default, [expand_default], where_self, True);  full_default = expand_default = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_4, -1)
        unsqueeze_default_1: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_2, sum_dim_int_list_4);  unsqueeze_default_1 = sum_dim_int_list_4 = None
        full_default_3: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default_3, [primals_4], where_self_1, True);  full_default_3 = primals_4 = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_2: "b8[32, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_2: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[32, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default_2, mul_tensor_6);  unsqueeze_default_2 = full_default_2 = mul_tensor_6 = None
        full_default_4: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[30522, 768]" = torch.ops.aten.index_put.default(full_default_4, [primals_1], where_self_2, True);  full_default_4 = primals_1 = where_self_2 = None
        add_tensor_5: "f32[30522, 768]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_default_2);  slice_tensor = index_put_default_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, index_put_default_1, add_tensor_5)


def _default_make_inputs():
    return [
    torch.randn([30524, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 512, 768], dtype=torch.bool, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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
