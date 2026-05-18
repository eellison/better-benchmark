"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['16', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['16', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '512', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_286: "f32[8192, 1024]", mm_288: "f32[8192, 1024]", mm_290: "f32[8192, 1024]", primals_7: "f32[1024]", mul_3: "f32[16, 512, 1024]", div_123: "f32[16, 512, 1]", add_342: "f32[16, 512, 1024]", gt: "b8[16, 512, 1024]", primals_4: "i64[1, 512]", full_default_3: "f32[]", full_default: "i64[16, 512]", primals_2: "i64[16, 512]", mm_1: "f32[29056, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:249 in forward, code: value_layer = self.value(current_states)
        reshape_default: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_286, _shape_param_0);  mm_286 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:245 in forward, code: key_layer = self.key(current_states)
        reshape_default_1: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_288, _shape_param_1);  mm_288 = _shape_param_1 = None
        add_tensor: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:221 in forward, code: query_layer = self.query(hidden_states)
        reshape_default_2: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_290, _shape_param_2);  mm_290 = _shape_param_2 = None
        add_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:365 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_tensor: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_7);  primals_7 = None
        mul_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_3);  mul_tensor = None
        sum_dim_int_list_1: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_3, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_123, sub_tensor_1);  div_123 = sub_tensor_1 = None
        mul_tensor_5: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_3);  mul_3 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1]);  add_tensor_1 = None
        add_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_342, mul_tensor_4);  add_342 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:176 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_6: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_7: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_6);  add_tensor_2 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:172 in forward, code: embeddings += position_embeddings
        sum_dim_int_list_4: "f32[1, 512, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_4, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default, full_default_3, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default: "f32[512, 1024]" = torch.ops.aten.full.default([512, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 1024]" = torch.ops.aten.index_put.default(full_default, [primals_4], where_self, True);  full_default = primals_4 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:167 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        full_default_4: "b8[16, 512, 1]" = torch.ops.aten.full.default([16, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[16, 512, 1024]" = torch.ops.aten.where.self(full_default_4, full_default_3, mul_tensor_7);  full_default_4 = None
        full_default_5: "f32[2, 1024]" = torch.ops.aten.full.default([2, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6 = full_default
        index_put_default_1: "f32[2, 1024]" = torch.ops.aten.index_put.default(full_default_5, [full_default_6], where_self_1, True);  full_default_5 = full_default_6 = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:166 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_1: "b8[16, 512]" = torch.ops.aten.eq.Scalar(primals_2, 0)
        unsqueeze_default_1: "b8[16, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_2: "f32[16, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_3, mul_tensor_7);  unsqueeze_default_1 = full_default_3 = mul_tensor_7 = None
        full_default_7: "f32[29056, 1024]" = torch.ops.aten.full.default([29056, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[29056, 1024]" = torch.ops.aten.index_put.default(full_default_7, [primals_2], where_self_2, True);  full_default_7 = primals_2 = where_self_2 = None
        add_tensor_3: "f32[29056, 1024]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, index_put_default_1, add_tensor_3)


def _default_make_inputs():
    return [
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 512, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [16, 512, 1024], dtype=torch.bool, device='cuda'),
    torch.randint(0, 16, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.tensor(1),  # full_default_6 (unknown shape)
    torch.randint(0, 29056, [16, 512], dtype=torch.int64, device='cuda'),
    torch.randn([29056, 1024], dtype=torch.float32, device='cuda'),
    [16, 512, 1024],  # _shape_param_0
    [16, 512, 1024],  # _shape_param_1
    [16, 512, 1024],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
