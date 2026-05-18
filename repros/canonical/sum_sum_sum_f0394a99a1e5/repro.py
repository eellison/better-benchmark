"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['64', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['64', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '512', '128'], reduction_ranges=[]
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
    def forward(self, mm_148: "f32[32768, 128]", gt: "b8[64, 512, 128]", primals_8: "f32[128]", mul_1: "f32[64, 512, 128]", div_63: "f32[64, 512, 1]", primals_5: "i64[1, 512]", full_default_2: "f32[]", primals_3: "i64[1, 512]", primals_2: "i64[64, 512]", mm_1: "f32[30522, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:787 in forward, code: hidden_states = self.embeddings_project(hidden_states)
        reshape_default: "f32[64, 512, 128]" = torch.ops.aten.reshape.default(mm_148, [64, 512, 128]);  mm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:194 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[64, 512, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:193 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_2: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_8);  primals_8 = None
        mul_tensor_3: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 128)
        sum_dim_int_list: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        mul_tensor_4: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_1);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_1: "f32[64, 512, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(div_63, sub_tensor_1);  div_63 = sub_tensor_1 = None
        mul_tensor_7: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_1);  mul_1 = None
        sum_dim_int_list_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:192 in forward, code: embeddings += position_embeddings
        sum_dim_int_list_4: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:191 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_5, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default, full_default_2, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 128]" = torch.ops.aten.index_put.default(full_default, [primals_5], where_self, True);  full_default = primals_5 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:758 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[64, 512]" = torch.ops.aten.expand.default(primals_3, [64, 512]);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_scalar_1: "b8[64, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default_1: "b8[64, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[64, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_2, mul_tensor_6);  unsqueeze_default_1 = None
        full_default_3: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[2, 128]" = torch.ops.aten.index_put.default(full_default_3, [expand_default], where_self_1, True);  full_default_3 = expand_default = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_2: "b8[64, 512]" = torch.ops.aten.eq.Scalar(primals_2, 0)
        unsqueeze_default_2: "b8[64, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[64, 512, 128]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default_2, mul_tensor_6);  unsqueeze_default_2 = full_default_2 = mul_tensor_6 = None
        full_default_4: "f32[30522, 128]" = torch.ops.aten.full.default([30522, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[30522, 128]" = torch.ops.aten.index_put.default(full_default_4, [primals_2], where_self_2, True);  full_default_4 = primals_2 = where_self_2 = None
        add_tensor: "f32[30522, 128]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, index_put_default, index_put_default_1, add_tensor)


def _default_make_inputs():
    return [
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [64, 512, 128], dtype=torch.bool, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([64, 512, 128], dtype=torch.float32, device='cuda'),
    torch.randn([64, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [64, 512], dtype=torch.int64, device='cuda'),
    torch.randn([30522, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
