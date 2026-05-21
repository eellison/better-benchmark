"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_DistilBert_train
Pattern hash: a615eda0cf46
Shape hash: b6a81ff5
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 768], f32), T([8, 512, 768], f32), T([4096, 768], f32), T([4096, 768], f32), T([8, 512, 768], b8), T([768], f32), T([8, 512, 768], f32), T([1, 512, 768], f32), T([8, 512, 1], f32), T([8, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([8, 512], i64, gen=Index(30522)), T([30522, 768], f32), S([8, 512, 768]), S([8, 512, 768]), S([8, 512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_70: "f32[4096, 768]", mul_264: "f32[8, 512, 768]", mm_72: "f32[4096, 768]", mm_74: "f32[4096, 768]", gt: "b8[8, 512, 768]", primals_5: "f32[768]", embedding: "f32[8, 512, 768]", embedding_1: "f32[1, 512, 768]", getitem_1: "f32[8, 512, 1]", rsqrt: "f32[8, 512, 1]", primals_3: "i64[1, 512]", full_default_1: "f32[]", primals_1: "i64[8, 512]", mm_1: "f32[30522, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_70, _shape_param_0);  mm_70 = _shape_param_0 = None
        add_tensor: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_264, reshape_default);  mul_264 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_72, _shape_param_1);  mm_72 = _shape_param_1 = None
        add_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_74, _shape_param_2);  mm_74 = _shape_param_2 = None
        add_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:121 in forward, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        convert_element_type_default: "f32[8, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor);  add_tensor_2 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:120 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        mul_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_5);  primals_5 = None
        mul_tensor_3: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 768)
        sum_dim_int_list: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:119 in forward, code: embeddings = inputs_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add_tensor_3: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:120 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        sub_tensor: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor_3, getitem_1);  add_tensor_3 = getitem_1 = None
        mul_tensor_4: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_5: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        mul_tensor_6: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_6);  sub_tensor_1 = mul_tensor_6 = None
        div_tensor: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_7: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_8: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:119 in forward, code: embeddings = inputs_embeds + position_embeddings  # (bs, max_seq_length, dim)
        sum_dim_int_list_4: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:117 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_3, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default, [primals_3], where_self, True);  full_default = primals_3 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        eq_scalar_1: "b8[8, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_1: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_1, mul_tensor_7);  unsqueeze_default_1 = full_default_1 = mul_tensor_7 = None
        full_default_2: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[30522, 768]" = torch.ops.aten.index_put.default(full_default_2, [primals_1], where_self_1, True);  full_default_2 = primals_1 = where_self_1 = None
        add_tensor_4: "f32[30522, 768]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_1);  mm_1 = index_put_default_1 = None
        return (sum_dim_int_list_3, sum_dim_int_list_2, add_tensor_4, index_put_default)



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
