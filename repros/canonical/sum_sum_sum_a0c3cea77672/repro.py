"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bert_large_train
Pattern hash: a0c3cea77672
Shape hash: 4a252802
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
_shapes_config = "(T([2048, 1024], f32), T([4, 512, 1024], f32), T([2048, 1024], f32), T([2048, 1024], f32), T([4, 512, 1024], b8), T([1024], f32), T([4, 512, 1024], f32), T([4, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([1, 512], i64, gen=Index(2)), T([4, 512], i64, gen=Index(30522)), T([30522, 1024], f32), S([4, 512, 1024]), S([4, 512, 1024]), S([4, 512, 1024]), S([4, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_286: "f32[2048, 1024]", mul_1096: "f32[4, 512, 1024]", mm_288: "f32[2048, 1024]", mm_290: "f32[2048, 1024]", gt: "b8[4, 512, 1024]", primals_7: "f32[1024]", mul: "f32[4, 512, 1024]", div_75: "f32[4, 512, 1]", primals_2: "i64[1, 512]", full_default_1: "f32[]", gather: "i64[1, 512]", primals_1: "i64[4, 512]", mm_1: "f32[30522, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_286, _shape_param_0);  mm_286 = _shape_param_0 = None
        add_tensor: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_1096, reshape_default);  mul_1096 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_288, _shape_param_1);  mm_288 = _shape_param_1 = None
        add_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_290, _shape_param_2);  mm_290 = _shape_param_2 = None
        add_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:111 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor);  add_tensor_2 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_7);  primals_7 = None
        mul_tensor_3: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1024)
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True)
        mul_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True);  mul_tensor_4 = None
        mul_tensor_5: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list);  mul_tensor_3 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_75, sub_tensor_1);  div_75 = sub_tensor_1 = None
        mul_tensor_7: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul);  mul = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:108 in forward, code: embeddings = embeddings + position_embeddings
        sum_dim_int_list_4: "f32[1, 512, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:107 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, sum_dim_int_list_4);  unsqueeze_default = sum_dim_int_list_4 = None
        full_default: "f32[512, 1024]" = torch.ops.aten.full.default([512, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 1024]" = torch.ops.aten.index_put.default(full_default, [primals_2], where_self, True);  full_default = primals_2 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:98 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[4, 512]" = torch.ops.aten.expand.default(gather, _shape_param_3);  gather = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:104 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_scalar_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default_1: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[4, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_1, mul_tensor_6);  unsqueeze_default_1 = None
        full_default_2: "f32[2, 1024]" = torch.ops.aten.full.default([2, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[2, 1024]" = torch.ops.aten.index_put.default(full_default_2, [expand_default], where_self_1, True);  full_default_2 = expand_default = where_self_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_2: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_2: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[4, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default_1, mul_tensor_6);  unsqueeze_default_2 = full_default_1 = mul_tensor_6 = None
        full_default_3: "f32[30522, 1024]" = torch.ops.aten.full.default([30522, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[30522, 1024]" = torch.ops.aten.index_put.default(full_default_3, [primals_1], where_self_2, True);  full_default_3 = primals_1 = where_self_2 = None
        add_tensor_3: "f32[30522, 1024]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_2);  mm_1 = index_put_default_2 = None
        return (sum_dim_int_list_3, sum_dim_int_list_2, index_put_default_1, index_put_default, add_tensor_3)



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
