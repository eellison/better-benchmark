"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_base_train
Pattern hash: 826140097d62
Shape hash: 199c262d
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
_shapes_config = "(T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([768], f32), T([8, 1024, 768], b8), T([8, 1024, 768], f32), T([8, 1024, 1], f32), T([8, 1024, 768], f32), T([8, 1024], i64, gen=Index(32128)), T([], f32), T([32128, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([768], f32), T([8, 1024, 768], b8), T([8, 1024, 768], f32), T([8, 1024, 1], f32), T([8, 1024, 768], f32), T([8, 1024], i64, gen=Index(32128)), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([8, 1024, 768]), S([768]), S([8, 1024, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_430: "f32[8192, 768]", mm_432: "f32[8192, 768]", mm_434: "f32[8192, 768]", primals_102: "f32[768]", gt_51: "b8[8, 1024, 768]", embedding_2: "f32[8, 1024, 768]", rsqrt_25: "f32[8, 1024, 1]", add_301: "f32[8, 1024, 768]", where_2: "i64[8, 1024]", full_default: "f32[]", mm_193: "f32[32128, 768]", mm_574: "f32[8192, 768]", mm_576: "f32[8192, 768]", mm_578: "f32[8192, 768]", primals_3: "f32[768]", gt: "b8[8, 1024, 768]", embedding: "f32[8, 1024, 768]", rsqrt: "f32[8, 1024, 1]", add_386: "f32[8, 1024, 768]", primals_1: "i64[8, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_430, _shape_param_0);  mm_430 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_1: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_432, _shape_param_1);  mm_432 = _shape_param_1 = None
        add_tensor: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_434, _shape_param_2);  mm_434 = _shape_param_2 = None
        add_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_102);  primals_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_51, embedding_2);  embedding_2 = None
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, rsqrt_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_3);  add_tensor_1 = mul_tensor_3 = None
        sum_dim_int_list: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True);  mul_tensor_4 = None
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_3);  sum_dim_int_list = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2)
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, rsqrt_25);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        add_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_301, mul_tensor_6);  add_301 = mul_tensor_6 = None
        pow_tensor_scalar: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_25, 3);  rsqrt_25 = None
        mul_scalar: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_7: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_tensor_7, _shape_param_4);  mul_tensor_7 = _shape_param_4 = None
        div_scalar: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_2, 1.0);  mul_tensor_2 = None
        mul_scalar_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_8);  add_tensor_2 = mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_default: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_51, torch.float32);  gt_51 = None
        mul_tensor_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_3, mul_tensor_9);  add_tensor_3 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        eq_scalar: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(where_2, -1)
        unsqueeze_default: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, mul_tensor_10);  unsqueeze_default = mul_tensor_10 = None
        full_default_1: "f32[32128, 768]" = torch.ops.aten.full.default([32128, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32128, 768]" = torch.ops.aten.index_put.default(full_default_1, [where_2], where_self, True);  where_2 = where_self = None
        add_tensor_4: "f32[32128, 768]" = torch.ops.aten.add.Tensor(mm_193, index_put_default);  mm_193 = index_put_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_4: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_574, _shape_param_5);  mm_574 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_5: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_576, _shape_param_6);  mm_576 = _shape_param_6 = None
        add_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(reshape_default_4, reshape_default_5);  reshape_default_4 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_6: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_578, _shape_param_7);  mm_578 = _shape_param_7 = None
        add_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_6);  add_tensor_5 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_11: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_6, primals_3);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_tensor_12: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_tensor_13: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 1.1111111111111112);  mul_tensor_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_14: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_13, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_15: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_6, mul_tensor_14);  add_tensor_6 = mul_tensor_14 = None
        sum_dim_int_list_2: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_7: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_8);  sum_dim_int_list_2 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_16: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_11, mul_tensor_13)
        mul_tensor_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_11, rsqrt);  mul_tensor_11 = None
        sum_dim_int_list_3: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [2], True);  mul_tensor_16 = None
        add_tensor_7: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_386, mul_tensor_17);  add_386 = mul_tensor_17 = None
        pow_tensor_scalar_2: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar_2: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_3, -0.5);  sum_dim_int_list_3 = None
        mul_tensor_18: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar_2, pow_tensor_scalar_2);  mul_scalar_2 = pow_tensor_scalar_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default_1: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_tensor_18, _shape_param_9);  mul_tensor_18 = _shape_param_9 = None
        div_scalar_1: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_default_1, 768);  expand_default_1 = None
        pow_tensor_scalar_3: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_13, 1.0);  mul_tensor_13 = None
        mul_scalar_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_3, 2.0);  pow_tensor_scalar_3 = None
        mul_tensor_19: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_3);  div_scalar_1 = mul_scalar_3 = None
        add_tensor_8: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_tensor_7, mul_tensor_19);  add_tensor_7 = mul_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_default_1: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_20: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_21: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_8, mul_tensor_20);  add_tensor_8 = mul_tensor_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        eq_scalar_1: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default_1: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_21);  unsqueeze_default_1 = full_default = mul_tensor_21 = None
        index_put_default_1: "f32[32128, 768]" = torch.ops.aten.index_put.default(full_default_1, [primals_1], where_self_1, True);  full_default_1 = primals_1 = where_self_1 = None
        add_tensor_9: "f32[32128, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, index_put_default_1);  add_tensor_4 = index_put_default_1 = None
        return (reshape_default_3, add_tensor_9, reshape_default_7)



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
