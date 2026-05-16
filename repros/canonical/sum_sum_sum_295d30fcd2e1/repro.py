"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '1024', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '1024', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_214: "f32[8192, 512]", mm_216: "f32[8192, 512]", mm_218: "f32[8192, 512]", primals_54: "f32[512]", gt_28: "b8[8, 1024, 512]", embedding: "f32[8, 1024, 512]", rsqrt_13: "f32[8, 1024, 1]", add_156: "f32[8, 1024, 512]", primals_1: "i64[8, 1024]", full_default_2: "f32[]", mm_97: "f32[32128, 512]", mm_286: "f32[8192, 512]", mm_288: "f32[8192, 512]", mm_290: "f32[8192, 512]", primals_3: "f32[512]", gt: "b8[8, 1024, 512]", rsqrt: "f32[8, 1024, 1]", add_199: "f32[8, 1024, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_214, [8, 1024, 512]);  mm_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_1: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_216, [8, 1024, 512]);  mm_216 = None
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:492 in forward, code: query_states = self.q(hidden_states)
        reshape_default_2: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_218, [8, 1024, 512]);  mm_218 = None
        add_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:262 in forward, code: return self.weight * hidden_states
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_54);  primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1074 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_28, embedding)
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:256 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:262 in forward, code: return self.weight * hidden_states
        mul_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_3);  add_tensor_1 = mul_tensor_3 = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True);  mul_tensor_4 = None
        reshape_default_3: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list, [512]);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:256 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2)
        mul_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, rsqrt_13);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        add_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_156, mul_tensor_6);  add_156 = mul_tensor_6 = None
        pow_tensor_scalar: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_13, 3);  rsqrt_13 = None
        mul_scalar: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_7: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:255 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_7, [8, 1024, 512]);  mul_tensor_7 = None
        div_scalar: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_2, 1.0);  mul_tensor_2 = None
        mul_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_8: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_8);  add_tensor_2 = mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1074 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_default: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_tensor_9: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_10: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_3, mul_tensor_9);  add_tensor_3 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1003 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        eq_scalar: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[8, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_default, full_default_2, mul_tensor_10);  mul_tensor_10 = None
        full_default: "f32[32128, 512]" = torch.ops.aten.full.default([32128, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32128, 512]" = torch.ops.aten.index_put.default(full_default, [primals_1], where_self, True);  where_self = None
        add_tensor_4: "f32[32128, 512]" = torch.ops.aten.add.Tensor(mm_97, index_put_default);  mm_97 = index_put_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_4: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_286, [8, 1024, 512]);  mm_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_5: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_288, [8, 1024, 512]);  mm_288 = None
        add_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(reshape_default_4, reshape_default_5);  reshape_default_4 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:492 in forward, code: query_states = self.q(hidden_states)
        reshape_default_6: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_290, [8, 1024, 512]);  mm_290 = None
        add_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_6);  add_tensor_5 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:262 in forward, code: return self.weight * hidden_states
        mul_tensor_11: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_6, primals_3);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1074 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_tensor_12: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_tensor_13: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 1.1111111111111112);  mul_tensor_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:256 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_14: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_13, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:262 in forward, code: return self.weight * hidden_states
        mul_tensor_15: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_6, mul_tensor_14);  add_tensor_6 = mul_tensor_14 = None
        sum_dim_int_list_2: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_7: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [512]);  sum_dim_int_list_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:256 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_16: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_11, mul_tensor_13)
        mul_tensor_17: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_11, rsqrt);  mul_tensor_11 = None
        sum_dim_int_list_3: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [2], True);  mul_tensor_16 = None
        add_tensor_7: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_199, mul_tensor_17);  add_199 = mul_tensor_17 = None
        pow_tensor_scalar_2: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar_2: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_3, -0.5);  sum_dim_int_list_3 = None
        mul_tensor_18: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar_2, pow_tensor_scalar_2);  mul_scalar_2 = pow_tensor_scalar_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:255 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default_1: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_18, [8, 1024, 512]);  mul_tensor_18 = None
        div_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default_1, 512);  expand_default_1 = None
        pow_tensor_scalar_3: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_13, 1.0);  mul_tensor_13 = None
        mul_scalar_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_3, 2.0);  pow_tensor_scalar_3 = None
        mul_tensor_19: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_3);  div_scalar_1 = mul_scalar_3 = None
        add_tensor_8: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_7, mul_tensor_19);  add_tensor_7 = mul_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1074 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_default_1: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_20: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_21: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_8, mul_tensor_20);  add_tensor_8 = mul_tensor_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1003 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        where_self_1: "f32[8, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_default, full_default_2, mul_tensor_21);  unsqueeze_default = full_default_2 = mul_tensor_21 = None
        index_put_default_1: "f32[32128, 512]" = torch.ops.aten.index_put.default(full_default, [primals_1], where_self_1, True);  full_default = primals_1 = where_self_1 = None
        add_tensor_9: "f32[32128, 512]" = torch.ops.aten.add.Tensor(add_tensor_4, index_put_default_1);  add_tensor_4 = index_put_default_1 = None
        return (reshape_default_3, reshape_default_7, add_tensor_9)


def _default_make_inputs():
    return [
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 512], dtype=torch.bool, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([32128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 512], dtype=torch.bool, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
