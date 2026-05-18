"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['8', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['8', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '1'], reduction_ranges=[]
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
    def forward(self, tangents_1: "f32[]", convert_element_type_1: "f32[]", eq_tensor: "b8[8, 2]", ne_5: "b8[8, 1]", index_2: "f32[8, 2]", tangents_2: "f32[8, 2]", iota_2: "i64[8]", argmax: "i64[8]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1422 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_1);  tangents_1 = convert_element_type_1 = None

        # No stacktrace found for following nodes
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1422 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        where_self: "f32[8, 2]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:96 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1422 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        where_self_1: "f32[8, 1]" = torch.ops.aten.where.self(ne_5, div_tensor, full_default);  ne_5 = div_tensor = full_default = None
        mul_tensor: "f32[8, 2]" = torch.ops.aten.mul.Tensor(where_self, where_self_1);  where_self = where_self_1 = None
        amax_default: "f32[8, 1]" = torch.ops.aten.amax.default(index_2, [1], True)
        sub_tensor: "f32[8, 2]" = torch.ops.aten.sub.Tensor(index_2, amax_default);  index_2 = amax_default = None
        exp_default: "f32[8, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[8, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[8, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        exp_default_1: "f32[8, 2]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list_1: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[8, 2]" = torch.ops.aten.mul.Tensor(exp_default_1, sum_dim_int_list_1);  exp_default_1 = sum_dim_int_list_1 = None
        sub_tensor_2: "f32[8, 2]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        add_tensor: "f32[8, 2]" = torch.ops.aten.add.Tensor(tangents_2, sub_tensor_2);  tangents_2 = sub_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1402 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        full_default_1: "f32[8, 1024, 2]" = torch.ops.aten.full.default([8, 1024, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[8, 1024, 2]" = torch.ops.aten.index_put.default(full_default_1, [iota_2, argmax], add_tensor, True);  full_default_1 = iota_2 = argmax = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:1379 in forward, code: logits = self.score(hidden_states)
        reshape_default: "f32[8192, 2]" = torch.ops.aten.reshape.default(index_put_default, _shape_param_0);  index_put_default = _shape_param_0 = None
        permute_default: "f32[2, 8192]" = torch.ops.aten.permute.default(reshape_default, [1, 0]);  reshape_default = None
        return permute_default


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 2], dtype=torch.bool, device='cuda'),
    torch.randint(0, 2, [8, 1], dtype=torch.bool, device='cuda'),
    torch.randn([8, 2], dtype=torch.float32, device='cuda'),
    torch.randn([8, 2], dtype=torch.float32, device='cuda'),
    torch.randint(0, 8, [8], dtype=torch.int64, device='cuda'),
    torch.randint(0, 1024, [8], dtype=torch.int64, device='cuda'),
    [8192, 2],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
