"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['4096', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[]", convert_element_type_6: "f32[]", primals_77: "i64[32, 128]", view_604: "f32[32, 128, 250112]", amax_24: "f32[4096, 1]", log_2: "f32[4096, 1]", tangents_2: "f32[32, 128, 250112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1823 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_6);  tangents_1 = convert_element_type_6 = None
        reshape_default: "i64[4096]" = torch.ops.aten.reshape.default(primals_77, [-1]);  primals_77 = None
        unsqueeze_default: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(reshape_default, 1);  reshape_default = None
        ne_scalar: "b8[4096, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4096, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = full_default = None

        # No stacktrace found for following nodes
        iota_default: "i64[250112]" = torch.ops.prims.iota.default(250112, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        reshape_default_1: "i64[1, 250112]" = torch.ops.aten.reshape.default(iota_default, [1, 250112]);  iota_default = None
        expand_default: "i64[4096, 250112]" = torch.ops.aten.expand.default(where_self, [4096, 250112]);  where_self = None
        eq_tensor: "b8[4096, 250112]" = torch.ops.aten.eq.Tensor(expand_default, reshape_default_1);  expand_default = reshape_default_1 = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1823 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_self_1: "f32[4096, 250112]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1259 in _prepare_4d_causal_attention_mask_with_cache_position, code: causal_mask = torch.triu(causal_mask, diagonal=1)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1823 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_self_2: "f32[4096, 1]" = torch.ops.aten.where.self(ne_scalar, div_tensor, full_default_1);  ne_scalar = div_tensor = full_default_1 = None
        mul_tensor: "f32[4096, 250112]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        reshape_default_2: "f32[4096, 250112]" = torch.ops.aten.reshape.default(view_604, [-1, 250112]);  view_604 = None
        sub_tensor: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(reshape_default_2, amax_24);  reshape_default_2 = amax_24 = None
        sub_tensor_1: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(sub_tensor, log_2);  sub_tensor = log_2 = None
        exp_default: "f32[4096, 250112]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[4096, 250112]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        reshape_default_3: "f32[32, 128, 250112]" = torch.ops.aten.reshape.default(sub_tensor_2, [32, 128, 250112]);  sub_tensor_2 = None
        add_tensor: "f32[32, 128, 250112]" = torch.ops.aten.add.Tensor(tangents_2, reshape_default_3);  tangents_2 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1816 in forward, code: lm_logits = self.lm_head(sequence_output)
        reshape_default_4: "f32[4096, 250112]" = torch.ops.aten.reshape.default(add_tensor, [4096, 250112]);  add_tensor = None
        permute_default: "f32[250112, 4096]" = torch.ops.aten.permute.default(reshape_default_4, [1, 0]);  reshape_default_4 = None
        return permute_default


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 128], dtype=torch.int64, device='cuda'),
    torch.randn([32, 128, 250112], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 250112], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
