"""
Standalone repro captured via capture_hook.
Label: hf_BertForMaskedLM_training
Pattern hash: b9b543f9e1e3
Shape hash: 902e0166
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_68: "f32[4096, 768]", _shape_param_0, _shape_param_1, bmm_22: "f32[96, 512, 512]", _shape_param_2, where: "f32[8, 1, 512, 512]", full_default_2: "f32[8, 12, 512, 512]", inductor_seeds_default: "i64[37]", _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(addmm_68, _shape_param_0);  addmm_68 = _shape_param_0 = None
        reshape_default_1: "f32[8, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[8, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default_2: "f32[8, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, _shape_param_2);  bmm_22 = _shape_param_2 = None
        add_tensor: "f32[8, 12, 512, 512]" = torch.ops.aten.add.Tensor(reshape_default_2, where);  reshape_default_2 = where = None
        amax_default: "f32[8, 12, 512, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[8, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
        exp_default: "f32[8, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[8, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[8, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default: "b8[8, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[8, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[8, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        where_self: "f32[8, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 12, 512, 512]" = torch.ops.prims.inductor_random.default([8, 12, 512, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, where_self);  gt_scalar = where_self = None
        mul_tensor_1: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        expand_default: "f32[8, 12, 512, 512]" = torch.ops.aten.expand.default(mul_tensor_1, _shape_param_3);  mul_tensor_1 = _shape_param_3 = None
        reshape_default_3: "f32[96, 512, 512]" = torch.ops.aten.reshape.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None
        expand_default_1: "f32[8, 12, 512, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[8, 12, 512, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f32[96, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        return (reshape_default_3, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [8, 512, 768],  # _shape_param_0
    [8, 512, -1, 64],  # _shape_param_1
    torch.randn([96, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 12, 512, 512],  # _shape_param_2
    torch.randn([8, 1, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 12, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [37], dtype=torch.int64, device='cuda'),
    [8, 12, 512, 512],  # _shape_param_3
    [96, 512, 512],  # _shape_param_4
    [8, 12, 512, 64],  # _shape_param_5
    [96, 512, 64],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
