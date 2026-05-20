"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: d4e7d15ad848
Shape hash: 1dd2ba9e
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([2048, 2560], f32), T([16, 1, 128, 128], b8, stride=(0, 128, 1, 0)), T([512, 128, 128], f32), S([16, 128, 2560]), S([16, 128, -1, 80]), S([16, 32, 128, 128]), S([16, 32, 128, 128]), S([512, 128, 128]), S([16, 32, 128, 80]), S([512, 128, 80]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f32[2048, 2560]", primals_10: "b8[16, 1, 128, 128]", bmm: "f32[512, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_0);  addmm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        reshape_default_1: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(primals_10, full_default_1, full_default);  primals_10 = full_default_1 = full_default = None
        reshape_default_2: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        add_tensor: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default_2, where_self);  reshape_default_2 = where_self = None
        amax_default: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  amax_default = None
        exp_default: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None
        expand_default: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_self_1, _shape_param_3);  where_self_1 = _shape_param_3 = None
        reshape_default_3: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None
        expand_default_1: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        return (reshape_default_3, reshape_default_4)


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
