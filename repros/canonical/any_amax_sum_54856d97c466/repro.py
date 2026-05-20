"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_infer
Pattern hash: 54856d97c466
Shape hash: e0dab0b3
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1024, 128, 128], f32), T([64, 1, 128, 128], b8, stride=(0, 128, 1, 0)), T([8192, 1024], f32), T([64, 128], i64, max=128), S([64, 16, 128, 128]), S([64, 16, 128, 128]), S([1024, 128, 128]), S([64, 128, 1024]), S([64, 128, -1, 64]), S([64, 16, 128, 64]), S([1024, 128, 64]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_22: "f32[1024, 128, 128]", expand: "b8[64, 1, 128, 128]", addmm_68: "f32[8192, 1024]", arg1_1: "i64[64, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[64, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, _shape_param_0);  bmm_22 = _shape_param_0 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[64, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default = full_default_1 = None
        add_tensor: "f32[64, 16, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, where_self);  reshape_default = where_self = None
        eq_scalar: "b8[64, 16, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor, -inf)
        logical_not_default: "b8[64, 16, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[64, 16, 128, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[64, 16, 128, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2: "f32[64, 16, 128, 128]" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default: "f32[64, 16, 128, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[64, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[64, 16, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[64, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[64, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self_1: "f32[64, 16, 128, 128]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None
        expand_default: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(where_self_1, _shape_param_1);  where_self_1 = _shape_param_1 = None
        reshape_default_1: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_2: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_68, _shape_param_3);  addmm_68 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        reshape_default_3: "f32[64, 128, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None
        permute_default: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_default_1: "f32[64, 16, 128, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[64, 16, 128, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f32[1024, 128, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:177 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne_scalar: "b8[64, 128]" = torch.ops.aten.ne.Scalar(arg1_1, 1);  arg1_1 = None
        convert_element_type_default: "i32[64, 128]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        return (reshape_default_1, reshape_default_4, convert_element_type_default)


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
