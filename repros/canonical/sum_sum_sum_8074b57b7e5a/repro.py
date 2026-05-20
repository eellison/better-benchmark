"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForQuestionAnswering_train
Pattern hash: 8074b57b7e5a
Shape hash: a6e5f445
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([], f32), T([1], b8), T([1], i64, max=1), T([1, 128], f32), T([1, 1], f32), T([1, 1], f32), T([1, 128], f32), T([1], b8), T([1], i64, max=1), T([1, 128], f32), T([1, 1], f32), T([1, 1], f32), T([1, 128], f32), T([2, 4096], f32), S([1, 128]), S([1, 128]), S([1, 128]), S([128, 2]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[]", ne_4: "b8[1]", primals_316: "i64[1]", clone_226: "f32[1, 128]", amax_29: "f32[1, 1]", log_1: "f32[1, 1]", tangents_3: "f32[1, 128]", ne_1: "b8[1]", primals_315: "i64[1]", clone_225: "f32[1, 128]", amax_28: "f32[1, 1]", log: "f32[1, 1]", tangents_2: "f32[1, 128]", primals_313: "f32[2, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:835 in forward, code: total_loss = (start_loss + end_loss) / 2
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_4);  ne_4 = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        div_tensor_1: "f32[]" = torch.ops.aten.div.Tensor(div_tensor, convert_element_type_default);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:830 in forward, code: end_positions = end_positions.clamp(0, ignored_index)
        clamp_min_default: "i64[1]" = torch.ops.aten.clamp_min.default(primals_316, 0);  primals_316 = None
        clamp_max_default: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 128);  clamp_min_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        unsqueeze_default: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_default, 1);  clamp_max_default = None
        ne_scalar: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default, 128)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        where_self: "i64[1, 1]" = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, full_default);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:502 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # No stacktrace found for following nodes
        reshape_default: "i64[1, 128]" = torch.ops.aten.reshape.default(iota_default, _shape_param_0);  iota_default = _shape_param_0 = None
        expand_default: "i64[1, 128]" = torch.ops.aten.expand.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        eq_tensor: "b8[1, 128]" = torch.ops.aten.eq.Tensor(expand_default, reshape_default);  expand_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        where_self_1: "f32[1, 128]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        where_self_2: "f32[1, 1]" = torch.ops.aten.where.self(ne_scalar, div_tensor_1, full_default_1);  ne_scalar = div_tensor_1 = None
        mul_tensor: "f32[1, 128]" = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        sub_tensor: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_226, amax_29);  clone_226 = amax_29 = None
        sub_tensor_1: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, log_1);  sub_tensor = log_1 = None
        exp_default: "f32[1, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True)
        mul_tensor_1: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2: "f32[1, 128]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        add_tensor: "f32[1, 128]" = torch.ops.aten.add.Tensor(tangents_3, sub_tensor_2);  tangents_3 = sub_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        sum_default_1: "i64[]" = torch.ops.aten.sum.default(ne_1);  ne_1 = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_1, torch.float32);  sum_default_1 = None
        div_tensor_2: "f32[]" = torch.ops.aten.div.Tensor(div_tensor, convert_element_type_default_1);  div_tensor = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:829 in forward, code: start_positions = start_positions.clamp(0, ignored_index)
        clamp_min_default_1: "i64[1]" = torch.ops.aten.clamp_min.default(primals_315, 0);  primals_315 = None
        clamp_max_default_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_default_1, 128);  clamp_min_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        unsqueeze_default_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_default_1, 1);  clamp_max_default_1 = None
        ne_scalar_1: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_default_1, 128)
        where_self_3: "i64[1, 1]" = torch.ops.aten.where.self(ne_scalar_1, unsqueeze_default_1, full_default);  unsqueeze_default_1 = full_default = None

        # No stacktrace found for following nodes
        expand_default_1: "i64[1, 128]" = torch.ops.aten.expand.default(where_self_3, _shape_param_2);  where_self_3 = _shape_param_2 = None
        eq_tensor_1: "b8[1, 128]" = torch.ops.aten.eq.Tensor(expand_default_1, reshape_default);  expand_default_1 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        where_self_4: "f32[1, 128]" = torch.ops.aten.where.self(eq_tensor_1, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor_1 = scalar_tensor_default_1 = scalar_tensor_default = None
        where_self_5: "f32[1, 1]" = torch.ops.aten.where.self(ne_scalar_1, div_tensor_2, full_default_1);  ne_scalar_1 = div_tensor_2 = full_default_1 = None
        mul_tensor_2: "f32[1, 128]" = torch.ops.aten.mul.Tensor(where_self_4, where_self_5);  where_self_4 = where_self_5 = None
        sub_tensor_3: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_225, amax_28);  clone_225 = amax_28 = None
        sub_tensor_4: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_3, log);  sub_tensor_3 = log = None
        exp_default_1: "f32[1, 128]" = torch.ops.aten.exp.default(sub_tensor_4);  sub_tensor_4 = None
        sum_dim_int_list_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [1], True)
        mul_tensor_3: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp_default_1, sum_dim_int_list_1);  exp_default_1 = sum_dim_int_list_1 = None
        sub_tensor_5: "f32[1, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        add_tensor_1: "f32[1, 128]" = torch.ops.aten.add.Tensor(tangents_2, sub_tensor_5);  tangents_2 = sub_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:818 in forward, code: end_logits = end_logits.squeeze(-1).contiguous()
        unsqueeze_default_2: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add_tensor, 2);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:817 in forward, code: start_logits = start_logits.squeeze(-1).contiguous()
        unsqueeze_default_3: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 2);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:816 in forward, code: start_logits, end_logits = logits.split(1, dim=-1)
        cat_default: "f32[1, 128, 2]" = torch.ops.aten.cat.default([unsqueeze_default_3, unsqueeze_default_2], 2);  unsqueeze_default_3 = unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:815 in forward, code: logits = self.qa_outputs(sequence_output)
        reshape_default_1: "f32[128, 2]" = torch.ops.aten.reshape.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None
        permute_default: "f32[4096, 2]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
        permute_default_1: "f32[2, 4096]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)


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
