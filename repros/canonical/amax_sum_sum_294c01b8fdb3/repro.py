"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForQuestionAnswering_training
Pattern hash: 294c01b8fdb3
Shape hash: aca77079
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_114: "f32[1, 128, 1]", getitem_115: "f32[1, 128, 1]", primals_315: "i64[1]", primals_316: "i64[1]", full_default_1: "f32[]", rsqrt_28: "f32[1, 128, 1]", view_775: "f32[16, 128, 256]", view_771: "f32[16, 128, 256]", view_772: "f32[16, 256, 128]", rsqrt_27: "f32[1, 128, 1]", view_747: "f32[16, 128, 256]", view_743: "f32[16, 128, 256]", view_744: "f32[16, 256, 128]", rsqrt_26: "f32[1, 128, 1]", view_719: "f32[16, 128, 256]", view_715: "f32[16, 128, 256]", view_716: "f32[16, 256, 128]", rsqrt_25: "f32[1, 128, 1]", view_691: "f32[16, 128, 256]", view_687: "f32[16, 128, 256]", view_688: "f32[16, 256, 128]", rsqrt_24: "f32[1, 128, 1]", view_663: "f32[16, 128, 256]", view_659: "f32[16, 128, 256]", view_660: "f32[16, 256, 128]", rsqrt_23: "f32[1, 128, 1]", view_635: "f32[16, 128, 256]", view_631: "f32[16, 128, 256]", view_632: "f32[16, 256, 128]", rsqrt_22: "f32[1, 128, 1]", view_607: "f32[16, 128, 256]", view_603: "f32[16, 128, 256]", view_604: "f32[16, 256, 128]", rsqrt_21: "f32[1, 128, 1]", view_579: "f32[16, 128, 256]", view_575: "f32[16, 128, 256]", view_576: "f32[16, 256, 128]", rsqrt_20: "f32[1, 128, 1]", view_551: "f32[16, 128, 256]", view_547: "f32[16, 128, 256]", view_548: "f32[16, 256, 128]", rsqrt_19: "f32[1, 128, 1]", view_523: "f32[16, 128, 256]", view_519: "f32[16, 128, 256]", view_520: "f32[16, 256, 128]", rsqrt_18: "f32[1, 128, 1]", view_495: "f32[16, 128, 256]", view_491: "f32[16, 128, 256]", view_492: "f32[16, 256, 128]", rsqrt_17: "f32[1, 128, 1]", view_467: "f32[16, 128, 256]", view_463: "f32[16, 128, 256]", view_464: "f32[16, 256, 128]", rsqrt_16: "f32[1, 128, 1]", view_439: "f32[16, 128, 256]", view_435: "f32[16, 128, 256]", view_436: "f32[16, 256, 128]", rsqrt_15: "f32[1, 128, 1]", view_411: "f32[16, 128, 256]", view_407: "f32[16, 128, 256]", view_408: "f32[16, 256, 128]", rsqrt_14: "f32[1, 128, 1]", view_383: "f32[16, 128, 256]", view_379: "f32[16, 128, 256]", view_380: "f32[16, 256, 128]", rsqrt_13: "f32[1, 128, 1]", view_355: "f32[16, 128, 256]", view_351: "f32[16, 128, 256]", view_352: "f32[16, 256, 128]", rsqrt_12: "f32[1, 128, 1]", view_327: "f32[16, 128, 256]", view_323: "f32[16, 128, 256]", view_324: "f32[16, 256, 128]", rsqrt_11: "f32[1, 128, 1]", view_299: "f32[16, 128, 256]", view_295: "f32[16, 128, 256]", view_296: "f32[16, 256, 128]", rsqrt_10: "f32[1, 128, 1]", view_271: "f32[16, 128, 256]", view_267: "f32[16, 128, 256]", view_268: "f32[16, 256, 128]", rsqrt_9: "f32[1, 128, 1]", view_243: "f32[16, 128, 256]", view_239: "f32[16, 128, 256]", view_240: "f32[16, 256, 128]", rsqrt_8: "f32[1, 128, 1]", view_215: "f32[16, 128, 256]", view_211: "f32[16, 128, 256]", view_212: "f32[16, 256, 128]", rsqrt_7: "f32[1, 128, 1]", view_187: "f32[16, 128, 256]", view_183: "f32[16, 128, 256]", view_184: "f32[16, 256, 128]", rsqrt_6: "f32[1, 128, 1]", view_159: "f32[16, 128, 256]", view_155: "f32[16, 128, 256]", view_156: "f32[16, 256, 128]", rsqrt_5: "f32[1, 128, 1]", view_131: "f32[16, 128, 256]", view_127: "f32[16, 128, 256]", view_128: "f32[16, 256, 128]", rsqrt_4: "f32[1, 128, 1]", view_103: "f32[16, 128, 256]", view_99: "f32[16, 128, 256]", view_100: "f32[16, 256, 128]", rsqrt_3: "f32[1, 128, 1]", view_75: "f32[16, 128, 256]", view_71: "f32[16, 128, 256]", view_72: "f32[16, 256, 128]", rsqrt_2: "f32[1, 128, 1]", view_47: "f32[16, 128, 256]", view_43: "f32[16, 128, 256]", view_44: "f32[16, 256, 128]", rsqrt_1: "f32[1, 128, 1]", view_19: "f32[16, 128, 256]", view_15: "f32[16, 128, 256]", view_16: "f32[16, 256, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:817 in forward, code: start_logits = start_logits.squeeze(-1).contiguous()
        squeeze_dim: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_114, -1);  getitem_114 = None
        clone_default: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:818 in forward, code: end_logits = end_logits.squeeze(-1).contiguous()
        squeeze_dim_1: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_115, -1);  getitem_115 = None
        clone_default_1: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_dim_1, memory_format = torch.contiguous_format);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:829 in forward, code: start_positions = start_positions.clamp(0, ignored_index)
        clamp_min_default: "i64[1]" = torch.ops.aten.clamp_min.default(primals_315, 0);  primals_315 = None
        clamp_max_default: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 128);  clamp_min_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:830 in forward, code: end_positions = end_positions.clamp(0, ignored_index)
        clamp_min_default_1: "i64[1]" = torch.ops.aten.clamp_min.default(primals_316, 0);  primals_316 = None
        clamp_max_default_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_default_1, 128);  clamp_min_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        amax_default: "f32[1, 1]" = torch.ops.aten.amax.default(clone_default, [1], True)
        sub_tensor: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_default, amax_default);  clone_default = amax_default = None
        exp_default: "f32[1, 128]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[1, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_default, 128)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[1]" = torch.ops.aten.where.self(ne_scalar, clamp_max_default, full_default);  clamp_max_default = None
        unsqueeze_default: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[1, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[1]" = torch.ops.aten.neg.default(squeeze_dim_2);  squeeze_dim_2 = None
        where_self_1: "f32[1]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        amax_default_1: "f32[1, 1]" = torch.ops.aten.amax.default(clone_default_1, [1], True)
        sub_tensor_2: "f32[1, 128]" = torch.ops.aten.sub.Tensor(clone_default_1, amax_default_1);  clone_default_1 = amax_default_1 = None
        exp_default_1: "f32[1, 128]" = torch.ops.aten.exp.default(sub_tensor_2)
        sum_dim_int_list_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [1], True);  exp_default_1 = None
        log_default_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_3: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_2, log_default_1);  sub_tensor_2 = log_default_1 = None
        ne_scalar_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_default_1, 128)
        where_self_2: "i64[1]" = torch.ops.aten.where.self(ne_scalar_1, clamp_max_default_1, full_default);  clamp_max_default_1 = full_default = None
        unsqueeze_default_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_self_2, 1);  where_self_2 = None
        gather_default_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_tensor_3, 1, unsqueeze_default_1);  sub_tensor_3 = unsqueeze_default_1 = None
        squeeze_dim_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_default_1, 1);  gather_default_1 = None
        neg_default_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_dim_3);  squeeze_dim_3 = None
        where_self_3: "f32[1]" = torch.ops.aten.where.self(ne_scalar_1, neg_default_1, full_default_1);  neg_default_1 = full_default_1 = None
        sum_default_2: "i64[]" = torch.ops.aten.sum.default(ne_scalar_1);  ne_scalar_1 = None
        convert_element_type_default_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default_2, torch.float32);  sum_default_2 = None
        sum_default_3: "f32[]" = torch.ops.aten.sum.default(where_self_3);  where_self_3 = None
        div_tensor_1: "f32[]" = torch.ops.aten.div.Tensor(sum_default_3, convert_element_type_default_1);  sum_default_3 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:835 in forward, code: total_loss = (start_loss + end_loss) / 2
        add_tensor: "f32[]" = torch.ops.aten.add.Tensor(div_tensor, div_tensor_1);  div_tensor = div_tensor_1 = None
        div_tensor_2: "f32[]" = torch.ops.aten.div.Tensor(add_tensor, 2);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        div_tensor_3: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 4096);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_775, [0, 2, 1]);  view_775 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_1: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_771, [0, 2, 1]);  view_771 = None
        permute_default_2: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_772, [0, 2, 1]);  view_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_4: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 4096);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_3: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_747, [0, 2, 1]);  view_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_4: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_743, [0, 2, 1]);  view_743 = None
        permute_default_5: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_744, [0, 2, 1]);  view_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_5: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 4096);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_6: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_719, [0, 2, 1]);  view_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_7: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_715, [0, 2, 1]);  view_715 = None
        permute_default_8: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_716, [0, 2, 1]);  view_716 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_6: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 4096);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_9: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_691, [0, 2, 1]);  view_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_10: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_687, [0, 2, 1]);  view_687 = None
        permute_default_11: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_688, [0, 2, 1]);  view_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_7: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 4096);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_12: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_663, [0, 2, 1]);  view_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_13: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_659, [0, 2, 1]);  view_659 = None
        permute_default_14: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_660, [0, 2, 1]);  view_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_8: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 4096);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_15: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_635, [0, 2, 1]);  view_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_16: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_631, [0, 2, 1]);  view_631 = None
        permute_default_17: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_632, [0, 2, 1]);  view_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_9: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 4096);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_18: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_607, [0, 2, 1]);  view_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_19: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_603, [0, 2, 1]);  view_603 = None
        permute_default_20: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_604, [0, 2, 1]);  view_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_10: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 4096);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_21: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_579, [0, 2, 1]);  view_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_22: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_575, [0, 2, 1]);  view_575 = None
        permute_default_23: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_576, [0, 2, 1]);  view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_11: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 4096);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_24: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_551, [0, 2, 1]);  view_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_25: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_547, [0, 2, 1]);  view_547 = None
        permute_default_26: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_548, [0, 2, 1]);  view_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_12: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 4096);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_27: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_523, [0, 2, 1]);  view_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_28: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None
        permute_default_29: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_520, [0, 2, 1]);  view_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_13: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 4096);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_30: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_31: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_491, [0, 2, 1]);  view_491 = None
        permute_default_32: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_492, [0, 2, 1]);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_14: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 4096);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_33: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_34: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_463, [0, 2, 1]);  view_463 = None
        permute_default_35: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_464, [0, 2, 1]);  view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_15: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 4096);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_36: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_37: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_435, [0, 2, 1]);  view_435 = None
        permute_default_38: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_436, [0, 2, 1]);  view_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_16: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 4096);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_39: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_40: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None
        permute_default_41: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_408, [0, 2, 1]);  view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_17: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 4096);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_42: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_43: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_379, [0, 2, 1]);  view_379 = None
        permute_default_44: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_380, [0, 2, 1]);  view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_18: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 4096);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_45: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_355, [0, 2, 1]);  view_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_46: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
        permute_default_47: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_19: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 4096);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_48: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_327, [0, 2, 1]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_49: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
        permute_default_50: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_20: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 4096);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_51: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_52: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
        permute_default_53: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_21: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 4096);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_54: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_55: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
        permute_default_56: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_22: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 4096);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_57: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_58: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
        permute_default_59: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_240, [0, 2, 1]);  view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_23: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 4096);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_60: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_215, [0, 2, 1]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_61: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        permute_default_62: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_24: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 4096);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_63: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_64: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        permute_default_65: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_184, [0, 2, 1]);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_25: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 4096);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_66: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_67: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
        permute_default_68: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_26: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 4096);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_69: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_70: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_default_71: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_27: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 4096);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_72: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_73: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        permute_default_74: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_28: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 4096);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_75: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_76: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
        permute_default_77: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_29: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 4096);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_78: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_79: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
        permute_default_80: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        div_tensor_30: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 4096);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        permute_default_81: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_19, [0, 2, 1]);  view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_82: "f32[16, 256, 128]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_default_83: "f32[16, 128, 256]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        return (div_tensor_2, div_tensor_3, permute_default, permute_default_1, permute_default_2, div_tensor_4, permute_default_3, permute_default_4, permute_default_5, div_tensor_5, permute_default_6, permute_default_7, permute_default_8, div_tensor_6, permute_default_9, permute_default_10, permute_default_11, div_tensor_7, permute_default_12, permute_default_13, permute_default_14, div_tensor_8, permute_default_15, permute_default_16, permute_default_17, div_tensor_9, permute_default_18, permute_default_19, permute_default_20, div_tensor_10, permute_default_21, permute_default_22, permute_default_23, div_tensor_11, permute_default_24, permute_default_25, permute_default_26, div_tensor_12, permute_default_27, permute_default_28, permute_default_29, div_tensor_13, permute_default_30, permute_default_31, permute_default_32, div_tensor_14, permute_default_33, permute_default_34, permute_default_35, div_tensor_15, permute_default_36, permute_default_37, permute_default_38, div_tensor_16, permute_default_39, permute_default_40, permute_default_41, div_tensor_17, permute_default_42, permute_default_43, permute_default_44, div_tensor_18, permute_default_45, permute_default_46, permute_default_47, div_tensor_19, permute_default_48, permute_default_49, permute_default_50, div_tensor_20, permute_default_51, permute_default_52, permute_default_53, div_tensor_21, permute_default_54, permute_default_55, permute_default_56, div_tensor_22, permute_default_57, permute_default_58, permute_default_59, div_tensor_23, permute_default_60, permute_default_61, permute_default_62, div_tensor_24, permute_default_63, permute_default_64, permute_default_65, div_tensor_25, permute_default_66, permute_default_67, permute_default_68, div_tensor_26, permute_default_69, permute_default_70, permute_default_71, div_tensor_27, permute_default_72, permute_default_73, permute_default_74, div_tensor_28, permute_default_75, permute_default_76, permute_default_77, div_tensor_29, permute_default_78, permute_default_79, permute_default_80, div_tensor_30, permute_default_81, permute_default_82, permute_default_83)


def _default_make_inputs():
    return [
    torch.randn(255, dtype=torch.float32, device='cuda').as_strided([1, 128, 1], [256, 2, 1]),  # getitem_114
    torch.randn(255, dtype=torch.float32, device='cuda').as_strided([1, 128, 1], [256, 2, 1]),  # getitem_115
    torch.randint(0, 2, [1], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_775
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_771
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_772
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_747
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_743
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_744
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_719
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_715
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_716
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_691
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_687
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_688
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_663
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_659
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_660
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_635
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_631
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_632
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_607
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_603
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_604
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_579
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_575
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_576
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_551
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_547
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_548
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_523
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_519
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_520
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_495
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_491
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_492
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_467
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_463
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_464
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_439
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_435
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_436
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_411
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_407
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_408
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_383
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_379
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_380
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_355
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_351
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_352
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_327
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_323
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_324
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_299
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_295
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_296
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_271
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_267
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_268
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_243
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_239
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_240
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_215
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_211
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_212
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_187
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_183
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_184
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_159
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_155
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_156
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_131
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_127
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_128
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_103
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_99
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_100
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_75
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_71
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_72
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_47
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_43
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_44
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_19
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 128, 256], [256, 4096, 1]),  # view_15
    torch.randn(524288, dtype=torch.float32, device='cuda').as_strided([16, 256, 128], [256, 1, 4096]),  # view_16
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
