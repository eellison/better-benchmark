"""
Standalone repro captured via capture_hook.
Label: hf_BertForMaskedLM_train
Pattern hash: 1d8303cdfc8a
Shape hash: c21c2f9a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_default: "f32[16384, 30524]", _shape_param_0, _shape_param_1, primals_206: "i64[32, 512]", full_default_1: "f32[]", rsqrt_24: "f32[32, 512, 1]", rsqrt_23: "f32[32, 512, 1]", view_254: "f32[384, 512, 512]", view_255: "f32[384, 512, 64]", view_251: "f32[384, 512, 64]", view_252: "f32[384, 64, 512]", rsqrt_22: "f32[32, 512, 1]", rsqrt_21: "f32[32, 512, 1]", view_232: "f32[384, 512, 512]", view_233: "f32[384, 512, 64]", view_229: "f32[384, 512, 64]", view_230: "f32[384, 64, 512]", rsqrt_20: "f32[32, 512, 1]", rsqrt_19: "f32[32, 512, 1]", view_210: "f32[384, 512, 512]", view_211: "f32[384, 512, 64]", view_207: "f32[384, 512, 64]", view_208: "f32[384, 64, 512]", rsqrt_18: "f32[32, 512, 1]", rsqrt_17: "f32[32, 512, 1]", view_188: "f32[384, 512, 512]", view_189: "f32[384, 512, 64]", view_185: "f32[384, 512, 64]", view_186: "f32[384, 64, 512]", rsqrt_16: "f32[32, 512, 1]", rsqrt_15: "f32[32, 512, 1]", view_166: "f32[384, 512, 512]", view_167: "f32[384, 512, 64]", view_163: "f32[384, 512, 64]", view_164: "f32[384, 64, 512]", rsqrt_14: "f32[32, 512, 1]", rsqrt_13: "f32[32, 512, 1]", view_144: "f32[384, 512, 512]", view_145: "f32[384, 512, 64]", view_141: "f32[384, 512, 64]", view_142: "f32[384, 64, 512]", rsqrt_12: "f32[32, 512, 1]", rsqrt_11: "f32[32, 512, 1]", view_122: "f32[384, 512, 512]", view_123: "f32[384, 512, 64]", view_119: "f32[384, 512, 64]", view_120: "f32[384, 64, 512]", rsqrt_10: "f32[32, 512, 1]", rsqrt_9: "f32[32, 512, 1]", view_100: "f32[384, 512, 512]", view_101: "f32[384, 512, 64]", view_97: "f32[384, 512, 64]", view_98: "f32[384, 64, 512]", rsqrt_8: "f32[32, 512, 1]", rsqrt_7: "f32[32, 512, 1]", view_78: "f32[384, 512, 512]", view_79: "f32[384, 512, 64]", view_75: "f32[384, 512, 64]", view_76: "f32[384, 64, 512]", rsqrt_6: "f32[32, 512, 1]", rsqrt_5: "f32[32, 512, 1]", view_56: "f32[384, 512, 512]", view_57: "f32[384, 512, 64]", view_53: "f32[384, 512, 64]", view_54: "f32[384, 64, 512]", rsqrt_4: "f32[32, 512, 1]", rsqrt_3: "f32[32, 512, 1]", view_34: "f32[384, 512, 512]", view_35: "f32[384, 512, 64]", view_31: "f32[384, 512, 64]", view_32: "f32[384, 64, 512]", rsqrt_2: "f32[32, 512, 1]", rsqrt_1: "f32[32, 512, 1]", view_12: "f32[384, 512, 512]", view_13: "f32[384, 512, 64]", view_9: "f32[384, 512, 64]", view_10: "f32[384, 64, 512]", rsqrt: "f32[32, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        slice_tensor: "f32[16384, 30522]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        reshape_default: "f32[32, 512, 30522]" = torch.ops.aten.reshape.default(slice_tensor, _shape_param_0);  slice_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:979 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default_1: "f32[16384, 30522]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        reshape_default_2: "i64[16384]" = torch.ops.aten.reshape.default(primals_206, [-1]);  primals_206 = None
        amax_default: "f32[16384, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[16384, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[16384]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[16384]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default);  reshape_default_2 = full_default = None
        unsqueeze_default: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[16384, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[16384]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[16384]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        where_self_1: "f32[16384]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_1: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_2: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_default_1: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
        permute_default_2: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_default_3: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_3: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_4: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_4: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        permute_default_5: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        permute_default_6: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_default_7: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_5: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_6: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_8: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_default_9: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        permute_default_10: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_default_11: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_7: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_8: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_12: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_default_13: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
        permute_default_14: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        permute_default_15: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_9: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_10: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_16: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        permute_default_17: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_default_18: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_default_19: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_11: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_12: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_20: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        permute_default_21: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        permute_default_22: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        permute_default_23: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_13: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_14: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_24: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_default_25: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        permute_default_26: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_default_27: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_15: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_16: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_28: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        permute_default_29: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
        permute_default_30: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_default_31: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_17: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_18: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_32: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        permute_default_33: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
        permute_default_34: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        permute_default_35: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_19: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_20: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_36: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        permute_default_37: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_default_38: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_default_39: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_21: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_22: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_40: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_default_41: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_default_42: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_default_43: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_23: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_tensor_24: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_44: "f32[384, 512, 512]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_default_45: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_default_46: "f32[384, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_default_47: "f32[384, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_tensor_25: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_tensor, div_tensor_1, div_tensor_2, permute_default, permute_default_1, permute_default_2, permute_default_3, div_tensor_3, div_tensor_4, permute_default_4, permute_default_5, permute_default_6, permute_default_7, div_tensor_5, div_tensor_6, permute_default_8, permute_default_9, permute_default_10, permute_default_11, div_tensor_7, div_tensor_8, permute_default_12, permute_default_13, permute_default_14, permute_default_15, div_tensor_9, div_tensor_10, permute_default_16, permute_default_17, permute_default_18, permute_default_19, div_tensor_11, div_tensor_12, permute_default_20, permute_default_21, permute_default_22, permute_default_23, div_tensor_13, div_tensor_14, permute_default_24, permute_default_25, permute_default_26, permute_default_27, div_tensor_15, div_tensor_16, permute_default_28, permute_default_29, permute_default_30, permute_default_31, div_tensor_17, div_tensor_18, permute_default_32, permute_default_33, permute_default_34, permute_default_35, div_tensor_19, div_tensor_20, permute_default_36, permute_default_37, permute_default_38, permute_default_39, div_tensor_21, div_tensor_22, permute_default_40, permute_default_41, permute_default_42, permute_default_43, div_tensor_23, div_tensor_24, permute_default_44, permute_default_45, permute_default_46, permute_default_47, div_tensor_25)


def _default_make_inputs():
    return [
    torch.randn([16384, 30524], dtype=torch.float32, device='cuda'),
    [32, 512, 30522],  # _shape_param_0
    [-1, 30522],  # _shape_param_1
    torch.randint(0, 2, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
