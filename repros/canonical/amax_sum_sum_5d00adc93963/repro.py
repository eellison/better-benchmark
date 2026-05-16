"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_training
Pattern hash: 5d00adc93963
Shape hash: 9f284891
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_74: "f32[4096, 30000]", _shape_param_0, _shape_param_1, primals_32: "i64[8, 512]", full_default_1: "f32[]", rsqrt_24: "f32[8, 512, 1]", rsqrt_23: "f32[8, 512, 1]", view_257: "f32[512, 512, 64]", view_253: "f32[512, 512, 64]", view_254: "f32[512, 64, 512]", rsqrt_22: "f32[8, 512, 1]", rsqrt_21: "f32[8, 512, 1]", view_235: "f32[512, 512, 64]", view_231: "f32[512, 512, 64]", view_232: "f32[512, 64, 512]", rsqrt_20: "f32[8, 512, 1]", rsqrt_19: "f32[8, 512, 1]", view_213: "f32[512, 512, 64]", view_209: "f32[512, 512, 64]", view_210: "f32[512, 64, 512]", rsqrt_18: "f32[8, 512, 1]", rsqrt_17: "f32[8, 512, 1]", view_191: "f32[512, 512, 64]", view_187: "f32[512, 512, 64]", view_188: "f32[512, 64, 512]", rsqrt_16: "f32[8, 512, 1]", rsqrt_15: "f32[8, 512, 1]", view_169: "f32[512, 512, 64]", view_165: "f32[512, 512, 64]", view_166: "f32[512, 64, 512]", rsqrt_14: "f32[8, 512, 1]", rsqrt_13: "f32[8, 512, 1]", view_147: "f32[512, 512, 64]", view_143: "f32[512, 512, 64]", view_144: "f32[512, 64, 512]", rsqrt_12: "f32[8, 512, 1]", rsqrt_11: "f32[8, 512, 1]", view_125: "f32[512, 512, 64]", view_121: "f32[512, 512, 64]", view_122: "f32[512, 64, 512]", rsqrt_10: "f32[8, 512, 1]", rsqrt_9: "f32[8, 512, 1]", view_103: "f32[512, 512, 64]", view_99: "f32[512, 512, 64]", view_100: "f32[512, 64, 512]", rsqrt_8: "f32[8, 512, 1]", rsqrt_7: "f32[8, 512, 1]", view_81: "f32[512, 512, 64]", view_77: "f32[512, 512, 64]", view_78: "f32[512, 64, 512]", rsqrt_6: "f32[8, 512, 1]", rsqrt_5: "f32[8, 512, 1]", view_59: "f32[512, 512, 64]", view_55: "f32[512, 512, 64]", view_56: "f32[512, 64, 512]", rsqrt_4: "f32[8, 512, 1]", rsqrt_3: "f32[8, 512, 1]", view_37: "f32[512, 512, 64]", view_33: "f32[512, 512, 64]", view_34: "f32[512, 64, 512]", rsqrt_2: "f32[8, 512, 1]", rsqrt_1: "f32[8, 512, 1]", view_15: "f32[512, 512, 64]", view_11: "f32[512, 512, 64]", view_12: "f32[512, 64, 512]", rsqrt: "f32[8, 512, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:541 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default: "f32[8, 512, 30000]" = torch.ops.aten.reshape.default(addmm_74, _shape_param_0);  addmm_74 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:650 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        reshape_default_1: "f32[4096, 30000]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        reshape_default_2: "i64[4096]" = torch.ops.aten.reshape.default(primals_32, [-1]);  primals_32 = None
        amax_default: "f32[4096, 1]" = torch.ops.aten.amax.default(reshape_default_1, [1], True)
        sub_tensor: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(reshape_default_1, amax_default);  reshape_default_1 = amax_default = None
        exp_default: "f32[4096, 30000]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[4096, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[4096]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4096]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default);  reshape_default_2 = full_default = None
        unsqueeze_default: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[4096]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[4096]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        where_self_1: "f32[4096]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_1: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 4096);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_2: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 4096);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
        permute_default_1: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        permute_default_2: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_3: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 4096);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_4: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 4096);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_3: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        permute_default_4: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        permute_default_5: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_5: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 4096);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_6: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 4096);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_6: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
        permute_default_7: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        permute_default_8: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_7: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 4096);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_8: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 4096);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_9: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        permute_default_10: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
        permute_default_11: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_9: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 4096);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_10: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 4096);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_12: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
        permute_default_13: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
        permute_default_14: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_11: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 4096);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_12: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 4096);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_15: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_default_16: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
        permute_default_17: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_13: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 4096);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_14: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 4096);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_18: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
        permute_default_19: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        permute_default_20: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_15: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 4096);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_16: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 4096);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_21: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
        permute_default_22: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        permute_default_23: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_17: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 4096);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_18: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 4096);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_24: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
        permute_default_25: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
        permute_default_26: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_19: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 4096);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_20: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 4096);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_27: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
        permute_default_28: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
        permute_default_29: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_21: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 4096);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_22: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 4096);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_30: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
        permute_default_31: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_default_32: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:235 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        div_tensor_23: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 4096);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:202 in forward, code: attn_output = self.LayerNorm(hidden_states + attn_output)
        div_tensor_24: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 4096);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_33: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_default_34: "f32[512, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_default_35: "f32[512, 512, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:108 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_tensor_25: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        return (div_tensor, div_tensor_1, div_tensor_2, permute_default, permute_default_1, permute_default_2, div_tensor_3, div_tensor_4, permute_default_3, permute_default_4, permute_default_5, div_tensor_5, div_tensor_6, permute_default_6, permute_default_7, permute_default_8, div_tensor_7, div_tensor_8, permute_default_9, permute_default_10, permute_default_11, div_tensor_9, div_tensor_10, permute_default_12, permute_default_13, permute_default_14, div_tensor_11, div_tensor_12, permute_default_15, permute_default_16, permute_default_17, div_tensor_13, div_tensor_14, permute_default_18, permute_default_19, permute_default_20, div_tensor_15, div_tensor_16, permute_default_21, permute_default_22, permute_default_23, div_tensor_17, div_tensor_18, permute_default_24, permute_default_25, permute_default_26, div_tensor_19, div_tensor_20, permute_default_27, permute_default_28, permute_default_29, div_tensor_21, div_tensor_22, permute_default_30, permute_default_31, permute_default_32, div_tensor_23, div_tensor_24, permute_default_33, permute_default_34, permute_default_35, div_tensor_25)


def _default_make_inputs():
    return [
    torch.randn([4096, 30000], dtype=torch.float32, device='cuda'),
    [8, 512, 30000],  # _shape_param_0
    [-1, 30000],  # _shape_param_1
    torch.randint(0, 2, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
