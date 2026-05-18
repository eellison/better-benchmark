"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_training
Pattern hash: 54778c120192
Shape hash: 6ee6170a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_139: "f32[1024, 30522]", _shape_param_0, view_141: "f32[8, 128, 768]", mul_85: "f32[8, 128, 768]", view_142: "f32[1024, 768]", _shape_param_1, view_144: "f32[8, 128, 768]", mul_80: "f32[8, 128, 768]", view_145: "f32[1024, 768]", _shape_param_2, view_148: "f32[1024, 3072]", _shape_param_3, add_61: "f32[8, 128, 768]", mul_73: "f32[8, 128, 768]", view_151: "f32[1024, 768]", _shape_param_4, view_162: "f32[1024, 768]", _shape_param_5, view_166: "f32[1024, 768]", _shape_param_6, view_170: "f32[1024, 768]", _shape_param_7, add_64: "f32[8, 128, 768]", mul_67: "f32[8, 128, 768]", view_173: "f32[1024, 768]", _shape_param_8, view_176: "f32[1024, 3072]", _shape_param_9, add_67: "f32[8, 128, 768]", mul_60: "f32[8, 128, 768]", view_179: "f32[1024, 768]", _shape_param_10, view_190: "f32[1024, 768]", _shape_param_11, view_194: "f32[1024, 768]", _shape_param_12, view_198: "f32[1024, 768]", _shape_param_13, add_70: "f32[8, 128, 768]", mul_54: "f32[8, 128, 768]", view_201: "f32[1024, 768]", _shape_param_14, view_204: "f32[1024, 3072]", _shape_param_15, add_73: "f32[8, 128, 768]", mul_47: "f32[8, 128, 768]", view_207: "f32[1024, 768]", _shape_param_16, view_218: "f32[1024, 768]", _shape_param_17, view_222: "f32[1024, 768]", _shape_param_18, view_226: "f32[1024, 768]", _shape_param_19, add_76: "f32[8, 128, 768]", mul_41: "f32[8, 128, 768]", view_229: "f32[1024, 768]", _shape_param_20, view_232: "f32[1024, 3072]", _shape_param_21, add_79: "f32[8, 128, 768]", mul_34: "f32[8, 128, 768]", view_235: "f32[1024, 768]", _shape_param_22, view_246: "f32[1024, 768]", _shape_param_23, view_250: "f32[1024, 768]", _shape_param_24, view_254: "f32[1024, 768]", _shape_param_25, add_82: "f32[8, 128, 768]", mul_28: "f32[8, 128, 768]", view_257: "f32[1024, 768]", _shape_param_26, view_260: "f32[1024, 3072]", _shape_param_27, add_85: "f32[8, 128, 768]", mul_21: "f32[8, 128, 768]", view_263: "f32[1024, 768]", _shape_param_28, view_274: "f32[1024, 768]", _shape_param_29, view_278: "f32[1024, 768]", _shape_param_30, view_282: "f32[1024, 768]", _shape_param_31, add_88: "f32[8, 128, 768]", mul_15: "f32[8, 128, 768]", view_285: "f32[1024, 768]", _shape_param_32, view_288: "f32[1024, 3072]", _shape_param_33, add_91: "f32[8, 128, 768]", mul_8: "f32[8, 128, 768]", view_291: "f32[1024, 768]", _shape_param_34, view_302: "f32[1024, 768]", _shape_param_35, mm_70: "f32[1024, 768]", _shape_param_36, mul_264: "f32[8, 128, 768]", view_306: "f32[1024, 768]", _shape_param_37, mm_72: "f32[1024, 768]", _shape_param_38, view_310: "f32[1024, 768]", _shape_param_39, mm_74: "f32[1024, 768]", _shape_param_40, gt: "b8[8, 128, 768]", primals_5: "f32[768]", embedding: "f32[8, 128, 768]", embedding_1: "f32[1, 128, 768]", getitem_1: "f32[8, 128, 1]", rsqrt: "f32[8, 128, 1]", primals_3: "i64[1, 512]", full_default_1: "f32[]", primals_1: "i64[8, 128]", mm_1: "f32[30522, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:517 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        sum_dim_int_list: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_139, [0], True);  view_139 = None
        reshape_default: "f32[30522]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:516 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        mul_tensor: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(view_141, mul_85);  mul_85 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_141, [0, 1]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:514 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        permute_default: "f32[768, 1024]" = torch.ops.aten.permute.default(view_142, [1, 0])
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_142, [0], True);  view_142 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_tensor_1: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(view_144, mul_80);  mul_80 = None
        sum_dim_int_list_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_144, [0, 1]);  view_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        permute_default_1: "f32[768, 1024]" = torch.ops.aten.permute.default(view_145, [1, 0])
        sum_dim_int_list_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_145, [0], True);  view_145 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        permute_default_2: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_148, [1, 0])
        sum_dim_int_list_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_148, [0], True);  view_148 = None
        reshape_default_3: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_tensor_2: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_61, mul_73);  mul_73 = None
        sum_dim_int_list_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_61, [0, 1]);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        permute_default_3: "f32[768, 1024]" = torch.ops.aten.permute.default(view_151, [1, 0])
        sum_dim_int_list_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_151, [0], True);  view_151 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_4);  sum_dim_int_list_10 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_4: "f32[768, 1024]" = torch.ops.aten.permute.default(view_162, [1, 0])
        sum_dim_int_list_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_162, [0], True);  view_162 = None
        reshape_default_5: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_5: "f32[768, 1024]" = torch.ops.aten.permute.default(view_166, [1, 0])
        sum_dim_int_list_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_166, [0], True);  view_166 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_6: "f32[768, 1024]" = torch.ops.aten.permute.default(view_170, [1, 0])
        sum_dim_int_list_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
        reshape_default_7: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_7);  sum_dim_int_list_13 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_tensor_3: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_64, mul_67);  mul_67 = None
        sum_dim_int_list_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_64, [0, 1]);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        permute_default_7: "f32[768, 1024]" = torch.ops.aten.permute.default(view_173, [1, 0])
        sum_dim_int_list_16: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        permute_default_8: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_176, [1, 0])
        sum_dim_int_list_17: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_176, [0], True);  view_176 = None
        reshape_default_9: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_9);  sum_dim_int_list_17 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_tensor_4: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_67, mul_60);  mul_60 = None
        sum_dim_int_list_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_67, [0, 1]);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        permute_default_9: "f32[768, 1024]" = torch.ops.aten.permute.default(view_179, [1, 0])
        sum_dim_int_list_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_179, [0], True);  view_179 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_10);  sum_dim_int_list_20 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_10: "f32[768, 1024]" = torch.ops.aten.permute.default(view_190, [1, 0])
        sum_dim_int_list_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_190, [0], True);  view_190 = None
        reshape_default_11: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_11);  sum_dim_int_list_21 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_11: "f32[768, 1024]" = torch.ops.aten.permute.default(view_194, [1, 0])
        sum_dim_int_list_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_194, [0], True);  view_194 = None
        reshape_default_12: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_12);  sum_dim_int_list_22 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_12: "f32[768, 1024]" = torch.ops.aten.permute.default(view_198, [1, 0])
        sum_dim_int_list_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
        reshape_default_13: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_13);  sum_dim_int_list_23 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_tensor_5: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_70, mul_54);  mul_54 = None
        sum_dim_int_list_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_70, [0, 1]);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        permute_default_13: "f32[768, 1024]" = torch.ops.aten.permute.default(view_201, [1, 0])
        sum_dim_int_list_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
        reshape_default_14: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_14);  sum_dim_int_list_26 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        permute_default_14: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_204, [1, 0])
        sum_dim_int_list_27: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
        reshape_default_15: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_15);  sum_dim_int_list_27 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_tensor_6: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_73, mul_47);  mul_47 = None
        sum_dim_int_list_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_73, [0, 1]);  add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        permute_default_15: "f32[768, 1024]" = torch.ops.aten.permute.default(view_207, [1, 0])
        sum_dim_int_list_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
        reshape_default_16: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_16);  sum_dim_int_list_30 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_16: "f32[768, 1024]" = torch.ops.aten.permute.default(view_218, [1, 0])
        sum_dim_int_list_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
        reshape_default_17: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_17);  sum_dim_int_list_31 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_17: "f32[768, 1024]" = torch.ops.aten.permute.default(view_222, [1, 0])
        sum_dim_int_list_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_18);  sum_dim_int_list_32 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_18: "f32[768, 1024]" = torch.ops.aten.permute.default(view_226, [1, 0])
        sum_dim_int_list_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
        reshape_default_19: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_19);  sum_dim_int_list_33 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_tensor_7: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_76, mul_41);  mul_41 = None
        sum_dim_int_list_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_76, [0, 1]);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        permute_default_19: "f32[768, 1024]" = torch.ops.aten.permute.default(view_229, [1, 0])
        sum_dim_int_list_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_229, [0], True);  view_229 = None
        reshape_default_20: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_20);  sum_dim_int_list_36 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        permute_default_20: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_232, [1, 0])
        sum_dim_int_list_37: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_232, [0], True);  view_232 = None
        reshape_default_21: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_21);  sum_dim_int_list_37 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_tensor_8: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_79, mul_34);  mul_34 = None
        sum_dim_int_list_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_79, [0, 1]);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        permute_default_21: "f32[768, 1024]" = torch.ops.aten.permute.default(view_235, [1, 0])
        sum_dim_int_list_40: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_235, [0], True);  view_235 = None
        reshape_default_22: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_22);  sum_dim_int_list_40 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_22: "f32[768, 1024]" = torch.ops.aten.permute.default(view_246, [1, 0])
        sum_dim_int_list_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
        reshape_default_23: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_23);  sum_dim_int_list_41 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_23: "f32[768, 1024]" = torch.ops.aten.permute.default(view_250, [1, 0])
        sum_dim_int_list_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True);  view_250 = None
        reshape_default_24: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_24);  sum_dim_int_list_42 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_24: "f32[768, 1024]" = torch.ops.aten.permute.default(view_254, [1, 0])
        sum_dim_int_list_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
        reshape_default_25: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_25);  sum_dim_int_list_43 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_tensor_9: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_82, mul_28);  mul_28 = None
        sum_dim_int_list_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_82, [0, 1]);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        permute_default_25: "f32[768, 1024]" = torch.ops.aten.permute.default(view_257, [1, 0])
        sum_dim_int_list_46: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_257, [0], True);  view_257 = None
        reshape_default_26: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_26);  sum_dim_int_list_46 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        permute_default_26: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_260, [1, 0])
        sum_dim_int_list_47: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True);  view_260 = None
        reshape_default_27: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_27);  sum_dim_int_list_47 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_tensor_10: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_85, mul_21);  mul_21 = None
        sum_dim_int_list_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_85, [0, 1]);  add_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        permute_default_27: "f32[768, 1024]" = torch.ops.aten.permute.default(view_263, [1, 0])
        sum_dim_int_list_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
        reshape_default_28: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_28);  sum_dim_int_list_50 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_28: "f32[768, 1024]" = torch.ops.aten.permute.default(view_274, [1, 0])
        sum_dim_int_list_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
        reshape_default_29: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_29);  sum_dim_int_list_51 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_29: "f32[768, 1024]" = torch.ops.aten.permute.default(view_278, [1, 0])
        sum_dim_int_list_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
        reshape_default_30: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_30);  sum_dim_int_list_52 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_30: "f32[768, 1024]" = torch.ops.aten.permute.default(view_282, [1, 0])
        sum_dim_int_list_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        reshape_default_31: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_31);  sum_dim_int_list_53 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:261 in forward, code: ffn_output = self.output_layer_norm(ffn_output + attention_output)
        mul_tensor_11: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_88, mul_15);  mul_15 = None
        sum_dim_int_list_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_88, [0, 1]);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:226 in ff_chunk, code: x = self.lin2(x)
        permute_default_31: "f32[768, 1024]" = torch.ops.aten.permute.default(view_285, [1, 0])
        sum_dim_int_list_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        reshape_default_32: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_32);  sum_dim_int_list_56 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:224 in ff_chunk, code: x = self.lin1(input)
        permute_default_32: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_288, [1, 0])
        sum_dim_int_list_57: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
        reshape_default_33: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_33);  sum_dim_int_list_57 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:257 in forward, code: attention_output = self.sa_layer_norm(attention_output + hidden_states)
        mul_tensor_12: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_91, mul_8);  mul_8 = None
        sum_dim_int_list_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_91, [0, 1]);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:206 in forward, code: attn_output = self.out_lin(attn_output)
        permute_default_33: "f32[768, 1024]" = torch.ops.aten.permute.default(view_291, [1, 0])
        sum_dim_int_list_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
        reshape_default_34: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_34);  sum_dim_int_list_60 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_34: "f32[768, 1024]" = torch.ops.aten.permute.default(view_302, [1, 0])
        sum_dim_int_list_61: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
        reshape_default_35: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, _shape_param_35);  sum_dim_int_list_61 = _shape_param_35 = None
        reshape_default_36: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(mm_70, _shape_param_36);  mm_70 = _shape_param_36 = None
        add_tensor: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_264, reshape_default_36);  mul_264 = reshape_default_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_35: "f32[768, 1024]" = torch.ops.aten.permute.default(view_306, [1, 0])
        sum_dim_int_list_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_306, [0], True);  view_306 = None
        reshape_default_37: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_37);  sum_dim_int_list_62 = _shape_param_37 = None
        reshape_default_38: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(mm_72, _shape_param_38);  mm_72 = _shape_param_38 = None
        add_tensor_1: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_38);  add_tensor = reshape_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_36: "f32[768, 1024]" = torch.ops.aten.permute.default(view_310, [1, 0])
        sum_dim_int_list_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        reshape_default_39: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_39);  sum_dim_int_list_63 = _shape_param_39 = None
        reshape_default_40: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(mm_74, _shape_param_40);  mm_74 = _shape_param_40 = None
        add_tensor_2: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_40);  add_tensor_1 = reshape_default_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:121 in forward, code: embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        convert_element_type_default: "f32[8, 128, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_13: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_14: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_13);  add_tensor_2 = mul_tensor_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:120 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        mul_tensor_15: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_14, primals_5);  primals_5 = None
        mul_tensor_16: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_15, 768)
        sum_dim_int_list_64: "f32[8, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:119 in forward, code: embeddings = inputs_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add_tensor_3: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:120 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        sub_tensor: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_tensor_3, getitem_1);  add_tensor_3 = getitem_1 = None
        mul_tensor_17: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_18: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_15, mul_tensor_17);  mul_tensor_15 = None
        sum_dim_int_list_65: "f32[8, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [2], True);  mul_tensor_18 = None
        mul_tensor_19: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_17, sum_dim_int_list_65);  sum_dim_int_list_65 = None
        sub_tensor_1: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_16, sum_dim_int_list_64);  mul_tensor_16 = sum_dim_int_list_64 = None
        sub_tensor_2: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_19);  sub_tensor_1 = mul_tensor_19 = None
        div_tensor: "f32[8, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_20: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_21: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_14, mul_tensor_17);  mul_tensor_17 = None
        sum_dim_int_list_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:119 in forward, code: embeddings = inputs_embeds + position_embeddings  # (bs, max_seq_length, dim)
        sum_dim_int_list_68: "f32[1, 128, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:112 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_tensor: "i64[1, 128]" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 128);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:117 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        eq_scalar: "b8[1, 128]" = torch.ops.aten.eq.Scalar(slice_tensor, -1)
        unsqueeze_default: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 128, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, sum_dim_int_list_68);  unsqueeze_default = sum_dim_int_list_68 = None
        full_default: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default, [slice_tensor], where_self, True);  full_default = slice_tensor = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        eq_scalar_1: "b8[8, 128]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_1: "b8[8, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 128, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_1, mul_tensor_20);  unsqueeze_default_1 = full_default_1 = mul_tensor_20 = None
        full_default_2: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[30522, 768]" = torch.ops.aten.index_put.default(full_default_2, [primals_1], where_self_1, True);  full_default_2 = primals_1 = where_self_1 = None
        add_tensor_4: "f32[30522, 768]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_1);  mm_1 = index_put_default_1 = None
        return (reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_1, reshape_default_2, permute_default_2, reshape_default_3, sum_dim_int_list_8, sum_dim_int_list_9, permute_default_3, reshape_default_4, permute_default_4, reshape_default_5, permute_default_5, reshape_default_6, permute_default_6, reshape_default_7, sum_dim_int_list_14, sum_dim_int_list_15, permute_default_7, reshape_default_8, permute_default_8, reshape_default_9, sum_dim_int_list_18, sum_dim_int_list_19, permute_default_9, reshape_default_10, permute_default_10, reshape_default_11, permute_default_11, reshape_default_12, permute_default_12, reshape_default_13, sum_dim_int_list_24, sum_dim_int_list_25, permute_default_13, reshape_default_14, permute_default_14, reshape_default_15, sum_dim_int_list_28, sum_dim_int_list_29, permute_default_15, reshape_default_16, permute_default_16, reshape_default_17, permute_default_17, reshape_default_18, permute_default_18, reshape_default_19, sum_dim_int_list_34, sum_dim_int_list_35, permute_default_19, reshape_default_20, permute_default_20, reshape_default_21, sum_dim_int_list_38, sum_dim_int_list_39, permute_default_21, reshape_default_22, permute_default_22, reshape_default_23, permute_default_23, reshape_default_24, permute_default_24, reshape_default_25, sum_dim_int_list_44, sum_dim_int_list_45, permute_default_25, reshape_default_26, permute_default_26, reshape_default_27, sum_dim_int_list_48, sum_dim_int_list_49, permute_default_27, reshape_default_28, permute_default_28, reshape_default_29, permute_default_29, reshape_default_30, permute_default_30, reshape_default_31, sum_dim_int_list_54, sum_dim_int_list_55, permute_default_31, reshape_default_32, permute_default_32, reshape_default_33, sum_dim_int_list_58, sum_dim_int_list_59, permute_default_33, reshape_default_34, permute_default_34, reshape_default_35, permute_default_35, reshape_default_37, permute_default_36, reshape_default_39, sum_dim_int_list_66, sum_dim_int_list_67, index_put_default, add_tensor_4)


def _default_make_inputs():
    return [
    torch.randn([1024, 30522], dtype=torch.float32, device='cuda'),
    [30522],  # _shape_param_0
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_1
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_2
    torch.randn([1024, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_3
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_4
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_5
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_6
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_7
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_8
    torch.randn([1024, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_9
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_10
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_11
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_12
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_13
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_14
    torch.randn([1024, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_15
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_16
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_17
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_18
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_19
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_20
    torch.randn([1024, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_21
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_22
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_23
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_24
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_25
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_26
    torch.randn([1024, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_27
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_28
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_29
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_30
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_31
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_32
    torch.randn([1024, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_33
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_34
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_35
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [8, 128, 768],  # _shape_param_36
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_37
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [8, 128, 768],  # _shape_param_38
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_39
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    [8, 128, 768],  # _shape_param_40
    torch.randint(0, 2, [8, 128, 768], dtype=torch.bool, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 128], dtype=torch.int64, device='cuda'),
    torch.randn([30522, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
