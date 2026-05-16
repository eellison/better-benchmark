"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_training
Pattern hash: 45ace8213b20
Shape hash: c72f7090
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_54: "f32[8, 512, 768]", mul_74: "f32[8, 512, 768]", view_79: "f32[4096, 768]", _shape_param_0, view_82: "f32[4096, 3072]", _shape_param_1, view_84: "f32[8, 512, 768]", mul_66: "f32[8, 512, 768]", getitem_60: "f32[8, 12, 512, 64]", _shape_param_2, _shape_param_3, view_85: "f32[4096, 768]", _shape_param_4, view_92: "f32[4096, 2304]", _shape_param_5, view_94: "f32[8, 512, 768]", mul_62: "f32[8, 512, 768]", view_95: "f32[4096, 768]", _shape_param_6, view_98: "f32[4096, 3072]", _shape_param_7, view_100: "f32[8, 512, 768]", mul_54: "f32[8, 512, 768]", getitem_49: "f32[8, 12, 512, 64]", _shape_param_8, _shape_param_9, view_101: "f32[4096, 768]", _shape_param_10, view_108: "f32[4096, 2304]", _shape_param_11, view_110: "f32[8, 512, 768]", mul_50: "f32[8, 512, 768]", view_111: "f32[4096, 768]", _shape_param_12, view_114: "f32[4096, 3072]", _shape_param_13, view_116: "f32[8, 512, 768]", mul_42: "f32[8, 512, 768]", getitem_38: "f32[8, 12, 512, 64]", _shape_param_14, _shape_param_15, view_117: "f32[4096, 768]", _shape_param_16, view_124: "f32[4096, 2304]", _shape_param_17, view_126: "f32[8, 512, 768]", mul_38: "f32[8, 512, 768]", view_127: "f32[4096, 768]", _shape_param_18, view_130: "f32[4096, 3072]", _shape_param_19, view_132: "f32[8, 512, 768]", mul_30: "f32[8, 512, 768]", getitem_27: "f32[8, 12, 512, 64]", _shape_param_20, _shape_param_21, view_133: "f32[4096, 768]", _shape_param_22, view_140: "f32[4096, 2304]", _shape_param_23, view_142: "f32[8, 512, 768]", mul_26: "f32[8, 512, 768]", view_143: "f32[4096, 768]", _shape_param_24, view_146: "f32[4096, 3072]", _shape_param_25, view_148: "f32[8, 512, 768]", mul_18: "f32[8, 512, 768]", getitem_16: "f32[8, 12, 512, 64]", _shape_param_26, _shape_param_27, view_149: "f32[4096, 768]", _shape_param_28, view_156: "f32[4096, 2304]", _shape_param_29, view_158: "f32[8, 512, 768]", mul_14: "f32[8, 512, 768]", view_159: "f32[4096, 768]", _shape_param_30, view_162: "f32[4096, 3072]", _shape_param_31, view_164: "f32[8, 512, 768]", mul_6: "f32[8, 512, 768]", getitem_5: "f32[8, 12, 512, 64]", _shape_param_32, _shape_param_33, view_165: "f32[4096, 768]", _shape_param_34, view_172: "f32[4096, 2304]", _shape_param_35, mm_49: "f32[4096, 768]", _shape_param_36, primals_4: "f32[768]", embedding: "f32[8, 512, 768]", embedding_1: "f32[1, 512, 768]", gt: "b8[8, 512, 768]", getitem_1: "f32[8, 512, 1]", rsqrt: "f32[8, 512, 1]", add_77: "f32[8, 512, 768]", unsqueeze: "i64[1, 512]", primals_1: "i64[8, 512]", mm_1: "f32[50257, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_tensor: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(add_54, mul_74);  mul_74 = None
        sum_dim_int_list: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_54, [0, 1]);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_2: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_79, [0], True);  view_79 = None
        reshape_default: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_0);  sum_dim_int_list_2 = _shape_param_0 = None
        sum_dim_int_list_3: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_82, [0], True);  view_82 = None
        reshape_default_1: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_84, mul_66);  mul_66 = None
        sum_dim_int_list_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_84, [0, 1]);  view_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_2: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_3: "f32[4096, 768]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[768, 4096]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0]);  reshape_default_3 = None
        sum_dim_int_list_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_85, [0], True);  view_85 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_4);  sum_dim_int_list_6 = _shape_param_4 = None
        sum_dim_int_list_7: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_92, [0], True);  view_92 = None
        reshape_default_5: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_5);  sum_dim_int_list_7 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_94, mul_62);  mul_62 = None
        sum_dim_int_list_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_94, [0, 1]);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_95, [0], True);  view_95 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_6);  sum_dim_int_list_10 = _shape_param_6 = None
        sum_dim_int_list_11: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True);  view_98 = None
        reshape_default_7: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_7);  sum_dim_int_list_11 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_3: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_100, mul_54);  mul_54 = None
        sum_dim_int_list_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_100, [0, 1]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_2: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_8: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_8);  permute_default_2 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_9: "f32[4096, 768]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_9);  reshape_default_8 = _shape_param_9 = None
        permute_default_3: "f32[768, 4096]" = torch.ops.aten.permute.default(reshape_default_9, [1, 0]);  reshape_default_9 = None
        sum_dim_int_list_14: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_101, [0], True);  view_101 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_10);  sum_dim_int_list_14 = _shape_param_10 = None
        sum_dim_int_list_15: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_108, [0], True);  view_108 = None
        reshape_default_11: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_11);  sum_dim_int_list_15 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_4: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_110, mul_50);  mul_50 = None
        sum_dim_int_list_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_110, [0, 1]);  view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_111, [0], True);  view_111 = None
        reshape_default_12: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_12);  sum_dim_int_list_18 = _shape_param_12 = None
        sum_dim_int_list_19: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True);  view_114 = None
        reshape_default_13: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_13);  sum_dim_int_list_19 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_5: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_116, mul_42);  mul_42 = None
        sum_dim_int_list_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_116, [0, 1]);  view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_4: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_14: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_14);  permute_default_4 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_15: "f32[4096, 768]" = torch.ops.aten.reshape.default(reshape_default_14, _shape_param_15);  reshape_default_14 = _shape_param_15 = None
        permute_default_5: "f32[768, 4096]" = torch.ops.aten.permute.default(reshape_default_15, [1, 0]);  reshape_default_15 = None
        sum_dim_int_list_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True);  view_117 = None
        reshape_default_16: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_16);  sum_dim_int_list_22 = _shape_param_16 = None
        sum_dim_int_list_23: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_124, [0], True);  view_124 = None
        reshape_default_17: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_17);  sum_dim_int_list_23 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_6: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_126, mul_38);  mul_38 = None
        sum_dim_int_list_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_126, [0, 1]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_127, [0], True);  view_127 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_18);  sum_dim_int_list_26 = _shape_param_18 = None
        sum_dim_int_list_27: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_130, [0], True);  view_130 = None
        reshape_default_19: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_19);  sum_dim_int_list_27 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_7: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_132, mul_30);  mul_30 = None
        sum_dim_int_list_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_132, [0, 1]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_6: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_20: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(permute_default_6, _shape_param_20);  permute_default_6 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_21: "f32[4096, 768]" = torch.ops.aten.reshape.default(reshape_default_20, _shape_param_21);  reshape_default_20 = _shape_param_21 = None
        permute_default_7: "f32[768, 4096]" = torch.ops.aten.permute.default(reshape_default_21, [1, 0]);  reshape_default_21 = None
        sum_dim_int_list_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_133, [0], True);  view_133 = None
        reshape_default_22: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_22);  sum_dim_int_list_30 = _shape_param_22 = None
        sum_dim_int_list_31: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_140, [0], True);  view_140 = None
        reshape_default_23: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_23);  sum_dim_int_list_31 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_8: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_142, mul_26);  mul_26 = None
        sum_dim_int_list_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_142, [0, 1]);  view_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_143, [0], True);  view_143 = None
        reshape_default_24: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_24);  sum_dim_int_list_34 = _shape_param_24 = None
        sum_dim_int_list_35: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_146, [0], True);  view_146 = None
        reshape_default_25: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_25);  sum_dim_int_list_35 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_9: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_148, mul_18);  mul_18 = None
        sum_dim_int_list_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_148, [0, 1]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_8: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_26: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(permute_default_8, _shape_param_26);  permute_default_8 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_27: "f32[4096, 768]" = torch.ops.aten.reshape.default(reshape_default_26, _shape_param_27);  reshape_default_26 = _shape_param_27 = None
        permute_default_9: "f32[768, 4096]" = torch.ops.aten.permute.default(reshape_default_27, [1, 0]);  reshape_default_27 = None
        sum_dim_int_list_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_149, [0], True);  view_149 = None
        reshape_default_28: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_28);  sum_dim_int_list_38 = _shape_param_28 = None
        sum_dim_int_list_39: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
        reshape_default_29: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_29);  sum_dim_int_list_39 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_10: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_158, mul_14);  mul_14 = None
        sum_dim_int_list_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_158, [0, 1]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_159, [0], True);  view_159 = None
        reshape_default_30: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_30);  sum_dim_int_list_42 = _shape_param_30 = None
        sum_dim_int_list_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_162, [0], True);  view_162 = None
        reshape_default_31: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_31);  sum_dim_int_list_43 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_11: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(view_164, mul_6);  mul_6 = None
        sum_dim_int_list_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_164, [0, 1]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_10: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_32: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(permute_default_10, _shape_param_32);  permute_default_10 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_33: "f32[4096, 768]" = torch.ops.aten.reshape.default(reshape_default_32, _shape_param_33);  reshape_default_32 = _shape_param_33 = None
        permute_default_11: "f32[768, 4096]" = torch.ops.aten.permute.default(reshape_default_33, [1, 0]);  reshape_default_33 = None
        sum_dim_int_list_46: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
        reshape_default_34: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_34);  sum_dim_int_list_46 = _shape_param_34 = None
        sum_dim_int_list_47: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_172, [0], True);  view_172 = None
        reshape_default_35: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_35);  sum_dim_int_list_47 = _shape_param_35 = None
        reshape_default_36: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_49, _shape_param_36);  mm_49 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_12: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default_36, primals_4);  primals_4 = None
        mul_tensor_13: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 768)
        sum_dim_int_list_48: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_tensor: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        mul_tensor_14: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_tensor);  add_tensor = None
        mul_tensor_15: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_14, 1.1111111111111112);  mul_tensor_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_tensor: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_15, getitem_1);  mul_tensor_15 = getitem_1 = None
        mul_tensor_16: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_17: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_tensor_16);  mul_tensor_12 = None
        sum_dim_int_list_49: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [2], True);  mul_tensor_17 = None
        mul_tensor_18: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_16, sum_dim_int_list_49);  sum_dim_int_list_49 = None
        sub_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_13, sum_dim_int_list_48);  mul_tensor_13 = sum_dim_int_list_48 = None
        sub_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_18);  sub_tensor_1 = mul_tensor_18 = None
        div_tensor: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_19: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_20: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default_36, mul_tensor_16);  mul_tensor_16 = None
        sum_dim_int_list_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_36, [0, 1]);  reshape_default_36 = None
        add_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(add_77, mul_tensor_19);  add_77 = mul_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        convert_element_type_default: "f32[8, 512, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_21: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_22: "f32[8, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_21);  add_tensor_1 = mul_tensor_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        sum_dim_int_list_52: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(unsqueeze, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        where_self: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, sum_dim_int_list_52);  unsqueeze_default = sum_dim_int_list_52 = None
        full_default_1: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_1, [unsqueeze], where_self, True);  full_default_1 = unsqueeze = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_scalar_1: "b8[8, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default_1: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_22);  unsqueeze_default_1 = full_default = mul_tensor_22 = None
        full_default_2: "f32[50257, 768]" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[50257, 768]" = torch.ops.aten.index_put.default(full_default_2, [primals_1], where_self_1, True);  full_default_2 = primals_1 = where_self_1 = None
        add_tensor_2: "f32[50257, 768]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_1);  mm_1 = index_put_default_1 = None
        return (sum_dim_int_list, sum_dim_int_list_1, reshape_default, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_1, reshape_default_4, reshape_default_5, sum_dim_int_list_8, sum_dim_int_list_9, reshape_default_6, reshape_default_7, sum_dim_int_list_12, sum_dim_int_list_13, permute_default_3, reshape_default_10, reshape_default_11, sum_dim_int_list_16, sum_dim_int_list_17, reshape_default_12, reshape_default_13, sum_dim_int_list_20, sum_dim_int_list_21, permute_default_5, reshape_default_16, reshape_default_17, sum_dim_int_list_24, sum_dim_int_list_25, reshape_default_18, reshape_default_19, sum_dim_int_list_28, sum_dim_int_list_29, permute_default_7, reshape_default_22, reshape_default_23, sum_dim_int_list_32, sum_dim_int_list_33, reshape_default_24, reshape_default_25, sum_dim_int_list_36, sum_dim_int_list_37, permute_default_9, reshape_default_28, reshape_default_29, sum_dim_int_list_40, sum_dim_int_list_41, reshape_default_30, reshape_default_31, sum_dim_int_list_44, sum_dim_int_list_45, permute_default_11, reshape_default_34, reshape_default_35, sum_dim_int_list_50, sum_dim_int_list_51, index_put_default, add_tensor_2)


def _default_make_inputs():
    return [
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_0
    torch.randn([4096, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_1
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([8, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_60
    [8, 512, -1],  # _shape_param_2
    [-1, 768],  # _shape_param_3
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_4
    torch.randn([4096, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_5
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_6
    torch.randn([4096, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_7
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([8, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_49
    [8, 512, -1],  # _shape_param_8
    [-1, 768],  # _shape_param_9
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_10
    torch.randn([4096, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_11
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_12
    torch.randn([4096, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_13
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([8, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_38
    [8, 512, -1],  # _shape_param_14
    [-1, 768],  # _shape_param_15
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_16
    torch.randn([4096, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_17
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_18
    torch.randn([4096, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_19
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([8, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_27
    [8, 512, -1],  # _shape_param_20
    [-1, 768],  # _shape_param_21
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_22
    torch.randn([4096, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_23
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_24
    torch.randn([4096, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_25
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([8, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_16
    [8, 512, -1],  # _shape_param_26
    [-1, 768],  # _shape_param_27
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_28
    torch.randn([4096, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_29
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_30
    torch.randn([4096, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_31
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([8, 12, 512, 64], [393216, 64, 768, 1]),  # getitem_5
    [8, 512, -1],  # _shape_param_32
    [-1, 768],  # _shape_param_33
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_34
    torch.randn([4096, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_35
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    [8, 512, 768],  # _shape_param_36
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 512, 768], dtype=torch.bool, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randn([50257, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
