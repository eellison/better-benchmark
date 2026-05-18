"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_train
Pattern hash: 9c6a87613ead
Shape hash: 4508b24d
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
    def forward(self, mm_default: "f32[4, 768]", view_152: "f32[8, 1024, 768]", mul_146: "f32[8, 1024, 768]", view_154: "f32[8192, 768]", _shape_param_0, view_157: "f32[8192, 3072]", _shape_param_1, view_159: "f32[8, 1024, 768]", mul_138: "f32[8, 1024, 768]", getitem_126: "f32[8, 12, 1024, 64]", _shape_param_2, _shape_param_3, view_160: "f32[8192, 768]", _shape_param_4, view_167: "f32[8192, 2304]", _shape_param_5, view_169: "f32[8, 1024, 768]", mul_134: "f32[8, 1024, 768]", view_170: "f32[8192, 768]", _shape_param_6, view_173: "f32[8192, 3072]", _shape_param_7, view_175: "f32[8, 1024, 768]", mul_126: "f32[8, 1024, 768]", getitem_115: "f32[8, 12, 1024, 64]", _shape_param_8, _shape_param_9, view_176: "f32[8192, 768]", _shape_param_10, view_183: "f32[8192, 2304]", _shape_param_11, view_185: "f32[8, 1024, 768]", mul_122: "f32[8, 1024, 768]", view_186: "f32[8192, 768]", _shape_param_12, view_189: "f32[8192, 3072]", _shape_param_13, view_191: "f32[8, 1024, 768]", mul_114: "f32[8, 1024, 768]", getitem_104: "f32[8, 12, 1024, 64]", _shape_param_14, _shape_param_15, view_192: "f32[8192, 768]", _shape_param_16, view_199: "f32[8192, 2304]", _shape_param_17, view_201: "f32[8, 1024, 768]", mul_110: "f32[8, 1024, 768]", view_202: "f32[8192, 768]", _shape_param_18, view_205: "f32[8192, 3072]", _shape_param_19, view_207: "f32[8, 1024, 768]", mul_102: "f32[8, 1024, 768]", getitem_93: "f32[8, 12, 1024, 64]", _shape_param_20, _shape_param_21, view_208: "f32[8192, 768]", _shape_param_22, view_215: "f32[8192, 2304]", _shape_param_23, view_217: "f32[8, 1024, 768]", mul_98: "f32[8, 1024, 768]", view_218: "f32[8192, 768]", _shape_param_24, view_221: "f32[8192, 3072]", _shape_param_25, view_223: "f32[8, 1024, 768]", mul_90: "f32[8, 1024, 768]", getitem_82: "f32[8, 12, 1024, 64]", _shape_param_26, _shape_param_27, view_224: "f32[8192, 768]", _shape_param_28, view_231: "f32[8192, 2304]", _shape_param_29, view_233: "f32[8, 1024, 768]", mul_86: "f32[8, 1024, 768]", view_234: "f32[8192, 768]", _shape_param_30, view_237: "f32[8192, 3072]", _shape_param_31, view_239: "f32[8, 1024, 768]", mul_78: "f32[8, 1024, 768]", getitem_71: "f32[8, 12, 1024, 64]", _shape_param_32, _shape_param_33, view_240: "f32[8192, 768]", _shape_param_34, view_247: "f32[8192, 2304]", _shape_param_35, view_249: "f32[8, 1024, 768]", mul_74: "f32[8, 1024, 768]", view_250: "f32[8192, 768]", _shape_param_36, view_253: "f32[8192, 3072]", _shape_param_37, view_255: "f32[8, 1024, 768]", mul_66: "f32[8, 1024, 768]", getitem_60: "f32[8, 12, 1024, 64]", _shape_param_38, _shape_param_39, view_256: "f32[8192, 768]", _shape_param_40, view_263: "f32[8192, 2304]", _shape_param_41, view_265: "f32[8, 1024, 768]", mul_62: "f32[8, 1024, 768]", view_266: "f32[8192, 768]", _shape_param_42, view_269: "f32[8192, 3072]", _shape_param_43, view_271: "f32[8, 1024, 768]", mul_54: "f32[8, 1024, 768]", getitem_49: "f32[8, 12, 1024, 64]", _shape_param_44, _shape_param_45, view_272: "f32[8192, 768]", _shape_param_46, view_279: "f32[8192, 2304]", _shape_param_47, view_281: "f32[8, 1024, 768]", mul_50: "f32[8, 1024, 768]", view_282: "f32[8192, 768]", _shape_param_48, view_285: "f32[8192, 3072]", _shape_param_49, view_287: "f32[8, 1024, 768]", mul_42: "f32[8, 1024, 768]", getitem_38: "f32[8, 12, 1024, 64]", _shape_param_50, _shape_param_51, view_288: "f32[8192, 768]", _shape_param_52, view_295: "f32[8192, 2304]", _shape_param_53, view_297: "f32[8, 1024, 768]", mul_38: "f32[8, 1024, 768]", view_298: "f32[8192, 768]", _shape_param_54, view_301: "f32[8192, 3072]", _shape_param_55, view_303: "f32[8, 1024, 768]", mul_30: "f32[8, 1024, 768]", getitem_27: "f32[8, 12, 1024, 64]", _shape_param_56, _shape_param_57, view_304: "f32[8192, 768]", _shape_param_58, view_311: "f32[8192, 2304]", _shape_param_59, view_313: "f32[8, 1024, 768]", mul_26: "f32[8, 1024, 768]", view_314: "f32[8192, 768]", _shape_param_60, view_317: "f32[8192, 3072]", _shape_param_61, view_319: "f32[8, 1024, 768]", mul_18: "f32[8, 1024, 768]", getitem_16: "f32[8, 12, 1024, 64]", _shape_param_62, _shape_param_63, view_320: "f32[8192, 768]", _shape_param_64, view_327: "f32[8192, 2304]", _shape_param_65, view_329: "f32[8, 1024, 768]", mul_14: "f32[8, 1024, 768]", view_330: "f32[8192, 768]", _shape_param_66, view_333: "f32[8192, 3072]", _shape_param_67, view_335: "f32[8, 1024, 768]", mul_6: "f32[8, 1024, 768]", getitem_5: "f32[8, 12, 1024, 64]", _shape_param_68, _shape_param_69, view_336: "f32[8192, 768]", _shape_param_70, view_343: "f32[8192, 2304]", _shape_param_71, mm_97: "f32[8192, 768]", _shape_param_72, primals_4: "f32[768]", embedding: "f32[8, 1024, 768]", embedding_1: "f32[1, 1024, 768]", gt: "b8[8, 1024, 768]", getitem_1: "f32[8, 1024, 1]", rsqrt: "f32[8, 1024, 1]", add_149: "f32[8, 1024, 768]", unsqueeze: "i64[1, 1024]", full_default_2: "f32[]", primals_1: "i64[8, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:920 in forward, code: logits = self.score(hidden_states)
        slice_tensor: "f32[2, 768]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -2);  mm_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_152, mul_146);  mul_146 = None
        sum_dim_int_list: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_152, [0, 1]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_2: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True);  view_154 = None
        reshape_default: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_0);  sum_dim_int_list_2 = _shape_param_0 = None
        sum_dim_int_list_3: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_157, [0], True);  view_157 = None
        reshape_default_1: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_159, mul_138);  mul_138 = None
        sum_dim_int_list_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_159, [0, 1]);  view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_2: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_3: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0]);  reshape_default_3 = None
        sum_dim_int_list_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_160, [0], True);  view_160 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_4);  sum_dim_int_list_6 = _shape_param_4 = None
        sum_dim_int_list_7: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_167, [0], True);  view_167 = None
        reshape_default_5: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_5);  sum_dim_int_list_7 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_169, mul_134);  mul_134 = None
        sum_dim_int_list_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_169, [0, 1]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_6);  sum_dim_int_list_10 = _shape_param_6 = None
        sum_dim_int_list_11: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
        reshape_default_7: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_7);  sum_dim_int_list_11 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_175, mul_126);  mul_126 = None
        sum_dim_int_list_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_175, [0, 1]);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_2: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_8: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_8);  permute_default_2 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_9: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_8, _shape_param_9);  reshape_default_8 = _shape_param_9 = None
        permute_default_3: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_9, [1, 0]);  reshape_default_9 = None
        sum_dim_int_list_14: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_176, [0], True);  view_176 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_10);  sum_dim_int_list_14 = _shape_param_10 = None
        sum_dim_int_list_15: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
        reshape_default_11: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_11);  sum_dim_int_list_15 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_185, mul_122);  mul_122 = None
        sum_dim_int_list_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_185, [0, 1]);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        reshape_default_12: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_12);  sum_dim_int_list_18 = _shape_param_12 = None
        sum_dim_int_list_19: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
        reshape_default_13: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_13);  sum_dim_int_list_19 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_191, mul_114);  mul_114 = None
        sum_dim_int_list_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_191, [0, 1]);  view_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_4: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_14: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_14);  permute_default_4 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_15: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_14, _shape_param_15);  reshape_default_14 = _shape_param_15 = None
        permute_default_5: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_15, [1, 0]);  reshape_default_15 = None
        sum_dim_int_list_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True);  view_192 = None
        reshape_default_16: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_16);  sum_dim_int_list_22 = _shape_param_16 = None
        sum_dim_int_list_23: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_199, [0], True);  view_199 = None
        reshape_default_17: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_17);  sum_dim_int_list_23 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_201, mul_110);  mul_110 = None
        sum_dim_int_list_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_201, [0, 1]);  view_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_202, [0], True);  view_202 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_18);  sum_dim_int_list_26 = _shape_param_18 = None
        sum_dim_int_list_27: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_205, [0], True);  view_205 = None
        reshape_default_19: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_19);  sum_dim_int_list_27 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_207, mul_102);  mul_102 = None
        sum_dim_int_list_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_207, [0, 1]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_6: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_20: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_6, _shape_param_20);  permute_default_6 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_21: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_20, _shape_param_21);  reshape_default_20 = _shape_param_21 = None
        permute_default_7: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_21, [1, 0]);  reshape_default_21 = None
        sum_dim_int_list_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_208, [0], True);  view_208 = None
        reshape_default_22: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_22);  sum_dim_int_list_30 = _shape_param_22 = None
        sum_dim_int_list_31: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
        reshape_default_23: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_23);  sum_dim_int_list_31 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_217, mul_98);  mul_98 = None
        sum_dim_int_list_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
        reshape_default_24: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_24);  sum_dim_int_list_34 = _shape_param_24 = None
        sum_dim_int_list_35: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True);  view_221 = None
        reshape_default_25: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_25);  sum_dim_int_list_35 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_223, mul_90);  mul_90 = None
        sum_dim_int_list_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_223, [0, 1]);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_8: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_26: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_8, _shape_param_26);  permute_default_8 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_27: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_26, _shape_param_27);  reshape_default_26 = _shape_param_27 = None
        permute_default_9: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_27, [1, 0]);  reshape_default_27 = None
        sum_dim_int_list_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
        reshape_default_28: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_28);  sum_dim_int_list_38 = _shape_param_28 = None
        sum_dim_int_list_39: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
        reshape_default_29: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_29);  sum_dim_int_list_39 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_233, mul_86);  mul_86 = None
        sum_dim_int_list_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_233, [0, 1]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
        reshape_default_30: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_30);  sum_dim_int_list_42 = _shape_param_30 = None
        sum_dim_int_list_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_237, [0], True);  view_237 = None
        reshape_default_31: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_31);  sum_dim_int_list_43 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_11: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_239, mul_78);  mul_78 = None
        sum_dim_int_list_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_239, [0, 1]);  view_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_10: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_32: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_10, _shape_param_32);  permute_default_10 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_33: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_32, _shape_param_33);  reshape_default_32 = _shape_param_33 = None
        permute_default_11: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_33, [1, 0]);  reshape_default_33 = None
        sum_dim_int_list_46: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True);  view_240 = None
        reshape_default_34: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_34);  sum_dim_int_list_46 = _shape_param_34 = None
        sum_dim_int_list_47: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_247, [0], True);  view_247 = None
        reshape_default_35: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_35);  sum_dim_int_list_47 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_12: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_249, mul_74);  mul_74 = None
        sum_dim_int_list_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_249, [0, 1]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True);  view_250 = None
        reshape_default_36: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_36);  sum_dim_int_list_50 = _shape_param_36 = None
        sum_dim_int_list_51: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_253, [0], True);  view_253 = None
        reshape_default_37: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_37);  sum_dim_int_list_51 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_13: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_255, mul_66);  mul_66 = None
        sum_dim_int_list_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_255, [0, 1]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_12: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_38: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_12, _shape_param_38);  permute_default_12 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_39: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_38, _shape_param_39);  reshape_default_38 = _shape_param_39 = None
        permute_default_13: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_39, [1, 0]);  reshape_default_39 = None
        sum_dim_int_list_54: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_256, [0], True);  view_256 = None
        reshape_default_40: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, _shape_param_40);  sum_dim_int_list_54 = _shape_param_40 = None
        sum_dim_int_list_55: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
        reshape_default_41: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_41);  sum_dim_int_list_55 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_14: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_265, mul_62);  mul_62 = None
        sum_dim_int_list_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_265, [0, 1]);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
        reshape_default_42: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, _shape_param_42);  sum_dim_int_list_58 = _shape_param_42 = None
        sum_dim_int_list_59: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
        reshape_default_43: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_43);  sum_dim_int_list_59 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_15: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_271, mul_54);  mul_54 = None
        sum_dim_int_list_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_14: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_44: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_14, _shape_param_44);  permute_default_14 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_45: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_44, _shape_param_45);  reshape_default_44 = _shape_param_45 = None
        permute_default_15: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_45, [1, 0]);  reshape_default_45 = None
        sum_dim_int_list_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
        reshape_default_46: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_46);  sum_dim_int_list_62 = _shape_param_46 = None
        sum_dim_int_list_63: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        reshape_default_47: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_47);  sum_dim_int_list_63 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_16: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_281, mul_50);  mul_50 = None
        sum_dim_int_list_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_281, [0, 1]);  view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        reshape_default_48: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_48);  sum_dim_int_list_66 = _shape_param_48 = None
        sum_dim_int_list_67: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        reshape_default_49: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_49);  sum_dim_int_list_67 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_287, mul_42);  mul_42 = None
        sum_dim_int_list_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_287, [0, 1]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_16: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_50: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_16, _shape_param_50);  permute_default_16 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_51: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_50, _shape_param_51);  reshape_default_50 = _shape_param_51 = None
        permute_default_17: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_51, [1, 0]);  reshape_default_51 = None
        sum_dim_int_list_70: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
        reshape_default_52: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_52);  sum_dim_int_list_70 = _shape_param_52 = None
        sum_dim_int_list_71: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
        reshape_default_53: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_53);  sum_dim_int_list_71 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_18: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_297, mul_38);  mul_38 = None
        sum_dim_int_list_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_297, [0, 1]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
        reshape_default_54: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_74, _shape_param_54);  sum_dim_int_list_74 = _shape_param_54 = None
        sum_dim_int_list_75: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True);  view_301 = None
        reshape_default_55: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_55);  sum_dim_int_list_75 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_19: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_303, mul_30);  mul_30 = None
        sum_dim_int_list_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_303, [0, 1]);  view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_18: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_56: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_18, _shape_param_56);  permute_default_18 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_57: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_56, _shape_param_57);  reshape_default_56 = _shape_param_57 = None
        permute_default_19: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_57, [1, 0]);  reshape_default_57 = None
        sum_dim_int_list_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
        reshape_default_58: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_78, _shape_param_58);  sum_dim_int_list_78 = _shape_param_58 = None
        sum_dim_int_list_79: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        reshape_default_59: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_59);  sum_dim_int_list_79 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_20: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_313, mul_26);  mul_26 = None
        sum_dim_int_list_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_313, [0, 1]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
        reshape_default_60: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_60);  sum_dim_int_list_82 = _shape_param_60 = None
        sum_dim_int_list_83: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
        reshape_default_61: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_61);  sum_dim_int_list_83 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_21: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_319, mul_18);  mul_18 = None
        sum_dim_int_list_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_319, [0, 1]);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_20: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_62: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_20, _shape_param_62);  permute_default_20 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_63: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_62, _shape_param_63);  reshape_default_62 = _shape_param_63 = None
        permute_default_21: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_63, [1, 0]);  reshape_default_63 = None
        sum_dim_int_list_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True);  view_320 = None
        reshape_default_64: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_64);  sum_dim_int_list_86 = _shape_param_64 = None
        sum_dim_int_list_87: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
        reshape_default_65: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_65);  sum_dim_int_list_87 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_22: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_329, mul_14);  mul_14 = None
        sum_dim_int_list_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_329, [0, 1]);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
        reshape_default_66: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_66);  sum_dim_int_list_90 = _shape_param_66 = None
        sum_dim_int_list_91: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
        reshape_default_67: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_67);  sum_dim_int_list_91 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_tensor_23: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_335, mul_6);  mul_6 = None
        sum_dim_int_list_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_335, [0, 1]);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_22: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        reshape_default_68: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_22, _shape_param_68);  permute_default_22 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        reshape_default_69: "f32[8192, 768]" = torch.ops.aten.reshape.default(reshape_default_68, _shape_param_69);  reshape_default_68 = _shape_param_69 = None
        permute_default_23: "f32[768, 8192]" = torch.ops.aten.permute.default(reshape_default_69, [1, 0]);  reshape_default_69 = None
        sum_dim_int_list_94: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_336, [0], True);  view_336 = None
        reshape_default_70: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_94, _shape_param_70);  sum_dim_int_list_94 = _shape_param_70 = None
        sum_dim_int_list_95: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_343, [0], True);  view_343 = None
        reshape_default_71: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, _shape_param_71);  sum_dim_int_list_95 = _shape_param_71 = None
        reshape_default_72: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_97, _shape_param_72);  mm_97 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_24: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(reshape_default_72, primals_4);  primals_4 = None
        mul_tensor_25: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_24, 768)
        sum_dim_int_list_96: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_tensor: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        mul_tensor_26: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, add_tensor);  add_tensor = None
        mul_tensor_27: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_26, 1.1111111111111112);  mul_tensor_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_tensor: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_27, getitem_1);  mul_tensor_27 = getitem_1 = None
        mul_tensor_28: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_29: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_24, mul_tensor_28);  mul_tensor_24 = None
        sum_dim_int_list_97: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [2], True);  mul_tensor_29 = None
        mul_tensor_30: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_28, sum_dim_int_list_97);  sum_dim_int_list_97 = None
        sub_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_25, sum_dim_int_list_96);  mul_tensor_25 = sum_dim_int_list_96 = None
        sub_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_30);  sub_tensor_1 = mul_tensor_30 = None
        div_tensor: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_31: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_32: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(reshape_default_72, mul_tensor_28);  mul_tensor_28 = None
        sum_dim_int_list_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_72, [0, 1]);  reshape_default_72 = None
        add_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_149, mul_tensor_31);  add_149 = mul_tensor_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        convert_element_type_default: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_33: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_34: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_33);  add_tensor_1 = mul_tensor_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        sum_dim_int_list_100: "f32[1, 1024, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        eq_scalar: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(unsqueeze, -1)
        unsqueeze_default: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_2, sum_dim_int_list_100);  unsqueeze_default = sum_dim_int_list_100 = None
        full_default: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default, [unsqueeze], where_self, True);  full_default = unsqueeze = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_scalar_1: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default_1: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_2, mul_tensor_34);  unsqueeze_default_1 = full_default_2 = mul_tensor_34 = None
        full_default_3: "f32[50257, 768]" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[50257, 768]" = torch.ops.aten.index_put.default(full_default_3, [primals_1], where_self_1, True);  full_default_3 = primals_1 = where_self_1 = None
        return (slice_tensor, sum_dim_int_list, sum_dim_int_list_1, reshape_default, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_1, reshape_default_4, reshape_default_5, sum_dim_int_list_8, sum_dim_int_list_9, reshape_default_6, reshape_default_7, sum_dim_int_list_12, sum_dim_int_list_13, permute_default_3, reshape_default_10, reshape_default_11, sum_dim_int_list_16, sum_dim_int_list_17, reshape_default_12, reshape_default_13, sum_dim_int_list_20, sum_dim_int_list_21, permute_default_5, reshape_default_16, reshape_default_17, sum_dim_int_list_24, sum_dim_int_list_25, reshape_default_18, reshape_default_19, sum_dim_int_list_28, sum_dim_int_list_29, permute_default_7, reshape_default_22, reshape_default_23, sum_dim_int_list_32, sum_dim_int_list_33, reshape_default_24, reshape_default_25, sum_dim_int_list_36, sum_dim_int_list_37, permute_default_9, reshape_default_28, reshape_default_29, sum_dim_int_list_40, sum_dim_int_list_41, reshape_default_30, reshape_default_31, sum_dim_int_list_44, sum_dim_int_list_45, permute_default_11, reshape_default_34, reshape_default_35, sum_dim_int_list_48, sum_dim_int_list_49, reshape_default_36, reshape_default_37, sum_dim_int_list_52, sum_dim_int_list_53, permute_default_13, reshape_default_40, reshape_default_41, sum_dim_int_list_56, sum_dim_int_list_57, reshape_default_42, reshape_default_43, sum_dim_int_list_60, sum_dim_int_list_61, permute_default_15, reshape_default_46, reshape_default_47, sum_dim_int_list_64, sum_dim_int_list_65, reshape_default_48, reshape_default_49, sum_dim_int_list_68, sum_dim_int_list_69, permute_default_17, reshape_default_52, reshape_default_53, sum_dim_int_list_72, sum_dim_int_list_73, reshape_default_54, reshape_default_55, sum_dim_int_list_76, sum_dim_int_list_77, permute_default_19, reshape_default_58, reshape_default_59, sum_dim_int_list_80, sum_dim_int_list_81, reshape_default_60, reshape_default_61, sum_dim_int_list_84, sum_dim_int_list_85, permute_default_21, reshape_default_64, reshape_default_65, sum_dim_int_list_88, sum_dim_int_list_89, reshape_default_66, reshape_default_67, sum_dim_int_list_92, sum_dim_int_list_93, permute_default_23, reshape_default_70, reshape_default_71, sum_dim_int_list_98, sum_dim_int_list_99, index_put_default, index_put_default_1)


def _default_make_inputs():
    return [
    torch.randn([4, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_0
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_1
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_126
    [8, 1024, -1],  # _shape_param_2
    [-1, 768],  # _shape_param_3
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_4
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_5
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_6
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_7
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_115
    [8, 1024, -1],  # _shape_param_8
    [-1, 768],  # _shape_param_9
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_10
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_11
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_12
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_13
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_104
    [8, 1024, -1],  # _shape_param_14
    [-1, 768],  # _shape_param_15
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_16
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_17
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_18
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_19
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_93
    [8, 1024, -1],  # _shape_param_20
    [-1, 768],  # _shape_param_21
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_22
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_23
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_24
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_25
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_82
    [8, 1024, -1],  # _shape_param_26
    [-1, 768],  # _shape_param_27
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_28
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_29
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_30
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_31
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_71
    [8, 1024, -1],  # _shape_param_32
    [-1, 768],  # _shape_param_33
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_34
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_35
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_36
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_37
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_60
    [8, 1024, -1],  # _shape_param_38
    [-1, 768],  # _shape_param_39
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_40
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_41
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_42
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_43
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_49
    [8, 1024, -1],  # _shape_param_44
    [-1, 768],  # _shape_param_45
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_46
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_47
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_48
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_49
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_38
    [8, 1024, -1],  # _shape_param_50
    [-1, 768],  # _shape_param_51
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_52
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_53
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_54
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_55
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_27
    [8, 1024, -1],  # _shape_param_56
    [-1, 768],  # _shape_param_57
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_58
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_59
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_60
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_61
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_16
    [8, 1024, -1],  # _shape_param_62
    [-1, 768],  # _shape_param_63
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_64
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_65
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_66
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_67
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([8, 12, 1024, 64], [786432, 64, 768, 1]),  # getitem_5
    [8, 1024, -1],  # _shape_param_68
    [-1, 768],  # _shape_param_69
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_70
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_71
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [8, 1024, 768],  # _shape_param_72
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 768], dtype=torch.bool, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 1024, [1, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 50257, [8, 1024], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
