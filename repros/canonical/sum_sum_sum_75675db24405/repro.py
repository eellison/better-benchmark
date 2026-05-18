"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_training
Pattern hash: 75675db24405
Shape hash: 086420a1
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
    def forward(self, tangents_1: "f32[32, 1000]", _shape_param_0, mm: "f32[32, 768]", mul_108: "f32[32, 768]", addmm_47: "f32[6304, 768]", _shape_param_1, slice_scatter: "f32[32, 197, 768]", _shape_param_2, view_147: "f32[6304, 768]", _shape_param_3, view_150: "f32[6304, 3072]", _shape_param_4, view_152: "f32[32, 197, 768]", mul_102: "f32[32, 197, 768]", addmm_45: "f32[6304, 768]", _shape_param_5, add_88: "f32[32, 197, 768]", _shape_param_6, view_154: "f32[6304, 768]", getitem_126: "f32[32, 12, 197, 64]", _shape_param_7, _shape_param_8, _shape_param_9, getitem_137: "f32[32, 12, 197, 197]", _shape_param_10, primals_211: "i64[197, 197]", view_161: "f32[6304, 2304]", _shape_param_11, view_163: "f32[32, 197, 768]", mul_99: "f32[32, 197, 768]", addmm_43: "f32[6304, 768]", _shape_param_12, add_89: "f32[32, 197, 768]", _shape_param_13, view_165: "f32[6304, 768]", _shape_param_14, view_168: "f32[6304, 3072]", _shape_param_15, view_170: "f32[32, 197, 768]", mul_93: "f32[32, 197, 768]", addmm_41: "f32[6304, 768]", _shape_param_16, add_92: "f32[32, 197, 768]", _shape_param_17, view_172: "f32[6304, 768]", getitem_115: "f32[32, 12, 197, 64]", _shape_param_18, _shape_param_19, _shape_param_20, getitem_141: "f32[32, 12, 197, 197]", _shape_param_21, primals_193: "i64[197, 197]", view_179: "f32[6304, 2304]", _shape_param_22, view_181: "f32[32, 197, 768]", mul_90: "f32[32, 197, 768]", addmm_39: "f32[6304, 768]", _shape_param_23, add_93: "f32[32, 197, 768]", _shape_param_24, view_183: "f32[6304, 768]", _shape_param_25, view_186: "f32[6304, 3072]", _shape_param_26, view_188: "f32[32, 197, 768]", mul_84: "f32[32, 197, 768]", addmm_37: "f32[6304, 768]", _shape_param_27, add_96: "f32[32, 197, 768]", _shape_param_28, view_190: "f32[6304, 768]", getitem_104: "f32[32, 12, 197, 64]", _shape_param_29, _shape_param_30, _shape_param_31, getitem_145: "f32[32, 12, 197, 197]", _shape_param_32, primals_175: "i64[197, 197]", view_197: "f32[6304, 2304]", _shape_param_33, view_199: "f32[32, 197, 768]", mul_81: "f32[32, 197, 768]", addmm_35: "f32[6304, 768]", _shape_param_34, add_97: "f32[32, 197, 768]", _shape_param_35, view_201: "f32[6304, 768]", _shape_param_36, view_204: "f32[6304, 3072]", _shape_param_37, view_206: "f32[32, 197, 768]", mul_75: "f32[32, 197, 768]", addmm_33: "f32[6304, 768]", _shape_param_38, add_100: "f32[32, 197, 768]", _shape_param_39, view_208: "f32[6304, 768]", getitem_93: "f32[32, 12, 197, 64]", _shape_param_40, _shape_param_41, _shape_param_42, getitem_149: "f32[32, 12, 197, 197]", _shape_param_43, primals_157: "i64[197, 197]", view_215: "f32[6304, 2304]", _shape_param_44, view_217: "f32[32, 197, 768]", mul_72: "f32[32, 197, 768]", addmm_31: "f32[6304, 768]", _shape_param_45, add_101: "f32[32, 197, 768]", _shape_param_46, view_219: "f32[6304, 768]", _shape_param_47, view_222: "f32[6304, 3072]", _shape_param_48, view_224: "f32[32, 197, 768]", mul_66: "f32[32, 197, 768]", addmm_29: "f32[6304, 768]", _shape_param_49, add_104: "f32[32, 197, 768]", _shape_param_50, view_226: "f32[6304, 768]", getitem_82: "f32[32, 12, 197, 64]", _shape_param_51, _shape_param_52, _shape_param_53, getitem_153: "f32[32, 12, 197, 197]", _shape_param_54, primals_139: "i64[197, 197]", view_233: "f32[6304, 2304]", _shape_param_55, view_235: "f32[32, 197, 768]", mul_63: "f32[32, 197, 768]", addmm_27: "f32[6304, 768]", _shape_param_56, add_105: "f32[32, 197, 768]", _shape_param_57, view_237: "f32[6304, 768]", _shape_param_58, view_240: "f32[6304, 3072]", _shape_param_59, view_242: "f32[32, 197, 768]", mul_57: "f32[32, 197, 768]", addmm_25: "f32[6304, 768]", _shape_param_60, add_108: "f32[32, 197, 768]", _shape_param_61, view_244: "f32[6304, 768]", getitem_71: "f32[32, 12, 197, 64]", _shape_param_62, _shape_param_63, _shape_param_64, getitem_157: "f32[32, 12, 197, 197]", _shape_param_65, primals_121: "i64[197, 197]", view_251: "f32[6304, 2304]", _shape_param_66, view_253: "f32[32, 197, 768]", mul_54: "f32[32, 197, 768]", addmm_23: "f32[6304, 768]", _shape_param_67, add_109: "f32[32, 197, 768]", _shape_param_68, view_255: "f32[6304, 768]", _shape_param_69, view_258: "f32[6304, 3072]", _shape_param_70, view_260: "f32[32, 197, 768]", mul_48: "f32[32, 197, 768]", addmm_21: "f32[6304, 768]", _shape_param_71, add_112: "f32[32, 197, 768]", _shape_param_72, view_262: "f32[6304, 768]", getitem_60: "f32[32, 12, 197, 64]", _shape_param_73, _shape_param_74, _shape_param_75, getitem_161: "f32[32, 12, 197, 197]", _shape_param_76, primals_103: "i64[197, 197]", view_269: "f32[6304, 2304]", _shape_param_77, view_271: "f32[32, 197, 768]", mul_45: "f32[32, 197, 768]", addmm_19: "f32[6304, 768]", _shape_param_78, add_113: "f32[32, 197, 768]", _shape_param_79, view_273: "f32[6304, 768]", _shape_param_80, view_276: "f32[6304, 3072]", _shape_param_81, view_278: "f32[32, 197, 768]", mul_39: "f32[32, 197, 768]", addmm_17: "f32[6304, 768]", _shape_param_82, add_116: "f32[32, 197, 768]", _shape_param_83, view_280: "f32[6304, 768]", getitem_49: "f32[32, 12, 197, 64]", _shape_param_84, _shape_param_85, _shape_param_86, getitem_165: "f32[32, 12, 197, 197]", _shape_param_87, primals_85: "i64[197, 197]", view_287: "f32[6304, 2304]", _shape_param_88, view_289: "f32[32, 197, 768]", mul_36: "f32[32, 197, 768]", addmm_15: "f32[6304, 768]", _shape_param_89, add_117: "f32[32, 197, 768]", _shape_param_90, view_291: "f32[6304, 768]", _shape_param_91, view_294: "f32[6304, 3072]", _shape_param_92, view_296: "f32[32, 197, 768]", mul_30: "f32[32, 197, 768]", addmm_13: "f32[6304, 768]", _shape_param_93, add_120: "f32[32, 197, 768]", _shape_param_94, view_298: "f32[6304, 768]", getitem_38: "f32[32, 12, 197, 64]", _shape_param_95, _shape_param_96, _shape_param_97, getitem_169: "f32[32, 12, 197, 197]", _shape_param_98, primals_67: "i64[197, 197]", view_305: "f32[6304, 2304]", _shape_param_99, view_307: "f32[32, 197, 768]", mul_27: "f32[32, 197, 768]", addmm_11: "f32[6304, 768]", _shape_param_100, add_121: "f32[32, 197, 768]", _shape_param_101, view_309: "f32[6304, 768]", _shape_param_102, view_312: "f32[6304, 3072]", _shape_param_103, view_314: "f32[32, 197, 768]", mul_21: "f32[32, 197, 768]", addmm_9: "f32[6304, 768]", _shape_param_104, add_124: "f32[32, 197, 768]", _shape_param_105, view_316: "f32[6304, 768]", getitem_27: "f32[32, 12, 197, 64]", _shape_param_106, _shape_param_107, _shape_param_108, getitem_173: "f32[32, 12, 197, 197]", _shape_param_109, primals_49: "i64[197, 197]", view_323: "f32[6304, 2304]", _shape_param_110, view_325: "f32[32, 197, 768]", mul_18: "f32[32, 197, 768]", addmm_7: "f32[6304, 768]", _shape_param_111, add_125: "f32[32, 197, 768]", _shape_param_112, view_327: "f32[6304, 768]", _shape_param_113, view_330: "f32[6304, 3072]", _shape_param_114, view_332: "f32[32, 197, 768]", mul_12: "f32[32, 197, 768]", addmm_5: "f32[6304, 768]", _shape_param_115, add_128: "f32[32, 197, 768]", _shape_param_116, view_334: "f32[6304, 768]", getitem_16: "f32[32, 12, 197, 64]", _shape_param_117, _shape_param_118, _shape_param_119, getitem_177: "f32[32, 12, 197, 197]", _shape_param_120, primals_31: "i64[197, 197]", view_341: "f32[6304, 2304]", _shape_param_121, view_343: "f32[32, 197, 768]", mul_9: "f32[32, 197, 768]", addmm_3: "f32[6304, 768]", _shape_param_122, add_129: "f32[32, 197, 768]", _shape_param_123, view_345: "f32[6304, 768]", _shape_param_124, view_348: "f32[6304, 3072]", _shape_param_125, view_350: "f32[32, 197, 768]", mul_3: "f32[32, 197, 768]", add_132: "f32[32, 197, 768]", view_8: "f32[32, 197, 768]", _shape_param_126, view_352: "f32[6304, 768]", getitem_5: "f32[32, 12, 197, 64]", _shape_param_127, _shape_param_128, _shape_param_129, getitem_181: "f32[32, 12, 197, 197]", _shape_param_130, primals_13: "i64[197, 197]", view_359: "f32[6304, 2304]", _shape_param_131, mm_96: "f32[6304, 768]", _shape_param_132, primals_6: "f32[768]", cat: "f32[32, 197, 768]", getitem_1: "f32[32, 197, 1]", rsqrt: "f32[32, 197, 1]", _shape_param_133):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:815 in forward_head, code: return x if pre_logits else self.head(x)
        permute_default: "f32[1000, 32]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[32, 768]" = torch.ops.aten.mul.Tensor(mm, mul_108);  mul_108 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mm, [0]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_1: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_47, _shape_param_1);  addmm_47 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_1: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter, reshape_default_1);  slice_scatter = reshape_default_1 = None
        sum_dim_int_list_3: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_2);  sum_dim_int_list_3 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_1: "f32[768, 6304]" = torch.ops.aten.permute.default(view_147, [1, 0])
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_147, [0], True);  view_147 = None
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_3);  sum_dim_int_list_4 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_2: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_150, [1, 0])
        sum_dim_int_list_5: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True);  view_150 = None
        reshape_default_4: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_4);  sum_dim_int_list_5 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_2: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_152, mul_102);  mul_102 = None
        sum_dim_int_list_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_152, [0, 1]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_5: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_45, _shape_param_5);  addmm_45 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_3: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_88, reshape_default_5);  add_88 = reshape_default_5 = None
        sum_dim_int_list_8: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_6);  sum_dim_int_list_8 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_3: "f32[768, 6304]" = torch.ops.aten.permute.default(view_154, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_4: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        reshape_default_7: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_7);  permute_default_4 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_8: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        sum_dim_int_list_9: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True);  view_154 = None
        reshape_default_9: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_9);  sum_dim_int_list_9 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_10: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_137, [0], True);  getitem_137 = None
        full_default: "f32[1, 12, 197, 200]" = torch.ops.aten.full.default([1, 12, 197, 200], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_10, -1, 0, 197);  sum_dim_int_list_10 = None
        constant_pad_nd_default: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default, [0, -3]);  slice_scatter_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default, 0);  constant_pad_nd_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_5: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_10: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_10);  permute_default_5 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default_1: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_11: "i64[38809]" = torch.ops.aten.reshape.default(primals_211, [-1]);  primals_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_11], reshape_default_10, True);  reshape_default_11 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_6: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_161, [1, 0])
        sum_dim_int_list_11: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_161, [0], True);  view_161 = None
        reshape_default_12: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_11);  sum_dim_int_list_11 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_12, 0, 0, 768)
        slice_tensor_1: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_12, 0, 1536, 2304);  reshape_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_4: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_163, mul_99);  mul_99 = None
        sum_dim_int_list_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_163, [0, 1]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_13: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_43, _shape_param_12);  addmm_43 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_5: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_89, reshape_default_13);  add_89 = reshape_default_13 = None
        sum_dim_int_list_14: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_14: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_13);  sum_dim_int_list_14 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_7: "f32[768, 6304]" = torch.ops.aten.permute.default(view_165, [1, 0])
        sum_dim_int_list_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
        reshape_default_15: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_14);  sum_dim_int_list_15 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_8: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_168, [1, 0])
        sum_dim_int_list_16: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
        reshape_default_16: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_15);  sum_dim_int_list_16 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_6: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_170, mul_93);  mul_93 = None
        sum_dim_int_list_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_170, [0, 1]);  view_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_17: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_41, _shape_param_16);  addmm_41 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_7: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_92, reshape_default_17);  add_92 = reshape_default_17 = None
        sum_dim_int_list_19: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_17);  sum_dim_int_list_19 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_9: "f32[768, 6304]" = torch.ops.aten.permute.default(view_172, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_10: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        reshape_default_19: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_10, _shape_param_18);  permute_default_10 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_20: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_19, _shape_param_19);  reshape_default_19 = _shape_param_19 = None
        sum_dim_int_list_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_172, [0], True);  view_172 = None
        reshape_default_21: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_20);  sum_dim_int_list_20 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_21: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_141, [0], True);  getitem_141 = None
        slice_scatter_default_1: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_21, -1, 0, 197);  sum_dim_int_list_21 = None
        constant_pad_nd_default_1: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_1, [0, -3]);  slice_scatter_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_1: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_1, 0);  constant_pad_nd_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_11: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_22: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_11, _shape_param_21);  permute_default_11 = _shape_param_21 = None
        reshape_default_23: "i64[38809]" = torch.ops.aten.reshape.default(primals_193, [-1]);  primals_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_1: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_23], reshape_default_22, True);  reshape_default_23 = reshape_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_12: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_179, [1, 0])
        sum_dim_int_list_22: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_179, [0], True);  view_179 = None
        reshape_default_24: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_22);  sum_dim_int_list_22 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor_2: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_24, 0, 0, 768)
        slice_tensor_3: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_24, 0, 1536, 2304);  reshape_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_8: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_181, mul_90);  mul_90 = None
        sum_dim_int_list_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_181, [0, 1]);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_25: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_39, _shape_param_23);  addmm_39 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_9: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_93, reshape_default_25);  add_93 = reshape_default_25 = None
        sum_dim_int_list_25: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_26: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, _shape_param_24);  sum_dim_int_list_25 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_13: "f32[768, 6304]" = torch.ops.aten.permute.default(view_183, [1, 0])
        sum_dim_int_list_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
        reshape_default_27: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_25);  sum_dim_int_list_26 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_14: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_186, [1, 0])
        sum_dim_int_list_27: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        reshape_default_28: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_26);  sum_dim_int_list_27 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_10: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_188, mul_84);  mul_84 = None
        sum_dim_int_list_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_188, [0, 1]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_29: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_37, _shape_param_27);  addmm_37 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_11: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_96, reshape_default_29);  add_96 = reshape_default_29 = None
        sum_dim_int_list_30: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_30: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_28);  sum_dim_int_list_30 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_15: "f32[768, 6304]" = torch.ops.aten.permute.default(view_190, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_16: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        reshape_default_31: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_16, _shape_param_29);  permute_default_16 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_32: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_31, _shape_param_30);  reshape_default_31 = _shape_param_30 = None
        sum_dim_int_list_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_190, [0], True);  view_190 = None
        reshape_default_33: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_31);  sum_dim_int_list_31 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_32: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_145, [0], True);  getitem_145 = None
        slice_scatter_default_2: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_32, -1, 0, 197);  sum_dim_int_list_32 = None
        constant_pad_nd_default_2: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_2, [0, -3]);  slice_scatter_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_2: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_2, 0);  constant_pad_nd_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_17: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_2, [1, 2, 0]);  squeeze_dim_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_34: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_17, _shape_param_32);  permute_default_17 = _shape_param_32 = None
        reshape_default_35: "i64[38809]" = torch.ops.aten.reshape.default(primals_175, [-1]);  primals_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_2: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_35], reshape_default_34, True);  reshape_default_35 = reshape_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_18: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_197, [1, 0])
        sum_dim_int_list_33: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
        reshape_default_36: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_33);  sum_dim_int_list_33 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor_4: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_36, 0, 0, 768)
        slice_tensor_5: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_36, 0, 1536, 2304);  reshape_default_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_12: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_199, mul_81);  mul_81 = None
        sum_dim_int_list_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_199, [0, 1]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_37: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_35, _shape_param_34);  addmm_35 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_13: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_97, reshape_default_37);  add_97 = reshape_default_37 = None
        sum_dim_int_list_36: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1], True);  mul_tensor_13 = None
        reshape_default_38: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_35);  sum_dim_int_list_36 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_19: "f32[768, 6304]" = torch.ops.aten.permute.default(view_201, [1, 0])
        sum_dim_int_list_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
        reshape_default_39: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_36);  sum_dim_int_list_37 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_20: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_204, [1, 0])
        sum_dim_int_list_38: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
        reshape_default_40: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_37);  sum_dim_int_list_38 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_14: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_206, mul_75);  mul_75 = None
        sum_dim_int_list_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_206, [0, 1]);  view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_41: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_33, _shape_param_38);  addmm_33 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_15: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_100, reshape_default_41);  add_100 = reshape_default_41 = None
        sum_dim_int_list_41: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_42: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_39);  sum_dim_int_list_41 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_21: "f32[768, 6304]" = torch.ops.aten.permute.default(view_208, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_22: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        reshape_default_43: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_22, _shape_param_40);  permute_default_22 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_44: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_43, _shape_param_41);  reshape_default_43 = _shape_param_41 = None
        sum_dim_int_list_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_208, [0], True);  view_208 = None
        reshape_default_45: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_42);  sum_dim_int_list_42 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_43: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_149, [0], True);  getitem_149 = None
        slice_scatter_default_3: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_43, -1, 0, 197);  sum_dim_int_list_43 = None
        constant_pad_nd_default_3: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_3, [0, -3]);  slice_scatter_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_3: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_3, 0);  constant_pad_nd_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_23: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_3, [1, 2, 0]);  squeeze_dim_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_46: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_23, _shape_param_43);  permute_default_23 = _shape_param_43 = None
        reshape_default_47: "i64[38809]" = torch.ops.aten.reshape.default(primals_157, [-1]);  primals_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_3: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_47], reshape_default_46, True);  reshape_default_47 = reshape_default_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_24: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_215, [1, 0])
        sum_dim_int_list_44: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
        reshape_default_48: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_44);  sum_dim_int_list_44 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor_6: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_48, 0, 0, 768)
        slice_tensor_7: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_48, 0, 1536, 2304);  reshape_default_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_16: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_217, mul_72);  mul_72 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_49: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_31, _shape_param_45);  addmm_31 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_17: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_101, reshape_default_49);  add_101 = reshape_default_49 = None
        sum_dim_int_list_47: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_50: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_46);  sum_dim_int_list_47 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_25: "f32[768, 6304]" = torch.ops.aten.permute.default(view_219, [1, 0])
        sum_dim_int_list_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_219, [0], True);  view_219 = None
        reshape_default_51: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_47);  sum_dim_int_list_48 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_26: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_222, [1, 0])
        sum_dim_int_list_49: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
        reshape_default_52: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, _shape_param_48);  sum_dim_int_list_49 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_18: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_224, mul_66);  mul_66 = None
        sum_dim_int_list_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_224, [0, 1]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_53: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_29, _shape_param_49);  addmm_29 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_19: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_104, reshape_default_53);  add_104 = reshape_default_53 = None
        sum_dim_int_list_52: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_54: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_50);  sum_dim_int_list_52 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_27: "f32[768, 6304]" = torch.ops.aten.permute.default(view_226, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_28: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        reshape_default_55: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_28, _shape_param_51);  permute_default_28 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_56: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_55, _shape_param_52);  reshape_default_55 = _shape_param_52 = None
        sum_dim_int_list_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
        reshape_default_57: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_53);  sum_dim_int_list_53 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_54: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_153, [0], True);  getitem_153 = None
        slice_scatter_default_4: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_54, -1, 0, 197);  sum_dim_int_list_54 = None
        constant_pad_nd_default_4: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_4, [0, -3]);  slice_scatter_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_4: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_4, 0);  constant_pad_nd_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_29: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_4, [1, 2, 0]);  squeeze_dim_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_58: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_29, _shape_param_54);  permute_default_29 = _shape_param_54 = None
        reshape_default_59: "i64[38809]" = torch.ops.aten.reshape.default(primals_139, [-1]);  primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_4: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_59], reshape_default_58, True);  reshape_default_59 = reshape_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_30: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_233, [1, 0])
        sum_dim_int_list_55: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
        reshape_default_60: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_55);  sum_dim_int_list_55 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor_8: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_60, 0, 0, 768)
        slice_tensor_9: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_60, 0, 1536, 2304);  reshape_default_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_20: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_235, mul_63);  mul_63 = None
        sum_dim_int_list_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_235, [0, 1]);  view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_61: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_27, _shape_param_56);  addmm_27 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_21: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_105, reshape_default_61);  add_105 = reshape_default_61 = None
        sum_dim_int_list_58: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], True);  mul_tensor_21 = None
        reshape_default_62: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, _shape_param_57);  sum_dim_int_list_58 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_31: "f32[768, 6304]" = torch.ops.aten.permute.default(view_237, [1, 0])
        sum_dim_int_list_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_237, [0], True);  view_237 = None
        reshape_default_63: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_58);  sum_dim_int_list_59 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_32: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_240, [1, 0])
        sum_dim_int_list_60: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True);  view_240 = None
        reshape_default_64: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_59);  sum_dim_int_list_60 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_22: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_242, mul_57);  mul_57 = None
        sum_dim_int_list_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_242, [0, 1]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_65: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_25, _shape_param_60);  addmm_25 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_23: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_108, reshape_default_65);  add_108 = reshape_default_65 = None
        sum_dim_int_list_63: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1], True);  mul_tensor_23 = None
        reshape_default_66: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_61);  sum_dim_int_list_63 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_33: "f32[768, 6304]" = torch.ops.aten.permute.default(view_244, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_34: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        reshape_default_67: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_34, _shape_param_62);  permute_default_34 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_68: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_67, _shape_param_63);  reshape_default_67 = _shape_param_63 = None
        sum_dim_int_list_64: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_244, [0], True);  view_244 = None
        reshape_default_69: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_64);  sum_dim_int_list_64 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_65: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_157, [0], True);  getitem_157 = None
        slice_scatter_default_5: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_65, -1, 0, 197);  sum_dim_int_list_65 = None
        constant_pad_nd_default_5: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_5, [0, -3]);  slice_scatter_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_5: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_5, 0);  constant_pad_nd_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_35: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_5, [1, 2, 0]);  squeeze_dim_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_70: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_35, _shape_param_65);  permute_default_35 = _shape_param_65 = None
        reshape_default_71: "i64[38809]" = torch.ops.aten.reshape.default(primals_121, [-1]);  primals_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_5: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_71], reshape_default_70, True);  reshape_default_71 = reshape_default_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_36: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_251, [1, 0])
        sum_dim_int_list_66: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True);  view_251 = None
        reshape_default_72: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_66);  sum_dim_int_list_66 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor_10: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_72, 0, 0, 768)
        slice_tensor_11: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_72, 0, 1536, 2304);  reshape_default_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_24: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_253, mul_54);  mul_54 = None
        sum_dim_int_list_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_253, [0, 1]);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_73: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_23, _shape_param_67);  addmm_23 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_25: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_109, reshape_default_73);  add_109 = reshape_default_73 = None
        sum_dim_int_list_69: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_74: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_69, _shape_param_68);  sum_dim_int_list_69 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_37: "f32[768, 6304]" = torch.ops.aten.permute.default(view_255, [1, 0])
        sum_dim_int_list_70: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_255, [0], True);  view_255 = None
        reshape_default_75: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_69);  sum_dim_int_list_70 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_38: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_258, [1, 0])
        sum_dim_int_list_71: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
        reshape_default_76: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_70);  sum_dim_int_list_71 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_26: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_260, mul_48);  mul_48 = None
        sum_dim_int_list_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None
        sum_dim_int_list_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_260, [0, 1]);  view_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_77: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_21, _shape_param_71);  addmm_21 = _shape_param_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_27: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_112, reshape_default_77);  add_112 = reshape_default_77 = None
        sum_dim_int_list_74: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_78: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_74, _shape_param_72);  sum_dim_int_list_74 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_39: "f32[768, 6304]" = torch.ops.aten.permute.default(view_262, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_40: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        reshape_default_79: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_40, _shape_param_73);  permute_default_40 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_80: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_79, _shape_param_74);  reshape_default_79 = _shape_param_74 = None
        sum_dim_int_list_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_262, [0], True);  view_262 = None
        reshape_default_81: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_75);  sum_dim_int_list_75 = _shape_param_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_76: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_161, [0], True);  getitem_161 = None
        slice_scatter_default_6: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_76, -1, 0, 197);  sum_dim_int_list_76 = None
        constant_pad_nd_default_6: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_6, [0, -3]);  slice_scatter_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_6: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_6, 0);  constant_pad_nd_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_41: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_6, [1, 2, 0]);  squeeze_dim_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_82: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_41, _shape_param_76);  permute_default_41 = _shape_param_76 = None
        reshape_default_83: "i64[38809]" = torch.ops.aten.reshape.default(primals_103, [-1]);  primals_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_6: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_83], reshape_default_82, True);  reshape_default_83 = reshape_default_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_42: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_269, [1, 0])
        sum_dim_int_list_77: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
        reshape_default_84: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_77);  sum_dim_int_list_77 = _shape_param_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor_12: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_84, 0, 0, 768)
        slice_tensor_13: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_84, 0, 1536, 2304);  reshape_default_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_28: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_271, mul_45);  mul_45 = None
        sum_dim_int_list_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 1]);  mul_tensor_28 = None
        sum_dim_int_list_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_85: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_19, _shape_param_78);  addmm_19 = _shape_param_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_29: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_113, reshape_default_85);  add_113 = reshape_default_85 = None
        sum_dim_int_list_80: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1], True);  mul_tensor_29 = None
        reshape_default_86: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_79);  sum_dim_int_list_80 = _shape_param_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_43: "f32[768, 6304]" = torch.ops.aten.permute.default(view_273, [1, 0])
        sum_dim_int_list_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
        reshape_default_87: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_80);  sum_dim_int_list_81 = _shape_param_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_44: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_276, [1, 0])
        sum_dim_int_list_82: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
        reshape_default_88: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_81);  sum_dim_int_list_82 = _shape_param_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_30: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_278, mul_39);  mul_39 = None
        sum_dim_int_list_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_89: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_17, _shape_param_82);  addmm_17 = _shape_param_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_31: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_116, reshape_default_89);  add_116 = reshape_default_89 = None
        sum_dim_int_list_85: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1], True);  mul_tensor_31 = None
        reshape_default_90: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_85, _shape_param_83);  sum_dim_int_list_85 = _shape_param_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_45: "f32[768, 6304]" = torch.ops.aten.permute.default(view_280, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_46: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        reshape_default_91: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_46, _shape_param_84);  permute_default_46 = _shape_param_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_92: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_91, _shape_param_85);  reshape_default_91 = _shape_param_85 = None
        sum_dim_int_list_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
        reshape_default_93: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_86);  sum_dim_int_list_86 = _shape_param_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_87: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_165, [0], True);  getitem_165 = None
        slice_scatter_default_7: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_87, -1, 0, 197);  sum_dim_int_list_87 = None
        constant_pad_nd_default_7: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_7, [0, -3]);  slice_scatter_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_7: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_7, 0);  constant_pad_nd_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_47: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_7, [1, 2, 0]);  squeeze_dim_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_94: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_47, _shape_param_87);  permute_default_47 = _shape_param_87 = None
        reshape_default_95: "i64[38809]" = torch.ops.aten.reshape.default(primals_85, [-1]);  primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_7: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_95], reshape_default_94, True);  reshape_default_95 = reshape_default_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_48: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_287, [1, 0])
        sum_dim_int_list_88: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
        reshape_default_96: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_88);  sum_dim_int_list_88 = _shape_param_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor_14: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_96, 0, 0, 768)
        slice_tensor_15: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_96, 0, 1536, 2304);  reshape_default_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_32: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_289, mul_36);  mul_36 = None
        sum_dim_int_list_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_289, [0, 1]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_97: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_15, _shape_param_89);  addmm_15 = _shape_param_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_33: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_117, reshape_default_97);  add_117 = reshape_default_97 = None
        sum_dim_int_list_91: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1], True);  mul_tensor_33 = None
        reshape_default_98: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_90);  sum_dim_int_list_91 = _shape_param_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_49: "f32[768, 6304]" = torch.ops.aten.permute.default(view_291, [1, 0])
        sum_dim_int_list_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
        reshape_default_99: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_91);  sum_dim_int_list_92 = _shape_param_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_50: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_294, [1, 0])
        sum_dim_int_list_93: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
        reshape_default_100: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_92);  sum_dim_int_list_93 = _shape_param_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_34: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_296, mul_30);  mul_30 = None
        sum_dim_int_list_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1]);  mul_tensor_34 = None
        sum_dim_int_list_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_296, [0, 1]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_101: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_13, _shape_param_93);  addmm_13 = _shape_param_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_35: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_120, reshape_default_101);  add_120 = reshape_default_101 = None
        sum_dim_int_list_96: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1], True);  mul_tensor_35 = None
        reshape_default_102: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_94);  sum_dim_int_list_96 = _shape_param_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_51: "f32[768, 6304]" = torch.ops.aten.permute.default(view_298, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_52: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        reshape_default_103: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_52, _shape_param_95);  permute_default_52 = _shape_param_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_104: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_103, _shape_param_96);  reshape_default_103 = _shape_param_96 = None
        sum_dim_int_list_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
        reshape_default_105: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_97);  sum_dim_int_list_97 = _shape_param_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_98: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_169, [0], True);  getitem_169 = None
        slice_scatter_default_8: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_98, -1, 0, 197);  sum_dim_int_list_98 = None
        constant_pad_nd_default_8: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_8, [0, -3]);  slice_scatter_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_8: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_8, 0);  constant_pad_nd_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_53: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_8, [1, 2, 0]);  squeeze_dim_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_106: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_53, _shape_param_98);  permute_default_53 = _shape_param_98 = None
        reshape_default_107: "i64[38809]" = torch.ops.aten.reshape.default(primals_67, [-1]);  primals_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_8: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_107], reshape_default_106, True);  reshape_default_107 = reshape_default_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_54: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_305, [1, 0])
        sum_dim_int_list_99: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
        reshape_default_108: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, _shape_param_99);  sum_dim_int_list_99 = _shape_param_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor_16: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_108, 0, 0, 768)
        slice_tensor_17: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_108, 0, 1536, 2304);  reshape_default_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_36: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_307, mul_27);  mul_27 = None
        sum_dim_int_list_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_36, [0, 1]);  mul_tensor_36 = None
        sum_dim_int_list_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1]);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_109: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_11, _shape_param_100);  addmm_11 = _shape_param_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_37: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_121, reshape_default_109);  add_121 = reshape_default_109 = None
        sum_dim_int_list_102: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1], True);  mul_tensor_37 = None
        reshape_default_110: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_101);  sum_dim_int_list_102 = _shape_param_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_55: "f32[768, 6304]" = torch.ops.aten.permute.default(view_309, [1, 0])
        sum_dim_int_list_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_309, [0], True);  view_309 = None
        reshape_default_111: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_102);  sum_dim_int_list_103 = _shape_param_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_56: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_312, [1, 0])
        sum_dim_int_list_104: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_312, [0], True);  view_312 = None
        reshape_default_112: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_103);  sum_dim_int_list_104 = _shape_param_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_38: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_314, mul_21);  mul_21 = None
        sum_dim_int_list_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [0, 1]);  mul_tensor_38 = None
        sum_dim_int_list_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_314, [0, 1]);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_113: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_9, _shape_param_104);  addmm_9 = _shape_param_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_39: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_124, reshape_default_113);  add_124 = reshape_default_113 = None
        sum_dim_int_list_107: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1], True);  mul_tensor_39 = None
        reshape_default_114: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_105);  sum_dim_int_list_107 = _shape_param_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_57: "f32[768, 6304]" = torch.ops.aten.permute.default(view_316, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_58: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        reshape_default_115: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_58, _shape_param_106);  permute_default_58 = _shape_param_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_116: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_115, _shape_param_107);  reshape_default_115 = _shape_param_107 = None
        sum_dim_int_list_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_316, [0], True);  view_316 = None
        reshape_default_117: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_108);  sum_dim_int_list_108 = _shape_param_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_109: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_173, [0], True);  getitem_173 = None
        slice_scatter_default_9: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_109, -1, 0, 197);  sum_dim_int_list_109 = None
        constant_pad_nd_default_9: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_9, [0, -3]);  slice_scatter_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_9: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_9, 0);  constant_pad_nd_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_59: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_9, [1, 2, 0]);  squeeze_dim_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_118: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_59, _shape_param_109);  permute_default_59 = _shape_param_109 = None
        reshape_default_119: "i64[38809]" = torch.ops.aten.reshape.default(primals_49, [-1]);  primals_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_9: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_119], reshape_default_118, True);  reshape_default_119 = reshape_default_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_60: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_323, [1, 0])
        sum_dim_int_list_110: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
        reshape_default_120: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, _shape_param_110);  sum_dim_int_list_110 = _shape_param_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor_18: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_120, 0, 0, 768)
        slice_tensor_19: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_120, 0, 1536, 2304);  reshape_default_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_40: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_325, mul_18);  mul_18 = None
        sum_dim_int_list_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_40, [0, 1]);  mul_tensor_40 = None
        sum_dim_int_list_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_325, [0, 1]);  view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_121: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_7, _shape_param_111);  addmm_7 = _shape_param_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_41: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_125, reshape_default_121);  add_125 = reshape_default_121 = None
        sum_dim_int_list_113: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1], True);  mul_tensor_41 = None
        reshape_default_122: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, _shape_param_112);  sum_dim_int_list_113 = _shape_param_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_61: "f32[768, 6304]" = torch.ops.aten.permute.default(view_327, [1, 0])
        sum_dim_int_list_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
        reshape_default_123: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_114, _shape_param_113);  sum_dim_int_list_114 = _shape_param_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_62: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_330, [1, 0])
        sum_dim_int_list_115: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
        reshape_default_124: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_115, _shape_param_114);  sum_dim_int_list_115 = _shape_param_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_42: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_332, mul_12);  mul_12 = None
        sum_dim_int_list_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [0, 1]);  mul_tensor_42 = None
        sum_dim_int_list_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_332, [0, 1]);  view_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_125: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_5, _shape_param_115);  addmm_5 = _shape_param_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_43: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_128, reshape_default_125);  add_128 = reshape_default_125 = None
        sum_dim_int_list_118: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1], True);  mul_tensor_43 = None
        reshape_default_126: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_118, _shape_param_116);  sum_dim_int_list_118 = _shape_param_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_63: "f32[768, 6304]" = torch.ops.aten.permute.default(view_334, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_64: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        reshape_default_127: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_64, _shape_param_117);  permute_default_64 = _shape_param_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_128: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_127, _shape_param_118);  reshape_default_127 = _shape_param_118 = None
        sum_dim_int_list_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
        reshape_default_129: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_119, _shape_param_119);  sum_dim_int_list_119 = _shape_param_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_120: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_177, [0], True);  getitem_177 = None
        slice_scatter_default_10: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_120, -1, 0, 197);  sum_dim_int_list_120 = None
        constant_pad_nd_default_10: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_10, [0, -3]);  slice_scatter_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_10: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_10, 0);  constant_pad_nd_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_65: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_10, [1, 2, 0]);  squeeze_dim_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_130: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_65, _shape_param_120);  permute_default_65 = _shape_param_120 = None
        reshape_default_131: "i64[38809]" = torch.ops.aten.reshape.default(primals_31, [-1]);  primals_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_10: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_131], reshape_default_130, True);  reshape_default_131 = reshape_default_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_66: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_341, [1, 0])
        sum_dim_int_list_121: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
        reshape_default_132: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_121, _shape_param_121);  sum_dim_int_list_121 = _shape_param_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor_20: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_132, 0, 0, 768)
        slice_tensor_21: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_132, 0, 1536, 2304);  reshape_default_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_44: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_343, mul_9);  mul_9 = None
        sum_dim_int_list_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_44, [0, 1]);  mul_tensor_44 = None
        sum_dim_int_list_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_343, [0, 1]);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_133: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(addmm_3, _shape_param_122);  addmm_3 = _shape_param_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:389 in forward, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        mul_tensor_45: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_129, reshape_default_133);  add_129 = reshape_default_133 = None
        sum_dim_int_list_124: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1], True);  mul_tensor_45 = None
        reshape_default_134: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_124, _shape_param_123);  sum_dim_int_list_124 = _shape_param_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_67: "f32[768, 6304]" = torch.ops.aten.permute.default(view_345, [1, 0])
        sum_dim_int_list_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_345, [0], True);  view_345 = None
        reshape_default_135: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_125, _shape_param_124);  sum_dim_int_list_125 = _shape_param_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_68: "f32[3072, 6304]" = torch.ops.aten.permute.default(view_348, [1, 0])
        sum_dim_int_list_126: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_348, [0], True);  view_348 = None
        reshape_default_136: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_126, _shape_param_125);  sum_dim_int_list_126 = _shape_param_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_46: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(view_350, mul_3);  mul_3 = None
        sum_dim_int_list_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [0, 1]);  mul_tensor_46 = None
        sum_dim_int_list_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_350, [0, 1]);  view_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:388 in forward, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
        mul_tensor_47: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(add_132, view_8);  view_8 = None
        sum_dim_int_list_129: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1], True);  mul_tensor_47 = None
        reshape_default_137: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_129, _shape_param_126);  sum_dim_int_list_129 = _shape_param_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        permute_default_69: "f32[768, 6304]" = torch.ops.aten.permute.default(view_352, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:249 in forward, code: x = x.transpose(1, 2).reshape(B, N, C)
        permute_default_70: "f32[32, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        reshape_default_138: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(permute_default_70, _shape_param_127);  permute_default_70 = _shape_param_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:250 in forward, code: x = self.proj(x)
        reshape_default_139: "f32[6304, 768]" = torch.ops.aten.reshape.default(reshape_default_138, _shape_param_128);  reshape_default_138 = _shape_param_128 = None
        sum_dim_int_list_130: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
        reshape_default_140: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_130, _shape_param_129);  sum_dim_int_list_130 = _shape_param_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        sum_dim_int_list_131: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_181, [0], True);  getitem_181 = None
        slice_scatter_default_11: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default, sum_dim_int_list_131, -1, 0, 197);  full_default = sum_dim_int_list_131 = None
        constant_pad_nd_default_11: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_default_11, [0, -3]);  slice_scatter_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_11: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default_11, 0);  constant_pad_nd_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_71: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_dim_11, [1, 2, 0]);  squeeze_dim_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_141: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_default_71, _shape_param_130);  permute_default_71 = _shape_param_130 = None
        reshape_default_142: "i64[38809]" = torch.ops.aten.reshape.default(primals_13, [-1]);  primals_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_11: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_142], reshape_default_141, True);  full_default_1 = reshape_default_142 = reshape_default_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        permute_default_72: "f32[2304, 6304]" = torch.ops.aten.permute.default(view_359, [1, 0])
        sum_dim_int_list_132: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
        reshape_default_143: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_132, _shape_param_131);  sum_dim_int_list_132 = _shape_param_131 = None
        reshape_default_144: "f32[32, 197, 768]" = torch.ops.aten.reshape.default(mm_96, _shape_param_132);  mm_96 = _shape_param_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:213 in forward, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        slice_tensor_22: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_143, 0, 0, 768)
        slice_tensor_23: "f32[768]" = torch.ops.aten.slice.Tensor(reshape_default_143, 0, 1536, 2304);  reshape_default_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor_48: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(reshape_default_144, primals_6);  primals_6 = None
        mul_tensor_49: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_48, 768)
        sum_dim_int_list_133: "f32[32, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_48, [2], True)
        sub_tensor: "f32[32, 197, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_1);  cat = getitem_1 = None
        mul_tensor_50: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_51: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_48, mul_tensor_50);  mul_tensor_48 = None
        sum_dim_int_list_134: "f32[32, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_51, [2], True);  mul_tensor_51 = None
        mul_tensor_52: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_50, sum_dim_int_list_134);  sum_dim_int_list_134 = None
        sub_tensor_1: "f32[32, 197, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_49, sum_dim_int_list_133);  mul_tensor_49 = sum_dim_int_list_133 = None
        sub_tensor_2: "f32[32, 197, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_52);  sub_tensor_1 = mul_tensor_52 = None
        div_tensor: "f32[32, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_tensor_53: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_54: "f32[32, 197, 768]" = torch.ops.aten.mul.Tensor(reshape_default_144, mul_tensor_50);  mul_tensor_50 = None
        sum_dim_int_list_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [0, 1]);  mul_tensor_54 = None
        sum_dim_int_list_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_144, [0, 1]);  reshape_default_144 = None
        add_tensor: "f32[32, 197, 768]" = torch.ops.aten.add.Tensor(add_132, mul_tensor_53);  add_132 = mul_tensor_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:787 in forward_features, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        slice_tensor_24: "f32[32, 1, 768]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 0, 1)
        slice_tensor_25: "f32[32, 196, 768]" = torch.ops.aten.slice.Tensor(add_tensor, 1, 1, 197);  add_tensor = None
        sum_dim_int_list_137: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_tensor_24, [0], True);  slice_tensor_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        permute_default_73: "f32[32, 768, 196]" = torch.ops.aten.permute.default(slice_tensor_25, [0, 2, 1]);  slice_tensor_25 = None
        reshape_default_145: "f32[32, 768, 14, 14]" = torch.ops.aten.reshape.default(permute_default_73, _shape_param_133);  permute_default_73 = _shape_param_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_dim_int_list_138: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_145, [0, 2, 3]);  reshape_default_145 = None
        return (permute_default, reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, reshape_default_2, permute_default_1, reshape_default_3, permute_default_2, reshape_default_4, sum_dim_int_list_6, sum_dim_int_list_7, reshape_default_6, permute_default_3, reshape_default_8, reshape_default_9, index_put_default, permute_default_6, slice_tensor, slice_tensor_1, sum_dim_int_list_12, sum_dim_int_list_13, reshape_default_14, permute_default_7, reshape_default_15, permute_default_8, reshape_default_16, sum_dim_int_list_17, sum_dim_int_list_18, reshape_default_18, permute_default_9, reshape_default_20, reshape_default_21, index_put_default_1, permute_default_12, slice_tensor_2, slice_tensor_3, sum_dim_int_list_23, sum_dim_int_list_24, reshape_default_26, permute_default_13, reshape_default_27, permute_default_14, reshape_default_28, sum_dim_int_list_28, sum_dim_int_list_29, reshape_default_30, permute_default_15, reshape_default_32, reshape_default_33, index_put_default_2, permute_default_18, slice_tensor_4, slice_tensor_5, sum_dim_int_list_34, sum_dim_int_list_35, reshape_default_38, permute_default_19, reshape_default_39, permute_default_20, reshape_default_40, sum_dim_int_list_39, sum_dim_int_list_40, reshape_default_42, permute_default_21, reshape_default_44, reshape_default_45, index_put_default_3, permute_default_24, slice_tensor_6, slice_tensor_7, sum_dim_int_list_45, sum_dim_int_list_46, reshape_default_50, permute_default_25, reshape_default_51, permute_default_26, reshape_default_52, sum_dim_int_list_50, sum_dim_int_list_51, reshape_default_54, permute_default_27, reshape_default_56, reshape_default_57, index_put_default_4, permute_default_30, slice_tensor_8, slice_tensor_9, sum_dim_int_list_56, sum_dim_int_list_57, reshape_default_62, permute_default_31, reshape_default_63, permute_default_32, reshape_default_64, sum_dim_int_list_61, sum_dim_int_list_62, reshape_default_66, permute_default_33, reshape_default_68, reshape_default_69, index_put_default_5, permute_default_36, slice_tensor_10, slice_tensor_11, sum_dim_int_list_67, sum_dim_int_list_68, reshape_default_74, permute_default_37, reshape_default_75, permute_default_38, reshape_default_76, sum_dim_int_list_72, sum_dim_int_list_73, reshape_default_78, permute_default_39, reshape_default_80, reshape_default_81, index_put_default_6, permute_default_42, slice_tensor_12, slice_tensor_13, sum_dim_int_list_78, sum_dim_int_list_79, reshape_default_86, permute_default_43, reshape_default_87, permute_default_44, reshape_default_88, sum_dim_int_list_83, sum_dim_int_list_84, reshape_default_90, permute_default_45, reshape_default_92, reshape_default_93, index_put_default_7, permute_default_48, slice_tensor_14, slice_tensor_15, sum_dim_int_list_89, sum_dim_int_list_90, reshape_default_98, permute_default_49, reshape_default_99, permute_default_50, reshape_default_100, sum_dim_int_list_94, sum_dim_int_list_95, reshape_default_102, permute_default_51, reshape_default_104, reshape_default_105, index_put_default_8, permute_default_54, slice_tensor_16, slice_tensor_17, sum_dim_int_list_100, sum_dim_int_list_101, reshape_default_110, permute_default_55, reshape_default_111, permute_default_56, reshape_default_112, sum_dim_int_list_105, sum_dim_int_list_106, reshape_default_114, permute_default_57, reshape_default_116, reshape_default_117, index_put_default_9, permute_default_60, slice_tensor_18, slice_tensor_19, sum_dim_int_list_111, sum_dim_int_list_112, reshape_default_122, permute_default_61, reshape_default_123, permute_default_62, reshape_default_124, sum_dim_int_list_116, sum_dim_int_list_117, reshape_default_126, permute_default_63, reshape_default_128, reshape_default_129, index_put_default_10, permute_default_66, slice_tensor_20, slice_tensor_21, sum_dim_int_list_122, sum_dim_int_list_123, reshape_default_134, permute_default_67, reshape_default_135, permute_default_68, reshape_default_136, sum_dim_int_list_127, sum_dim_int_list_128, reshape_default_137, permute_default_69, reshape_default_139, reshape_default_140, index_put_default_11, permute_default_72, slice_tensor_22, slice_tensor_23, sum_dim_int_list_135, sum_dim_int_list_136, sum_dim_int_list_137, sum_dim_int_list_138)


def _default_make_inputs():
    return [
    torch.randn([32, 1000], dtype=torch.float32, device='cuda'),
    [1000],  # _shape_param_0
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_1
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_2
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_3
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_4
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_5
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_6
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_126
    [32, 197, 768],  # _shape_param_7
    [6304, 768],  # _shape_param_8
    [768],  # _shape_param_9
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_137
    [38809, 12],  # _shape_param_10
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_11
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_12
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_13
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_14
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_15
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_16
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_17
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_115
    [32, 197, 768],  # _shape_param_18
    [6304, 768],  # _shape_param_19
    [768],  # _shape_param_20
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_141
    [38809, 12],  # _shape_param_21
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_22
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_23
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_24
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_25
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_26
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_27
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_28
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_104
    [32, 197, 768],  # _shape_param_29
    [6304, 768],  # _shape_param_30
    [768],  # _shape_param_31
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_145
    [38809, 12],  # _shape_param_32
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_33
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_34
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_35
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_36
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_37
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_38
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_39
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_93
    [32, 197, 768],  # _shape_param_40
    [6304, 768],  # _shape_param_41
    [768],  # _shape_param_42
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_149
    [38809, 12],  # _shape_param_43
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_44
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_45
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_46
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_47
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_48
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_49
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_50
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_82
    [32, 197, 768],  # _shape_param_51
    [6304, 768],  # _shape_param_52
    [768],  # _shape_param_53
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_153
    [38809, 12],  # _shape_param_54
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_55
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_56
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_57
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_58
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_59
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_60
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_61
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_71
    [32, 197, 768],  # _shape_param_62
    [6304, 768],  # _shape_param_63
    [768],  # _shape_param_64
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_157
    [38809, 12],  # _shape_param_65
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_66
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_67
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_68
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_69
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_70
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_71
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_72
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_60
    [32, 197, 768],  # _shape_param_73
    [6304, 768],  # _shape_param_74
    [768],  # _shape_param_75
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_161
    [38809, 12],  # _shape_param_76
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_77
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_78
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_79
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_80
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_81
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_82
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_83
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_49
    [32, 197, 768],  # _shape_param_84
    [6304, 768],  # _shape_param_85
    [768],  # _shape_param_86
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_165
    [38809, 12],  # _shape_param_87
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_88
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_89
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_90
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_91
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_92
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_93
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_94
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_38
    [32, 197, 768],  # _shape_param_95
    [6304, 768],  # _shape_param_96
    [768],  # _shape_param_97
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_169
    [38809, 12],  # _shape_param_98
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_99
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_100
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_101
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_102
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_103
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_104
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_105
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_27
    [32, 197, 768],  # _shape_param_106
    [6304, 768],  # _shape_param_107
    [768],  # _shape_param_108
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_173
    [38809, 12],  # _shape_param_109
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_110
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_111
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_112
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_113
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_114
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_115
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_116
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_16
    [32, 197, 768],  # _shape_param_117
    [6304, 768],  # _shape_param_118
    [768],  # _shape_param_119
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_177
    [38809, 12],  # _shape_param_120
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_121
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_122
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_123
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_124
    torch.randn([6304, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_125
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_126
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_5
    [32, 197, 768],  # _shape_param_127
    [6304, 768],  # _shape_param_128
    [768],  # _shape_param_129
    torch.randn(15734773, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 197], [491712, 40976, 208, 1]),  # getitem_181
    [38809, 12],  # _shape_param_130
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [2304],  # _shape_param_131
    torch.randn([6304, 768], dtype=torch.float32, device='cuda'),
    [32, 197, 768],  # _shape_param_132
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 197, 1], dtype=torch.float32, device='cuda'),
    [32, 768, 14, 14],  # _shape_param_133
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
