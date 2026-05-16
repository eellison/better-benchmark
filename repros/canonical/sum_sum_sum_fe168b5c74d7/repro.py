"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_training
Pattern hash: fe168b5c74d7
Shape hash: 9950b0b0
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[8, 1024, 768]", mul_166: "f32[8, 1024, 768]", view_1380: "f32[8192, 768]", _shape_param_0, view_1383: "f32[8192, 3072]", _shape_param_1, add_182: "f32[8, 1024, 768]", mul_159: "f32[8, 1024, 768]", mul_192: "f32[8, 1024, 768]", _shape_param_2, view_1387: "f32[8192, 768]", view_1401: "f32[1024, 8, 768]", _shape_param_3, view_1460: "f32[8192, 768]", view_1464: "f32[1024, 8, 768]", _shape_param_4, view_1470: "f32[8192, 768]", div_122: "f32[1024, 8, 768]", _shape_param_5, view_1477: "f32[8192, 768]", add_190: "f32[8, 1024, 768]", mul_152: "f32[8, 1024, 768]", view_1479: "f32[8192, 768]", _shape_param_6, view_1482: "f32[8192, 3072]", _shape_param_7, add_193: "f32[8, 1024, 768]", mul_145: "f32[8, 1024, 768]", mul_220: "f32[8, 1024, 768]", _shape_param_8, view_1486: "f32[8192, 768]", view_1500: "f32[1024, 8, 768]", _shape_param_9, view_1559: "f32[8192, 768]", view_1563: "f32[1024, 8, 768]", _shape_param_10, view_1569: "f32[8192, 768]", div_125: "f32[1024, 8, 768]", _shape_param_11, view_1576: "f32[8192, 768]", add_201: "f32[8, 1024, 768]", mul_138: "f32[8, 1024, 768]", view_1578: "f32[8192, 768]", _shape_param_12, view_1581: "f32[8192, 3072]", _shape_param_13, add_204: "f32[8, 1024, 768]", mul_131: "f32[8, 1024, 768]", mul_248: "f32[8, 1024, 768]", _shape_param_14, view_1585: "f32[8192, 768]", view_1599: "f32[1024, 8, 768]", _shape_param_15, view_1658: "f32[8192, 768]", view_1662: "f32[1024, 8, 768]", _shape_param_16, view_1668: "f32[8192, 768]", div_128: "f32[1024, 8, 768]", _shape_param_17, view_1675: "f32[8192, 768]", add_212: "f32[8, 1024, 768]", mul_124: "f32[8, 1024, 768]", view_1677: "f32[8192, 768]", _shape_param_18, view_1680: "f32[8192, 3072]", _shape_param_19, add_215: "f32[8, 1024, 768]", mul_117: "f32[8, 1024, 768]", mul_276: "f32[8, 1024, 768]", _shape_param_20, view_1684: "f32[8192, 768]", view_1698: "f32[1024, 8, 768]", _shape_param_21, view_1757: "f32[8192, 768]", view_1761: "f32[1024, 8, 768]", _shape_param_22, view_1767: "f32[8192, 768]", div_131: "f32[1024, 8, 768]", _shape_param_23, view_1774: "f32[8192, 768]", add_223: "f32[8, 1024, 768]", mul_110: "f32[8, 1024, 768]", view_1776: "f32[8192, 768]", _shape_param_24, view_1779: "f32[8192, 3072]", _shape_param_25, add_226: "f32[8, 1024, 768]", mul_103: "f32[8, 1024, 768]", mul_304: "f32[8, 1024, 768]", _shape_param_26, view_1783: "f32[8192, 768]", view_1797: "f32[1024, 8, 768]", _shape_param_27, view_1856: "f32[8192, 768]", view_1860: "f32[1024, 8, 768]", _shape_param_28, view_1866: "f32[8192, 768]", div_134: "f32[1024, 8, 768]", _shape_param_29, view_1873: "f32[8192, 768]", add_234: "f32[8, 1024, 768]", mul_96: "f32[8, 1024, 768]", view_1875: "f32[8192, 768]", _shape_param_30, view_1878: "f32[8192, 3072]", _shape_param_31, add_237: "f32[8, 1024, 768]", mul_89: "f32[8, 1024, 768]", mul_332: "f32[8, 1024, 768]", _shape_param_32, view_1882: "f32[8192, 768]", view_1896: "f32[1024, 8, 768]", _shape_param_33, view_1955: "f32[8192, 768]", view_1959: "f32[1024, 8, 768]", _shape_param_34, view_1965: "f32[8192, 768]", div_137: "f32[1024, 8, 768]", _shape_param_35, view_1972: "f32[8192, 768]", add_245: "f32[8, 1024, 768]", mul_82: "f32[8, 1024, 768]", view_1974: "f32[8192, 768]", _shape_param_36, view_1977: "f32[8192, 3072]", _shape_param_37, add_248: "f32[8, 1024, 768]", mul_75: "f32[8, 1024, 768]", mul_360: "f32[8, 1024, 768]", _shape_param_38, view_1981: "f32[8192, 768]", view_1995: "f32[1024, 8, 768]", _shape_param_39, view_2054: "f32[8192, 768]", view_2058: "f32[1024, 8, 768]", _shape_param_40, view_2064: "f32[8192, 768]", div_140: "f32[1024, 8, 768]", _shape_param_41, view_2071: "f32[8192, 768]", add_256: "f32[8, 1024, 768]", mul_68: "f32[8, 1024, 768]", view_2073: "f32[8192, 768]", _shape_param_42, view_2076: "f32[8192, 3072]", _shape_param_43, add_259: "f32[8, 1024, 768]", mul_61: "f32[8, 1024, 768]", mul_388: "f32[8, 1024, 768]", _shape_param_44, view_2080: "f32[8192, 768]", view_2094: "f32[1024, 8, 768]", _shape_param_45, view_2153: "f32[8192, 768]", view_2157: "f32[1024, 8, 768]", _shape_param_46, view_2163: "f32[8192, 768]", div_143: "f32[1024, 8, 768]", _shape_param_47, view_2170: "f32[8192, 768]", add_267: "f32[8, 1024, 768]", mul_54: "f32[8, 1024, 768]", view_2172: "f32[8192, 768]", _shape_param_48, view_2175: "f32[8192, 3072]", _shape_param_49, add_270: "f32[8, 1024, 768]", mul_47: "f32[8, 1024, 768]", mul_416: "f32[8, 1024, 768]", _shape_param_50, view_2179: "f32[8192, 768]", view_2193: "f32[1024, 8, 768]", _shape_param_51, view_2252: "f32[8192, 768]", view_2256: "f32[1024, 8, 768]", _shape_param_52, view_2262: "f32[8192, 768]", div_146: "f32[1024, 8, 768]", _shape_param_53, view_2269: "f32[8192, 768]", add_278: "f32[8, 1024, 768]", mul_40: "f32[8, 1024, 768]", view_2271: "f32[8192, 768]", _shape_param_54, view_2274: "f32[8192, 3072]", _shape_param_55, add_281: "f32[8, 1024, 768]", mul_33: "f32[8, 1024, 768]", mul_444: "f32[8, 1024, 768]", _shape_param_56, view_2278: "f32[8192, 768]", view_2292: "f32[1024, 8, 768]", _shape_param_57, view_2351: "f32[8192, 768]", view_2355: "f32[1024, 8, 768]", _shape_param_58, view_2361: "f32[8192, 768]", div_149: "f32[1024, 8, 768]", _shape_param_59, view_2368: "f32[8192, 768]", add_289: "f32[8, 1024, 768]", mul_26: "f32[8, 1024, 768]", view_2370: "f32[8192, 768]", _shape_param_60, view_2373: "f32[8192, 3072]", _shape_param_61, add_292: "f32[8, 1024, 768]", mul_19: "f32[8, 1024, 768]", mul_472: "f32[8, 1024, 768]", _shape_param_62, view_2377: "f32[8192, 768]", view_2391: "f32[1024, 8, 768]", _shape_param_63, view_2450: "f32[8192, 768]", view_2454: "f32[1024, 8, 768]", _shape_param_64, view_2460: "f32[8192, 768]", div_152: "f32[1024, 8, 768]", _shape_param_65, view_2467: "f32[8192, 768]", add_300: "f32[8, 1024, 768]", mul_12: "f32[8, 1024, 768]", view_2469: "f32[8192, 768]", _shape_param_66, view_2472: "f32[8192, 3072]", _shape_param_67, add_303: "f32[8, 1024, 768]", mul_5: "f32[8, 1024, 768]", mul_500: "f32[8, 1024, 768]", _shape_param_68, view_2476: "f32[8192, 768]", view_2490: "f32[1024, 8, 768]", _shape_param_69, view_2549: "f32[8192, 768]", mm_187: "f32[8192, 768]", _shape_param_70, view_2553: "f32[1024, 8, 768]", _shape_param_71, view_2559: "f32[8192, 768]", mm_189: "f32[8192, 768]", _shape_param_72, div_155: "f32[1024, 8, 768]", _shape_param_73, view_2566: "f32[8192, 768]", mm_191: "f32[8192, 768]", _shape_param_74, mul_497: "f32[8, 1024, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(tangents_1, mul_166);  mul_166 = None
        sum_dim_int_list: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1380, [1, 0])
        sum_dim_int_list_2: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1380, [0], True);  view_1380 = None
        reshape_default: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_0);  sum_dim_int_list_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_1: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1383, [1, 0])
        sum_dim_int_list_3: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1383, [0], True);  view_1383 = None
        reshape_default_1: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_182, mul_159);  mul_159 = None
        sum_dim_int_list_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1]);  add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_6: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_192, [0, 1], True);  mul_192 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None
        permute_default_2: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1387, [1, 0]);  view_1387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_7: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1401, [0, 1], True);  view_1401 = None
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None
        permute_default_3: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1460, [1, 0]);  view_1460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_8: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1464, [0, 1], True);  view_1464 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_4);  sum_dim_int_list_8 = _shape_param_4 = None
        permute_default_4: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1470, [1, 0]);  view_1470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_9: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_122, [0, 1], True);  div_122 = None
        reshape_default_5: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_5);  sum_dim_int_list_9 = _shape_param_5 = None
        permute_default_5: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1477, [1, 0]);  view_1477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_190, mul_152);  mul_152 = None
        sum_dim_int_list_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_190, [0, 1]);  add_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_6: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1479, [1, 0])
        sum_dim_int_list_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1479, [0], True);  view_1479 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_7: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1482, [1, 0])
        sum_dim_int_list_13: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1482, [0], True);  view_1482 = None
        reshape_default_7: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_7);  sum_dim_int_list_13 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_193, mul_145);  mul_145 = None
        sum_dim_int_list_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_193, [0, 1]);  add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_16: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1], True);  mul_220 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None
        permute_default_8: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1486, [1, 0]);  view_1486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_17: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1500, [0, 1], True);  view_1500 = None
        reshape_default_9: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_9);  sum_dim_int_list_17 = _shape_param_9 = None
        permute_default_9: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1559, [1, 0]);  view_1559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_18: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1563, [0, 1], True);  view_1563 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_10);  sum_dim_int_list_18 = _shape_param_10 = None
        permute_default_10: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1569, [1, 0]);  view_1569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_19: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_125, [0, 1], True);  div_125 = None
        reshape_default_11: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_11);  sum_dim_int_list_19 = _shape_param_11 = None
        permute_default_11: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1576, [1, 0]);  view_1576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_201, mul_138);  mul_138 = None
        sum_dim_int_list_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_201, [0, 1]);  add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_12: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1578, [1, 0])
        sum_dim_int_list_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1578, [0], True);  view_1578 = None
        reshape_default_12: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_12);  sum_dim_int_list_22 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_13: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1581, [1, 0])
        sum_dim_int_list_23: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1581, [0], True);  view_1581 = None
        reshape_default_13: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_13);  sum_dim_int_list_23 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_204, mul_131);  mul_131 = None
        sum_dim_int_list_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_204, [0, 1]);  add_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_26: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1], True);  mul_248 = None
        reshape_default_14: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_14);  sum_dim_int_list_26 = _shape_param_14 = None
        permute_default_14: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1585, [1, 0]);  view_1585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_27: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1599, [0, 1], True);  view_1599 = None
        reshape_default_15: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_15);  sum_dim_int_list_27 = _shape_param_15 = None
        permute_default_15: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1658, [1, 0]);  view_1658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_28: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1662, [0, 1], True);  view_1662 = None
        reshape_default_16: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_16);  sum_dim_int_list_28 = _shape_param_16 = None
        permute_default_16: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1668, [1, 0]);  view_1668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_29: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_128, [0, 1], True);  div_128 = None
        reshape_default_17: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_17);  sum_dim_int_list_29 = _shape_param_17 = None
        permute_default_17: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1675, [1, 0]);  view_1675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_212, mul_124);  mul_124 = None
        sum_dim_int_list_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_18: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1677, [1, 0])
        sum_dim_int_list_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1677, [0], True);  view_1677 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_18);  sum_dim_int_list_32 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_19: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1680, [1, 0])
        sum_dim_int_list_33: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1680, [0], True);  view_1680 = None
        reshape_default_19: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_19);  sum_dim_int_list_33 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_215, mul_117);  mul_117 = None
        sum_dim_int_list_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_215, [0, 1]);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_36: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1], True);  mul_276 = None
        reshape_default_20: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_20);  sum_dim_int_list_36 = _shape_param_20 = None
        permute_default_20: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1684, [1, 0]);  view_1684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_37: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1698, [0, 1], True);  view_1698 = None
        reshape_default_21: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_21);  sum_dim_int_list_37 = _shape_param_21 = None
        permute_default_21: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1757, [1, 0]);  view_1757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_38: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1761, [0, 1], True);  view_1761 = None
        reshape_default_22: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_22);  sum_dim_int_list_38 = _shape_param_22 = None
        permute_default_22: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1767, [1, 0]);  view_1767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_39: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_131, [0, 1], True);  div_131 = None
        reshape_default_23: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_23);  sum_dim_int_list_39 = _shape_param_23 = None
        permute_default_23: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1774, [1, 0]);  view_1774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_8: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_223, mul_110);  mul_110 = None
        sum_dim_int_list_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_223, [0, 1]);  add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_24: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1776, [1, 0])
        sum_dim_int_list_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1776, [0], True);  view_1776 = None
        reshape_default_24: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_24);  sum_dim_int_list_42 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_25: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1779, [1, 0])
        sum_dim_int_list_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1779, [0], True);  view_1779 = None
        reshape_default_25: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_25);  sum_dim_int_list_43 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_226, mul_103);  mul_103 = None
        sum_dim_int_list_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_226, [0, 1]);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_46: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1], True);  mul_304 = None
        reshape_default_26: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_26);  sum_dim_int_list_46 = _shape_param_26 = None
        permute_default_26: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1783, [1, 0]);  view_1783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_47: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1797, [0, 1], True);  view_1797 = None
        reshape_default_27: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_27);  sum_dim_int_list_47 = _shape_param_27 = None
        permute_default_27: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1856, [1, 0]);  view_1856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_48: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1860, [0, 1], True);  view_1860 = None
        reshape_default_28: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_28);  sum_dim_int_list_48 = _shape_param_28 = None
        permute_default_28: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1866, [1, 0]);  view_1866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_49: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_134, [0, 1], True);  div_134 = None
        reshape_default_29: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, _shape_param_29);  sum_dim_int_list_49 = _shape_param_29 = None
        permute_default_29: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1873, [1, 0]);  view_1873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_234, mul_96);  mul_96 = None
        sum_dim_int_list_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_234, [0, 1]);  add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_30: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1875, [1, 0])
        sum_dim_int_list_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1875, [0], True);  view_1875 = None
        reshape_default_30: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_30);  sum_dim_int_list_52 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_31: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1878, [1, 0])
        sum_dim_int_list_53: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1878, [0], True);  view_1878 = None
        reshape_default_31: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_31);  sum_dim_int_list_53 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_11: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_237, mul_89);  mul_89 = None
        sum_dim_int_list_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_237, [0, 1]);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_56: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1], True);  mul_332 = None
        reshape_default_32: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_32);  sum_dim_int_list_56 = _shape_param_32 = None
        permute_default_32: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1882, [1, 0]);  view_1882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_57: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1896, [0, 1], True);  view_1896 = None
        reshape_default_33: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_33);  sum_dim_int_list_57 = _shape_param_33 = None
        permute_default_33: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1955, [1, 0]);  view_1955 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_58: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1959, [0, 1], True);  view_1959 = None
        reshape_default_34: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, _shape_param_34);  sum_dim_int_list_58 = _shape_param_34 = None
        permute_default_34: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1965, [1, 0]);  view_1965 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_59: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_137, [0, 1], True);  div_137 = None
        reshape_default_35: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_35);  sum_dim_int_list_59 = _shape_param_35 = None
        permute_default_35: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1972, [1, 0]);  view_1972 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_12: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_245, mul_82);  mul_82 = None
        sum_dim_int_list_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_36: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1974, [1, 0])
        sum_dim_int_list_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1974, [0], True);  view_1974 = None
        reshape_default_36: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_36);  sum_dim_int_list_62 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_37: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1977, [1, 0])
        sum_dim_int_list_63: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1977, [0], True);  view_1977 = None
        reshape_default_37: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_37);  sum_dim_int_list_63 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_13: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_248, mul_75);  mul_75 = None
        sum_dim_int_list_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_66: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1], True);  mul_360 = None
        reshape_default_38: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_38);  sum_dim_int_list_66 = _shape_param_38 = None
        permute_default_38: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1981, [1, 0]);  view_1981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_67: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1995, [0, 1], True);  view_1995 = None
        reshape_default_39: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_39);  sum_dim_int_list_67 = _shape_param_39 = None
        permute_default_39: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2054, [1, 0]);  view_2054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_68: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2058, [0, 1], True);  view_2058 = None
        reshape_default_40: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_40);  sum_dim_int_list_68 = _shape_param_40 = None
        permute_default_40: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2064, [1, 0]);  view_2064 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_69: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_140, [0, 1], True);  div_140 = None
        reshape_default_41: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_69, _shape_param_41);  sum_dim_int_list_69 = _shape_param_41 = None
        permute_default_41: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2071, [1, 0]);  view_2071 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_14: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_256, mul_68);  mul_68 = None
        sum_dim_int_list_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_256, [0, 1]);  add_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_42: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2073, [1, 0])
        sum_dim_int_list_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2073, [0], True);  view_2073 = None
        reshape_default_42: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_42);  sum_dim_int_list_72 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_43: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_2076, [1, 0])
        sum_dim_int_list_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2076, [0], True);  view_2076 = None
        reshape_default_43: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_43);  sum_dim_int_list_73 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_15: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_259, mul_61);  mul_61 = None
        sum_dim_int_list_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_259, [0, 1]);  add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_76: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1], True);  mul_388 = None
        reshape_default_44: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_44);  sum_dim_int_list_76 = _shape_param_44 = None
        permute_default_44: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2080, [1, 0]);  view_2080 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_77: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2094, [0, 1], True);  view_2094 = None
        reshape_default_45: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_45);  sum_dim_int_list_77 = _shape_param_45 = None
        permute_default_45: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2153, [1, 0]);  view_2153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_78: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2157, [0, 1], True);  view_2157 = None
        reshape_default_46: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_78, _shape_param_46);  sum_dim_int_list_78 = _shape_param_46 = None
        permute_default_46: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2163, [1, 0]);  view_2163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_79: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_143, [0, 1], True);  div_143 = None
        reshape_default_47: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_47);  sum_dim_int_list_79 = _shape_param_47 = None
        permute_default_47: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2170, [1, 0]);  view_2170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_16: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_267, mul_54);  mul_54 = None
        sum_dim_int_list_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_267, [0, 1]);  add_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_48: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2172, [1, 0])
        sum_dim_int_list_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2172, [0], True);  view_2172 = None
        reshape_default_48: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_48);  sum_dim_int_list_82 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_49: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_2175, [1, 0])
        sum_dim_int_list_83: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2175, [0], True);  view_2175 = None
        reshape_default_49: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_49);  sum_dim_int_list_83 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_270, mul_47);  mul_47 = None
        sum_dim_int_list_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_270, [0, 1]);  add_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_86: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1], True);  mul_416 = None
        reshape_default_50: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_50);  sum_dim_int_list_86 = _shape_param_50 = None
        permute_default_50: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2179, [1, 0]);  view_2179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_87: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2193, [0, 1], True);  view_2193 = None
        reshape_default_51: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_51);  sum_dim_int_list_87 = _shape_param_51 = None
        permute_default_51: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2252, [1, 0]);  view_2252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_88: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2256, [0, 1], True);  view_2256 = None
        reshape_default_52: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_52);  sum_dim_int_list_88 = _shape_param_52 = None
        permute_default_52: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2262, [1, 0]);  view_2262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_89: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_146, [0, 1], True);  div_146 = None
        reshape_default_53: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_89, _shape_param_53);  sum_dim_int_list_89 = _shape_param_53 = None
        permute_default_53: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2269, [1, 0]);  view_2269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_18: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_278, mul_40);  mul_40 = None
        sum_dim_int_list_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_54: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2271, [1, 0])
        sum_dim_int_list_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2271, [0], True);  view_2271 = None
        reshape_default_54: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_54);  sum_dim_int_list_92 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_55: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_2274, [1, 0])
        sum_dim_int_list_93: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2274, [0], True);  view_2274 = None
        reshape_default_55: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_55);  sum_dim_int_list_93 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_19: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_281, mul_33);  mul_33 = None
        sum_dim_int_list_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_96: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 1], True);  mul_444 = None
        reshape_default_56: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_56);  sum_dim_int_list_96 = _shape_param_56 = None
        permute_default_56: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2278, [1, 0]);  view_2278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_97: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2292, [0, 1], True);  view_2292 = None
        reshape_default_57: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_57);  sum_dim_int_list_97 = _shape_param_57 = None
        permute_default_57: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2351, [1, 0]);  view_2351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_98: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2355, [0, 1], True);  view_2355 = None
        reshape_default_58: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_98, _shape_param_58);  sum_dim_int_list_98 = _shape_param_58 = None
        permute_default_58: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2361, [1, 0]);  view_2361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_99: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_149, [0, 1], True);  div_149 = None
        reshape_default_59: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, _shape_param_59);  sum_dim_int_list_99 = _shape_param_59 = None
        permute_default_59: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2368, [1, 0]);  view_2368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_20: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_289, mul_26);  mul_26 = None
        sum_dim_int_list_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_289, [0, 1]);  add_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_60: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2370, [1, 0])
        sum_dim_int_list_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2370, [0], True);  view_2370 = None
        reshape_default_60: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_60);  sum_dim_int_list_102 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_61: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_2373, [1, 0])
        sum_dim_int_list_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2373, [0], True);  view_2373 = None
        reshape_default_61: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_61);  sum_dim_int_list_103 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_21: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_292, mul_19);  mul_19 = None
        sum_dim_int_list_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_292, [0, 1]);  add_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_106: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1], True);  mul_472 = None
        reshape_default_62: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, _shape_param_62);  sum_dim_int_list_106 = _shape_param_62 = None
        permute_default_62: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2377, [1, 0]);  view_2377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_107: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2391, [0, 1], True);  view_2391 = None
        reshape_default_63: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_63);  sum_dim_int_list_107 = _shape_param_63 = None
        permute_default_63: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2450, [1, 0]);  view_2450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_108: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2454, [0, 1], True);  view_2454 = None
        reshape_default_64: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_64);  sum_dim_int_list_108 = _shape_param_64 = None
        permute_default_64: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2460, [1, 0]);  view_2460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_109: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_152, [0, 1], True);  div_152 = None
        reshape_default_65: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_109, _shape_param_65);  sum_dim_int_list_109 = _shape_param_65 = None
        permute_default_65: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2467, [1, 0]);  view_2467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_22: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_300, mul_12);  mul_12 = None
        sum_dim_int_list_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_300, [0, 1]);  add_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_66: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2469, [1, 0])
        sum_dim_int_list_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2469, [0], True);  view_2469 = None
        reshape_default_66: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_66);  sum_dim_int_list_112 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_67: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_2472, [1, 0])
        sum_dim_int_list_113: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2472, [0], True);  view_2472 = None
        reshape_default_67: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, _shape_param_67);  sum_dim_int_list_113 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_23: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_303, mul_5);  mul_5 = None
        sum_dim_int_list_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_303, [0, 1]);  add_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_dim_int_list_116: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1], True);  mul_500 = None
        reshape_default_68: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, _shape_param_68);  sum_dim_int_list_116 = _shape_param_68 = None
        permute_default_68: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2476, [1, 0]);  view_2476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_dim_int_list_117: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2490, [0, 1], True);  view_2490 = None
        reshape_default_69: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, _shape_param_69);  sum_dim_int_list_117 = _shape_param_69 = None
        permute_default_69: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2549, [1, 0]);  view_2549 = None
        reshape_default_70: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_187, _shape_param_70);  mm_187 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_dim_int_list_118: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2553, [0, 1], True);  view_2553 = None
        reshape_default_71: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_118, _shape_param_71);  sum_dim_int_list_118 = _shape_param_71 = None
        permute_default_70: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2559, [1, 0]);  view_2559 = None
        reshape_default_72: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_189, _shape_param_72);  mm_189 = _shape_param_72 = None
        add_tensor: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(reshape_default_70, reshape_default_72);  reshape_default_70 = reshape_default_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_dim_int_list_119: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_155, [0, 1], True);  div_155 = None
        reshape_default_73: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_119, _shape_param_73);  sum_dim_int_list_119 = _shape_param_73 = None
        permute_default_71: "f32[768, 8192]" = torch.ops.aten.permute.default(view_2566, [1, 0]);  view_2566 = None
        reshape_default_74: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_191, _shape_param_74);  mm_191 = _shape_param_74 = None
        add_tensor_1: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_74);  add_tensor = reshape_default_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_default_72: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None
        add_tensor_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_497, permute_default_72);  mul_497 = permute_default_72 = None
        return (sum_dim_int_list, sum_dim_int_list_1, permute_default, reshape_default, permute_default_1, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, reshape_default_2, permute_default_2, reshape_default_3, permute_default_3, reshape_default_4, permute_default_4, reshape_default_5, permute_default_5, sum_dim_int_list_10, sum_dim_int_list_11, permute_default_6, reshape_default_6, permute_default_7, reshape_default_7, sum_dim_int_list_14, sum_dim_int_list_15, reshape_default_8, permute_default_8, reshape_default_9, permute_default_9, reshape_default_10, permute_default_10, reshape_default_11, permute_default_11, sum_dim_int_list_20, sum_dim_int_list_21, permute_default_12, reshape_default_12, permute_default_13, reshape_default_13, sum_dim_int_list_24, sum_dim_int_list_25, reshape_default_14, permute_default_14, reshape_default_15, permute_default_15, reshape_default_16, permute_default_16, reshape_default_17, permute_default_17, sum_dim_int_list_30, sum_dim_int_list_31, permute_default_18, reshape_default_18, permute_default_19, reshape_default_19, sum_dim_int_list_34, sum_dim_int_list_35, reshape_default_20, permute_default_20, reshape_default_21, permute_default_21, reshape_default_22, permute_default_22, reshape_default_23, permute_default_23, sum_dim_int_list_40, sum_dim_int_list_41, permute_default_24, reshape_default_24, permute_default_25, reshape_default_25, sum_dim_int_list_44, sum_dim_int_list_45, reshape_default_26, permute_default_26, reshape_default_27, permute_default_27, reshape_default_28, permute_default_28, reshape_default_29, permute_default_29, sum_dim_int_list_50, sum_dim_int_list_51, permute_default_30, reshape_default_30, permute_default_31, reshape_default_31, sum_dim_int_list_54, sum_dim_int_list_55, reshape_default_32, permute_default_32, reshape_default_33, permute_default_33, reshape_default_34, permute_default_34, reshape_default_35, permute_default_35, sum_dim_int_list_60, sum_dim_int_list_61, permute_default_36, reshape_default_36, permute_default_37, reshape_default_37, sum_dim_int_list_64, sum_dim_int_list_65, reshape_default_38, permute_default_38, reshape_default_39, permute_default_39, reshape_default_40, permute_default_40, reshape_default_41, permute_default_41, sum_dim_int_list_70, sum_dim_int_list_71, permute_default_42, reshape_default_42, permute_default_43, reshape_default_43, sum_dim_int_list_74, sum_dim_int_list_75, reshape_default_44, permute_default_44, reshape_default_45, permute_default_45, reshape_default_46, permute_default_46, reshape_default_47, permute_default_47, sum_dim_int_list_80, sum_dim_int_list_81, permute_default_48, reshape_default_48, permute_default_49, reshape_default_49, sum_dim_int_list_84, sum_dim_int_list_85, reshape_default_50, permute_default_50, reshape_default_51, permute_default_51, reshape_default_52, permute_default_52, reshape_default_53, permute_default_53, sum_dim_int_list_90, sum_dim_int_list_91, permute_default_54, reshape_default_54, permute_default_55, reshape_default_55, sum_dim_int_list_94, sum_dim_int_list_95, reshape_default_56, permute_default_56, reshape_default_57, permute_default_57, reshape_default_58, permute_default_58, reshape_default_59, permute_default_59, sum_dim_int_list_100, sum_dim_int_list_101, permute_default_60, reshape_default_60, permute_default_61, reshape_default_61, sum_dim_int_list_104, sum_dim_int_list_105, reshape_default_62, permute_default_62, reshape_default_63, permute_default_63, reshape_default_64, permute_default_64, reshape_default_65, permute_default_65, sum_dim_int_list_110, sum_dim_int_list_111, permute_default_66, reshape_default_66, permute_default_67, reshape_default_67, sum_dim_int_list_114, sum_dim_int_list_115, reshape_default_68, permute_default_68, reshape_default_69, permute_default_69, reshape_default_71, permute_default_70, reshape_default_73, permute_default_71, add_tensor_2)


def _default_make_inputs():
    return [
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_0
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_1
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_2
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_3
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_4
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_5
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_6
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_7
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_8
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_9
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_10
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_11
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_12
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_13
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_14
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_15
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_16
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_17
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_18
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_19
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_20
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_21
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_22
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_23
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_24
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_25
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_26
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_27
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_28
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_29
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_30
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_31
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_32
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_33
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_34
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_35
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_36
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_37
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_38
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_39
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_40
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_41
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_42
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_43
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_44
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_45
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_46
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_47
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_48
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_49
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_50
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_51
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_52
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_53
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_54
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_55
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_56
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_57
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_58
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_59
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_60
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_61
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_62
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_63
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_64
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_65
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_66
    torch.randn([8192, 3072], dtype=torch.float32, device='cuda'),
    [3072],  # _shape_param_67
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_68
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_69
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [1024, 8, 768],  # _shape_param_70
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_71
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [1024, 8, 768],  # _shape_param_72
    torch.randn([1024, 8, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_73
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [1024, 8, 768],  # _shape_param_74
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
