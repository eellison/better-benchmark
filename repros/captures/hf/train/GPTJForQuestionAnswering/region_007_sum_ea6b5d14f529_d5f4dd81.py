"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForQuestionAnswering_train
Pattern hash: ea6b5d14f529
Shape hash: d5f4dd81
"""
_shapes_config = "(T([128, 2], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([4096], f32), T([1, 128, 4096], f32), T([1, 128, 1], f32), T([1, 128, 1], f32), T([1, 128, 4096], f32), T([1, 128], i64), T([], f32), S([2]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 4096]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, view_787: "f32[128, 2]", view_789: "f32[1, 128, 4096]", mul_280: "f32[1, 128, 4096]", view_791: "f32[128, 4096]", view_794: "f32[128, 16384]", view_811: "f32[128, 4096]", view_813: "f32[128, 4096]", view_815: "f32[128, 4096]", add_270: "f32[1, 128, 4096]", mul_270: "f32[1, 128, 4096]", view_817: "f32[128, 4096]", view_820: "f32[128, 16384]", view_837: "f32[128, 4096]", view_839: "f32[128, 4096]", view_841: "f32[128, 4096]", add_282: "f32[1, 128, 4096]", mul_260: "f32[1, 128, 4096]", view_843: "f32[128, 4096]", view_846: "f32[128, 16384]", view_863: "f32[128, 4096]", view_865: "f32[128, 4096]", view_867: "f32[128, 4096]", add_294: "f32[1, 128, 4096]", mul_250: "f32[1, 128, 4096]", view_869: "f32[128, 4096]", view_872: "f32[128, 16384]", view_889: "f32[128, 4096]", view_891: "f32[128, 4096]", view_893: "f32[128, 4096]", add_306: "f32[1, 128, 4096]", mul_240: "f32[1, 128, 4096]", view_895: "f32[128, 4096]", view_898: "f32[128, 16384]", view_915: "f32[128, 4096]", view_917: "f32[128, 4096]", view_919: "f32[128, 4096]", add_318: "f32[1, 128, 4096]", mul_230: "f32[1, 128, 4096]", view_921: "f32[128, 4096]", view_924: "f32[128, 16384]", view_941: "f32[128, 4096]", view_943: "f32[128, 4096]", view_945: "f32[128, 4096]", add_330: "f32[1, 128, 4096]", mul_220: "f32[1, 128, 4096]", view_947: "f32[128, 4096]", view_950: "f32[128, 16384]", view_967: "f32[128, 4096]", view_969: "f32[128, 4096]", view_971: "f32[128, 4096]", add_342: "f32[1, 128, 4096]", mul_210: "f32[1, 128, 4096]", view_973: "f32[128, 4096]", view_976: "f32[128, 16384]", view_993: "f32[128, 4096]", view_995: "f32[128, 4096]", view_997: "f32[128, 4096]", add_354: "f32[1, 128, 4096]", mul_200: "f32[1, 128, 4096]", view_999: "f32[128, 4096]", view_1002: "f32[128, 16384]", view_1019: "f32[128, 4096]", view_1021: "f32[128, 4096]", view_1023: "f32[128, 4096]", add_366: "f32[1, 128, 4096]", mul_190: "f32[1, 128, 4096]", view_1025: "f32[128, 4096]", view_1028: "f32[128, 16384]", view_1045: "f32[128, 4096]", view_1047: "f32[128, 4096]", view_1049: "f32[128, 4096]", add_378: "f32[1, 128, 4096]", mul_180: "f32[1, 128, 4096]", view_1051: "f32[128, 4096]", view_1054: "f32[128, 16384]", view_1071: "f32[128, 4096]", view_1073: "f32[128, 4096]", view_1075: "f32[128, 4096]", add_390: "f32[1, 128, 4096]", mul_170: "f32[1, 128, 4096]", view_1077: "f32[128, 4096]", view_1080: "f32[128, 16384]", view_1097: "f32[128, 4096]", view_1099: "f32[128, 4096]", view_1101: "f32[128, 4096]", add_402: "f32[1, 128, 4096]", mul_160: "f32[1, 128, 4096]", view_1103: "f32[128, 4096]", view_1106: "f32[128, 16384]", view_1123: "f32[128, 4096]", view_1125: "f32[128, 4096]", view_1127: "f32[128, 4096]", add_414: "f32[1, 128, 4096]", mul_150: "f32[1, 128, 4096]", view_1129: "f32[128, 4096]", view_1132: "f32[128, 16384]", view_1149: "f32[128, 4096]", view_1151: "f32[128, 4096]", view_1153: "f32[128, 4096]", add_426: "f32[1, 128, 4096]", mul_140: "f32[1, 128, 4096]", view_1155: "f32[128, 4096]", view_1158: "f32[128, 16384]", view_1175: "f32[128, 4096]", view_1177: "f32[128, 4096]", view_1179: "f32[128, 4096]", add_438: "f32[1, 128, 4096]", mul_130: "f32[1, 128, 4096]", view_1181: "f32[128, 4096]", view_1184: "f32[128, 16384]", view_1201: "f32[128, 4096]", view_1203: "f32[128, 4096]", view_1205: "f32[128, 4096]", add_450: "f32[1, 128, 4096]", mul_120: "f32[1, 128, 4096]", view_1207: "f32[128, 4096]", view_1210: "f32[128, 16384]", view_1227: "f32[128, 4096]", view_1229: "f32[128, 4096]", view_1231: "f32[128, 4096]", add_462: "f32[1, 128, 4096]", mul_110: "f32[1, 128, 4096]", view_1233: "f32[128, 4096]", view_1236: "f32[128, 16384]", view_1253: "f32[128, 4096]", view_1255: "f32[128, 4096]", view_1257: "f32[128, 4096]", add_474: "f32[1, 128, 4096]", mul_100: "f32[1, 128, 4096]", view_1259: "f32[128, 4096]", view_1262: "f32[128, 16384]", view_1279: "f32[128, 4096]", view_1281: "f32[128, 4096]", view_1283: "f32[128, 4096]", add_486: "f32[1, 128, 4096]", mul_90: "f32[1, 128, 4096]", view_1285: "f32[128, 4096]", view_1288: "f32[128, 16384]", view_1305: "f32[128, 4096]", view_1307: "f32[128, 4096]", view_1309: "f32[128, 4096]", add_498: "f32[1, 128, 4096]", mul_80: "f32[1, 128, 4096]", view_1311: "f32[128, 4096]", view_1314: "f32[128, 16384]", view_1331: "f32[128, 4096]", view_1333: "f32[128, 4096]", view_1335: "f32[128, 4096]", add_510: "f32[1, 128, 4096]", mul_70: "f32[1, 128, 4096]", view_1337: "f32[128, 4096]", view_1340: "f32[128, 16384]", view_1357: "f32[128, 4096]", view_1359: "f32[128, 4096]", view_1361: "f32[128, 4096]", add_522: "f32[1, 128, 4096]", mul_60: "f32[1, 128, 4096]", view_1363: "f32[128, 4096]", view_1366: "f32[128, 16384]", view_1383: "f32[128, 4096]", view_1385: "f32[128, 4096]", view_1387: "f32[128, 4096]", add_534: "f32[1, 128, 4096]", mul_50: "f32[1, 128, 4096]", view_1389: "f32[128, 4096]", view_1392: "f32[128, 16384]", view_1409: "f32[128, 4096]", view_1411: "f32[128, 4096]", view_1413: "f32[128, 4096]", add_546: "f32[1, 128, 4096]", mul_40: "f32[1, 128, 4096]", view_1415: "f32[128, 4096]", view_1418: "f32[128, 16384]", view_1435: "f32[128, 4096]", view_1437: "f32[128, 4096]", view_1439: "f32[128, 4096]", add_558: "f32[1, 128, 4096]", mul_30: "f32[1, 128, 4096]", view_1441: "f32[128, 4096]", view_1444: "f32[128, 16384]", view_1461: "f32[128, 4096]", view_1463: "f32[128, 4096]", view_1465: "f32[128, 4096]", add_570: "f32[1, 128, 4096]", mul_20: "f32[1, 128, 4096]", view_1467: "f32[128, 4096]", view_1470: "f32[128, 16384]", view_1487: "f32[128, 4096]", view_1489: "f32[128, 4096]", view_1491: "f32[128, 4096]", add_582: "f32[1, 128, 4096]", mul_10: "f32[1, 128, 4096]", view_1493: "f32[128, 4096]", view_1496: "f32[128, 16384]", mm_440: "f32[128, 4096]", view_1513: "f32[128, 4096]", mm_445: "f32[128, 4096]", view_1515: "f32[128, 4096]", mm_447: "f32[128, 4096]", view_1517: "f32[128, 4096]", mm_449: "f32[128, 4096]", primals_3: "f32[4096]", embedding: "f32[1, 128, 4096]", getitem_1: "f32[1, 128, 1]", rsqrt: "f32[1, 128, 1]", add_583: "f32[1, 128, 4096]", primals_1: "i64[1, 128]", full_default_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:815 in forward, code: logits = self.qa_outputs(sequence_output)
        permute_default: "f32[2, 128]" = torch.ops.aten.permute.default(view_787, [1, 0])
        sum_dim_int_list: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_787, [0], True);  view_787 = None
        reshape_default: "f32[2]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_tensor: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_789, mul_280);  mul_280 = None
        sum_dim_int_list_1: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[4096]" = torch.ops.aten.sum.dim_IntList(view_789, [0, 1]);  view_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_1: "f32[4096, 128]" = torch.ops.aten.permute.default(view_791, [1, 0])
        sum_dim_int_list_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_791, [0], True);  view_791 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_2: "f32[16384, 128]" = torch.ops.aten.permute.default(view_794, [1, 0])
        sum_dim_int_list_4: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_794, [0], True);  view_794 = None
        reshape_default_2: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_3: "f32[4096, 128]" = torch.ops.aten.permute.default(view_811, [1, 0]);  view_811 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_4: "f32[4096, 128]" = torch.ops.aten.permute.default(view_813, [1, 0]);  view_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_5: "f32[4096, 128]" = torch.ops.aten.permute.default(view_815, [1, 0]);  view_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_270, mul_270);  mul_270 = None
        sum_dim_int_list_5: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_6: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_270, [0, 1]);  add_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_6: "f32[4096, 128]" = torch.ops.aten.permute.default(view_817, [1, 0])
        sum_dim_int_list_7: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_817, [0], True);  view_817 = None
        reshape_default_3: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_7: "f32[16384, 128]" = torch.ops.aten.permute.default(view_820, [1, 0])
        sum_dim_int_list_8: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_820, [0], True);  view_820 = None
        reshape_default_4: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_4);  sum_dim_int_list_8 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_8: "f32[4096, 128]" = torch.ops.aten.permute.default(view_837, [1, 0]);  view_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_9: "f32[4096, 128]" = torch.ops.aten.permute.default(view_839, [1, 0]);  view_839 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_10: "f32[4096, 128]" = torch.ops.aten.permute.default(view_841, [1, 0]);  view_841 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_282, mul_260);  mul_260 = None
        sum_dim_int_list_9: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_10: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_282, [0, 1]);  add_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_11: "f32[4096, 128]" = torch.ops.aten.permute.default(view_843, [1, 0])
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_843, [0], True);  view_843 = None
        reshape_default_5: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_12: "f32[16384, 128]" = torch.ops.aten.permute.default(view_846, [1, 0])
        sum_dim_int_list_12: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
        reshape_default_6: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_13: "f32[4096, 128]" = torch.ops.aten.permute.default(view_863, [1, 0]);  view_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_14: "f32[4096, 128]" = torch.ops.aten.permute.default(view_865, [1, 0]);  view_865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_15: "f32[4096, 128]" = torch.ops.aten.permute.default(view_867, [1, 0]);  view_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_294, mul_250);  mul_250 = None
        sum_dim_int_list_13: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_14: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_294, [0, 1]);  add_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_16: "f32[4096, 128]" = torch.ops.aten.permute.default(view_869, [1, 0])
        sum_dim_int_list_15: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_869, [0], True);  view_869 = None
        reshape_default_7: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_7);  sum_dim_int_list_15 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_17: "f32[16384, 128]" = torch.ops.aten.permute.default(view_872, [1, 0])
        sum_dim_int_list_16: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_872, [0], True);  view_872 = None
        reshape_default_8: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_18: "f32[4096, 128]" = torch.ops.aten.permute.default(view_889, [1, 0]);  view_889 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_19: "f32[4096, 128]" = torch.ops.aten.permute.default(view_891, [1, 0]);  view_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_20: "f32[4096, 128]" = torch.ops.aten.permute.default(view_893, [1, 0]);  view_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_4: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_306, mul_240);  mul_240 = None
        sum_dim_int_list_17: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_18: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_306, [0, 1]);  add_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_21: "f32[4096, 128]" = torch.ops.aten.permute.default(view_895, [1, 0])
        sum_dim_int_list_19: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_895, [0], True);  view_895 = None
        reshape_default_9: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_9);  sum_dim_int_list_19 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_22: "f32[16384, 128]" = torch.ops.aten.permute.default(view_898, [1, 0])
        sum_dim_int_list_20: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
        reshape_default_10: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_10);  sum_dim_int_list_20 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_23: "f32[4096, 128]" = torch.ops.aten.permute.default(view_915, [1, 0]);  view_915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_24: "f32[4096, 128]" = torch.ops.aten.permute.default(view_917, [1, 0]);  view_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_25: "f32[4096, 128]" = torch.ops.aten.permute.default(view_919, [1, 0]);  view_919 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_5: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_318, mul_230);  mul_230 = None
        sum_dim_int_list_21: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_22: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_318, [0, 1]);  add_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_26: "f32[4096, 128]" = torch.ops.aten.permute.default(view_921, [1, 0])
        sum_dim_int_list_23: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_921, [0], True);  view_921 = None
        reshape_default_11: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_11);  sum_dim_int_list_23 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_27: "f32[16384, 128]" = torch.ops.aten.permute.default(view_924, [1, 0])
        sum_dim_int_list_24: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_924, [0], True);  view_924 = None
        reshape_default_12: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_12);  sum_dim_int_list_24 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_28: "f32[4096, 128]" = torch.ops.aten.permute.default(view_941, [1, 0]);  view_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_29: "f32[4096, 128]" = torch.ops.aten.permute.default(view_943, [1, 0]);  view_943 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_30: "f32[4096, 128]" = torch.ops.aten.permute.default(view_945, [1, 0]);  view_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_6: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_330, mul_220);  mul_220 = None
        sum_dim_int_list_25: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_26: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_330, [0, 1]);  add_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_31: "f32[4096, 128]" = torch.ops.aten.permute.default(view_947, [1, 0])
        sum_dim_int_list_27: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_947, [0], True);  view_947 = None
        reshape_default_13: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_13);  sum_dim_int_list_27 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_32: "f32[16384, 128]" = torch.ops.aten.permute.default(view_950, [1, 0])
        sum_dim_int_list_28: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_950, [0], True);  view_950 = None
        reshape_default_14: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_14);  sum_dim_int_list_28 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_33: "f32[4096, 128]" = torch.ops.aten.permute.default(view_967, [1, 0]);  view_967 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_34: "f32[4096, 128]" = torch.ops.aten.permute.default(view_969, [1, 0]);  view_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_35: "f32[4096, 128]" = torch.ops.aten.permute.default(view_971, [1, 0]);  view_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_7: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_342, mul_210);  mul_210 = None
        sum_dim_int_list_29: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_30: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_342, [0, 1]);  add_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_36: "f32[4096, 128]" = torch.ops.aten.permute.default(view_973, [1, 0])
        sum_dim_int_list_31: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_973, [0], True);  view_973 = None
        reshape_default_15: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_15);  sum_dim_int_list_31 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_37: "f32[16384, 128]" = torch.ops.aten.permute.default(view_976, [1, 0])
        sum_dim_int_list_32: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_976, [0], True);  view_976 = None
        reshape_default_16: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_16);  sum_dim_int_list_32 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_38: "f32[4096, 128]" = torch.ops.aten.permute.default(view_993, [1, 0]);  view_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_39: "f32[4096, 128]" = torch.ops.aten.permute.default(view_995, [1, 0]);  view_995 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_40: "f32[4096, 128]" = torch.ops.aten.permute.default(view_997, [1, 0]);  view_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_8: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_354, mul_200);  mul_200 = None
        sum_dim_int_list_33: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_34: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_354, [0, 1]);  add_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_41: "f32[4096, 128]" = torch.ops.aten.permute.default(view_999, [1, 0])
        sum_dim_int_list_35: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_999, [0], True);  view_999 = None
        reshape_default_17: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_17);  sum_dim_int_list_35 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_42: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1002, [1, 0])
        sum_dim_int_list_36: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1002, [0], True);  view_1002 = None
        reshape_default_18: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_18);  sum_dim_int_list_36 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_43: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1019, [1, 0]);  view_1019 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_44: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1021, [1, 0]);  view_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_45: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1023, [1, 0]);  view_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_9: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_366, mul_190);  mul_190 = None
        sum_dim_int_list_37: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_38: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_366, [0, 1]);  add_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_46: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1025, [1, 0])
        sum_dim_int_list_39: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1025, [0], True);  view_1025 = None
        reshape_default_19: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_19);  sum_dim_int_list_39 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_47: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1028, [1, 0])
        sum_dim_int_list_40: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1028, [0], True);  view_1028 = None
        reshape_default_20: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_20);  sum_dim_int_list_40 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_48: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1045, [1, 0]);  view_1045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_49: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1047, [1, 0]);  view_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_50: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1049, [1, 0]);  view_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_10: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_378, mul_180);  mul_180 = None
        sum_dim_int_list_41: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_42: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_378, [0, 1]);  add_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_51: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1051, [1, 0])
        sum_dim_int_list_43: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1051, [0], True);  view_1051 = None
        reshape_default_21: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_21);  sum_dim_int_list_43 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_52: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1054, [1, 0])
        sum_dim_int_list_44: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1054, [0], True);  view_1054 = None
        reshape_default_22: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_22);  sum_dim_int_list_44 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_53: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1071, [1, 0]);  view_1071 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_54: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1073, [1, 0]);  view_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_55: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1075, [1, 0]);  view_1075 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_11: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_390, mul_170);  mul_170 = None
        sum_dim_int_list_45: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_46: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_390, [0, 1]);  add_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_56: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1077, [1, 0])
        sum_dim_int_list_47: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1077, [0], True);  view_1077 = None
        reshape_default_23: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_23);  sum_dim_int_list_47 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_57: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1080, [1, 0])
        sum_dim_int_list_48: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1080, [0], True);  view_1080 = None
        reshape_default_24: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_24);  sum_dim_int_list_48 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_58: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1097, [1, 0]);  view_1097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_59: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1099, [1, 0]);  view_1099 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_60: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1101, [1, 0]);  view_1101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_12: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_402, mul_160);  mul_160 = None
        sum_dim_int_list_49: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_50: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_402, [0, 1]);  add_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_61: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1103, [1, 0])
        sum_dim_int_list_51: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1103, [0], True);  view_1103 = None
        reshape_default_25: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_25);  sum_dim_int_list_51 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_62: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1106, [1, 0])
        sum_dim_int_list_52: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1106, [0], True);  view_1106 = None
        reshape_default_26: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_26);  sum_dim_int_list_52 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_63: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1123, [1, 0]);  view_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_64: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1125, [1, 0]);  view_1125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_65: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1127, [1, 0]);  view_1127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_13: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_414, mul_150);  mul_150 = None
        sum_dim_int_list_53: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_54: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_414, [0, 1]);  add_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_66: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1129, [1, 0])
        sum_dim_int_list_55: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1129, [0], True);  view_1129 = None
        reshape_default_27: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_27);  sum_dim_int_list_55 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_67: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1132, [1, 0])
        sum_dim_int_list_56: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True);  view_1132 = None
        reshape_default_28: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_28);  sum_dim_int_list_56 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_68: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1149, [1, 0]);  view_1149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_69: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1151, [1, 0]);  view_1151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_70: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1153, [1, 0]);  view_1153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_14: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_426, mul_140);  mul_140 = None
        sum_dim_int_list_57: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_58: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_426, [0, 1]);  add_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_71: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1155, [1, 0])
        sum_dim_int_list_59: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1155, [0], True);  view_1155 = None
        reshape_default_29: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_29);  sum_dim_int_list_59 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_72: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1158, [1, 0])
        sum_dim_int_list_60: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1158, [0], True);  view_1158 = None
        reshape_default_30: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_30);  sum_dim_int_list_60 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_73: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1175, [1, 0]);  view_1175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_74: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1177, [1, 0]);  view_1177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_75: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1179, [1, 0]);  view_1179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_15: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_438, mul_130);  mul_130 = None
        sum_dim_int_list_61: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_62: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_438, [0, 1]);  add_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_76: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1181, [1, 0])
        sum_dim_int_list_63: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1181, [0], True);  view_1181 = None
        reshape_default_31: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_31);  sum_dim_int_list_63 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_77: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1184, [1, 0])
        sum_dim_int_list_64: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1184, [0], True);  view_1184 = None
        reshape_default_32: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_32);  sum_dim_int_list_64 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_78: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1201, [1, 0]);  view_1201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_79: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1203, [1, 0]);  view_1203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_80: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1205, [1, 0]);  view_1205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_16: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_450, mul_120);  mul_120 = None
        sum_dim_int_list_65: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_66: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_450, [0, 1]);  add_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_81: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1207, [1, 0])
        sum_dim_int_list_67: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1207, [0], True);  view_1207 = None
        reshape_default_33: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_33);  sum_dim_int_list_67 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_82: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1210, [1, 0])
        sum_dim_int_list_68: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
        reshape_default_34: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_34);  sum_dim_int_list_68 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_83: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1227, [1, 0]);  view_1227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_84: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1229, [1, 0]);  view_1229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_85: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1231, [1, 0]);  view_1231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_17: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_462, mul_110);  mul_110 = None
        sum_dim_int_list_69: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_70: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_462, [0, 1]);  add_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_86: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1233, [1, 0])
        sum_dim_int_list_71: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1233, [0], True);  view_1233 = None
        reshape_default_35: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_35);  sum_dim_int_list_71 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_87: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1236, [1, 0])
        sum_dim_int_list_72: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1236, [0], True);  view_1236 = None
        reshape_default_36: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_36);  sum_dim_int_list_72 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_88: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1253, [1, 0]);  view_1253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_89: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1255, [1, 0]);  view_1255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_90: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1257, [1, 0]);  view_1257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_18: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_474, mul_100);  mul_100 = None
        sum_dim_int_list_73: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_74: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_474, [0, 1]);  add_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_91: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1259, [1, 0])
        sum_dim_int_list_75: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1259, [0], True);  view_1259 = None
        reshape_default_37: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_37);  sum_dim_int_list_75 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_92: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1262, [1, 0])
        sum_dim_int_list_76: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1262, [0], True);  view_1262 = None
        reshape_default_38: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_38);  sum_dim_int_list_76 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_93: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1279, [1, 0]);  view_1279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_94: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1281, [1, 0]);  view_1281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_95: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1283, [1, 0]);  view_1283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_19: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_486, mul_90);  mul_90 = None
        sum_dim_int_list_77: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_78: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_486, [0, 1]);  add_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_96: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1285, [1, 0])
        sum_dim_int_list_79: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1285, [0], True);  view_1285 = None
        reshape_default_39: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_39);  sum_dim_int_list_79 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_97: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1288, [1, 0])
        sum_dim_int_list_80: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1288, [0], True);  view_1288 = None
        reshape_default_40: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_40);  sum_dim_int_list_80 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_98: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1305, [1, 0]);  view_1305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_99: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1307, [1, 0]);  view_1307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_100: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1309, [1, 0]);  view_1309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_20: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_498, mul_80);  mul_80 = None
        sum_dim_int_list_81: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_82: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_498, [0, 1]);  add_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_101: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1311, [1, 0])
        sum_dim_int_list_83: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1311, [0], True);  view_1311 = None
        reshape_default_41: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_41);  sum_dim_int_list_83 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_102: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1314, [1, 0])
        sum_dim_int_list_84: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1314, [0], True);  view_1314 = None
        reshape_default_42: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, _shape_param_42);  sum_dim_int_list_84 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_103: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1331, [1, 0]);  view_1331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_104: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1333, [1, 0]);  view_1333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_105: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1335, [1, 0]);  view_1335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_21: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_510, mul_70);  mul_70 = None
        sum_dim_int_list_85: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_86: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_510, [0, 1]);  add_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_106: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1337, [1, 0])
        sum_dim_int_list_87: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1337, [0], True);  view_1337 = None
        reshape_default_43: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_43);  sum_dim_int_list_87 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_107: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1340, [1, 0])
        sum_dim_int_list_88: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1340, [0], True);  view_1340 = None
        reshape_default_44: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_44);  sum_dim_int_list_88 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_108: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1357, [1, 0]);  view_1357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_109: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1359, [1, 0]);  view_1359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_110: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1361, [1, 0]);  view_1361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_22: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_522, mul_60);  mul_60 = None
        sum_dim_int_list_89: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_90: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_522, [0, 1]);  add_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_111: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1363, [1, 0])
        sum_dim_int_list_91: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1363, [0], True);  view_1363 = None
        reshape_default_45: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_45);  sum_dim_int_list_91 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_112: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1366, [1, 0])
        sum_dim_int_list_92: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1366, [0], True);  view_1366 = None
        reshape_default_46: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_46);  sum_dim_int_list_92 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_113: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1383, [1, 0]);  view_1383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_114: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1385, [1, 0]);  view_1385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_115: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1387, [1, 0]);  view_1387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_23: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_534, mul_50);  mul_50 = None
        sum_dim_int_list_93: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_94: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_534, [0, 1]);  add_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_116: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1389, [1, 0])
        sum_dim_int_list_95: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1389, [0], True);  view_1389 = None
        reshape_default_47: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, _shape_param_47);  sum_dim_int_list_95 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_117: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1392, [1, 0])
        sum_dim_int_list_96: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1392, [0], True);  view_1392 = None
        reshape_default_48: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_48);  sum_dim_int_list_96 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_118: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1409, [1, 0]);  view_1409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_119: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1411, [1, 0]);  view_1411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_120: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1413, [1, 0]);  view_1413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_24: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_546, mul_40);  mul_40 = None
        sum_dim_int_list_97: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_98: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_546, [0, 1]);  add_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_121: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1415, [1, 0])
        sum_dim_int_list_99: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1415, [0], True);  view_1415 = None
        reshape_default_49: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, _shape_param_49);  sum_dim_int_list_99 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_122: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1418, [1, 0])
        sum_dim_int_list_100: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1418, [0], True);  view_1418 = None
        reshape_default_50: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_50);  sum_dim_int_list_100 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_123: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1435, [1, 0]);  view_1435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_124: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1437, [1, 0]);  view_1437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_125: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1439, [1, 0]);  view_1439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_25: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_558, mul_30);  mul_30 = None
        sum_dim_int_list_101: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1]);  mul_tensor_25 = None
        sum_dim_int_list_102: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_558, [0, 1]);  add_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_126: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1441, [1, 0])
        sum_dim_int_list_103: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1441, [0], True);  view_1441 = None
        reshape_default_51: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_51);  sum_dim_int_list_103 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_127: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1444, [1, 0])
        sum_dim_int_list_104: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1444, [0], True);  view_1444 = None
        reshape_default_52: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_52);  sum_dim_int_list_104 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_128: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1461, [1, 0]);  view_1461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_129: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1463, [1, 0]);  view_1463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_130: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1465, [1, 0]);  view_1465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_26: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_570, mul_20);  mul_20 = None
        sum_dim_int_list_105: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None
        sum_dim_int_list_106: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_570, [0, 1]);  add_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_131: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1467, [1, 0])
        sum_dim_int_list_107: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1467, [0], True);  view_1467 = None
        reshape_default_53: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_53);  sum_dim_int_list_107 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_132: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1470, [1, 0])
        sum_dim_int_list_108: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1470, [0], True);  view_1470 = None
        reshape_default_54: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_54);  sum_dim_int_list_108 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_133: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1487, [1, 0]);  view_1487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_134: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1489, [1, 0]);  view_1489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_135: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1491, [1, 0]);  view_1491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_27: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_582, mul_10);  mul_10 = None
        sum_dim_int_list_109: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1]);  mul_tensor_27 = None
        sum_dim_int_list_110: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_582, [0, 1]);  add_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_136: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1493, [1, 0])
        sum_dim_int_list_111: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1493, [0], True);  view_1493 = None
        reshape_default_55: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_55);  sum_dim_int_list_111 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_137: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1496, [1, 0])
        sum_dim_int_list_112: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1496, [0], True);  view_1496 = None
        reshape_default_56: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_56);  sum_dim_int_list_112 = _shape_param_56 = None
        reshape_default_57: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_440, _shape_param_57);  mm_440 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_138: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1513, [1, 0]);  view_1513 = None
        reshape_default_58: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_445, _shape_param_58);  mm_445 = _shape_param_58 = None
        add_tensor: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(reshape_default_57, reshape_default_58);  reshape_default_57 = reshape_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_139: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1515, [1, 0]);  view_1515 = None
        reshape_default_59: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_447, _shape_param_59);  mm_447 = _shape_param_59 = None
        add_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_59);  add_tensor = reshape_default_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_140: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1517, [1, 0]);  view_1517 = None
        reshape_default_60: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_449, _shape_param_60);  mm_449 = _shape_param_60 = None
        add_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_60);  add_tensor_1 = reshape_default_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_28: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_2, primals_3);  primals_3 = None
        mul_tensor_29: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_28, 4096)
        sum_dim_int_list_113: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [2], True)
        sub_tensor: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(embedding, getitem_1);  embedding = getitem_1 = None
        mul_tensor_30: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_31: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_28, mul_tensor_30);  mul_tensor_28 = None
        sum_dim_int_list_114: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [2], True);  mul_tensor_31 = None
        mul_tensor_32: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor_30, sum_dim_int_list_114);  sum_dim_int_list_114 = None
        sub_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_tensor_29, sum_dim_int_list_113);  mul_tensor_29 = sum_dim_int_list_113 = None
        sub_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_32);  sub_tensor_1 = mul_tensor_32 = None
        div_tensor: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 4096);  rsqrt = None
        mul_tensor_33: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_34: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_30);  mul_tensor_30 = None
        sum_dim_int_list_115: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1]);  mul_tensor_34 = None
        sum_dim_int_list_116: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1]);  add_tensor_2 = None
        add_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_583, mul_tensor_33);  add_583 = mul_tensor_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:494 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_scalar: "b8[1, 128]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 128, 4096]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, add_tensor_3);  unsqueeze_default = full_default_1 = add_tensor_3 = None
        full_default: "f32[50400, 4096]" = torch.ops.aten.full.default([50400, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[50400, 4096]" = torch.ops.aten.index_put.default(full_default, [primals_1], where_self, True);  full_default = primals_1 = where_self = None
        return (permute_default, reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default_1, reshape_default_1, permute_default_2, reshape_default_2, permute_default_3, permute_default_4, permute_default_5, sum_dim_int_list_5, sum_dim_int_list_6, permute_default_6, reshape_default_3, permute_default_7, reshape_default_4, permute_default_8, permute_default_9, permute_default_10, sum_dim_int_list_9, sum_dim_int_list_10, permute_default_11, reshape_default_5, permute_default_12, reshape_default_6, permute_default_13, permute_default_14, permute_default_15, sum_dim_int_list_13, sum_dim_int_list_14, permute_default_16, reshape_default_7, permute_default_17, reshape_default_8, permute_default_18, permute_default_19, permute_default_20, sum_dim_int_list_17, sum_dim_int_list_18, permute_default_21, reshape_default_9, permute_default_22, reshape_default_10, permute_default_23, permute_default_24, permute_default_25, sum_dim_int_list_21, sum_dim_int_list_22, permute_default_26, reshape_default_11, permute_default_27, reshape_default_12, permute_default_28, permute_default_29, permute_default_30, sum_dim_int_list_25, sum_dim_int_list_26, permute_default_31, reshape_default_13, permute_default_32, reshape_default_14, permute_default_33, permute_default_34, permute_default_35, sum_dim_int_list_29, sum_dim_int_list_30, permute_default_36, reshape_default_15, permute_default_37, reshape_default_16, permute_default_38, permute_default_39, permute_default_40, sum_dim_int_list_33, sum_dim_int_list_34, permute_default_41, reshape_default_17, permute_default_42, reshape_default_18, permute_default_43, permute_default_44, permute_default_45, sum_dim_int_list_37, sum_dim_int_list_38, permute_default_46, reshape_default_19, permute_default_47, reshape_default_20, permute_default_48, permute_default_49, permute_default_50, sum_dim_int_list_41, sum_dim_int_list_42, permute_default_51, reshape_default_21, permute_default_52, reshape_default_22, permute_default_53, permute_default_54, permute_default_55, sum_dim_int_list_45, sum_dim_int_list_46, permute_default_56, reshape_default_23, permute_default_57, reshape_default_24, permute_default_58, permute_default_59, permute_default_60, sum_dim_int_list_49, sum_dim_int_list_50, permute_default_61, reshape_default_25, permute_default_62, reshape_default_26, permute_default_63, permute_default_64, permute_default_65, sum_dim_int_list_53, sum_dim_int_list_54, permute_default_66, reshape_default_27, permute_default_67, reshape_default_28, permute_default_68, permute_default_69, permute_default_70, sum_dim_int_list_57, sum_dim_int_list_58, permute_default_71, reshape_default_29, permute_default_72, reshape_default_30, permute_default_73, permute_default_74, permute_default_75, sum_dim_int_list_61, sum_dim_int_list_62, permute_default_76, reshape_default_31, permute_default_77, reshape_default_32, permute_default_78, permute_default_79, permute_default_80, sum_dim_int_list_65, sum_dim_int_list_66, permute_default_81, reshape_default_33, permute_default_82, reshape_default_34, permute_default_83, permute_default_84, permute_default_85, sum_dim_int_list_69, sum_dim_int_list_70, permute_default_86, reshape_default_35, permute_default_87, reshape_default_36, permute_default_88, permute_default_89, permute_default_90, sum_dim_int_list_73, sum_dim_int_list_74, permute_default_91, reshape_default_37, permute_default_92, reshape_default_38, permute_default_93, permute_default_94, permute_default_95, sum_dim_int_list_77, sum_dim_int_list_78, permute_default_96, reshape_default_39, permute_default_97, reshape_default_40, permute_default_98, permute_default_99, permute_default_100, sum_dim_int_list_81, sum_dim_int_list_82, permute_default_101, reshape_default_41, permute_default_102, reshape_default_42, permute_default_103, permute_default_104, permute_default_105, sum_dim_int_list_85, sum_dim_int_list_86, permute_default_106, reshape_default_43, permute_default_107, reshape_default_44, permute_default_108, permute_default_109, permute_default_110, sum_dim_int_list_89, sum_dim_int_list_90, permute_default_111, reshape_default_45, permute_default_112, reshape_default_46, permute_default_113, permute_default_114, permute_default_115, sum_dim_int_list_93, sum_dim_int_list_94, permute_default_116, reshape_default_47, permute_default_117, reshape_default_48, permute_default_118, permute_default_119, permute_default_120, sum_dim_int_list_97, sum_dim_int_list_98, permute_default_121, reshape_default_49, permute_default_122, reshape_default_50, permute_default_123, permute_default_124, permute_default_125, sum_dim_int_list_101, sum_dim_int_list_102, permute_default_126, reshape_default_51, permute_default_127, reshape_default_52, permute_default_128, permute_default_129, permute_default_130, sum_dim_int_list_105, sum_dim_int_list_106, permute_default_131, reshape_default_53, permute_default_132, reshape_default_54, permute_default_133, permute_default_134, permute_default_135, sum_dim_int_list_109, sum_dim_int_list_110, permute_default_136, reshape_default_55, permute_default_137, reshape_default_56, permute_default_138, permute_default_139, permute_default_140, sum_dim_int_list_115, sum_dim_int_list_116, index_put_default)



def make_inputs():
    return [
    torch.randn([128, 2], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16384], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 4096], dtype=torch.float32, device='cuda'),
    torch.randint(0, 50400, [1, 128], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    [2],  # _shape_param_0
    [4096],  # _shape_param_1
    [16384],  # _shape_param_2
    [4096],  # _shape_param_3
    [16384],  # _shape_param_4
    [4096],  # _shape_param_5
    [16384],  # _shape_param_6
    [4096],  # _shape_param_7
    [16384],  # _shape_param_8
    [4096],  # _shape_param_9
    [16384],  # _shape_param_10
    [4096],  # _shape_param_11
    [16384],  # _shape_param_12
    [4096],  # _shape_param_13
    [16384],  # _shape_param_14
    [4096],  # _shape_param_15
    [16384],  # _shape_param_16
    [4096],  # _shape_param_17
    [16384],  # _shape_param_18
    [4096],  # _shape_param_19
    [16384],  # _shape_param_20
    [4096],  # _shape_param_21
    [16384],  # _shape_param_22
    [4096],  # _shape_param_23
    [16384],  # _shape_param_24
    [4096],  # _shape_param_25
    [16384],  # _shape_param_26
    [4096],  # _shape_param_27
    [16384],  # _shape_param_28
    [4096],  # _shape_param_29
    [16384],  # _shape_param_30
    [4096],  # _shape_param_31
    [16384],  # _shape_param_32
    [4096],  # _shape_param_33
    [16384],  # _shape_param_34
    [4096],  # _shape_param_35
    [16384],  # _shape_param_36
    [4096],  # _shape_param_37
    [16384],  # _shape_param_38
    [4096],  # _shape_param_39
    [16384],  # _shape_param_40
    [4096],  # _shape_param_41
    [16384],  # _shape_param_42
    [4096],  # _shape_param_43
    [16384],  # _shape_param_44
    [4096],  # _shape_param_45
    [16384],  # _shape_param_46
    [4096],  # _shape_param_47
    [16384],  # _shape_param_48
    [4096],  # _shape_param_49
    [16384],  # _shape_param_50
    [4096],  # _shape_param_51
    [16384],  # _shape_param_52
    [4096],  # _shape_param_53
    [16384],  # _shape_param_54
    [4096],  # _shape_param_55
    [16384],  # _shape_param_56
    [1, 128, 4096],  # _shape_param_57
    [1, 128, 4096],  # _shape_param_58
    [1, 128, 4096],  # _shape_param_59
    [1, 128, 4096],  # _shape_param_60
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
